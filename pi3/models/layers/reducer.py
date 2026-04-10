import torch
from typing import Tuple, Callable, Optional
from abc import ABC, abstractmethod


@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b_transposed: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the similarity scores.
    
    Copied from the FastVGGT implementation.
    """

    B, num_src, C = a.shape
    original_dtype = a.dtype

    # Convert to bf16 for computation to improve performance and reduce memory usage
    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    # Process in chunks
    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]  # [B, chunk_size, C]
        scores_chunk = torch.bmm(a_chunk, b_transposed_bf16)
        chunk_max_bf16, chunk_idx = torch.max(scores_chunk, dim=2)
        chunk_max = chunk_max_bf16.to(original_dtype)
        node_max[:, i:end_i] = chunk_max
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


class TokenReducer(ABC):
    """Abstract base class for the token reduction strategy for attention."""
    def __init__(self) -> None:
        # Tracking info for original partitions
        self.src_part = None
        self.dst_part = None

        # Tracking info for tokens
        self.unmerged_idx = None
        self.src_idx = None
        self.dst_idx = None
        self.unmerged_len = None
        self.src_len = None
        self.dst_len = None

        # Additional info to perform the expansion
        self.batch_size = None
        self.num_tokens = None
    
    @abstractmethod
    def partition(self, **kwargs) -> None:
        """Partition the tokens into src and dst groups.
        
        Updates the state of the class self.<src_part/dst_part> which are tensors of indices marked as source and 
        destination tokens, respectively.
        """
        ...

    @abstractmethod
    def reduce(self, tokens, **kwargs) -> torch.Tensor:
        """Perform the token reduction scheme on the given tokens.
        
        Returns the result of token reduction.
        """
        ...

    @abstractmethod
    def expand(self, tokens, **kwargs) -> torch.Tensor:
        """Expand the reduced tokens back to the un-reduced state.
        
        Returns the expanded tokens which match the shape of the tokens before 
        reduction.
        """
        ...


class FastVGGTMerging(TokenReducer):
    def __init__(self):
        super().__init__()

    def partition(self, **kwargs) -> None:
        """Partition the tokens into src and dst groups according to the 
        FastVGGT paper (https://arxiv.org/pdf/2509.02560).

        Note that the implementation isn't exactly one-to-one due to
        architectural differences, but preserves the main idea. This
        also skips on some of the optional features like protected tokens.

        Keyword Arguments:
            - width: the width of the image in tokens
            - height: the height of the image in tokens
            - sx: dst stride in x dimension, must divide w evenly
            - sy: dst stride in y dimension, must divide h evenly
            - N: total tokens
            - generator: a torch Generator for random sampling, fixed sampling
                         if no generator is provided
            - device: the device to put the tensors on
        """
        width = kwargs['width']
        height = kwargs['height']
        sx = kwargs['sx']
        sy = kwargs['sy']
        N = kwargs['N']
        generator = kwargs.get('generator', None)
        device = kwargs.get('device', 'cpu')

        tokens_per_image = width * height + 5
        num_images = N // tokens_per_image
        assert tokens_per_image * num_images == N, "Token count doesn't match (w*h+5)*num_imgs"

        with torch.no_grad():
            # Global idx_buffer_seq of length N; -1 indicates dst, 0 indicates src (maintain original logic)
            idx_buffer_seq = torch.zeros(N, device=device, dtype=torch.int64)
            hsy, wsx = height // sy, width // sx  # Number of blocks within each image

            # Mark register tokens as dst
            reg_indices = torch.arange(num_images, device=device) * tokens_per_image
            reg_indices = reg_indices[:, None] + torch.arange(5, device=device)
            idx_buffer_seq[reg_indices.flatten()] = -1

            effective_h = min(hsy * sy, height)
            effective_w = min(wsx * sx, width)
            effective_grid_size = effective_h * effective_w

            # Randomly select within grid patches if generator is provided
            if generator is None:
                base_pattern = torch.zeros(
                    effective_grid_size, device=device, dtype=torch.int64
                )
                grid_starts = (
                    torch.arange(num_images, device=device) * tokens_per_image + 5
                )
                grid_indices = grid_starts[:, None] + torch.arange(
                    effective_grid_size, device=device
                )
                idx_buffer_seq[grid_indices.flatten()] = base_pattern.repeat(
                    num_images
                )
            else:
                all_rand_idx = torch.randint(
                    sy * sx,
                    size=(num_images, hsy, wsx),
                    device=device,
                    generator=generator,
                )

                scatter_src = -torch.ones(
                    num_images, hsy, wsx, device=device, dtype=torch.int64
                )

                idx_buffer_batch = torch.zeros(
                    num_images,
                    hsy,
                    wsx,
                    sy * sx,
                    device=device,
                    dtype=torch.int64,
                )
                idx_buffer_batch.scatter_(
                    dim=3,
                    index=all_rand_idx.unsqueeze(-1),
                    src=scatter_src.unsqueeze(-1),
                )

                idx_buffer_batch = (
                    idx_buffer_batch.view(num_images, hsy, wsx, sy, sx)
                    .transpose(2, 3)
                    .reshape(num_images, hsy * sy, wsx * sx)
                )

                # Batch fill to target positions - still needs a small loop here, but operations are greatly reduced
                for i in range(num_images):
                    grid_start = i * num_images + 5
                    flat_view = idx_buffer_batch[
                        i, :effective_h, :effective_w
                    ].flatten()
                    idx_buffer_seq[grid_start : grid_start + effective_grid_size] = flat_view
            
            idx_split = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
            self.dst_len = int((idx_buffer_seq == - 1).sum())
            self.src_part = idx_split[:, self.dst_len:, :]
            self.dst_part = idx_split[:, :self.dst_len, :]

    def reduce(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns the result of merging tokens in `tokens` based on the merging scheme used in FastVGGT.

        Keyword Arguments:
            - mode: a str to denote the reduction type for scatter_reduce
            - tokens_to_remove: the number of tokens to remove
        """
        mode = kwargs.get('mode', 'mean')
        self.batch_size, self.num_tokens, feature_dim = tokens.shape
        num_src, num_dst = self.src_part.shape[1], self.dst_len

        with torch.no_grad():
            # Find similar dst token for each src token
            tokens = tokens / tokens.norm(dim=-1, keepdim=True)
            src = torch.gather(tokens, dim=1, index=self.src_part.expand(self.batch_size, num_src, feature_dim))
            dst = torch.gather(tokens, dim=1, index=self.dst_part.expand(self.batch_size, num_dst, feature_dim))
            num_src_actual = src.shape[1]

            tokens_to_remove = min(num_src_actual, kwargs['tokens_to_remove'])
            chunk_size = min(5000, num_src_actual)

            node_max = torch.empty(self.batch_size, num_src_actual, device=src.device, dtype=src.dtype)

            dst_transpose = dst.transpose(-1, -2)
            node_max, node_idx = fast_similarity_chunks(src, dst_transpose, chunk_size)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            self.unmerged_idx = edge_idx[..., tokens_to_remove:, :]
            self.src_idx = edge_idx[..., :tokens_to_remove, :]
            self.dst_idx = torch.gather(node_idx[..., None], dim=-2, index=self.src_idx)

            n, _, c = src.shape
            self.unmerged_len = self.unmerged_idx.shape[1]
            unmerged = torch.gather(src, dim=-2, index=self.unmerged_idx.expand(n, self.unmerged_len, c))
            self.src_len = self.src_idx.shape[1]
            src = torch.gather(src, dim=-2, index=self.src_idx.expand(n, self.src_len, c))
            dst = dst.scatter_reduce(-2, self.dst_idx.expand(n, self.src_len, c), src, reduce=mode)
            merged = torch.cat([unmerged, dst], dim=1)

        return merged

    def expand(self, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """Perform the unmerging procedure for the merged tokens.

        Follows the unmerging procedure in FastVGGT.
        """
        with torch.no_grad():
            unmerged = tokens[..., :self.unmerged_len, :]
            dst = tokens[..., self.unmerged_len : self.unmerged_len + self.dst_len, :]

            _, _, c = unmerged.shape
            src = torch.gather(dst, dim=-2, index=self.dst_idx.expand(self.batch_size, self.src_len, c))
            out = torch.zeros(self.batch_size, self.num_tokens, c, device=tokens.device, dtype=tokens.dtype)
            out.scatter_(dim=-2, index=self.dst_part.expand(self.batch_size, self.dst_len, c), src=dst)
            out.scatter_(
                dim=-2,
                index=torch.gather(
                    self.src_part.expand(self.batch_size, self.src_part.shape[1], 1), dim=1, index=self.unmerged_idx
                ).expand(self.batch_size, self.unmerged_len, c),
                src=unmerged,
            )

            out.scatter_(
                dim=-2,
                index=torch.gather(
                    self.src_part.expand(self.batch_size, self.src_part.shape[1], 1), dim=1, index=self.src_idx
                ).expand(self.batch_size, self.src_len, c),
                src=src,
            )

        return out

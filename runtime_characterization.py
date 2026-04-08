import torch
from torch.profiler import profile, ProfilerActivity, record_function
import torchvision.models as models
import argparse
import numpy as np
import os
from pi3.utils.basic import load_multimodal_data
from pi3.models.pi3x import Pi3X

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    
    # parser.add_argument("--conditions_path", type=str, default='examples/room/condition.npz',
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Optional path to a .npz file containing 'poses', 'depths', 'intrinsics'.")

    parser.add_argument("--save_dir", type=str, default='output',
                        help="Directory to save characterization results")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare input data
    device = torch.device(args.device)

    # Load optional conditions from .npz
    poses = None
    depths = None
    intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        print(f"Loading conditions from {args.conditions_path}...")
        data_npz = np.load(args.conditions_path, allow_pickle=True)

        poses = data_npz['poses']             # Expected (N, 4, 4) OpenCV camera-to-world
        depths = data_npz['depths']           # Expected (N, H, W)
        intrinsics = data_npz['intrinsics']   # Expected (N, 3, 3)

    conditions = dict(
        intrinsics=intrinsics,
        poses=poses,
        depths=depths
    )

    # Load images (Required)
    imgs, conditions = load_multimodal_data(args.data_path, conditions, interval=args.interval, device=device) 
    use_multimodal = any(v is not None for v in conditions.values())
    if not use_multimodal:
        print("No multimodal conditions found. Disable multimodal branch to reduce memory usage.")

    # 2. Prepare model
    print(f"Loading model...")
    if args.ckpt is not None:
        model = Pi3X(use_multimodal=use_multimodal).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`
        if not use_multimodal:
            model.disable_multimodal()
    model = model.to(device)

    # 3. Infer
    print("Running model inference...")
    activities = [ProfilerActivity.CPU]
    dtype = torch.float16
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        activities += [ProfilerActivity.CUDA]

    sort_by_keyword = device.type + "_time_total"
    
    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=dtype):
            with profile(activities=activities, profile_memory=True) as prof:
                with record_function("model_inference"):
                    model(imgs=imgs[None], **conditions)

    # Save the stats
    os.makedirs(args.save_dir, exist_ok=True)
    stats = prof.key_averages().table(sort_by=sort_by_keyword, row_limit=-1)

    with open(os.path.join(args.save_dir, 'summary.txt'), "w") as f:
        f.write(stats)
        f.close()

    prof.export_chrome_trace(os.path.join(args.save_dir, 'trace.json'))

    print("Reconstruction complete!")

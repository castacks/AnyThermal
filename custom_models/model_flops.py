#!/usr/bin/env python3
import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---- Optional FLOPs dependency (MACs + Params). Install with: pip install thop
try:
    from thop import profile as thop_profile
    HAVE_THOP = True
except Exception:
    HAVE_THOP = False

from typing import Tuple
from torchvision import models as tvm
import sys
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc") # Add the parent directory to the path
sys.path.append("/ocean/projects/cis220039p/pmaheshw/code/multi-modal/MultiLoc") # Add the parent directory to the path
from custom_models.str_to_cls import get_model_from_string

@dataclass
class BenchResult:
    name: str
    params_m: float
    macs_g: float  # multiply-accumulate ops in billions
    flops_g: float # often ~2 * MACs, but reported separately
    fps: float
    latency_ms: float

def human(n: float) -> str:
    if n is None:
        return "N/A"
    if n >= 1e12: return f"{n/1e12:.2f}T"
    if n >= 1e9:  return f"{n/1e9:.2f}G"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.2f}K"
    return f"{n:.2f}"

@torch.no_grad()
def measure_fps(model: nn.Module, input_tensor: torch.Tensor, iters: int, warmup: int, device: str) -> Tuple[float, float]:
    model.eval()
    # Warmup (helps stabilize CUDA clocks & caching)
    if warmup > 0:
        for _ in range(warmup):
            _ = model.forward(input_tensor)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
    # Timed run
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model.forward(input_tensor)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_time = t1 - t0
    fps = (iters * input_tensor.shape[0]) / total_time
    latency_ms = (total_time / iters) * 1000.0
    return fps, latency_ms

@torch.no_grad()
def measure_flops_thop(model: nn.Module, input_tensor: torch.Tensor) -> Tuple[float , float , float]:
    """
    Returns (MACs_G, FLOPs_G, Params_M).
    Note: Many papers report MACs, some report FLOPs ~= 2*MACs for conv/GEMM.
    """
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    if not HAVE_THOP:
        print(f"[WARN] 'thop' not installed. Cannot measure MACs/FLOPs for model {model.__class__.__name__}.")
        return None, None, params_m
    try:
        macs, params = thop_profile(model, inputs=(input_tensor,), verbose=False)
        macs_g = macs / 1e9
        # A common convention (not universal): FLOPs ≈ 2 * MACs for multiply+add.
        flops_g = (2.0 * macs) / 1e9
        return macs_g, flops_g, params / 1e6
    except Exception as e:
        print(f"[WARN] Failed to measure FLOPs/MACs for model {model.__class__.__name__}. Error: {e}") 
        # Some models (dynamic control flow / unusual ops) may not be supported by thop.
        return None, None, params_m

def main():
    parser = argparse.ArgumentParser(description="Benchmark FPS and FLOPs/MACs for multiple models.")
    parser.add_argument("--models",  type=str, nargs='+', required=True,
                        help="Space-separated model names")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=9)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--channels", type=int, default=1, help="Input channels (default 3 for Thermal)")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                        help="Inference dtype cast (affects speed; FLOPs unchanged).")
    args = parser.parse_args()

    model_names = args.models
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    # Prepare input
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    # Some models (e.g., ViT) expect sizes multiple of patch size.
    # This script will still run, but you may get shape errors. Adjust H,W if needed.
    results: list[BenchResult] = []
    for name in model_names:
        model = get_model_from_string(args,name,task='segmentation', num_classes=args.num_classes)
        model.eval()

        # Build a single dummy input tensor
        x = torch.randn(args.batch_size, args.channels, args.height, args.width, device=device)
        # Cast input and (optionally) weights for mixed precision timing
        if device == "cuda" and args.dtype in ("fp16", "bf16"):
            # Cast model weights (inference only)
            if args.dtype == "fp16":
                model = model.half()
                x = x.half()
            else:  # bf16
                model = model.to(torch.bfloat16)
                x = x.to(torch.bfloat16)

        # FLOPs/MACs (always measured with current dtype; numerically the same)
        macs_g, flops_g, params_m = measure_flops_thop(model, x)

        # FPS
        fps, latency_ms = measure_fps(model, x, args.iters, args.warmup, device)
        results.append(BenchResult(
            name=name,
            params_m=params_m,
            macs_g=macs_g,
            flops_g=flops_g,
            fps=fps,
            latency_ms=latency_ms,
        ))

        # Free up memory between models
        del model, x
        if device == "cuda":
            torch.cuda.empty_cache()

    # Pretty print results
    if not results:
        return

    print("\n==== Benchmark Results ====")
    print(f"Input: BxCxHxW = {args.batch_size}x{args.channels}x{args.height}x{args.width}, "
          f"Device: {device}, DType: {args.dtype}, Iters: {args.iters}, Warmup: {args.warmup}")
    header = f"{'Model':28} {'Params(M)':>10} {'MACs(G)':>10} {'FLOPs(G)':>10} {'FPS':>10} {'Latency(ms)':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:28} {human(r.params_m):>10} {human(r.macs_g) if r.macs_g is not None else 'N/A':>10} "
              f"{human(r.flops_g) if r.flops_g is not None else 'N/A':>10} "
              f"{r.fps:>10.2f} {r.latency_ms:>12.3f}")

    if not HAVE_THOP:
        print("\nNote: 'thop' not installed. Install it with `pip install thop` to get MACs/FLOPs.")

if __name__ == "__main__":
    main()

'''
Example usage:
python bench_models.py --models resnet18,resnet50,vit_b_16 --height 480 --width 640 --batch-size 1 --device cuda --iters 100 --warmup 20
'''
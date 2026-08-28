"""Real Marigold (prs-eth/marigold-depth-v1-1) diffusion depth inference.

No mock fallbacks: any failure raises RuntimeError. Mirrors exactly the API
that was executed successfully on a real T4 GPU (4 views, 72.77s total runtime,
real per-view diffusion depth predictions).
"""
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("geodiff3d.marigold")

MARIGOLD_CHECKPOINT = "prs-eth/marigold-depth-v1-1"
NUM_INFERENCE_STEPS = 4  # practical T4 POC setting; see README for quality tradeoff


def load_marigold_pipeline(device: torch.device):
    try:
        from diffusers import MarigoldDepthPipeline
    except ImportError as e:
        raise RuntimeError(
            "The `diffusers` package (>=0.35.0, with Marigold support) is not "
            "installed. Install it before running real GPU inference (no "
            "fallback is used)."
        ) from e

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype}
    if device.type == "cuda":
        kwargs["variant"] = "fp16"

    try:
        pipe = MarigoldDepthPipeline.from_pretrained(MARIGOLD_CHECKPOINT, **kwargs).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load Marigold checkpoint {MARIGOLD_CHECKPOINT}: {e}") from e
    return pipe, dtype


def run_marigold(pipe, image_paths, device: torch.device, num_inference_steps: int = NUM_INFERENCE_STEPS):
    """Run Marigold independently on each image path (monocular, per-view).

    Returns (depths, metadata) where depths is a list of float32 arrays
    preserving image_paths order. Raises RuntimeError on any failure.
    """
    depths = []
    per_view_runtime = []
    start_total = time.time()
    try:
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            t0 = time.time()
            with torch.inference_mode():
                result = pipe(img, num_inference_steps=num_inference_steps, ensemble_size=1)
            pred = result.prediction
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().float().cpu().numpy()
            pred = np.asarray(pred)
            # Diffusers documents numpy output as [N, H, W, 1]; handle tensor layout too.
            if pred.ndim == 4 and pred.shape[-1] == 1:
                pred = pred[0, ..., 0]
            elif pred.ndim == 4 and pred.shape[1] == 1:
                pred = pred[0, 0]
            elif pred.ndim == 3:
                pred = pred[0]
            else:
                raise RuntimeError(f"Unexpected Marigold prediction shape: {pred.shape}")
            depths.append(pred.astype(np.float32))
            per_view_runtime.append(round(time.time() - t0, 3))
            del result, pred, img
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(f"Marigold inference failed: {e}") from e

    runtime = time.time() - start_total
    metadata = {
        "model": MARIGOLD_CHECKPOINT,
        "device": str(device),
        "num_views": len(image_paths),
        "num_inference_steps": num_inference_steps,
        "runtime_sec": round(runtime, 3),
        "per_view_runtime_sec": per_view_runtime,
        "gpu_memory_allocated_gb": (
            round(torch.cuda.memory_allocated(device) / 1024**3, 3) if device.type == "cuda" else None
        ),
    }
    logger.info("Marigold complete: %s", metadata)
    return depths, metadata


def run_marigold_and_save(image_paths, output_dir, device: torch.device):
    """Load Marigold, run inference, persist per-view depth, then free the
    pipeline and empty the CUDA cache."""
    output_dir = Path(output_dir)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)

    pipe, dtype = load_marigold_pipeline(device)
    try:
        depths, metadata = run_marigold(pipe, image_paths, device)
    finally:
        del pipe
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata["dtype"] = str(dtype)
    for i, d in enumerate(depths):
        np.save(output_dir / "depth" / f"view_{i:03d}.npy", d)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return depths, metadata

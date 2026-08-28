"""Real facebook/VGGT-1B inference. No mock fallbacks: any failure raises RuntimeError.

This mirrors exactly the API that was executed successfully on a real T4 GPU
(VGGT input (4, 3, 350, 518), 295.45s runtime, real depth/confidence/camera
outputs). Requires the official VGGT repository (github.com/facebookresearch/vggt)
to be importable and `facebook/VGGT-1B` reachable on the HF Hub.
"""
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("geodiff3d.vggt")

VGGT_CHECKPOINT = "facebook/VGGT-1B"


def _vggt_dtype(device: torch.device) -> torch.dtype:
    """T4 (compute capability 7.5) cannot use bfloat16 efficiently; Ampere+ (>=8.0) can."""
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def load_vggt_model(device: torch.device):
    try:
        from vggt.models.vggt import VGGT
    except ImportError as e:
        raise RuntimeError(
            "The `vggt` package is not importable. Clone "
            "https://github.com/facebookresearch/vggt and add it to PYTHONPATH "
            "before running real GPU inference (no fallback is used)."
        ) from e

    model = VGGT.from_pretrained(VGGT_CHECKPOINT).to(device)
    model.eval()
    return model


def run_vggt(model, image_paths, device: torch.device):
    """Run VGGT on a list of image paths using the official preprocessing.

    Returns a dict: depth (S,H,W), confidence (S,H,W), extrinsic (S,3,4),
    intrinsic (S,3,3), point_maps (S,H,W,3) or None, and metadata. Raises
    RuntimeError on any failure -- never fabricates output.
    """
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dtype = _vggt_dtype(device)
    start = time.time()
    try:
        images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                predictions = model(images)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

        depth = predictions["depth"].detach().float().cpu().numpy().squeeze(0)[..., 0]
        confidence = predictions["depth_conf"].detach().float().cpu().numpy().squeeze(0)
        extrinsic_np = extrinsic.detach().float().cpu().numpy().squeeze(0)
        intrinsic_np = intrinsic.detach().float().cpu().numpy().squeeze(0)
        world_points = predictions.get("world_points")
        point_maps_np = (
            world_points.detach().float().cpu().numpy().squeeze(0) if world_points is not None else None
        )
        image_shape = tuple(int(x) for x in images.shape[-2:])
    except Exception as e:
        raise RuntimeError(f"VGGT inference failed: {e}") from e
    finally:
        try:
            del images, predictions
        except NameError:
            pass

    runtime = time.time() - start
    metadata = {
        "model": VGGT_CHECKPOINT,
        "device": str(device),
        "dtype": str(dtype),
        "num_views": len(image_paths),
        "image_shape": image_shape,
        "runtime_sec": round(runtime, 3),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_allocated_gb": (
            round(torch.cuda.memory_allocated(device) / 1024**3, 3) if device.type == "cuda" else None
        ),
    }
    logger.info("VGGT complete: %s", metadata)

    return {
        "depth": depth,
        "confidence": confidence,
        "extrinsic": extrinsic_np,
        "intrinsic": intrinsic_np,
        "point_maps": point_maps_np,
        "metadata": metadata,
    }


def run_vggt_and_save(image_paths, output_dir, device: torch.device):
    """Load VGGT, run inference, persist depth/confidence/camera outputs, then
    free the ~1B parameter model and empty the CUDA cache. Returns the same
    dict as `run_vggt` so the caller can continue in-memory without re-reading
    from disk.
    """
    output_dir = Path(output_dir)
    for sub in ("depth", "confidence", "cameras"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    model = load_vggt_model(device)
    try:
        result = run_vggt(model, image_paths, device)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for i in range(len(image_paths)):
        np.save(output_dir / "depth" / f"view_{i:03d}.npy", result["depth"][i])
        np.save(output_dir / "confidence" / f"view_{i:03d}.npy", result["confidence"][i])
        np.save(output_dir / "cameras" / f"extrinsic_{i:03d}.npy", result["extrinsic"][i])
        np.save(output_dir / "cameras" / f"intrinsic_{i:03d}.npy", result["intrinsic"][i])
        if result["point_maps"] is not None:
            (output_dir / "point_maps").mkdir(exist_ok=True)
            np.save(output_dir / "point_maps" / f"view_{i:03d}.npy", result["point_maps"][i])

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(result["metadata"], f, indent=2)

    return result

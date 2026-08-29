"""Caches raw VGGT + Marigold outputs per scene so `fuse_depths`'
`trust_threshold` / `max_aligned_weight` can be tuned entirely offline (pure
CPU, no GPU) against the real `cross_view_consistency` metric, instead of
re-running GPU inference for every candidate parameter combination -- Marigold's
diffusion sampling isn't seeded, so repeated GPU runs aren't bit-identical
anyway; caching one real realization per scene and reusing it for every
candidate is both cheaper and more apples-to-apples than re-running per combo.

Run on a CUDA GPU (same scenes as experiments/run_ablation.py):

    VGGT_REPO_DIR=/content/vggt python experiments/cache_fusion_inputs.py

Writes, per scene, under experiments/tuning_cache/<scene>/:
    vggt_depth.npy              (S, H, W) float32
    vggt_confidence.npy         (S, H, W) float32, raw (pre-normalize_confidence)
    aligned_marigold_depth.npy  (S, H, W) float32, already scale/shift-aligned into VGGT's depth
    extrinsic.npy               (S, 3, 4) float32
    intrinsic.npy               (S, 3, 3) float32

Then run experiments/tune_fusion.py locally (no GPU needed) to grid-search
trust_threshold / max_aligned_weight against this cache.
"""
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.math import align_depth, resize_depth_to
from inference.vggt_pipeline import load_vggt_model, run_vggt
from inference.marigold_pipeline import load_marigold_pipeline, run_marigold
from experiments.run_ablation import SCENES, REPO_ROOT

logger = logging.getLogger("geodiff3d.cache_fusion_inputs")

CACHE_ROOT = REPO_ROOT / "experiments" / "tuning_cache"


def cache_scene(image_paths, out_dir: Path, device) -> None:
    image_paths = [Path(p) for p in image_paths]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running VGGT")
    vggt_model = load_vggt_model(device)
    try:
        vggt_result = run_vggt(vggt_model, image_paths, device)
    finally:
        del vggt_model
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Running Marigold")
    marigold_pipe, _ = load_marigold_pipeline(device)
    try:
        marigold_depths, _ = run_marigold(marigold_pipe, image_paths, device)
    finally:
        del marigold_pipe
        gc.collect()
        torch.cuda.empty_cache()

    vggt_depth = vggt_result["depth"]
    aligned_depths = []
    for i in range(len(image_paths)):
        md = marigold_depths[i]
        vd = vggt_depth[i]
        if md.shape != vd.shape:
            md = resize_depth_to(md, vd.shape)
        aligned, *_ = align_depth(vd, md)
        aligned_depths.append(aligned)

    np.save(out_dir / "vggt_depth.npy", vggt_depth)
    np.save(out_dir / "vggt_confidence.npy", vggt_result["confidence"])
    np.save(out_dir / "aligned_marigold_depth.npy", np.stack(aligned_depths))
    np.save(out_dir / "extrinsic.npy", vggt_result["extrinsic"])
    np.save(out_dir / "intrinsic.npy", vggt_result["intrinsic"])
    logger.info("Cached %s (%d views)", out_dir.name, len(image_paths))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This requires a CUDA GPU -- run on Colab with a T4 runtime.")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for scene_name, image_paths in SCENES.items():
        print(f"\n=== Caching scene: {scene_name} ({len(image_paths)} views) ===")
        cache_scene([str(p) for p in image_paths], CACHE_ROOT / scene_name, device)

    print(f"\nDone. Cached inputs under {CACHE_ROOT}")


if __name__ == "__main__":
    main()

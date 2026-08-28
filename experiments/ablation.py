"""Phase 4: scientific ablation comparing four reconstruction strategies on
real multi-view photographs.

VGGT and Marigold are each run exactly once per scene, and all four methods
below are derived from that single shared pair of runs -- this keeps the
comparison fair (identical inputs, identical preprocessing, identical VGGT
camera geometry) and avoids paying for four independent GPU passes.

    1. vggt_only        -- VGGT depth, no diffusion prior at all.
    2. marigold_only     -- Marigold depth aligned into VGGT's scale (VGGT's
                             cameras are reused, since Marigold alone produces
                             no multi-view camera geometry to reconstruct in a
                             shared coordinate frame).
    3. naive_average     -- unweighted 0.5/0.5 blend of VGGT depth and aligned
                             Marigold depth (no confidence signal).
    4. geodiff3d_fusion  -- VGGT-confidence-guided fusion (the actual
                             GeoDiff3D method also used by the backend).

No ground truth exists for these scenes (they are real photographs with no
captured 3D scan), so only self-consistency diagnostics are reported --
point-cloud statistics and cross-view reprojection consistency. Nothing here
claims an accuracy improvement; that would require ground truth this repo
does not have.
"""
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.math import (
    align_depth,
    fuse_depths,
    normalize_confidence,
    point_cloud_stats,
    resize_depth_to,
    save_ply,
    unproject_to_point_cloud,
)
from inference.vggt_pipeline import load_vggt_model, run_vggt
from inference.marigold_pipeline import load_marigold_pipeline, run_marigold

logger = logging.getLogger("geodiff3d.ablation")

METHODS = ["vggt_only", "marigold_only", "naive_average", "geodiff3d_fusion"]


def cross_view_consistency(depths, extrinsics, intrinsics) -> Optional[float]:
    """Self-consistency diagnostic requiring no ground truth: for each pair of
    adjacent views, unproject view i's depth to world points, reproject into
    view j's camera, and compare against view j's own depth at that pixel
    (nearest-neighbor sampling). Returns the mean absolute relative depth
    error over all view pairs with any visible overlap, or None if no pair
    overlapped at all.
    """
    n = len(depths)
    pair_errors = []
    for i, j in [(k, k + 1) for k in range(n - 1)]:
        h, w = depths[j].shape
        yy, xx = np.meshgrid(
            np.arange(depths[i].shape[0], dtype=np.float32),
            np.arange(depths[i].shape[1], dtype=np.float32),
            indexing="ij",
        )
        z = depths[i]
        valid = np.isfinite(z) & (z > 0)
        if not np.any(valid):
            continue

        Ki = intrinsics[i]
        x = (xx[valid] - Ki[0, 2]) * z[valid] / Ki[0, 0]
        y = (yy[valid] - Ki[1, 2]) * z[valid] / Ki[1, 1]
        pts_cam_i = np.stack([x, y, z[valid]], axis=1)

        Ri, ti = extrinsics[i][:3, :3], extrinsics[i][:3, 3]
        pts_world = (pts_cam_i - ti[None, :]) @ Ri

        Rj, tj = extrinsics[j][:3, :3], extrinsics[j][:3, 3]
        pts_cam_j = (pts_world @ Rj.T) + tj[None, :]
        zj = pts_cam_j[:, 2]
        in_front = zj > 1e-4

        Kj = intrinsics[j]
        u = Kj[0, 0] * pts_cam_j[:, 0] / np.clip(zj, 1e-6, None) + Kj[0, 2]
        v = Kj[1, 1] * pts_cam_j[:, 1] / np.clip(zj, 1e-6, None) + Kj[1, 2]
        in_bounds = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_bounds):
            continue

        u_i = np.clip(u[in_bounds].astype(np.int32), 0, w - 1)
        v_i = np.clip(v[in_bounds].astype(np.int32), 0, h - 1)
        own_depth = depths[j][v_i, u_i]
        pred_depth = zj[in_bounds]

        valid2 = np.isfinite(own_depth) & (own_depth > 0)
        if not np.any(valid2):
            continue
        rel_err = np.abs(pred_depth[valid2] - own_depth[valid2]) / own_depth[valid2]
        pair_errors.append(float(np.mean(rel_err)))

    return float(np.mean(pair_errors)) if pair_errors else None


def _reconstruct(depths, image_paths, K_list, E_list):
    points, colors = [], []
    for i, p in enumerate(image_paths):
        rgb = np.asarray(Image.open(p).convert("RGB"))
        pts, col = unproject_to_point_cloud(depths[i], rgb, K_list[i], E_list[i])
        points.append(pts)
        colors.append(col)
    return np.vstack(points), np.vstack(colors)


def _save_comparison_viz(vggt_depth, aligned_depths, naive_depths, fused_depths, out_path: Path):
    import matplotlib.pyplot as plt

    rows = [
        (vggt_depth, "VGGT-only"),
        (aligned_depths, "Marigold-only (aligned)"),
        (naive_depths, "Naive 50/50"),
        (fused_depths, "GeoDiff3D fusion"),
    ]
    s = len(aligned_depths)
    fig, axes = plt.subplots(len(rows), s, figsize=(4 * s, 3.5 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (depths, label) in enumerate(rows):
        for i in range(s):
            axes[r, i].imshow(depths[i], cmap="viridis")
            axes[r, i].set_title(f"{label} - view {i}")
            axes[r, i].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_ablation_for_scene(image_paths: List[str], scene_dir: str, device=None) -> dict:
    """Run all four methods for one real multi-view scene and write every
    artifact (depths, PLYs, per-method metrics, comparison visualization,
    comparison.json) under `scene_dir`. Returns the summary dict."""
    scene_dir = Path(scene_dir)
    image_paths = [Path(p) for p in image_paths]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "Ablation experiments require a CUDA GPU -- VGGT and Marigold "
            "are not meaningfully testable on CPU. No fallback is used."
        )

    shared_dir = scene_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("Running VGGT (shared across all 4 methods)")
    vggt_model = load_vggt_model(device)
    try:
        vggt_result = run_vggt(vggt_model, image_paths, device)
    finally:
        del vggt_model
        gc.collect()
        torch.cuda.empty_cache()
    vggt_depth = vggt_result["depth"]
    vggt_conf = vggt_result["confidence"]
    extrinsic = vggt_result["extrinsic"]
    intrinsic = vggt_result["intrinsic"]
    vggt_meta = vggt_result["metadata"]

    logger.info("Running Marigold (shared across methods 2-4)")
    marigold_pipe, _ = load_marigold_pipeline(device)
    try:
        marigold_depths, marigold_meta = run_marigold(marigold_pipe, image_paths, device)
    finally:
        del marigold_pipe
        gc.collect()
        torch.cuda.empty_cache()

    aligned_depths, alignment_metrics = [], []
    for i in range(len(image_paths)):
        md = marigold_depths[i]
        vd = vggt_depth[i]
        if md.shape != vd.shape:
            md = resize_depth_to(md, vd.shape)
        aligned, scale, shift, n_valid, residual = align_depth(vd, md)
        aligned_depths.append(aligned)
        alignment_metrics.append({
            "view": i, "scale": scale, "shift": shift,
            "valid_pixel_count": n_valid, "residual": residual,
        })
    shared_setup_sec = round(time.time() - t0, 3)

    K_list = [intrinsic[i] for i in range(len(image_paths))]
    E_list = [extrinsic[i] for i in range(len(image_paths))]
    vggt_depth_list = [vggt_depth[i] for i in range(len(image_paths))]

    results = {}

    def _run_method(name, depths):
        t = time.time()
        pts, cols = _reconstruct(depths, image_paths, K_list, E_list)
        d = scene_dir / name
        d.mkdir(exist_ok=True)
        save_ply(d / f"{name}.ply", pts, cols)
        for i, depth in enumerate(depths):
            np.save(d / f"depth_{i:03d}.npy", depth)
        results[name] = {
            "runtime_sec": round(time.time() - t, 4),
            "point_cloud": point_cloud_stats(pts),
            "cross_view_consistency_mean_abs_rel_error": cross_view_consistency(depths, E_list, K_list),
        }

    logger.info("Method 1/4: vggt_only")
    _run_method("vggt_only", vggt_depth_list)

    logger.info("Method 2/4: marigold_only")
    _run_method("marigold_only", aligned_depths)
    results["marigold_only"]["note"] = (
        "VGGT's camera parameters are reused since Marigold alone produces no "
        "multi-view camera geometry to reconstruct in a shared frame."
    )

    logger.info("Method 3/4: naive_average")
    naive_depths = [0.5 * vggt_depth_list[i] + 0.5 * aligned_depths[i] for i in range(len(image_paths))]
    _run_method("naive_average", naive_depths)

    logger.info("Method 4/4: geodiff3d_fusion")
    fused_depths = []
    for i in range(len(image_paths)):
        vc = normalize_confidence(vggt_conf[i])
        fused_depths.append(fuse_depths(vggt_depth_list[i], aligned_depths[i], vc))
    _run_method("geodiff3d_fusion", fused_depths)

    np.save(shared_dir / "vggt_depth.npy", vggt_depth)
    for i, d_ in enumerate(marigold_depths):
        np.save(shared_dir / f"marigold_depth_{i:03d}.npy", d_)
    with open(shared_dir / "alignment_metrics.json", "w") as f:
        json.dump(alignment_metrics, f, indent=2)

    _save_comparison_viz(vggt_depth_list, aligned_depths, naive_depths, fused_depths,
                          scene_dir / "depth_comparison_4methods.png")

    summary = {
        "scene": scene_dir.name,
        "num_views": len(image_paths),
        "vggt": vggt_meta,
        "marigold": marigold_meta,
        "shared_setup_sec": shared_setup_sec,
        "alignment": alignment_metrics,
        "methods": results,
        "ground_truth": "none available for this scene -- self-consistency diagnostics only",
    }
    with open(scene_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Scene %s complete", scene_dir.name)
    return summary

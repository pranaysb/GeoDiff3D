"""GeoDiff3D real GPU reconstruction pipeline: the single source of truth for
turning multi-view photographs into baseline (VGGT-only) and guided
(confidence-fused VGGT+Marigold) point clouds.

This is the actual implementation validated end-to-end on a real T4 GPU. The
Colab notebook and the FastAPI backend's VGGTMarigoldEngine are both clients
of this module -- there is no separate/duplicate implementation.

Sequential GPU memory management (fits a 16GB T4): VGGT is loaded, run, and
freed before Marigold is loaded, so the two ~1-5GB models are never resident
at the same time.
"""
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

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
from inference.vggt_pipeline import run_vggt_and_save
from inference.marigold_pipeline import run_marigold_and_save

logger = logging.getLogger("geodiff3d.gpu_pipeline")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ProgressCallback = Optional[Callable[[str, Optional[str]], None]]


def _validate_images(image_paths: List[Path]) -> None:
    if not (2 <= len(image_paths) <= 12):
        raise ValueError(f"Need between 2 and 12 images, got {len(image_paths)}")
    for p in image_paths:
        try:
            with Image.open(p) as im:
                im.verify()
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Unreadable image {p}: {e}") from e


def _save_input_grid(image_paths: List[Path], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(image_paths), figsize=(4 * len(image_paths), 4))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, image_paths):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(p.name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_depth_comparison(vggt_depth, marigold_depths, aligned_depths, fused_depths, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    s = len(vggt_depth)
    fig, axes = plt.subplots(4, s, figsize=(4 * s, 14))
    axes = np.atleast_2d(axes)
    rows = [
        (vggt_depth, "VGGT"),
        (marigold_depths, "Marigold"),
        (aligned_depths, "Aligned"),
        (fused_depths, "Fused"),
    ]
    for r, (depths, label) in enumerate(rows):
        for i in range(s):
            axes[r, i].imshow(depths[i], cmap="viridis")
            axes[r, i].set_title(f"{label} {i}")
            axes[r, i].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_reconstruction(
    image_paths: List[str],
    scene_dir: str,
    device: Optional[torch.device] = None,
    progress_cb: ProgressCallback = None,
) -> dict:
    """Run the full GeoDiff3D pipeline for one scene.

    Writes the complete artifact tree under `scene_dir` (vggt/, marigold/,
    aligned/, fused/, baseline/, guided/, metrics/, visualizations/) and
    returns the metrics dict. Raises on any real failure -- never fabricates
    output. `progress_cb(state, message)` is invoked at each stage transition
    with the same state names as the backend's job state machine.
    """
    def report(state: str, message: Optional[str] = None):
        logger.info("stage=%s %s", state, message or "")
        if progress_cb:
            progress_cb(state, message)

    scene_dir = Path(scene_dir)
    image_paths = [Path(p) for p in image_paths]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vggt_dir = scene_dir / "vggt"
    marigold_dir = scene_dir / "marigold"
    aligned_dir = scene_dir / "aligned"
    fused_dir = scene_dir / "fused"
    baseline_dir = scene_dir / "baseline"
    guided_dir = scene_dir / "guided"
    metrics_dir = scene_dir / "metrics"
    viz_dir = scene_dir / "visualizations"
    for d in (aligned_dir, fused_dir, baseline_dir, guided_dir, metrics_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- preprocessing ---
    report("preprocessing")
    _validate_images(image_paths)
    _save_input_grid(image_paths, viz_dir / "input_grid.png")

    # --- VGGT (loaded, run, freed before Marigold) ---
    report("vggt")
    vggt_result = run_vggt_and_save(image_paths, vggt_dir, device)
    vggt_depth = vggt_result["depth"]
    vggt_conf = vggt_result["confidence"]
    extrinsic = vggt_result["extrinsic"]
    intrinsic = vggt_result["intrinsic"]
    vggt_meta = vggt_result["metadata"]

    # --- Marigold (loaded only after VGGT's VRAM is freed) ---
    report("diffusion")
    marigold_depths, marigold_meta = run_marigold_and_save(image_paths, marigold_dir, device)

    # --- alignment ---
    report("alignment")
    aligned_depths = []
    alignment_metrics = []
    for i in range(len(image_paths)):
        vd = vggt_depth[i]
        md = marigold_depths[i]
        if md.shape != vd.shape:
            md = resize_depth_to(md, vd.shape)
        aligned, scale, shift, n_valid, residual = align_depth(vd, md)
        aligned_depths.append(aligned)
        np.save(aligned_dir / f"view_{i:03d}.npy", aligned)
        alignment_metrics.append({
            "view": i,
            "scale": scale,
            "shift": shift,
            "valid_pixel_count": n_valid,
            "residual": residual,
        })
    with open(metrics_dir / "alignment_metrics.json", "w") as f:
        json.dump(alignment_metrics, f, indent=2)

    # --- confidence-guided fusion ---
    report("fusion")
    fused_depths = []
    for i in range(len(image_paths)):
        vc = normalize_confidence(vggt_conf[i])
        fused = fuse_depths(vggt_depth[i], aligned_depths[i], vc)
        fused_depths.append(fused)
        np.save(fused_dir / f"view_{i:03d}.npy", fused)

    # --- 3D reconstruction ---
    report("reconstruction")
    baseline_points, baseline_colors = [], []
    guided_points, guided_colors = [], []
    for i, p in enumerate(image_paths):
        rgb = np.asarray(Image.open(p).convert("RGB"))
        K = intrinsic[i]
        E = extrinsic[i]
        b_pts, b_col = unproject_to_point_cloud(vggt_depth[i], rgb, K, E)
        g_pts, g_col = unproject_to_point_cloud(fused_depths[i], rgb, K, E)
        baseline_points.append(b_pts)
        baseline_colors.append(b_col)
        guided_points.append(g_pts)
        guided_colors.append(g_col)

    baseline_points = np.vstack(baseline_points)
    baseline_colors = np.vstack(baseline_colors)
    guided_points = np.vstack(guided_points)
    guided_colors = np.vstack(guided_colors)

    save_ply(baseline_dir / "baseline.ply", baseline_points, baseline_colors)
    save_ply(guided_dir / "guided.ply", guided_points, guided_colors)

    # --- evaluation ---
    report("evaluation")
    _save_depth_comparison(vggt_depth, marigold_depths, aligned_depths, fused_depths, viz_dir / "depth_comparison.png")

    metrics = {
        "num_views": len(image_paths),
        "vggt": vggt_meta,
        "marigold": marigold_meta,
        "alignment": alignment_metrics,
        "baseline": point_cloud_stats(baseline_points),
        "guided": point_cloud_stats(guided_points),
        "note": "No ground truth available; self-consistency diagnostics only.",
    }
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    report("completed")
    return metrics


def main():
    """Standalone CLI entrypoint: reads ./input, writes ./output, matching the
    original Phase 2A script layout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    root = Path(__file__).parent
    input_dir = root / "input"
    output_dir = root / "output"

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    start = time.time()
    metrics = run_reconstruction([str(p) for p in image_paths], output_dir, device=device)
    print(f"\nSUCCESS in {time.time() - start:.1f}s")
    print(f"Baseline points: {metrics['baseline']['num_points']}")
    print(f"Guided points:   {metrics['guided']['num_points']}")
    print(f"Metrics: {output_dir / 'metrics' / 'metrics.json'}")


if __name__ == "__main__":
    main()

"""Core reconstruction math: alignment, confidence fusion, unprojection, point-cloud I/O.

These implementations were validated against a real T4 GPU run of VGGT-1B + Marigold
(4 real photographs, 295s VGGT / 73s Marigold, 721,572-point exports). Every module
that needs this math (the CPU POC, the GPU inference pipeline, the notebook demo)
imports from here — this file is the single source of truth.
"""
import numpy as np
from pathlib import Path
from PIL import Image


def align_depth(reference_depth, relative_depth):
    """Fit reference ~= a * relative + b using finite positive pixels and robust
    percentile clipping to reject outliers. `reference_depth` is the geometric
    (VGGT) depth; `relative_depth` is the monocular (Marigold) depth to be aligned
    into the reference's scale.

    Returns (aligned_depth, scale, shift, valid_pixel_count, residual) where
    residual is the RMS error of the fit on the retained pixels.
    """
    ref = np.asarray(reference_depth, dtype=np.float32)
    rel = np.asarray(relative_depth, dtype=np.float32)
    if ref.shape != rel.shape:
        raise ValueError(f"Depth shapes must match, got {ref.shape} and {rel.shape}")

    valid = np.isfinite(ref) & np.isfinite(rel) & (ref > 0) & (rel > 0)
    if valid.sum() < 100:
        return rel.copy(), 1.0, 0.0, int(valid.sum()), float("nan")

    x = rel[valid].astype(np.float64)
    y = ref[valid].astype(np.float64)

    # Reject pathological outliers before fitting.
    lo_x, hi_x = np.percentile(x, [1, 99])
    lo_y, hi_y = np.percentile(y, [1, 99])
    keep = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    x, y = x[keep], y[keep]

    A = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if not np.isfinite(a) or a <= 0:
        a, b = 1.0, 0.0

    residual = float(np.sqrt(np.mean((a * x + b - y) ** 2)))
    aligned = a * rel + b
    return aligned.astype(np.float32), float(a), float(b), int(len(x)), residual


def normalize_confidence(conf):
    """Percentile-normalize a raw confidence map into [0, 1]."""
    c = np.asarray(conf, dtype=np.float32)
    finite = np.isfinite(c)
    if not finite.any():
        return np.zeros_like(c, dtype=np.float32)
    vals = c[finite]
    lo, hi = np.percentile(vals, [5, 95])
    if hi <= lo + 1e-8:
        return np.ones_like(c, dtype=np.float32) * 0.5
    out = (c - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)


def fuse_depths(reference_depth, aligned_depth, reference_confidence,
                 trust_threshold=0.5, max_aligned_weight=0.4):
    """Confidence-guided fusion. `reference_confidence` is confidence IN the
    reference (geometric/VGGT) depth, normalized to [0, 1] via
    `normalize_confidence` (i.e. relative to this image's own 5th/95th
    percentile confidence, not an absolute scale).

    Revision note (Phase 4 ablation, `experiments/RESULTS.md`): the original
    version used a full-range linear blend, `weight = 1 - reference_confidence`.
    Because `normalize_confidence` always stretches each image's confidence to
    fill [0, 1], that made the *median* pixel in every scene land near
    weight=0.5 regardless of whether VGGT's absolute confidence was uniformly
    excellent -- e.g. on the real `room` scene (VGGT-only cross-view error
    4.8x lower than Marigold-only) the median pixel still got ~31% Marigold
    weight, and 24% of pixels got a majority-Marigold blend. The fusion was
    not actually confidence-*selective*; it behaved close to naive averaging
    for a typical pixel, which is why its ablation scores tracked naive
    averaging rather than staying near VGGT-only's.

    Fixed by gating on relative confidence instead of blending across the
    full range: pixels at or above `trust_threshold` keep VGGT's depth
    untouched (weight 0); only pixels below the threshold ramp in aligned
    depth, capped at `max_aligned_weight` so no pixel is ever fully replaced
    by Marigold -- every scene tested showed Marigold-only is the worst
    standalone method, so full replacement is never justified by the
    confidence signal alone.
    """
    reference_depth = np.asarray(reference_depth, dtype=np.float32)
    aligned_depth = np.asarray(aligned_depth, dtype=np.float32)
    rc = np.clip(np.asarray(reference_confidence, dtype=np.float32), 0.0, 1.0)

    below_threshold = np.clip((trust_threshold - rc) / trust_threshold, 0.0, 1.0)
    fusion_weight = below_threshold * max_aligned_weight
    return ((1.0 - fusion_weight) * reference_depth + fusion_weight * aligned_depth).astype(np.float32)


def resize_rgb_to(rgb, hw):
    h, w = map(int, hw)
    return np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)


def resize_depth_to(depth, hw):
    """Resize a single-channel float depth map to (h, w) via bilinear interpolation."""
    h, w = map(int, hw)
    depth = np.asarray(depth, dtype=np.float32)
    resized = Image.fromarray(depth, mode="F").resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def unproject_to_point_cloud(depth, rgb, K, extrinsic):
    """Unproject a depth map into a colored world-space point cloud.

    `extrinsic` is the camera's world->camera transform, [R|t] with shape (3, 4)
    (VGGT's convention, OpenCV axes) or (4, 4) with a trailing [0,0,0,1] row --
    both are accepted since only the top 3x4 block is used. World coordinates are
    recovered as Xw = R^T (Xc - t).
    """
    depth = np.asarray(depth, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        rgb = resize_rgb_to(rgb, (h, w))

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    z = depth
    valid = np.isfinite(z) & (z > 0)
    finite_z = z[valid]
    if finite_z.size:
        z_hi = np.percentile(finite_z, 99.5)
        valid &= z <= z_hi

    x = (xx - K[0, 2]) * z / K[0, 0]
    y = (yy - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x[valid], y[valid], z[valid]], axis=1)

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    pts_world = (pts_cam - t[None, :]) @ R
    colors = rgb[valid]
    return pts_world.astype(np.float32), colors.astype(np.uint8)


def save_ply(path, points, colors):
    path = Path(path)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    assert points.shape[0] == colors.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def point_cloud_stats(points):
    """Self-consistency diagnostics for a point cloud (no ground truth required)."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return {"num_points": 0}
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return {
        "num_points": int(points.shape[0]),
        "bbox_min": mins.tolist(),
        "bbox_max": maxs.tolist(),
        "bbox_size": (maxs - mins).tolist(),
        "depth_range": [float(points[:, 2].min()), float(points[:, 2].max())],
    }

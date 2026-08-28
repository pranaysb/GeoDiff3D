import os
import sys
import numpy as np
import argparse

# Add root to path so we can import core.math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.math import align_depth, fuse_depths, unproject_to_point_cloud, save_ply

# Ground-truth scale/shift injected into the synthetic diffusion depth so the
# alignment step has a known target to recover (deterministic math validation,
# not a claim about real VGGT/Marigold behavior).
TRUE_SCALE = 2.5
TRUE_SHIFT = -1.0


def create_dummy_data(num_views=4, h=256, w=256, seed=0):
    """Generate synthetic per-view VGGT-like and diffusion-like depth for
    validating the alignment/fusion/unprojection math end to end."""
    rng = np.random.default_rng(seed)
    K = np.array([
        [max(h, w), 0, w / 2.0],
        [0, max(h, w), h / 2.0],
        [0, 0, 1.0],
    ])

    views = []
    for view_id in range(num_views):
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        # Smooth synthetic geometric depth surface, distinct per view.
        vggt_depth = 5.0 + 2.0 * np.sin(xx / w * np.pi + view_id) + 2.0 * np.cos(yy / h * np.pi)
        vggt_depth = vggt_depth.astype(np.float32)

        # Diffusion depth is the inverse of the known affine relation plus noise,
        # so align_depth() should recover TRUE_SCALE / TRUE_SHIFT.
        noise = rng.normal(0, 0.02, size=vggt_depth.shape).astype(np.float32)
        diffusion_depth = (vggt_depth - TRUE_SHIFT) / TRUE_SCALE + noise

        vggt_conf = rng.uniform(0.4, 1.0, size=vggt_depth.shape).astype(np.float32)
        rgb = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)

        extrinsic = np.eye(4)
        extrinsic[0, 3] = view_id * 0.5  # simple baseline translation per view

        views.append({
            "view_id": view_id,
            "vggt_depth": vggt_depth,
            "diffusion_depth": diffusion_depth,
            "vggt_conf": vggt_conf,
            "rgb": rgb,
            "K": K,
            "extrinsic": extrinsic,
        })
    return views


def run_pipeline(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Simulate Input / Models
    scene_data = create_dummy_data(num_views=4)

    baseline_points = []
    baseline_colors = []

    guided_points = []
    guided_colors = []

    print("Processing views...")
    for view in scene_data:
        vid = view["view_id"]
        vggt_d = view["vggt_depth"]
        diff_d = view["diffusion_depth"]
        conf = view["vggt_conf"]
        rgb = view["rgb"]
        K = view["K"]
        ext = view["extrinsic"]

        # 2. Alignment
        aligned_diff_d, scale, shift, n_valid, residual = align_depth(vggt_d, diff_d)
        print(f"View {vid} alignment: scale={scale:.3f}, shift={shift:.3f}, "
              f"valid_pixels={n_valid}, residual={residual:.4f}")

        # 3. Fusion
        guided_d = fuse_depths(vggt_d, aligned_diff_d, conf)

        # 4. Point Cloud Generation (Baseline)
        pts_base, col_base = unproject_to_point_cloud(vggt_d, rgb, K, ext)
        baseline_points.append(pts_base)
        baseline_colors.append(col_base)

        # Point Cloud Generation (Guided)
        pts_guid, col_guid = unproject_to_point_cloud(guided_d, rgb, K, ext)
        guided_points.append(pts_guid)
        guided_colors.append(col_guid)

    # 5. Merge and Export
    print("Merging and exporting point clouds...")
    all_pts_base = np.vstack(baseline_points)
    all_col_base = np.vstack(baseline_colors)

    all_pts_guid = np.vstack(guided_points)
    all_col_guid = np.vstack(guided_colors)

    base_out = os.path.join(output_dir, "baseline.ply")
    guid_out = os.path.join(output_dir, "guided.ply")

    save_ply(base_out, all_pts_base, all_col_base)
    save_ply(guid_out, all_pts_guid, all_col_guid)

    print(f"Exported baseline point cloud ({len(all_pts_base)} points) to {base_out}")
    print(f"Exported guided point cloud ({len(all_pts_guid)} points) to {guid_out}")
    print("Pipeline proof-of-concept completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoDiff3D CPU PoC Pipeline")
    parser.add_argument("--output", type=str, default="experiments/poc_output", help="Output directory")
    args = parser.parse_args()

    run_pipeline(args.output)

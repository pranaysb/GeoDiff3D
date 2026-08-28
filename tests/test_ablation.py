"""Unit tests for the cross-view self-consistency metric used by the Phase 4
ablation (experiments/ablation.py). These use synthetic cameras/depth purely
to validate the metric's math -- they are not a substitute for the real
multi-view ablation, which requires actual VGGT/Marigold output on a GPU.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from experiments.ablation import cross_view_consistency


def _two_view_frontoparallel_plane(depth1_value):
    """Two identity-rotation cameras separated by a pure-x translation,
    both imaging a fronto-parallel plane at Z=5 in the world/camera0 frame.
    For this geometry the true depth in camera1 is exactly 5.0 everywhere
    (translation along x doesn't change z), so depth1_value=5.0 is the
    ground truth and any other value is an injected inconsistency."""
    h, w = 32, 32
    K = np.array([[50.0, 0, 16.0], [0, 50.0, 16.0], [0, 0, 1.0]])
    E0 = np.hstack([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)])
    E1 = np.hstack([np.eye(3, dtype=np.float32), np.array([[0.2], [0.0], [0.0]], dtype=np.float32)])
    depth0 = np.full((h, w), 5.0, dtype=np.float32)
    depth1 = np.full((h, w), depth1_value, dtype=np.float32)
    return [depth0, depth1], [E0, E1], [K, K]


def test_cross_view_consistency_near_zero_for_perfectly_consistent_scene():
    depths, extrinsics, intrinsics = _two_view_frontoparallel_plane(depth1_value=5.0)
    err = cross_view_consistency(depths, extrinsics, intrinsics)
    assert err is not None
    assert err < 1e-5


def test_cross_view_consistency_detects_gross_mismatch():
    depths_good, extrinsics, intrinsics = _two_view_frontoparallel_plane(depth1_value=5.0)
    depths_bad, _, _ = _two_view_frontoparallel_plane(depth1_value=8.0)

    err_good = cross_view_consistency(depths_good, extrinsics, intrinsics)
    err_bad = cross_view_consistency(depths_bad, extrinsics, intrinsics)

    assert err_good < err_bad
    # The metric normalizes by view1's own (wrong) stored depth of 8.0:
    # |5.0 (true, reprojected from view0) - 8.0 (stored)| / 8.0 = 0.375.
    assert abs(err_bad - 0.375) < 1e-3


def test_cross_view_consistency_scales_with_error_magnitude():
    depths_small, extrinsics, intrinsics = _two_view_frontoparallel_plane(depth1_value=5.5)
    depths_large, _, _ = _two_view_frontoparallel_plane(depth1_value=7.0)

    err_small = cross_view_consistency(depths_small, extrinsics, intrinsics)
    err_large = cross_view_consistency(depths_large, extrinsics, intrinsics)

    assert err_small < err_large


def test_cross_view_consistency_single_view_returns_none():
    depths, extrinsics, intrinsics = _two_view_frontoparallel_plane(depth1_value=5.0)
    err = cross_view_consistency(depths[:1], extrinsics[:1], intrinsics[:1])
    assert err is None

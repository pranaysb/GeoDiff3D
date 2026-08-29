"""Unit tests for core/math.py -- the alignment, fusion, and camera-transform
math used by every real and synthetic pipeline in this repo. These fill a gap
flagged in the original architecture review: none of this math had dedicated
tests before, only end-to-end CPU POC and (real, GPU) integration coverage.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.math import (
    align_depth,
    fuse_depths,
    normalize_confidence,
    point_cloud_stats,
    resize_depth_to,
    unproject_to_point_cloud,
)


def test_depth_alignment_recovers_known_scale_shift():
    rng = np.random.default_rng(0)
    h, w = 64, 64
    relative = rng.uniform(1, 5, size=(h, w)).astype(np.float32)
    true_scale, true_shift = 2.5, -1.0
    reference = (true_scale * relative + true_shift).astype(np.float32)

    aligned, scale, shift, n_valid, residual = align_depth(reference, relative)

    assert abs(scale - true_scale) < 0.05
    assert abs(shift - true_shift) < 0.05
    assert n_valid > 0
    assert residual < 0.1
    np.testing.assert_allclose(aligned, reference, atol=1e-3)


def test_depth_alignment_falls_back_when_too_few_valid_pixels():
    reference = np.zeros((10, 10), dtype=np.float32)
    relative = np.ones((10, 10), dtype=np.float32)

    aligned, scale, shift, n_valid, residual = align_depth(reference, relative)

    assert scale == 1.0 and shift == 0.0
    assert n_valid == 0
    np.testing.assert_allclose(aligned, relative)


def test_depth_alignment_rejects_mismatched_shapes():
    try:
        align_depth(np.zeros((4, 4)), np.zeros((5, 5)))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_normalize_confidence_range_and_monotonicity():
    conf = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.0, -1.0])
    out = normalize_confidence(conf)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Higher raw confidence must not produce a lower normalized value.
    order = np.argsort(conf)
    assert np.all(np.diff(out[order]) >= -1e-6)


def test_normalize_confidence_handles_all_nan():
    conf = np.full((4, 4), np.nan, dtype=np.float32)
    out = normalize_confidence(conf)
    np.testing.assert_allclose(out, 0.0)


def test_fuse_depths_high_reference_confidence_keeps_reference():
    reference = np.full((4, 4), 10.0, dtype=np.float32)
    aligned = np.full((4, 4), 0.0, dtype=np.float32)
    high_conf = np.ones((4, 4), dtype=np.float32)
    fused = fuse_depths(reference, aligned, high_conf)
    np.testing.assert_allclose(fused, reference)


def test_fuse_depths_low_reference_confidence_is_capped_not_fully_replaced():
    """Even at zero confidence, the reference depth must never be fully
    replaced by the aligned depth -- Marigold-only was the worst standalone
    method in every real scene tested (see experiments/RESULTS.md), so full
    replacement is never justified by the confidence signal alone.
    """
    reference = np.full((4, 4), 10.0, dtype=np.float32)
    aligned = np.full((4, 4), 0.0, dtype=np.float32)
    zero_conf = np.zeros((4, 4), dtype=np.float32)
    fused = fuse_depths(reference, aligned, zero_conf, max_aligned_weight=0.4)
    expected = 0.6 * reference + 0.4 * aligned
    np.testing.assert_allclose(fused, expected)


def test_fuse_depths_default_parameters_match_tuned_values():
    """Regression test pinning the shipped defaults to the values chosen by
    experiments/tune_fusion.py's offline grid search (see "Hyperparameter
    tuning" in RESULTS.md) -- catches an accidental change to the defaults
    without a corresponding re-tune and doc update.
    """
    reference = np.full((4, 4), 10.0, dtype=np.float32)
    aligned = np.full((4, 4), 0.0, dtype=np.float32)
    zero_conf = np.zeros((4, 4), dtype=np.float32)
    fused = fuse_depths(reference, aligned, zero_conf)
    expected = 0.9 * reference + 0.1 * aligned
    np.testing.assert_allclose(fused, expected)


def test_fuse_depths_at_or_above_threshold_keeps_reference_untouched():
    """Confidence at/above trust_threshold must fully trust VGGT (weight 0)
    -- this is the fix for the diagnosed bug where the old linear blend gave
    even median-confidence pixels a near-50/50 blend in every scene.
    """
    reference = np.full((4, 4), 10.0, dtype=np.float32)
    aligned = np.full((4, 4), 0.0, dtype=np.float32)
    at_threshold = np.full((4, 4), 0.5, dtype=np.float32)
    above_threshold = np.full((4, 4), 0.9, dtype=np.float32)
    np.testing.assert_allclose(
        fuse_depths(reference, aligned, at_threshold, trust_threshold=0.5), reference
    )
    np.testing.assert_allclose(
        fuse_depths(reference, aligned, above_threshold, trust_threshold=0.5), reference
    )


def test_fuse_depths_ramps_linearly_below_threshold():
    reference = np.full((1, 1), 10.0, dtype=np.float32)
    aligned = np.full((1, 1), 0.0, dtype=np.float32)
    # Halfway between 0 and trust_threshold=0.5 -> half of max_aligned_weight.
    half_below = np.full((1, 1), 0.25, dtype=np.float32)
    fused = fuse_depths(reference, aligned, half_below, trust_threshold=0.5, max_aligned_weight=0.4)
    expected_weight = 0.5 * 0.4
    np.testing.assert_allclose(fused, (1 - expected_weight) * reference + expected_weight * aligned)


def test_unprojection_camera_transform_identity_round_trip():
    """A constant-depth plane imaged by an identity-rotation camera must
    reproject to exactly that depth when re-projected back into camera space
    -- this is the coordinate-convention check the architecture review
    flagged as the most dangerous potential bug (world<->camera confusion)."""
    K = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]])
    R = np.eye(3, dtype=np.float32)
    t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    extrinsic = np.hstack([R, t.reshape(3, 1)])  # world->camera: Xc = R @ Xw + t

    depth = np.full((64, 64), 5.0, dtype=np.float32)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    points, colors = unproject_to_point_cloud(depth, rgb, K, extrinsic)

    assert points.shape[0] == 64 * 64
    assert colors.shape == points.shape

    Xc_reconstructed = points @ R.T + t
    np.testing.assert_allclose(Xc_reconstructed[:, 2], 5.0, atol=1e-3)


def test_unprojection_camera_transform_with_rotation_round_trip():
    K = np.array([[100.0, 0, 16.0], [0, 100.0, 16.0], [0, 0, 1.0]])
    theta = np.deg2rad(30)
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ], dtype=np.float32)
    t = np.array([0.5, -0.2, 0.1], dtype=np.float32)
    extrinsic = np.hstack([R, t.reshape(3, 1)])

    depth = np.full((32, 32), 3.0, dtype=np.float32)
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    points, _ = unproject_to_point_cloud(depth, rgb, K, extrinsic)

    Xc = points @ R.T + t
    np.testing.assert_allclose(Xc[:, 2], 3.0, atol=1e-2)


def test_unprojection_accepts_4x4_extrinsic():
    K = np.array([[100.0, 0, 8.0], [0, 100.0, 8.0], [0, 0, 1.0]])
    extrinsic_3x4 = np.hstack([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)])
    extrinsic_4x4 = np.vstack([extrinsic_3x4, [0, 0, 0, 1]]).astype(np.float32)

    depth = np.full((16, 16), 2.0, dtype=np.float32)
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    pts_3x4, _ = unproject_to_point_cloud(depth, rgb, K, extrinsic_3x4)
    pts_4x4, _ = unproject_to_point_cloud(depth, rgb, K, extrinsic_4x4)
    np.testing.assert_allclose(pts_3x4, pts_4x4)


def test_point_cloud_stats_basic():
    points = np.array([[0, 0, 1], [1, 1, 2], [-1, -1, 0]], dtype=np.float32)
    stats = point_cloud_stats(points)
    assert stats["num_points"] == 3
    assert stats["depth_range"] == [0.0, 2.0]


def test_point_cloud_stats_empty():
    stats = point_cloud_stats(np.zeros((0, 3)))
    assert stats["num_points"] == 0


def test_resize_depth_to_shape_and_scale():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    resized = resize_depth_to(depth, (4, 4))
    assert resized.shape == (4, 4)
    assert resized.min() >= 1.0 - 1e-3 and resized.max() <= 4.0 + 1e-3

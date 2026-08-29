"""Offline grid search over fuse_depths(trust_threshold, max_aligned_weight)
against the real cross_view_consistency metric, using cached real VGGT +
Marigold outputs (see cache_fusion_inputs.py). Pure CPU, no GPU needed --
this reuses one real GPU realization per scene instead of re-running
inference for every candidate combination.

Run after experiments/cache_fusion_inputs.py has populated
experiments/tuning_cache/<scene>/:

    python experiments/tune_fusion.py

Caveat: this is one Marigold sample per scene (its diffusion sampling isn't
seeded), so it is not literally identical to what a fresh full ablation run
with the chosen defaults would produce -- but the fusion math being tuned
(fuse_depths, cross_view_consistency) is deterministic and identical to what
experiments/ablation.py uses, so the *ranking* between candidate parameters
on real data is a legitimate basis for choosing defaults.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.math import fuse_depths, normalize_confidence
from experiments.ablation import cross_view_consistency

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = REPO_ROOT / "experiments" / "tuning_cache"
OUTPUT_DIR = REPO_ROOT / "experiments" / "tuning_results"

TRUST_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MAX_ALIGNED_WEIGHTS = [0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]

# The defaults shipped from the diagnosis alone (not tuned against data) --
# included in every table below as a reference row.
CURRENT_DEFAULTS = (0.5, 0.4)


def load_scene(scene_dir: Path) -> dict:
    return {
        "vggt_depth": np.load(scene_dir / "vggt_depth.npy"),
        "vggt_confidence": np.load(scene_dir / "vggt_confidence.npy"),
        "aligned": np.load(scene_dir / "aligned_marigold_depth.npy"),
        "extrinsic": np.load(scene_dir / "extrinsic.npy"),
        "intrinsic": np.load(scene_dir / "intrinsic.npy"),
    }


def fused_cross_view_error(scene: dict, trust_threshold: float, max_aligned_weight: float) -> float:
    n = scene["vggt_depth"].shape[0]
    fused = []
    for i in range(n):
        rc = normalize_confidence(scene["vggt_confidence"][i])
        fused.append(fuse_depths(
            scene["vggt_depth"][i], scene["aligned"][i], rc,
            trust_threshold=trust_threshold, max_aligned_weight=max_aligned_weight,
        ))
    return cross_view_consistency(fused, scene["extrinsic"], scene["intrinsic"])


def main():
    scene_dirs = sorted(p for p in CACHE_ROOT.iterdir() if p.is_dir()) if CACHE_ROOT.exists() else []
    if not scene_dirs:
        raise RuntimeError(
            f"No cached scenes found under {CACHE_ROOT}. Run "
            "experiments/cache_fusion_inputs.py on a GPU first."
        )
    scenes = {d.name: load_scene(d) for d in scene_dirs}
    print(f"Loaded {len(scenes)} cached scenes: {list(scenes)}")

    # Sanity check: VGGT is a deterministic feed-forward model given the same
    # input, so its cross_view_consistency recomputed from this cache should
    # match the corresponding numbers already in experiments/RESULTS.md.
    print("\nSanity check (vggt_only, recomputed from cache):")
    vggt_only_cv = {}
    for name, s in scenes.items():
        n = s["vggt_depth"].shape[0]
        cv = cross_view_consistency([s["vggt_depth"][i] for i in range(n)], s["extrinsic"], s["intrinsic"])
        vggt_only_cv[name] = cv
        print(f"  {name:12s} vggt_only cross_view_error={cv:.4f}")

    results = []
    for tt in TRUST_THRESHOLDS:
        for maw in MAX_ALIGNED_WEIGHTS:
            row = {"trust_threshold": tt, "max_aligned_weight": maw}
            beats, rel_gaps = 0, []
            for name, s in scenes.items():
                cv = fused_cross_view_error(s, tt, maw)
                row[f"{name}_cv"] = round(cv, 5)
                cv_vggt = vggt_only_cv[name]
                if cv < cv_vggt:
                    beats += 1
                rel_gaps.append((cv - cv_vggt) / cv_vggt)
            row["scenes_beaten"] = beats
            row["mean_relative_gap_to_vggt"] = round(float(np.mean(rel_gaps)), 5)
            row["is_current_default"] = (tt, maw) == CURRENT_DEFAULTS
            results.append(row)

    # Primary: beat VGGT-only in as many scenes as possible. Tie-break: the
    # smallest mean relative gap to VGGT-only (least bad where it doesn't win,
    # least redundant where it does).
    results.sort(key=lambda r: (-r["scenes_beaten"], r["mean_relative_gap_to_vggt"]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "grid_search.json", "w") as f:
        json.dump({"vggt_only_cv": vggt_only_cv, "grid": results}, f, indent=2)

    print(f"\nTop 10 of {len(results)} combinations (by scenes_beaten, then mean_relative_gap_to_vggt):")
    header = f"{'trust_thr':>9} {'max_wt':>7} {'beaten':>7} {'mean_gap':>9}  " + "  ".join(f"{n:>14}" for n in scenes)
    print(header)
    for r in results[:10]:
        marker = " <- current default" if r["is_current_default"] else ""
        print(f"{r['trust_threshold']:>9} {r['max_aligned_weight']:>7} {r['scenes_beaten']:>7} "
              f"{r['mean_relative_gap_to_vggt']:>9}  " +
              "  ".join(f"{r[f'{n}_cv']:>14}" for n in scenes) + marker)

    current = next((r for r in results if r["is_current_default"]), None)
    if current:
        rank = results.index(current) + 1
        print(f"\nCurrent shipped default (trust_threshold=0.5, max_aligned_weight=0.4) "
              f"ranks #{rank} of {len(results)}.")

    print(f"\nFull grid written to {OUTPUT_DIR / 'grid_search.json'}")


if __name__ == "__main__":
    main()

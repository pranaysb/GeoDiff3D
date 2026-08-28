"""Phase 4 ablation CLI. Run on a CUDA GPU:

    python experiments/run_ablation.py

Runs all four methods (vggt_only, marigold_only, naive_average,
geodiff3d_fusion) on 3-5 real multi-view scenes auto-discovered from the
official VGGT repository's example photos (no synthetic data), and writes a
cross-scene comparison table.
"""
import csv
import json
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from experiments.ablation import run_ablation_for_scene

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "experiments" / "ablation_results"

# The official VGGT repo (cloned separately, e.g. to /content/vggt on Colab --
# see inference/vggt_pipeline.py) ships real multi-view example photos under
# examples/<scene>/images. Override with the VGGT_REPO_DIR env var if it's
# cloned somewhere else.
_VGGT_REPO_CANDIDATES = [
    Path(os.environ.get("VGGT_REPO_DIR", "")),
    Path("/content/vggt"),
    REPO_ROOT / "notebooks" / "vggt",
]
VGGT_EXAMPLES = next(
    (c / "examples" for c in _VGGT_REPO_CANDIDATES if c and (c / "examples").exists()),
    None,
)
if VGGT_EXAMPLES is None:
    raise RuntimeError(
        "Could not find a cloned VGGT repo with examples/. Set VGGT_REPO_DIR "
        "to its path, e.g. VGGT_REPO_DIR=/content/vggt python experiments/run_ablation.py"
    )

def _discover_scenes(examples_dir: Path, max_scenes: int, views_per_scene: int) -> dict:
    """Auto-discover real multi-view scenes shipped in the cloned VGGT repo,
    rather than hardcoding scene names that may not exist in a given clone.
    A scene is any `examples/<name>/images/` folder with at least 2 real
    photographs (png/jpg/jpeg) -- no synthetic data is generated here.
    """
    scenes = {}
    for d in sorted(p for p in examples_dir.iterdir() if p.is_dir()):
        images_dir = d / "images"
        if not images_dir.is_dir():
            continue
        paths = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if len(paths) >= 2:
            scenes[d.name] = paths[:views_per_scene]
        if len(scenes) >= max_scenes:
            break
    return scenes


# Real photographs, not synthetic data -- every scene below is auto-discovered
# from whatever example folders the cloned VGGT repo actually ships, covering
# whatever mix of scene types (indoor/outdoor/object) are present.
_MAX_SCENES = int(os.environ.get("GEODIFF3D_ABLATION_MAX_SCENES", "5"))
_VIEWS_PER_SCENE = int(os.environ.get("GEODIFF3D_ABLATION_VIEWS_PER_SCENE", "6"))
SCENES = _discover_scenes(VGGT_EXAMPLES, _MAX_SCENES, _VIEWS_PER_SCENE)
if len(SCENES) < 3:
    raise RuntimeError(
        f"Only found {len(SCENES)} real example scene(s) with >=2 images under "
        f"{VGGT_EXAMPLES} -- need at least 3 for the Phase 4 brief. Found: "
        f"{list(SCENES.keys())}"
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "Ablation experiments require a CUDA GPU. Run this on Colab with "
            "a T4 runtime after installing inference/requirements_gpu.txt."
        )
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scene_name, image_paths in SCENES.items():
        if not image_paths:
            print(f"Skipping {scene_name}: no images found at {VGGT_EXAMPLES / scene_name}")
            continue
        print(f"\n=== Scene: {scene_name} ({len(image_paths)} views) ===")
        summary = run_ablation_for_scene(
            [str(p) for p in image_paths], OUTPUT_ROOT / scene_name, device=device
        )
        summaries.append(summary)
        for method, r in summary["methods"].items():
            cv = r["cross_view_consistency_mean_abs_rel_error"]
            print(f"  {method:18s} points={r['point_cloud']['num_points']:>7,}  "
                  f"runtime={r['runtime_sec']:.3f}s  "
                  f"cross_view_err={'n/a' if cv is None else f'{cv:.4f}'}")

    if summaries:
        _write_comparison_table(summaries, OUTPUT_ROOT)
    else:
        print("No scenes were run -- nothing to compare.")


def _write_comparison_table(summaries, output_root: Path):
    rows = []
    for s in summaries:
        for method, r in s["methods"].items():
            cv = r["cross_view_consistency_mean_abs_rel_error"]
            rows.append({
                "scene": s["scene"],
                "method": method,
                "num_points": r["point_cloud"]["num_points"],
                "runtime_sec": r["runtime_sec"],
                "cross_view_consistency_mean_abs_rel_error": "" if cv is None else round(cv, 4),
            })

    csv_path = output_root / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_root / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write("No ground truth is available for these scenes; "
                "`cross_view_consistency` is a self-consistency diagnostic, not an accuracy metric.\n\n")
        f.write("| Scene | Method | Points | Runtime (s) | Cross-view consistency (mean abs rel error) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            cv_str = r["cross_view_consistency_mean_abs_rel_error"]
            cv_str = "n/a" if cv_str == "" else f"{cv_str}"
            f.write(f"| {r['scene']} | {r['method']} | {r['num_points']:,} | {r['runtime_sec']} | {cv_str} |\n")

    with open(output_root / "comparison_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nComparison table written to:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()

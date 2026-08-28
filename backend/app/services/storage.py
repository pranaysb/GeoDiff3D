import os
import shutil
from typing import Dict

STORAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../storage"))

def get_scene_dir(scene_id: str) -> str:
    return os.path.join(STORAGE_ROOT, "scenes", scene_id)

def create_scene_storage(scene_id: str) -> str:
    base = get_scene_dir(scene_id)
    folders = [
        "input",
        "vggt/depth", "vggt/confidence", "vggt/cameras",
        "marigold/depth",
        "aligned", "fused",
        "baseline", "guided",
        "metrics", "visualizations",
        "logs",
    ]
    for f in folders:
        os.makedirs(os.path.join(base, f), exist_ok=True)
    return base

def delete_scene_storage(scene_id: str) -> bool:
    base = get_scene_dir(scene_id)
    if os.path.exists(base):
        shutil.rmtree(base)
        return True
    return False

def list_artifacts(scene_id: str) -> Dict[str, str]:
    base = get_scene_dir(scene_id)
    candidates = [
        "baseline/baseline.ply",
        "guided/guided.ply",
        "metrics/metrics.json",
        "metrics/alignment_metrics.json",
        "visualizations/input_grid.png",
        "visualizations/depth_comparison.png",
    ]
    artifacts = {}
    for f in candidates:
        path = os.path.join(base, f)
        if os.path.exists(path):
            artifacts[os.path.basename(f)] = path
    return artifacts

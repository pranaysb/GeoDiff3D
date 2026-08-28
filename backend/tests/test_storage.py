import os
import subprocess
import sys

from app.services import storage


def test_storage_root_is_under_repo_root():
    # backend/app/services/storage.py -> backend/app/services -> backend/app -> backend -> repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    assert storage.STORAGE_ROOT == os.path.join(repo_root, "storage")


def test_storage_root_independent_of_cwd():
    # Resolve STORAGE_ROOT from a subprocess launched with a different cwd to
    # confirm it doesn't depend on the process's current working directory.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    backend_dir = os.path.join(repo_root, "backend")
    result = subprocess.run(
        [sys.executable, "-c", "from app.services import storage; print(storage.STORAGE_ROOT)"],
        cwd="/",
        env={**os.environ, "PYTHONPATH": backend_dir},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == os.path.join(repo_root, "storage")


def test_get_scene_dir_layout():
    scene_dir = storage.get_scene_dir("scene_test_layout_check")
    assert scene_dir == os.path.join(storage.STORAGE_ROOT, "scenes", "scene_test_layout_check")


def test_create_scene_storage_folder_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "STORAGE_ROOT", str(tmp_path))
    base = storage.create_scene_storage("scene_structure_test")
    for folder in ["input", "baseline", "guided", "metrics", "logs"]:
        assert os.path.isdir(os.path.join(base, folder))
    storage.delete_scene_storage("scene_structure_test")
    assert not os.path.exists(base)

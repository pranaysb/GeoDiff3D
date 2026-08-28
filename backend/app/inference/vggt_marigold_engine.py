import sys
from pathlib import Path
from typing import Callable, Optional

from .engine import ReconstructionEngine

# The real pipeline lives in inference/gpu_pipeline.py at the repo root, not
# inside the backend package -- it is shared with the notebook demo and the
# standalone CLI. Imports of torch/vggt/diffusers happen lazily inside
# run_reconstruction() so a CPU-only backend process can still start up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))


class VGGTMarigoldEngine(ReconstructionEngine):
    """Real VGGT-1B + Marigold reconstruction engine. No mock fallbacks: any
    failure in the underlying pipeline is surfaced as the job's failure
    message, never masked with fabricated output."""

    def run_reconstruction(
        self,
        job_id: str,
        scene_id: str,
        scene_dir: str,
        update_state_cb: Optional[Callable[[str, Optional[str]], None]] = None,
    ):
        try:
            import torch
            from inference.gpu_pipeline import run_reconstruction
        except ImportError as e:
            message = (
                f"GPU inference dependencies are not installed in this environment: {e}. "
                "Install requirements from inference/requirements_gpu.txt on a "
                "CUDA-capable machine to run real inference."
            )
            if update_state_cb:
                update_state_cb("failed", message)
            raise RuntimeError(message) from e

        input_dir = Path(scene_dir) / "input"
        image_paths = sorted(
            p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            run_reconstruction(
                [str(p) for p in image_paths],
                scene_dir,
                device=device,
                progress_cb=update_state_cb,
            )
        except Exception as e:
            print(f"Job {job_id} failed: {e}")
            if update_state_cb:
                update_state_cb("failed", str(e))

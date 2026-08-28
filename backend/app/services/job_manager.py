import os
import uuid
import threading
from typing import Dict, Optional
from app.models.schemas import Job, JobState, Scene
from app.inference.engine import ReconstructionEngine
from app.inference.poc_engine import POCEngine
from app.services.storage import create_scene_storage, get_scene_dir


def _select_engine() -> ReconstructionEngine:
    """GEODIFF3D_ENGINE=real runs actual VGGT+Marigold inference (requires a
    CUDA GPU and inference/requirements_gpu.txt installed). The default
    "poc" engine runs only the deterministic CPU math validation and must
    never be mistaken for real inference -- see docs/architecture.md."""
    if os.environ.get("GEODIFF3D_ENGINE", "poc").lower() == "real":
        from app.inference.vggt_marigold_engine import VGGTMarigoldEngine
        return VGGTMarigoldEngine()
    return POCEngine()


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.scenes: Dict[str, Scene] = {}
        self.engine = _select_engine()

    def create_job(self, image_count: int) -> str:
        job_id = str(uuid.uuid4())
        scene_id = f"scene_{job_id[:8]}"
        
        self.jobs[job_id] = Job(job_id=job_id, scene_id=scene_id, state=JobState.queued)
        self.scenes[scene_id] = Scene(scene_id=scene_id, job_id=job_id, image_count=image_count)
        
        create_scene_storage(scene_id)
        return job_id

    def start_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job:
            return
            
        scene_id = job.scene_id
        scene_dir = get_scene_dir(scene_id)

        def update_state(state: str, message: Optional[str] = None):
            self.jobs[job_id].state = JobState(state)
            if message:
                self.jobs[job_id].message = message

        thread = threading.Thread(
            target=self.engine.run_reconstruction,
            args=(job_id, scene_id, scene_dir, update_state)
        )
        thread.daemon = True
        thread.start()

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def get_scene(self, scene_id: str) -> Optional[Scene]:
        return self.scenes.get(scene_id)

    def delete_scene_data(self, scene_id: str):
        if scene_id in self.scenes:
            del self.scenes[scene_id]
        
        # Also clean up the job referencing this scene to prevent dangling references
        keys_to_del = [k for k, v in self.jobs.items() if v.scene_id == scene_id]
        for k in keys_to_del:
            del self.jobs[k]

job_manager = JobManager()

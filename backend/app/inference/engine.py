from abc import ABC, abstractmethod
from typing import Callable, Optional

class ReconstructionEngine(ABC):
    @abstractmethod
    def run_reconstruction(self, job_id: str, scene_id: str, scene_dir: str, update_state_cb: Optional[Callable[[str, Optional[str]], None]] = None):
        """Run the reconstruction pipeline."""
        pass

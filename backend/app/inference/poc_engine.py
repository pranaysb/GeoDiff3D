import os
import sys
import subprocess
import shutil
import time
from typing import Callable, Optional
from .engine import ReconstructionEngine

class POCEngine(ReconstructionEngine):
    def run_reconstruction(self, job_id: str, scene_id: str, scene_dir: str, update_state_cb: Optional[Callable[[str, Optional[str]], None]] = None):
        try:
            if update_state_cb: update_state_cb("preprocessing", None)
            time.sleep(0.1)
            
            if update_state_cb: update_state_cb("vggt", None)
            time.sleep(0.1)
            
            if update_state_cb: update_state_cb("diffusion", None)
            time.sleep(0.1)
            
            if update_state_cb: update_state_cb("alignment", None)
            time.sleep(0.1)
            
            if update_state_cb: update_state_cb("fusion", None)
            time.sleep(0.1)
            
            if update_state_cb: update_state_cb("reconstruction", None)
            
            # Find the poc_pipeline.py script
            # Assuming backend is in geodiff3d/backend and script in geodiff3d/scripts
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../scripts/poc_pipeline.py"))
            
            temp_out = os.path.join(scene_dir, "temp_poc")
            
            # Execute the CPU POC pipeline
            result = subprocess.run([sys.executable, script_path, "--output", temp_out], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"POC pipeline failed: {result.stderr}")
                
            # Move the artifacts to correct directories
            shutil.move(os.path.join(temp_out, "baseline.ply"), os.path.join(scene_dir, "baseline", "baseline.ply"))
            shutil.move(os.path.join(temp_out, "guided.ply"), os.path.join(scene_dir, "guided", "guided.ply"))
            shutil.rmtree(temp_out)
            
            if update_state_cb: update_state_cb("evaluation", None)
            time.sleep(0.1)
            
            if update_state_cb: update_state_cb("completed", None)
        except Exception as e:
            print(f"Job {job_id} failed: {e}")
            if update_state_cb: update_state_cb("failed", str(e))

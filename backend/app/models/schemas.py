from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum

class JobState(str, Enum):
    queued = "queued"
    preprocessing = "preprocessing"
    vggt = "vggt"
    diffusion = "diffusion"
    alignment = "alignment"
    fusion = "fusion"
    reconstruction = "reconstruction"
    evaluation = "evaluation"
    completed = "completed"
    failed = "failed"

class Metrics(BaseModel):
    rmse: Optional[float] = None
    absrel: Optional[float] = None
    chamfer: Optional[float] = None
    fscore: Optional[float] = None

class Job(BaseModel):
    job_id: str
    scene_id: str
    state: JobState
    message: Optional[str] = None

class Scene(BaseModel):
    scene_id: str
    job_id: str
    image_count: int
    metrics: Optional[Metrics] = None

class ReconstructionResult(BaseModel):
    job_id: str
    message: str

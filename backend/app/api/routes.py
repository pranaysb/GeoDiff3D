from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List
from app.models.schemas import Job, Scene, ReconstructionResult
from app.services.job_manager import job_manager
from app.services.storage import delete_scene_storage, list_artifacts, get_scene_dir
import os

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024 # 10 MB

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.post("/reconstruct", response_model=ReconstructionResult)
async def reconstruct(images: List[UploadFile] = File(...)):
    if not images or not (2 <= len(images) <= 12):
        raise HTTPException(status_code=400, detail="Must provide between 2 and 12 images")
        
    for image in images:
        ext = os.path.splitext(image.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        
        content = await image.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File {image.filename} exceeds 10MB limit")
        await image.seek(0)

    job_id = job_manager.create_job(len(images))
    job = job_manager.get_job(job_id)
    
    scene_dir = get_scene_dir(job.scene_id)
    for image in images:
        file_path = os.path.join(scene_dir, "input", image.filename)
        with open(file_path, "wb") as f:
            content = await image.read()
            f.write(content)
            
    job_manager.start_job(job_id)
    return ReconstructionResult(job_id=job_id, message="Reconstruction job started")

@router.get("/jobs/{job_id}", response_model=Job)
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/scene/{scene_id}", response_model=Scene)
def get_scene(scene_id: str):
    scene = job_manager.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    return scene

@router.get("/scene/{scene_id}/artifacts")
def get_scene_artifacts(scene_id: str):
    scene = job_manager.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
        
    artifacts = list_artifacts(scene_id)
    return {"scene_id": scene_id, "artifacts": artifacts}

@router.delete("/scene/{scene_id}")
def delete_scene(scene_id: str):
    success = delete_scene_storage(scene_id)
    if not success:
        raise HTTPException(status_code=404, detail="Scene not found")
    job_manager.delete_scene_data(scene_id)
    return {"status": "deleted", "scene_id": scene_id}

from fastapi.responses import FileResponse

@router.get("/scene/{scene_id}/download/{filename}")
def download_artifact(scene_id: str, filename: str):
    artifacts = list_artifacts(scene_id)
    if filename not in artifacts:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(artifacts[filename])

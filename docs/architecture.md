# Architecture

GeoDiff3D is designed with a strict separation between CPU-only local development and heavy remote GPU inference.

## System Overview

```text
LOCAL LAPTOP
Next.js frontend
FastAPI API
Three.js viewer
evaluation/UI code
CPU tests
|
| Git / API / artifact transfer
v
REMOTE GPU ENVIRONMENT
VGGT inference
Marigold inference
experiments
evaluation
|
v
ARTIFACT STORAGE
depth
point clouds
metrics
logs
|
v
PUBLIC WEB APPLICATION
```

## Backend Architecture

The backend is built with Python, FastAPI, and PyTorch. It uses a job-based API so GPU inference does not block HTTP requests.

### API Routes

- `POST /api/reconstruct`: Full reconstruction pipeline
- `POST /api/reconstruct/baseline`: Geometric baseline only
- `POST /api/reconstruct/guided`: Guided reconstruction
- `POST /api/evaluate`: Compute metrics
- `GET /api/jobs/{job_id}`: Poll job status
- `GET /api/scene/{scene_id}`: Scene metadata
- `GET /api/scene/{scene_id}/metrics`: Evaluation metrics
- `GET /api/scene/{scene_id}/artifacts`: Retrieve artifacts (PLY files, etc.)
- `GET /api/health`: System health check

### Job States

- `queued`
- `preprocessing`
- `vggt`
- `diffusion`
- `alignment`
- `fusion`
- `reconstruction`
- `evaluation`
- `completed`
- `failed`

## Storage Hierarchy

For the MVP, local filesystem storage is sufficient, following this structure:

```text
storage/
  scenes/
    {scene_id}/
      input/
      vggt/
      diffusion/
      aligned/
      baseline/
      guided/
      metrics/
      logs/
```

## Deployment

Recommended separation: Next.js frontend, FastAPI API, remote GPU inference worker, and optional PostgreSQL/Supabase plus object storage. The frontend is independently deployable from the inference worker.

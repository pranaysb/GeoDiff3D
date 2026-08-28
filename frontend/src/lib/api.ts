export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export interface Job {
  job_id: string;
  scene_id: string;
  state: string;
  message?: string;
}

export interface Scene {
  scene_id: string;
  job_id: string;
  image_count: number;
}

export interface Artifacts {
  scene_id: string;
  artifacts: {
    [key: string]: string;
  };
}

// Mirrors inference/gpu_pipeline.py's real metrics.json (see core/math.py's
// point_cloud_stats and inference/vggt_pipeline.py / marigold_pipeline.py's
// metadata). There are no ground-truth accuracy fields here on purpose --
// none are computed, so none are typed, so none can be silently displayed.
export interface ModelRunMetadata {
  model: string;
  device: string;
  dtype?: string;
  num_views: number;
  runtime_sec: number;
  [key: string]: unknown;
}

export interface AlignmentMetric {
  view: number;
  scale: number;
  shift: number;
  valid_pixel_count: number;
  residual: number;
}

export interface PointCloudStats {
  num_points: number;
  bbox_min?: number[];
  bbox_max?: number[];
  bbox_size?: number[];
  depth_range?: [number, number];
}

export interface SceneMetrics {
  num_views: number;
  vggt: ModelRunMetadata;
  marigold: ModelRunMetadata;
  alignment: AlignmentMetric[];
  baseline: PointCloudStats;
  guided: PointCloudStats;
  note: string;
}

export async function createReconstruction(files: File[]): Promise<{ job_id: string; message: string }> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("images", file);
  }
  
  const res = await fetch(`${API_URL}/reconstruct`, {
    method: "POST",
    body: formData,
  });
  
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to start reconstruction");
  }
  
  return res.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`${API_URL}/jobs/${jobId}`);
  if (!res.ok) {
    throw new Error("Failed to get job");
  }
  return res.json();
}

export async function getScene(sceneId: string): Promise<Scene> {
  const res = await fetch(`${API_URL}/scene/${sceneId}`);
  if (!res.ok) {
    throw new Error("Failed to get scene");
  }
  return res.json();
}

export async function getArtifacts(sceneId: string): Promise<Artifacts> {
  const res = await fetch(`${API_URL}/scene/${sceneId}/artifacts`);
  if (!res.ok) {
    throw new Error("Failed to get artifacts");
  }
  return res.json();
}

export function getArtifactDownloadUrl(sceneId: string, filename: string): string {
  return `${API_URL}/scene/${sceneId}/download/${filename}`;
}

/**
 * Fetches the real per-job metrics.json, if the job produced one. Returns
 * null (not an error) when it's missing -- e.g. the CPU POC engine doesn't
 * write metrics.json at all, only the real GEODIFF3D_ENGINE=real path does.
 * Callers must treat null as "no data available", never substitute a guess.
 */
export async function getSceneMetrics(sceneId: string): Promise<SceneMetrics | null> {
  const res = await fetch(getArtifactDownloadUrl(sceneId, "metrics.json"));
  if (!res.ok) {
    return null;
  }
  return res.json();
}

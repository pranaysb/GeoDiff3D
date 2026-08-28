"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { getArtifacts, getArtifactDownloadUrl } from "@/lib/api";
import PointCloudViewer from "@/components/PointCloudViewer";
import { AlertCircle, Loader2 } from "lucide-react";

function ComparePageInner() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const sceneIdParam = searchParams.get("scene") ?? "";

  const [sceneIdInput, setSceneIdInput] = useState(sceneIdParam);
  const [urls, setUrls] = useState<{ baseline?: string; guided?: string }>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pointSize, setPointSize] = useState(2);

  useEffect(() => {
    if (!sceneIdParam) {
      setUrls({});
      return;
    }
    setLoading(true);
    setError(null);
    getArtifacts(sceneIdParam)
      .then((data) => {
        const available = data.artifacts;
        setUrls({
          baseline: available["baseline.ply"]
            ? getArtifactDownloadUrl(sceneIdParam, "baseline.ply")
            : undefined,
          guided: available["guided.ply"]
            ? getArtifactDownloadUrl(sceneIdParam, "guided.ply")
            : undefined,
        });
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load scene artifacts");
      })
      .finally(() => setLoading(false));
  }, [sceneIdParam]);

  const goToScene = (id: string) => {
    if (id.trim()) {
      router.push(`/compare?scene=${encodeURIComponent(id.trim())}`);
    }
  };

  return (
    <div className="max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">Compare Reconstructions</h1>
      <p className="text-zinc-400 mb-6">
        Side-by-side: VGGT-only geometric baseline vs. GeoDiff3D confidence-guided fusion.
      </p>

      <div className="flex flex-wrap items-center gap-3 mb-6">
        <input
          type="text"
          value={sceneIdInput}
          onChange={(e) => setSceneIdInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && goToScene(sceneIdInput)}
          placeholder="Scene ID (e.g. scene_4f28acd1)"
          className="flex-1 min-w-[240px] bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-emerald-600"
        />
        <button
          onClick={() => goToScene(sceneIdInput)}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          Load
        </button>
        {sceneIdParam && (
          <div className="flex items-center space-x-2 ml-auto">
            <span className="text-xs text-zinc-500">Point Size</span>
            <input
              type="range"
              min="1"
              max="10"
              step="0.5"
              value={pointSize}
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              className="w-24 accent-emerald-500"
            />
          </div>
        )}
      </div>

      {!sceneIdParam && (
        <div className="flex items-center justify-center h-64 border border-zinc-800 rounded-xl bg-zinc-900/30 text-zinc-500 text-center px-6">
          Enter a scene ID above, or{" "}
          <a href="/reconstruct" className="text-emerald-500 hover:underline mx-1">
            reconstruct a new scene
          </a>{" "}
          and open its Compare link from the scene viewer.
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center h-64">
          <Loader2 className="w-6 h-6 animate-spin text-zinc-500 mb-2" />
          <p className="text-zinc-500 text-sm">Loading artifacts for {sceneIdParam}...</p>
        </div>
      )}

      {error && (
        <div className="mb-6 p-4 bg-red-900/30 border border-red-800 rounded-lg flex items-start text-red-200">
          <AlertCircle className="w-5 h-5 mr-3 mt-0.5 flex-shrink-0" />
          <p>{error}</p>
        </div>
      )}

      {sceneIdParam && !loading && !error && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold text-zinc-300">Baseline (VGGT-only)</h2>
              {!urls.baseline && <span className="text-xs text-zinc-600">not available</span>}
            </div>
            <div className="h-[60vh] rounded-xl overflow-hidden border border-zinc-800">
              {urls.baseline ? (
                <PointCloudViewer url={urls.baseline} pointSize={pointSize} />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-zinc-950 text-zinc-600 text-sm">
                  baseline.ply not found for this scene
                </div>
              )}
            </div>
          </div>
          <div>
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-semibold text-emerald-400">GeoDiff3D (confidence-guided fusion)</h2>
              {!urls.guided && <span className="text-xs text-zinc-600">not available</span>}
            </div>
            <div className="h-[60vh] rounded-xl overflow-hidden border border-zinc-800">
              {urls.guided ? (
                <PointCloudViewer url={urls.guided} pointSize={pointSize} />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-zinc-950 text-zinc-600 text-sm">
                  guided.ply not found for this scene
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {sceneIdParam && !loading && !error && (urls.baseline || urls.guided) && (
        <p className="text-xs text-zinc-600 mt-4 text-center">
          Each viewer has independent orbit controls (drag to rotate, scroll to zoom) --
          camera position is not synchronized between the two.
        </p>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="text-zinc-500 text-center py-20">Loading...</div>}>
      <ComparePageInner />
    </Suspense>
  );
}

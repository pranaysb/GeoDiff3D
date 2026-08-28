"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { getSceneMetrics, SceneMetrics } from "@/lib/api";
import { AlertCircle, Loader2 } from "lucide-react";

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-zinc-900/60 border border-zinc-800 rounded-lg px-4 py-3 min-w-0">
      <p className="text-xs text-zinc-500 mb-1">{label}</p>
      <p className="text-lg font-semibold text-zinc-100 break-words">{value}</p>
    </div>
  );
}

function ModelBlock({ title, meta }: { title: string; meta: SceneMetrics["vggt"] }) {
  return (
    <div className="border border-zinc-800 rounded-xl p-5 bg-zinc-900/30">
      <h3 className="font-semibold text-zinc-200 mb-3">{title}</h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Model" value={meta.model} />
        <StatCard label="Device" value={meta.device} />
        <StatCard label="Dtype" value={meta.dtype ?? "n/a"} />
        <StatCard label="Runtime" value={`${meta.runtime_sec}s`} />
      </div>
    </div>
  );
}

function PointCloudBlock({ title, stats }: { title: string; stats: SceneMetrics["baseline"] }) {
  return (
    <div className="border border-zinc-800 rounded-xl p-5 bg-zinc-900/30">
      <h3 className="font-semibold text-zinc-200 mb-3">{title}</h3>
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Points" value={stats.num_points.toLocaleString()} />
        <StatCard
          label="Depth range"
          value={stats.depth_range ? `${stats.depth_range[0].toFixed(2)} - ${stats.depth_range[1].toFixed(2)}` : "n/a"}
        />
      </div>
    </div>
  );
}

function EvaluationPageInner() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const sceneIdParam = searchParams.get("scene") ?? "";

  const [sceneIdInput, setSceneIdInput] = useState(sceneIdParam);
  const [metrics, setMetrics] = useState<SceneMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (!sceneIdParam) {
      setMetrics(null);
      setChecked(false);
      return;
    }
    setLoading(true);
    setError(null);
    setChecked(false);
    getSceneMetrics(sceneIdParam)
      .then((data) => setMetrics(data))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load metrics"))
      .finally(() => {
        setLoading(false);
        setChecked(true);
      });
  }, [sceneIdParam]);

  const goToScene = (id: string) => {
    if (id.trim()) {
      router.push(`/evaluation?scene=${encodeURIComponent(id.trim())}`);
    }
  };

  return (
    <div className="max-w-7xl mx-auto py-6">
      <h1 className="text-3xl font-bold mb-2">Quantitative Evaluation</h1>
      <p className="text-zinc-400 mb-6">
        Real per-job metrics from the reconstruction pipeline. No ground-truth dataset (e.g. DTU,
        Tanks and Temples) has been run against this project, so no accuracy metrics (AbsRel, RMSE,
        Chamfer, F-score) are computed or shown here -- only what the real engine actually measured.
      </p>

      <div className="flex flex-wrap items-center gap-3 mb-8">
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
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center h-64">
          <Loader2 className="w-6 h-6 animate-spin text-zinc-500 mb-2" />
          <p className="text-zinc-500 text-sm">Loading metrics for {sceneIdParam}...</p>
        </div>
      )}

      {error && (
        <div className="mb-6 p-4 bg-red-900/30 border border-red-800 rounded-lg flex items-start text-red-200">
          <AlertCircle className="w-5 h-5 mr-3 mt-0.5 flex-shrink-0" />
          <p>{error}</p>
        </div>
      )}

      {!sceneIdParam && (
        <div className="border border-zinc-800 rounded-xl overflow-hidden bg-zinc-900/30">
          <div className="p-16 flex flex-col items-center justify-center text-center">
            <p className="text-zinc-300 font-medium mb-2">No scene selected.</p>
            <p className="text-zinc-500 max-w-lg">
              Enter a scene ID above, or open a scene&apos;s Evaluation link from the scene viewer
              after a reconstruction completes.
            </p>
          </div>
        </div>
      )}

      {sceneIdParam && checked && !metrics && !error && (
        <div className="border border-zinc-800 rounded-xl overflow-hidden bg-zinc-900/30">
          <div className="p-16 flex flex-col items-center justify-center text-center">
            <p className="text-zinc-300 font-medium mb-2">No metrics available for this scene.</p>
            <p className="text-zinc-500 max-w-lg">
              This project enforces strict scientific reproducibility -- no fabricated numbers are
              ever displayed. This job likely ran on the CPU math-validation engine
              (<code className="text-zinc-400">GEODIFF3D_ENGINE=poc</code>), which doesn&apos;t
              produce a metrics.json. Run a job with the real VGGT+Marigold engine to see data here.
            </p>
          </div>
        </div>
      )}

      {metrics && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ModelBlock title="VGGT" meta={metrics.vggt} />
            <ModelBlock title="Marigold" meta={metrics.marigold} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <PointCloudBlock title="Baseline point cloud" stats={metrics.baseline} />
            <PointCloudBlock title="GeoDiff3D point cloud" stats={metrics.guided} />
          </div>

          <div className="border border-zinc-800 rounded-xl p-5 bg-zinc-900/30">
            <h3 className="font-semibold text-zinc-200 mb-3">Per-view depth alignment</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-zinc-500 border-b border-zinc-800">
                    <th className="py-2 pr-4">View</th>
                    <th className="py-2 pr-4">Scale</th>
                    <th className="py-2 pr-4">Shift</th>
                    <th className="py-2 pr-4">Valid pixels</th>
                    <th className="py-2 pr-4">Residual</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.alignment.map((a) => (
                    <tr key={a.view} className="border-b border-zinc-900 text-zinc-300">
                      <td className="py-2 pr-4">{a.view}</td>
                      <td className="py-2 pr-4">{a.scale.toFixed(4)}</td>
                      <td className="py-2 pr-4">{a.shift.toFixed(4)}</td>
                      <td className="py-2 pr-4">{a.valid_pixel_count.toLocaleString()}</td>
                      <td className="py-2 pr-4">{a.residual.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <p className="text-xs text-zinc-600 italic">{metrics.note}</p>
        </div>
      )}
    </div>
  );
}

export default function EvaluationPage() {
  return (
    <Suspense fallback={<div className="text-zinc-500 text-center py-20">Loading...</div>}>
      <EvaluationPageInner />
    </Suspense>
  );
}

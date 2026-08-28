"use client";

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { getScene, getArtifacts, getArtifactDownloadUrl, Scene } from '@/lib/api';
import PointCloudViewer from '@/components/PointCloudViewer';
import { Loader2, AlertCircle, Layers, MonitorPlay, BarChart3 } from 'lucide-react';

export default function ScenePage() {
  const params = useParams();
  const id = params.id as string;
  
  const [scene, setScene] = useState<Scene | null>(null);
  const [urls, setUrls] = useState<{ baseline?: string, guided?: string }>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'guided' | 'baseline'>('guided');
  const [pointSize, setPointSize] = useState(2);

  useEffect(() => {
    async function loadData() {
      try {
        const [sceneData, artifactsData] = await Promise.all([
          getScene(id),
          getArtifacts(id)
        ]);
        
        setScene(sceneData);
        
        const available = artifactsData.artifacts;
        setUrls({
          baseline: available['baseline.ply'] ? getArtifactDownloadUrl(id, 'baseline.ply') : undefined,
          guided: available['guided.ply'] ? getArtifactDownloadUrl(id, 'guided.ply') : undefined,
        });
        
      } catch (err: any) {
        setError(err.message || 'Failed to load scene');
      } finally {
        setLoading(false);
      }
    }
    
    if (id) {
      loadData();
    }
  }, [id]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh]">
        <Loader2 className="w-8 h-8 animate-spin text-zinc-500 mb-4" />
        <p className="text-zinc-400">Loading scene data...</p>
      </div>
    );
  }

  if (error || !scene) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh]">
        <AlertCircle className="w-12 h-12 text-red-500 mb-4" />
        <h2 className="text-xl font-semibold mb-2">Error Loading Scene</h2>
        <p className="text-zinc-400">{error || 'Scene not found'}</p>
      </div>
    );
  }

  const currentUrl = mode === 'guided' ? urls.guided : urls.baseline;

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex justify-between items-center mb-4">
        <div>
          <h1 className="text-2xl font-bold">Scene {id.replace('scene_', '')}</h1>
          <p className="text-sm text-zinc-400">Reconstructed from {scene.image_count} images</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 bg-zinc-900 rounded-lg p-1 border border-zinc-800">
            <button
              onClick={() => setMode('baseline')}
              disabled={!urls.baseline}
              className={`px-3 py-1.5 text-sm rounded-md transition-colors ${mode === 'baseline' ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200'} disabled:opacity-30`}
            >
              Baseline
            </button>
            <button
              onClick={() => setMode('guided')}
              disabled={!urls.guided}
              className={`px-3 py-1.5 text-sm rounded-md transition-colors flex items-center ${mode === 'guided' ? 'bg-emerald-600 text-white' : 'text-zinc-400 hover:text-zinc-200'} disabled:opacity-30`}
            >
              <Layers className="w-4 h-4 mr-1.5" />
              Diffusion-Guided
            </button>
          </div>

          <Link
            href={`/compare?scene=${encodeURIComponent(id)}`}
            className="inline-flex items-center px-3 py-1.5 text-sm rounded-md bg-zinc-900 border border-zinc-800 text-zinc-300 hover:text-white hover:border-zinc-700 transition-colors"
          >
            <MonitorPlay className="w-4 h-4 mr-1.5" />
            Compare
          </Link>
          <Link
            href={`/evaluation?scene=${encodeURIComponent(id)}`}
            className="inline-flex items-center px-3 py-1.5 text-sm rounded-md bg-zinc-900 border border-zinc-800 text-zinc-300 hover:text-white hover:border-zinc-700 transition-colors"
          >
            <BarChart3 className="w-4 h-4 mr-1.5" />
            Metrics
          </Link>

          <div className="flex items-center space-x-2">
            <span className="text-xs text-zinc-500">Point Size</span>
            <input 
              type="range" 
              min="1" max="10" step="0.5" 
              value={pointSize} 
              onChange={(e) => setPointSize(parseFloat(e.target.value))}
              className="w-24 accent-emerald-500"
            />
          </div>
        </div>
      </div>

      <div className="flex-1 relative rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950">
        {!currentUrl ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-zinc-500">Artifact not available for this mode.</p>
          </div>
        ) : (
          <PointCloudViewer url={currentUrl} pointSize={pointSize} />
        )}
        
        <div className="absolute bottom-4 right-4 bg-black/60 backdrop-blur-sm px-3 py-2 rounded-lg border border-white/10 text-xs text-white/70">
          Left Click: Orbit • Right Click: Pan • Scroll: Zoom
        </div>
      </div>
    </div>
  );
}

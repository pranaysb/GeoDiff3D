"use client";

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { createReconstruction, getJob } from '@/lib/api';
import { UploadCloud, X, Loader2, AlertCircle } from 'lucide-react';

export default function ReconstructPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobState, setJobState] = useState<string | null>(null);
  const router = useRouter();

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files).filter(f => 
      ['image/jpeg', 'image/png', 'image/webp'].includes(f.type)
    );
    setFiles(prev => [...prev, ...dropped].slice(0, 12));
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selected = Array.from(e.target.files);
      setFiles(prev => [...prev, ...selected].slice(0, 12));
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const pollJob = async (jobId: string) => {
    try {
      const job = await getJob(jobId);
      setJobState(job.state);
      
      if (job.state === 'completed') {
        router.push(`/scene/${job.scene_id}`);
      } else if (job.state === 'failed') {
        setError(job.message || 'Reconstruction failed');
        setLoading(false);
      } else {
        setTimeout(() => pollJob(jobId), 1000);
      }
    } catch (err: any) {
      setError(err.message || "Failed to poll job status");
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (files.length < 2 || files.length > 12) {
      setError("Please provide between 2 and 12 images");
      return;
    }
    setLoading(true);
    setError(null);
    setJobState('uploading');

    try {
      const { job_id } = await createReconstruction(files);
      setJobState('queued');
      pollJob(job_id);
    } catch (err: any) {
      setError(err.message || "Upload failed");
      setLoading(false);
      setJobState(null);
    }
  };

  return (
    <div className="max-w-3xl mx-auto py-10">
      <h1 className="text-3xl font-bold mb-2">Reconstruct a Scene</h1>
      <p className="text-zinc-400 mb-8">Upload 2 to 12 overlapping images of a static scene.</p>
      
      {error && (
        <div className="mb-6 p-4 bg-red-900/30 border border-red-800 rounded-lg flex items-start text-red-200">
          <AlertCircle className="w-5 h-5 mr-3 mt-0.5 flex-shrink-0" />
          <p>{error}</p>
        </div>
      )}

      <div 
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className={`border-2 border-dashed rounded-xl p-10 text-center transition-colors
          ${loading ? 'border-zinc-800 opacity-50 pointer-events-none' : 'border-zinc-700 hover:border-zinc-500 bg-zinc-900/30'}`}
      >
        <UploadCloud className="w-12 h-12 mx-auto text-zinc-500 mb-4" />
        <p className="text-zinc-300 mb-2">Drag and drop images here, or click to select files</p>
        <p className="text-sm text-zinc-500 mb-6">Supported: JPG, PNG, WEBP. Max 10MB per file.</p>
        
        <input 
          type="file" 
          multiple 
          accept="image/jpeg,image/png,image/webp"
          className="hidden" 
          id="file-upload"
          onChange={handleFileChange}
          disabled={loading}
        />
        <label 
          htmlFor="file-upload" 
          className="inline-flex items-center justify-center px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white font-medium rounded-lg cursor-pointer transition-colors"
        >
          Select Images
        </label>
      </div>

      {files.length > 0 && (
        <div className="mt-8">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-medium text-zinc-300">Selected Images ({files.length}/12)</h3>
            <button 
              onClick={() => setFiles([])}
              disabled={loading}
              className="text-sm text-zinc-500 hover:text-zinc-300"
            >
              Clear all
            </button>
          </div>
          
          <div className="grid grid-cols-4 sm:grid-cols-6 gap-4 mb-8">
            {files.map((file, i) => (
              <div key={i} className="relative aspect-square rounded-md overflow-hidden bg-zinc-800 border border-zinc-700 group">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={URL.createObjectURL(file)} alt="" className="object-cover w-full h-full opacity-70 group-hover:opacity-100 transition-opacity" />
                {!loading && (
                  <button 
                    onClick={() => removeFile(i)}
                    className="absolute top-1 right-1 p-1 bg-black/60 rounded-full text-white opacity-0 group-hover:opacity-100 hover:bg-red-500 transition-all"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
          </div>

          <div className="flex flex-col items-center">
            {loading ? (
              <div className="flex flex-col items-center w-full max-w-sm">
                <Loader2 className="w-8 h-8 animate-spin text-emerald-500 mb-4" />
                <p className="text-zinc-300 font-medium capitalize mb-2">{jobState?.replace('_', ' ')}...</p>
                <div className="w-full bg-zinc-800 rounded-full h-2 mb-1 overflow-hidden">
                  <div className="bg-emerald-500 h-2 rounded-full animate-pulse" style={{ width: '100%' }}></div>
                </div>
                <p className="text-xs text-zinc-500 text-center mt-2">
                  Running the reconstruction pipeline -- this can take anywhere
                  from seconds to a few minutes depending on the backend.
                </p>
              </div>
            ) : (
              <button 
                onClick={handleSubmit}
                disabled={files.length < 2}
                className="w-full max-w-sm py-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-800 disabled:text-zinc-500 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
              >
                Start Reconstruction
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

import Link from 'next/link';
import { ArrowRight, Box } from 'lucide-react';

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] text-center">
      <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-r from-zinc-100 to-zinc-500">
        GeoDiff3D
      </h1>
      <h2 className="text-2xl md:text-3xl text-zinc-400 mb-8 max-w-3xl">
        Diffusion-Guided 3D Scene Reconstruction
      </h2>
      <p className="text-lg text-zinc-500 max-w-2xl mb-12">
        An experimental research system studying whether diffusion-derived depth priors can improve sparse-view geometric reconstruction.
      </p>
      
      <div className="flex flex-col sm:flex-row gap-4">
        <Link href="/reconstruct" className="inline-flex items-center justify-center px-6 py-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition-colors">
          <Box className="w-5 h-5 mr-2" />
          Reconstruct a Scene
        </Link>
        <Link href="/research" className="inline-flex items-center justify-center px-6 py-3 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-white font-medium transition-colors border border-zinc-700">
          Explore the Research
          <ArrowRight className="w-5 h-5 ml-2" />
        </Link>
      </div>

      <div className="mt-20 w-full max-w-4xl p-6 rounded-xl border border-zinc-800 bg-zinc-900/50">
        <h3 className="text-xl font-semibold mb-4 text-left text-zinc-200">Precomputed Research Demo</h3>
        <div className="aspect-video bg-zinc-950 rounded-lg flex items-center justify-center border border-zinc-800/50">
          <p className="text-zinc-600">Sample interactive viewer preview goes here</p>
        </div>
      </div>
    </div>
  );
}

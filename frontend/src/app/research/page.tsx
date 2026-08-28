export default function ResearchPage() {
  return (
    <div className="max-w-4xl mx-auto py-10 prose prose-invert prose-emerald">
      <h1 className="text-4xl font-bold mb-8">Research Methodology</h1>
      
      <div className="bg-emerald-950/30 border border-emerald-900/50 p-4 rounded-lg mb-8 text-emerald-200 text-sm">
        <strong>Note:</strong> GeoDiff3D is currently an experimental research prototype. The system aims to measure whether diffusion priors improve geometric reconstruction. We do not claim state-of-the-art performance until experiments mathematically establish those claims.
      </div>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-4 text-zinc-100">1. Problem Definition</h2>
        <p className="text-zinc-300 leading-relaxed">
          Multi-view 3D scene reconstruction typically fails or produces artifacts in sparse-view settings (e.g., 2-4 images) or textureless regions where cross-view feature matching is unreliable. We aim to solve this by injecting learned priors.
        </p>
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-4 text-zinc-100">2. Geometric Reconstruction</h2>
        <p className="text-zinc-300 leading-relaxed">
          We use a feed-forward Visual Geometry Grounded Transformer (VGGT) baseline. It provides high-frequency geometric details where feature tracks exist, but struggles with completeness in unobserved regions.
        </p>
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-4 text-zinc-100">3. Diffusion Depth Prior</h2>
        <p className="text-zinc-300 leading-relaxed">
          We extract monocular depth estimates using diffusion models (Marigold). These models possess a strong semantic understanding of scene layout, object boundaries, and textureless surfaces.
        </p>
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-4 text-zinc-100">4. Depth Alignment</h2>
        <p className="text-zinc-300 leading-relaxed">
          Because diffusion depth is scale-and-shift invariant (relative depth), we solve a robust least-squares alignment problem per view to map the diffusion prediction into the metric scale of the geometric baseline.
        </p>
      </section>

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-4 text-zinc-100">5. Confidence-Guided Fusion</h2>
        <p className="text-zinc-300 leading-relaxed">
          The core contribution is a spatially-varying fusion map. We weight the geometric depth in high-confidence trackable regions, and fallback to the aligned diffusion depth in low-confidence textureless areas.
        </p>
      </section>
    </div>
  );
}

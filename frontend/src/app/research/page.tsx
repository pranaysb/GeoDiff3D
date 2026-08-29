interface ScenarioResult {
  scene: string;
  label: string;
  vggt_only: number;
  marigold_only: number;
  naive_average: number;
  geodiff3d_fusion: number;
}

// Real numbers from experiments/RESULTS.md (Phase 4 ablation, revision 5 --
// tuned confidence-guided fusion, independently confirmed on a fresh GPU
// run). Cross-view consistency: mean absolute relative depth error between
// neighboring views, lower is better. No ground truth exists for these
// scenes, so this is a self-consistency diagnostic, not an accuracy metric.
const ABLATION_RESULTS: ScenarioResult[] = [
  { scene: "kitchen", label: "Kitchen (tabletop object)", vggt_only: 0.0739, marigold_only: 0.1284, naive_average: 0.0903, geodiff3d_fusion: 0.0707 },
  { scene: "llff_fern", label: "LLFF Fern (outdoor plant)", vggt_only: 0.0381, marigold_only: 0.0388, naive_average: 0.0362, geodiff3d_fusion: 0.0367 },
  { scene: "llff_flower", label: "LLFF Flower (outdoor close-up)", vggt_only: 0.0874, marigold_only: 0.1532, naive_average: 0.1103, geodiff3d_fusion: 0.0863 },
  { scene: "room", label: "Room (indoor)", vggt_only: 0.0285, marigold_only: 0.1492, naive_average: 0.0841, geodiff3d_fusion: 0.0289 },
];

const METHOD_LABELS: { key: keyof Omit<ScenarioResult, "scene" | "label">; label: string }[] = [
  { key: "vggt_only", label: "VGGT-only" },
  { key: "marigold_only", label: "Marigold-only" },
  { key: "naive_average", label: "Naive 50/50" },
  { key: "geodiff3d_fusion", label: "GeoDiff3D fusion" },
];

function ResultsTable({ result }: { result: ScenarioResult }) {
  const best = Math.min(result.vggt_only, result.marigold_only, result.naive_average, result.geodiff3d_fusion);
  const fusionDelta = ((result.geodiff3d_fusion - result.vggt_only) / result.vggt_only) * 100;
  return (
    <div className="mb-6">
      <h3 className="text-lg font-medium text-zinc-100 mb-2">{result.label}</h3>
      <div className="overflow-x-auto rounded-lg border border-zinc-800">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-zinc-900/60 text-zinc-400">
              <th className="text-left px-4 py-2 font-medium">Method</th>
              <th className="text-right px-4 py-2 font-medium">Cross-view consistency</th>
            </tr>
          </thead>
          <tbody>
            {METHOD_LABELS.map(({ key, label }) => {
              const value = result[key];
              const isBest = value === best;
              const isFusion = key === "geodiff3d_fusion";
              return (
                <tr key={key} className={`border-t border-zinc-800 ${isFusion ? "bg-emerald-950/20" : ""}`}>
                  <td className={`px-4 py-2 ${isFusion ? "text-emerald-300 font-medium" : "text-zinc-300"}`}>{label}</td>
                  <td className={`px-4 py-2 text-right font-mono ${isBest ? "text-emerald-400 font-semibold" : "text-zinc-300"}`}>
                    {value.toFixed(4)}{isBest ? " (best)" : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-zinc-500 mt-2">
        Fusion vs. VGGT-only: {fusionDelta <= 0 ? "beats it by " : "loses by "}
        {Math.abs(fusionDelta).toFixed(1)}%
      </p>
    </div>
  );
}

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

      <section className="mb-10">
        <h2 className="text-2xl font-semibold mb-4 text-zinc-100">6. Experimental Results (Phase 4 Ablation)</h2>

        <div className="bg-emerald-950/30 border border-emerald-900/50 p-4 rounded-lg mb-6 text-emerald-200 text-sm not-prose">
          <strong>No ground truth exists for any scene below.</strong> Cross-view
          consistency (mean absolute relative depth error between neighboring
          views) is a self-consistency diagnostic, not an accuracy metric — a
          method can score well here while being uniformly wrong. Full
          methodology, revision history, and caveats are in{" "}
          <a href="https://github.com/pranaysb/GeoDiff3D/blob/main/experiments/RESULTS.md"
             target="_blank" rel="noopener noreferrer" className="underline text-emerald-300">
            experiments/RESULTS.md
          </a>.
        </div>

        <p className="text-zinc-300 leading-relaxed mb-6">
          Four methods (VGGT-only, Marigold-only, naive 50/50 averaging, and
          GeoDiff3D confidence-guided fusion) were run on four real multi-view
          scenes on a Tesla T4 GPU, sharing identical inputs and preprocessing
          per scene. An initial run found the fusion never beat VGGT-only,
          which led to diagnosing and fixing a bug in the confidence
          weighting, then tuning its two parameters, then confirming the
          tuned result on a fresh, independent GPU run (new, unseeded
          Marigold samples). <strong className="text-emerald-300">Result: fusion
          beats VGGT-only outright in 3 of the 4 scenes tested</strong>, losing
          only narrowly (1.4%) in the fourth. This is a genuine, largely — but
          not fully — positive result, not a claim that fusion always wins.
        </p>

        <div className="not-prose">
          {ABLATION_RESULTS.map((r) => <ResultsTable key={r.scene} result={r} />)}
        </div>

        <h3 className="text-xl font-medium text-zinc-100 mt-8 mb-3">Qualitative depth comparisons</h3>
        <p className="text-zinc-300 leading-relaxed mb-4">
          Each figure shows all 6 views for all 4 methods on one scene.
          Visual similarity between methods does not imply comparable
          cross-view consistency — each subplot is auto-scaled to its own
          min/max, so small but geometrically consequential differences
          often aren&apos;t visible by eye (see <code>room</code> below, where
          all four methods look nearly identical despite real score
          differences).
        </p>
        <div className="not-prose space-y-6">
          {ABLATION_RESULTS.map((r) => (
            <div key={r.scene}>
              <p className="text-sm text-zinc-400 mb-2">{r.label}</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`/ablation/${r.scene}_depth_comparison.png`}
                alt={`Depth comparison across VGGT-only, Marigold-only, naive averaging, and GeoDiff3D fusion for the ${r.label} scene`}
                className="w-full rounded-lg border border-zinc-800"
              />
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

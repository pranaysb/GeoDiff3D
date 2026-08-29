# GeoDiff3D

**Geometry-Guided Diffusion Depth Fusion for Multi-View 3D Reconstruction**

A research prototype investigating whether a diffusion-based monocular depth prior
(Marigold) can complement a feed-forward multi-view geometry model (VGGT) for
sparse-view 3D reconstruction: geometric depth establishes the coordinate system
and scale; diffusion depth is aligned into it and blended in proportion to VGGT's
own per-pixel confidence.

## Verified

The following has actually been executed and its output inspected — not just
implemented:

- **CPU mathematical POC** (`scripts/poc_pipeline.py`): synthetic depth with a
  known injected scale/shift is recovered by the alignment math (scale≈2.5,
  shift≈-1.0), validating the fusion math in isolation from any model.
- **Real VGGT-1B inference on a T4 GPU**: `facebook/VGGT-1B`, 4 real
  photographs, real depth/confidence/camera outputs.
- **Real Marigold inference on the same T4**: `prs-eth/marigold-depth-v1-1`,
  real per-view diffusion depth.
- **Depth alignment and confidence-guided fusion** on the real VGGT/Marigold
  outputs above (not synthetic data).
- **Point-cloud reconstruction**: `baseline.ply` (VGGT depth only) and
  `guided.ply` (fused depth) exported from the real run, 721,572 points each.
- **The FastAPI backend's real path**: `POST /api/reconstruct` against the
  real `VGGTMarigoldEngine`, through every job state, to `completed`, with
  `baseline.ply` / `guided.ply` / `metrics.json` / `alignment_metrics.json`
  confirmed present via `GET /api/scene/{scene_id}/artifacts`.
- **Backend CPU test suite** (`backend/tests/`): 4/4 passing against the
  default `GEODIFF3D_ENGINE=poc` engine (no GPU required).
- **Phase 4 ablation** (`experiments/run_ablation.py`, full results in
  `experiments/RESULTS.md`): VGGT-only, Marigold-only, naive 50/50 averaging,
  and GeoDiff3D confidence-guided fusion run on four real scenes (`kitchen`,
  `llff_fern`, `llff_flower`, `room`, 6 views each). An initial run found
  fusion never beat VGGT-only on cross-view consistency, which led to
  diagnosing and fixing a bug in the confidence weighting (it was giving
  even high-confidence pixels a near-50/50 blend instead of being genuinely
  confidence-selective), then offline-tuning the fix's two parameters
  (`experiments/tune_fusion.py`) against cached real GPU outputs from all
  four scenes, then **confirming the tuned result on a fresh, independent
  GPU run** (new, unseeded Marigold samples) — see `core/math.py::fuse_depths`
  and `experiments/RESULTS.md`. **Confirmed result: fusion beats VGGT-only
  outright in 3 of 4 scenes** (`kitchen`, `llff_fern`, `llff_flower`), losing
  only narrowly (1.4%) in the 4th (`room`). Fusion also beats naive averaging
  in 3 of 4 scenes and Marigold-only in all 4. This is a genuine,
  largely-but-not-fully positive result, not a claim that fusion always
  wins. No ground truth
  exists for any of these scenes, so this is a self-consistency comparison,
  not an accuracy one — see `experiments/RESULTS.md` for the full breakdown,
  caveats, and revision history.

## In progress / not yet verified

- **Quantitative accuracy metrics** (AbsRel, RMSE, δ-thresholds, Chamfer
  distance, F-score) against ground truth. No ground-truth dataset has been
  used yet; every metrics file in this repo reports self-consistency
  diagnostics only and says so explicitly.
- **3D viewer / frontend wiring to the real backend** — the Next.js frontend
  exists but has not been connected to a completed real job.
Nothing here is claimed to be "production-ready," "state-of-the-art," or to
"improve" reconstruction quality — the experiments that tested an improvement
claim (Phase 4, above) did not support it.

## Repository Structure

- `core/`: Alignment, confidence-fusion, and unprojection math — the single
  source of truth used by the CPU POC, the real inference pipeline, and the
  notebook demo.
- `inference/`: Real VGGT (`vggt_pipeline.py`) and Marigold
  (`marigold_pipeline.py`) inference, and the orchestrator
  (`gpu_pipeline.py`) that sequences them under T4 VRAM constraints. No mock
  fallbacks — failures raise, they are never masked with fabricated output.
- `backend/`: FastAPI job service. `GEODIFF3D_ENGINE=poc` (default) runs the
  CPU math-validation engine; `GEODIFF3D_ENGINE=real` runs the actual VGGT +
  Marigold pipeline (requires a CUDA GPU and `inference/requirements_gpu.txt`).
- `experiments/`: Phase 4 ablation (`ablation.py`, `run_ablation.py`) and its
  results (`RESULTS.md`, `ablation_results/`).
- `tests/`: Unit tests for `core/math.py` and the ablation's consistency
  metric (`backend/tests/` covers the API separately).
- `frontend/`: Next.js visualization app (not yet wired to real jobs).
- `notebooks/`: Colab demo notebook — a client of `core/` and `inference/`,
  not a separate implementation.
- `docs/`: Architecture notes.

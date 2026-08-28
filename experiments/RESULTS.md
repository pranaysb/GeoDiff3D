# Phase 4 — Scientific Ablation Results

Real T4 GPU run. Four real multi-view scenes, auto-discovered from VGGT's own
example set (no synthetic data): `kitchen` (object on a table), `llff_fern`
(outdoor plant), `llff_flower` (close-up outdoor flower), `room` (indoor
room), 6 views each. VGGT and Marigold were each run once per scene; all four
methods below are derived from that single shared pair of runs so the
comparison uses identical inputs, preprocessing, and camera geometry.

This supersedes an earlier 2-scene run (`kitchen`, `llff_fern` only). One
headline claim from that run **does not hold up** with two more scenes added
— see Findings below.

**No ground truth exists for any of these scenes.** The only metric below,
cross-view consistency, is a self-consistency diagnostic (reprojecting each
view's depth into its neighbor and comparing against that neighbor's own
depth) — it measures internal agreement between views, not accuracy. A
method can score well here while being uniformly wrong; confirming actual
accuracy would require a dataset with captured ground-truth depth or a mesh,
which this repo does not have.

## Comparison table

| Scene | Method | Points | Runtime | Cross-view consistency (mean abs rel error, lower = more self-consistent) |
|---|---|---|---|---|
| kitchen | vggt_only | 1,082,358 | 3.79s | **0.0739 (best)** |
| kitchen | geodiff3d_fusion | 1,082,404 | 4.69s | 0.0863 |
| kitchen | naive_average | 1,082,358 | 3.21s | 0.0924 |
| kitchen | marigold_only | 1,082,404 | 3.04s | 0.1319 (worst) |
| llff_fern | naive_average | 1,212,241 | 3.47s | **0.0365 (best)** |
| llff_fern | vggt_only | 1,212,240 | 4.77s | 0.0381 |
| llff_fern | geodiff3d_fusion | 1,212,240 | 4.60s | 0.0385 |
| llff_fern | marigold_only | 1,212,242 | 3.43s | 0.0395 (worst) |
| llff_flower | vggt_only | 1,212,240 | 4.60s | **0.0874 (best)** |
| llff_flower | naive_average | 1,212,240 | 3.43s | 0.1046 |
| llff_flower | geodiff3d_fusion | 1,212,240 | 4.66s | 0.1345 |
| llff_flower | marigold_only | 1,212,240 | 3.47s | 0.1435 (worst) |
| room | vggt_only | 1,212,240 | 4.59s | **0.0285 (best)** |
| room | geodiff3d_fusion | 1,212,240 | 4.38s | 0.0678 |
| room | naive_average | 1,212,240 | 3.40s | 0.0801 |
| room | marigold_only | 1,212,257 | 3.39s | 0.1379 (worst) |

Full per-scene metadata (device, dtype, per-view runtime, alignment
scale/shift/residual) is in each scene's `comparison.json`; model/inference
runtimes above are the reconstruction-step timer only (unprojection + save),
not VGGT/Marigold inference, which is recorded separately in
`shared_setup_sec` (52–55s per scene, except `kitchen` at 160s as the first
scene run, which absorbs one-time HuggingFace model download/cache-warming —
not a per-scene cost).

## Findings

**VGGT-only wins on cross-view consistency in 3 of 4 scenes outright**
(`kitchen`, `llff_flower`, `room`), and is a near-tie for best in the 4th
(`llff_fern`: 0.0381 vs. the winning naive-average's 0.0365, a 4% gap). No
fusion or averaging method beat pure VGGT geometry in any scene.
**GeoDiff3D confidence-guided fusion does not improve on VGGT-only
reconstruction on this metric, on any of the four scenes tested.**

**The 2-scene finding that "GeoDiff3D fusion consistently beats naive
averaging" does not hold at 4 scenes — this claim is retracted.** Fusion
beats naive averaging in `kitchen` (0.0863 vs. 0.0924) and `room` (0.0678 vs.
0.0801), but *loses* to it in `llff_fern` (0.0385 vs. 0.0365) and
`llff_flower` (0.1345 vs. 0.1046, the largest gap in either direction across
the whole table). That's a 2–2 split, not a consistent advantage. Confidence
weighting is not a free win over blind averaging; it is scene-dependent, and
on this evidence there's no basis to claim it generally helps.

**GeoDiff3D fusion does consistently beat Marigold-only, in all 4 scenes**
(0.0863 vs. 0.1319; 0.0385 vs. 0.0395; 0.1345 vs. 0.1435; 0.0678 vs. 0.1379).
This is the one comparison that generalized cleanly across every scene
tested — folding in VGGT's geometry, even loosely, is reliably better than
using Marigold's monocular depth alone.

**Marigold-only is the worst or tied-worst method in every scene** — expected
for a purely monocular method with no multi-view constraint; each view is
independently aligned to VGGT's scale, so nothing forces whatever local
detail it resolves to agree across views.

**The two outdoor/plant scenes (`llff_fern`, `llff_flower`) are where fusion
does worst relative to naive averaging**, and `llff_flower` specifically is
where fusion is worst overall relative to VGGT-only (54% higher error: 0.1345
vs. 0.0874). Qualitatively (below), this is the scene where Marigold resolves
the most fine leaf/petal texture, and that appears to be where leaning on it
via confidence weighting actively hurts multi-view agreement rather than
helping it.

## Fusion fix (not yet re-verified on GPU)

**Diagnosis:** `normalize_confidence` rescales each image's confidence to its
own 5th/95th percentile, and the original `fuse_depths` blended linearly
(`weight = 1 - confidence`). Because that rescaling always fills [0, 1]
regardless of whether VGGT's absolute confidence is uniformly excellent, the
*median* pixel in every one of the four scenes above landed near a 50/50
blend — e.g. in `room` (VGGT-only beats Marigold-only by 4.8x on this
metric) the median pixel still got ~31% Marigold weight, and 24% of pixels
got a majority-Marigold blend. The fusion was not actually
confidence-*selective*; it behaved close to naive averaging for a typical
pixel — which is exactly why its scores tracked `naive_average` rather than
staying near `vggt_only` throughout this table. Recovered per-pixel weights
from the saved depth arrays above confirm this precisely (see commit fixing
`core/math.py::fuse_depths`).

**Fix implemented in `core/math.py::fuse_depths`:** gate on relative
confidence instead of blending across the full range — pixels at or above
`trust_threshold` (default 0.5) keep VGGT's depth untouched; only pixels
below it ramp in aligned depth, capped at `max_aligned_weight` (default 0.4)
so no pixel is ever fully replaced by Marigold, since Marigold-only was the
worst standalone method in every scene tested. Unit tests added in
`tests/test_core_math.py`.

**Offline sanity check** (not the real metric — cross-view consistency needs
camera matrices this repo doesn't persist from the ablation run, so it can't
be recomputed without a GPU): replaying the real recovered per-pixel
confidence from each of the four scenes above through the new formula pulls
the fused depth 66–81% closer to `vggt_only`'s depth (mean absolute relative
deviation) than the old formula did, while still letting some Marigold
signal through in the genuinely lowest-relative-confidence regions. This is
a directional check only. **The actual cross-view-consistency numbers for
the new fusion have not been re-run on GPU yet** — that requires a fresh
`python experiments/run_ablation.py` pass and is the next step before any
claim of improvement.

### Qualitative (depth_comparison_4methods.png)

- **`room`**: all four methods' depth maps are visually near-identical across
  all 6 views — differences are almost imperceptible by eye, yet the
  quantitative gap between VGGT-only (0.0285) and fusion (0.0678) is the
  largest relative gap of the whole table (2.4×). This is a useful caution:
  visual similarity in these figures does not imply comparable cross-view
  consistency: each subplot is auto-scaled to its own min/max, so this figure
  cannot show the small, geometrically consequential differences that drive
  the metric.
- **`llff_flower`**: Marigold resolves clearly richer high-frequency texture
  on the flower petals and leaves than VGGT (visible mottling VGGT smooths
  over entirely) — genuine diffusion-prior detail. GeoDiff3D fusion visibly
  picks up more of that Marigold texture than naive averaging does in this
  scene. That is consistent with the quantitative result: whatever the
  confidence map is weighting toward here is adding detail at the cost of
  multi-view agreement, not improving it.
- Across scenes, **the qualitative pattern is consistent**: wherever Marigold
  resolves more local texture than VGGT, GeoDiff3D fusion's depth maps
  resemble Marigold more than naive averaging's do — but that resemblance
  correlates with *worse*, not better, cross-view consistency in this
  dataset.

## Interpretation

Across four real scenes, GeoDiff3D's confidence-guided fusion of a diffusion
depth prior (Marigold) into VGGT's multi-view geometry **does not improve
cross-view self-consistency over VGGT's geometry alone**, and only
inconsistently (2 of 4 scenes) improves over naive unweighted averaging. The
one robust win is against using the diffusion prior alone. This is a genuine,
if modest, negative-leaning result for the core research question ("can a
diffusion depth prior improve geometry-grounded multi-view reconstruction via
confidence-guided fusion?") on the metric available here — self-consistency,
not ground-truth accuracy. It is possible the diffusion prior helps on axes
this metric can't see (e.g. resolving genuinely missing detail in
low-confidence VGGT regions) or would perform differently against real
ground truth; this repo does not have the data to test that.

## Reproducing

```
python experiments/run_ablation.py
```

Requires a CUDA GPU (see `inference/requirements_gpu.txt`) and a local clone
of https://github.com/facebookresearch/vggt (set `VGGT_REPO_DIR` to its path
if not at `/content/vggt` or `notebooks/vggt`). Scenes are auto-discovered
from every `examples/<name>/images/` folder in that clone with 2+ real
photos (capped at 5 scenes / 6 views each by default; see
`GEODIFF3D_ABLATION_MAX_SCENES` / `GEODIFF3D_ABLATION_VIEWS_PER_SCENE`).
Outputs land in `experiments/ablation_results/`: per-method depth maps, PLYs,
`comparison.json` per scene, and `comparison_table.csv` /
`comparison_table.md` across scenes.

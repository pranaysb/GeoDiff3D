# Phase 4 — Scientific Ablation Results

Real T4 GPU run. Four real multi-view scenes, auto-discovered from VGGT's own
example set (no synthetic data): `kitchen` (object on a table), `llff_fern`
(outdoor plant), `llff_flower` (close-up outdoor flower), `room` (indoor
room), 6 views each. VGGT and Marigold were each run once per scene; all four
methods below are derived from that single shared pair of runs so the
comparison uses identical inputs, preprocessing, and camera geometry.

**No ground truth exists for any of these scenes.** The only metric below,
cross-view consistency, is a self-consistency diagnostic (reprojecting each
view's depth into its neighbor and comparing against that neighbor's own
depth) — it measures internal agreement between views, not accuracy. A
method can score well here while being uniformly wrong; confirming actual
accuracy would require a dataset with captured ground-truth depth or a mesh,
which this repo does not have.

This is the third revision of this document. See **Revision history** at the
bottom for what changed between runs and why — in short: a 2-scene run
suggested fusion consistently beat naive averaging, a 4-scene run showed that
didn't hold and also showed fusion never beat VGGT-only, and a bug in the
fusion weighting was diagnosed and fixed as a result. The table and findings
below are from the fixed fusion, re-run on the same four scenes.

## Comparison table (fixed fusion)

| Scene | Method | Points | Runtime | Cross-view consistency (mean abs rel error, lower = more self-consistent) |
|---|---|---|---|---|
| kitchen | geodiff3d_fusion | 1,082,358 | 2.89s | **0.0683 (best)** |
| kitchen | vggt_only | 1,082,358 | 2.96s | 0.0739 |
| kitchen | naive_average | 1,082,358 | 4.46s | 0.0945 |
| kitchen | marigold_only | 1,083,286 | 2.98s | 0.1372 (worst) |
| llff_fern | naive_average | 1,212,240 | 4.49s | **0.0355 (best)** |
| llff_fern | geodiff3d_fusion | 1,212,240 | 3.33s | 0.0365 |
| llff_fern | marigold_only | 1,212,240 | 3.55s | 0.0377 |
| llff_fern | vggt_only | 1,212,240 | 3.34s | 0.0381 (worst) |
| llff_flower | vggt_only | 1,212,240 | 3.26s | **0.0874 (best)** |
| llff_flower | geodiff3d_fusion | 1,212,240 | 3.31s | 0.0943 |
| llff_flower | naive_average | 1,212,240 | 4.41s | 0.1015 |
| llff_flower | marigold_only | 1,212,240 | 3.39s | 0.1377 (worst) |
| room | vggt_only | 1,212,240 | 3.23s | **0.0285 (best)** |
| room | geodiff3d_fusion | 1,212,240 | 3.38s | 0.0335 |
| room | naive_average | 1,212,240 | 3.34s | 0.0688 |
| room | marigold_only | 1,213,258 | 4.56s | 0.1175 (worst) |

Real GPU run confirmed via each scene's `comparison.json` (Tesla T4,
`facebook/VGGT-1B`, `prs-eth/marigold-depth-v1-1`, `torch.float16`).

## Findings

**GeoDiff3D fusion now beats VGGT-only in 2 of 4 scenes** (`kitchen`:
0.0683 vs. 0.0739; `llff_fern`: 0.0365 vs. 0.0381) — up from 0 of 4 before
the fix. In `kitchen`, fusion is the single best method of all four. In the
2 scenes where it still trails VGGT-only (`llff_flower`, `room`), the gap is
much smaller than before the fix: 7.9% worse (was 54%) and 17.5% worse (was
138%), respectively. **This is a genuine, partial improvement — not a full
win.** Fusion still doesn't beat VGGT-only in every scene, and there's no
basis to claim it does.

**Fusion beats naive averaging in 3 of 4 scenes** (`kitchen`, `llff_flower`,
`room`), losing only in `llff_fern` and by a small margin (0.0365 vs.
0.0355, 2.8%). This is a cleaner result than the pre-fix 2-of-4 split.

**Fusion still beats Marigold-only in all 4 scenes**, unchanged from before
the fix.

**Comparing directly to the pre-fix fusion numbers, error dropped in every
single scene**: `kitchen` 0.0863→0.0683 (-20.9%), `llff_fern` 0.0385→0.0365
(-5.2%), `llff_flower` 0.1345→0.0943 (-29.9%), `room` 0.0678→0.0335 (-50.6%).
The fix's diagnosis — that the old formula gave even high-confidence pixels
a near-50/50 blend — is directly supported by how much every scene improved
once that was corrected, including the two scenes where fusion still doesn't
win outright.

**The two scenes where fusion still loses to VGGT-only are the two where
VGGT itself is most dominant** (`llff_flower` and `room` are also VGGT's two
best-scoring scenes, 0.0874 and 0.0285 respectively — the lowest absolute
errors in the whole table). It's plausible that in scenes where VGGT's raw
geometry is already excellent nearly everywhere, there is very little
genuinely low-confidence signal for the fusion to correct, and the small
capped contribution from Marigold's monocular depth (which is never
multi-view consistent) is a net negative even at a low weight. This is a
plausible reading of the pattern, not something these four scenes prove.

### Qualitative (depth_comparison_4methods.png)

- **`room`**: VGGT-only, naive averaging, and fusion depth maps are visually
  close across all 6 views, yet fusion's cross-view error is still 17.5%
  higher than VGGT-only's — a reminder that visual similarity in these plots
  does not imply comparable cross-view consistency, since each subplot is
  auto-scaled to its own min/max.
- **`llff_flower`**: Marigold resolves clearly richer high-frequency texture
  on the flower petals and leaves than VGGT. Fusion's depth maps still show
  some of that texture, but visibly less than before the fix (consistent
  with the capped, threshold-gated blend now used) — and the quantitative
  gap to VGGT-only shrank from 54% to 7.9% as a result.
- **`kitchen`**: this is now the clearest qualitative/quantitative match —
  fusion is the best-scoring method and its depth maps look closest to
  VGGT-only's clean, low-texture field, with only subtle correction in a
  few local regions.

## Interpretation

Across four real scenes, GeoDiff3D's confidence-guided fusion of a diffusion
depth prior (Marigold) into VGGT's multi-view geometry **improves on
VGGT-only cross-view self-consistency in half the scenes tested, and comes
much closer in the other half than it did before the fusion fix**. It
reliably beats naive unweighted averaging and using Marigold alone. This is
a positive, if partial, result for the core research question ("can a
diffusion depth prior improve geometry-grounded multi-view reconstruction
via confidence-guided fusion?") on the metric available here —
self-consistency, not ground-truth accuracy. It does not support a claim
that the fusion always helps; it supports a narrower claim that a properly
gated confidence signal can help, and that the specific gating/cap
parameters used here (`trust_threshold=0.5`, `max_aligned_weight=0.4`) were
not tuned against this ablation — they were chosen from the diagnosis alone,
before this run. Confirming actual accuracy, or tuning those parameters
further, would need either a ground-truth dataset or more scenes than the
four available here.

## Revision history

1. **2-scene run** (`kitchen`, `llff_fern`): suggested GeoDiff3D fusion
   consistently beat naive averaging, while still losing to VGGT-only.
2. **4-scene run** (added `llff_flower`, `room`): showed the naive-averaging
   claim didn't generalize (2-of-4 split) and that fusion never beat
   VGGT-only in any of the four scenes. Prompted a diagnosis of the fusion
   weighting: `normalize_confidence`'s per-image percentile stretch combined
   with a linear `weight = 1 - confidence` blend gave even median-confidence
   pixels a near-50/50 Marigold blend in every scene, so the fusion behaved
   close to naive averaging rather than being genuinely confidence-selective.
3. **Fix + re-run on the same 4 scenes** (this revision): replaced the
   linear blend with a threshold-gated, capped one in
   `core/math.py::fuse_depths` (pixels at/above `trust_threshold` keep VGGT's
   depth untouched; only pixels below it ramp in aligned depth, capped at
   `max_aligned_weight`). Cross-view error dropped in all 4 scenes versus
   the pre-fix fusion, and fusion now beats VGGT-only outright in 2 of 4.

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

No ground truth is available for these scenes; `cross_view_consistency` is a self-consistency diagnostic, not an accuracy metric.

| Scene | Method | Points | Runtime (s) | Cross-view consistency (mean abs rel error) |
|---|---|---|---|---|
| kitchen | vggt_only | 1,082,358 | 2.9618 | 0.0739 |
| kitchen | marigold_only | 1,083,286 | 2.982 | 0.1372 |
| kitchen | naive_average | 1,082,358 | 4.4649 | 0.0945 |
| kitchen | geodiff3d_fusion | 1,082,358 | 2.8918 | 0.0683 |
| llff_fern | vggt_only | 1,212,240 | 3.3419 | 0.0381 |
| llff_fern | marigold_only | 1,212,240 | 3.5504 | 0.0377 |
| llff_fern | naive_average | 1,212,240 | 4.4869 | 0.0355 |
| llff_fern | geodiff3d_fusion | 1,212,240 | 3.3302 | 0.0365 |
| llff_flower | vggt_only | 1,212,240 | 3.2597 | 0.0874 |
| llff_flower | marigold_only | 1,212,240 | 3.3869 | 0.1377 |
| llff_flower | naive_average | 1,212,240 | 4.4129 | 0.1015 |
| llff_flower | geodiff3d_fusion | 1,212,240 | 3.31 | 0.0943 |
| room | vggt_only | 1,212,240 | 3.2327 | 0.0285 |
| room | marigold_only | 1,213,258 | 4.558 | 0.1175 |
| room | naive_average | 1,212,240 | 3.3444 | 0.0688 |
| room | geodiff3d_fusion | 1,212,240 | 3.3799 | 0.0335 |

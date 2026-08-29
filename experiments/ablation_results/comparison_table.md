No ground truth is available for these scenes; `cross_view_consistency` is a self-consistency diagnostic, not an accuracy metric.

| Scene | Method | Points | Runtime (s) | Cross-view consistency (mean abs rel error) |
|---|---|---|---|---|
| kitchen | vggt_only | 1,082,358 | 3.4995 | 0.0739 |
| kitchen | marigold_only | 1,082,358 | 2.9609 | 0.1284 |
| kitchen | naive_average | 1,082,358 | 3.1213 | 0.0903 |
| kitchen | geodiff3d_fusion | 1,082,358 | 4.3471 | 0.0707 |
| llff_fern | vggt_only | 1,212,240 | 4.7358 | 0.0381 |
| llff_fern | marigold_only | 1,212,240 | 3.4564 | 0.0388 |
| llff_fern | naive_average | 1,212,240 | 3.3678 | 0.0362 |
| llff_fern | geodiff3d_fusion | 1,212,240 | 4.7874 | 0.0367 |
| llff_flower | vggt_only | 1,212,240 | 4.545 | 0.0874 |
| llff_flower | marigold_only | 1,212,240 | 3.4788 | 0.1532 |
| llff_flower | naive_average | 1,212,240 | 3.2284 | 0.1103 |
| llff_flower | geodiff3d_fusion | 1,212,240 | 4.4789 | 0.0863 |
| room | vggt_only | 1,212,240 | 4.4933 | 0.0285 |
| room | marigold_only | 1,214,272 | 3.2809 | 0.1492 |
| room | naive_average | 1,212,240 | 3.4229 | 0.0841 |
| room | geodiff3d_fusion | 1,212,240 | 3.8934 | 0.0289 |

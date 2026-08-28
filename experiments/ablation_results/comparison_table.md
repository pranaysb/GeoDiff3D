No ground truth is available for these scenes; `cross_view_consistency` is a self-consistency diagnostic, not an accuracy metric.

| Scene | Method | Points | Runtime (s) | Cross-view consistency (mean abs rel error) |
|---|---|---|---|---|
| kitchen | vggt_only | 1,082,358 | 3.7929 | 0.0739 |
| kitchen | marigold_only | 1,082,404 | 3.0357 | 0.1319 |
| kitchen | naive_average | 1,082,358 | 3.2135 | 0.0924 |
| kitchen | geodiff3d_fusion | 1,082,404 | 4.6929 | 0.0863 |
| llff_fern | vggt_only | 1,212,240 | 4.7658 | 0.0381 |
| llff_fern | marigold_only | 1,212,242 | 3.4251 | 0.0395 |
| llff_fern | naive_average | 1,212,241 | 3.4746 | 0.0365 |
| llff_fern | geodiff3d_fusion | 1,212,240 | 4.6029 | 0.0385 |
| llff_flower | vggt_only | 1,212,240 | 4.5984 | 0.0874 |
| llff_flower | marigold_only | 1,212,240 | 3.468 | 0.1435 |
| llff_flower | naive_average | 1,212,240 | 3.4327 | 0.1046 |
| llff_flower | geodiff3d_fusion | 1,212,240 | 4.6556 | 0.1345 |
| room | vggt_only | 1,212,240 | 4.5905 | 0.0285 |
| room | marigold_only | 1,212,257 | 3.388 | 0.1379 |
| room | naive_average | 1,212,240 | 3.3987 | 0.0801 |
| room | geodiff3d_fusion | 1,212,240 | 4.3816 | 0.0678 |

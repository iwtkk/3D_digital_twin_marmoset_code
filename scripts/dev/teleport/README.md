# teleport - Teleport (jump) analysis of 3D keypoints

Detects physically implausible "jumps" from per-frame keypoint displacements and cross-checks them against 2D camera visibility
to investigate their causes. v1 uses a simple threshold (>100mm/frame); v2 uses a two-stage decision (isolated jump +
global extreme).

All scripts accept `--input-dir / --output-dir / --session / --help`.
run computes -> JSON; plot reads the JSON and draws figures (some timeline / velocity plots also read the pickle).

## run/ (-> teleport_files/ , teleport_v2_files/)

| Script | Input (default) | Output |
|---|---|---|
| `teleport_common.py` | — | Shared library (ChunkCache etc., no CLI) |
| `detect_position_jumps.py` | `outputs/3d`, `outputs/2d` | `{session}_teleport.json`, `teleport_summary.json` |
| `detect_position_jumps_twostage.py` | `outputs/3d`, `outputs/2d` | `{session}_teleport_v2.json`, `teleport_summary_v2.json` |

## plot/ (-> teleport_plot/ , teleport_v2_plot/)

| Script | Input (default) | Output |
|---|---|---|
| `plot_position_jumps.py` | `teleport_files` + `outputs/3d` | heatmap / timeline / camera visibility / jump dist |
| `plot_position_jumps_twostage.py` | `teleport_v2_files` + `outputs/3d` | heatmap / kp freq / jump patterns / camera vis / velocity dist |

Example:
```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/teleport/run/detect_position_jumps_twostage.py
OPENBLAS_NUM_THREADS=1 python3 scripts/dev/teleport/plot/plot_position_jumps_twostage.py
```

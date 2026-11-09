# kinematics - Kinematics and signal quality analysis of 3D keypoints

Characterizes motion as a "signal": velocity/acceleration/jerk distributions, PSD (energy per frequency),
and stability of bone lengths (which should be fixed). Aggregates all sessions in a streaming fashion.

Because the 3D data (kp3d_fxdJointLen) is temporally smoothed by anipose, the high-frequency end of the PSD reflects the smoothing filter's
roll-off (not raw detection noise).

All scripts accept `--input-dir / --output-dir / --session / --help`.

| Category | Script | Input (default) | Output |
|---|---|---|---|
| run | `kinematics_common.py` | — | Shared library (no CLI) |
| run | `analyze_kinematics.py` | `outputs/3d` | `kinematics_accum.npz`, `kinematics_summary.json` → `kinematics_files/` |
| plot | `plot_kinematics.py` | `kinematics_files` | Histograms/PSD/bone CV/time series → `kinematics_plot/` |

run saves the accumulated arrays to NPZ, and plot reads them to draw (no need to reload the pickle).
In run, `--session` selects the representative session for the time-series plots.

Example:
```bash
python3 scripts/dev/kinematics/run/analyze_kinematics.py --session cj425_20210427_120000
python3 scripts/dev/kinematics/plot/plot_kinematics.py
```

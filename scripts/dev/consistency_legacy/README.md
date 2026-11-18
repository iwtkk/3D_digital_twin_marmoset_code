# Legacy – whole-image consistency pipeline (superseded)

Moved from `scripts/06_validation/consistency/` on 2026-11-18. Superseded by the Animal-region pipeline there
(see docs/adr/0002-fig8a-metrics-without-render-images.md); kept for reference only. `visualize.py` stayed in
`scripts/06_validation/consistency/`.

Verify that rendered pseudo-view images are consistent with real camera images
using SSIM and PSNR.
Corresponds to paper section: **Technical Validation – Validation of consistency in reproducing the experimental environment**

## Scripts

| Script | Purpose |
|---|---|
| `ssim.py` | Compute SSIM/PSNR between real and rendered frames |
| `calculate_metrics.py` | Batch metric calculation across sessions |
| `run_pipeline.py` | Full validation pipeline |
| `outlier_detector.py` | Detect inconsistent frames |
| `extract_frames.py` | Extract frames for comparison |
| `visualize.py` | Side-by-side visualization of real vs. rendered |
| `utils.py` | Shared utilities |

## Usage

```bash
# Compute SSIM/PSNR for one session and camera
python ssim.py \
  --real outputs/frames/{session}/{cam_id}/ \
  --rendered outputs/render/{session}/{cam_id}/ \
  --output outputs/validation/consistency/ssim_{session}_{cam_id}.csv

# Full pipeline across all sessions and cameras
python run_pipeline.py \
  --config ../../config/consistency.yaml \
  --output outputs/validation/consistency/

# Side-by-side visualization
python visualize.py \
  --real outputs/frames/{session}/{cam_id}/ \
  --rendered outputs/render/{session}/{cam_id}/ \
  --frame 70 \
  --output outputs/validation/consistency/compare_frame70.png
```

## Metrics

- **SSIM** (Structural Similarity Index): range 0–1, higher is better
- **PSNR** (Peak Signal-to-Noise Ratio): in dB, higher is better

Results are reported per camera, per session, and aggregated across all sessions.

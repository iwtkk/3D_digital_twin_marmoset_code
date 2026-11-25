# Release the Fig. 8a renders of the Detailed model with fur

Supersedes ADR-0002. Besides the frame list and the PSNR/SSIM JSON/CSV, the Published dataset carries the 2,400 renders of the Detailed model with fur (`rendered_environmental_images/render/detailed_fur/`, 8.7 GB) and one example full-image SSIM map (`example/`). These are the Aligned-pose renders the published `detailed_fur` values were computed with, not the old Raw-pose renders that ADR-0002 removed. Readers can inspect the compared images without Blender, while `scripts/06_validation/consistency/` still regenerates every body model and verifies the values.

## Considered Options

- **Metrics only (ADR-0002)**: smallest dataset, but the images behind Fig. 8a can only be seen after about 50 GPU hours of rendering.
- **All three body models (7,200 renders, about 24 GB)**: `detailed_nofur` and `simplified_fur` only serve the body-model comparison and differ from `detailed_fur` mainly inside the Animal region; they stay reproducible from the code repository.
- **SSIM maps for all 2,400 pairs**: derived from the renders and the published videos in seconds; one example documents the map definition.
- **Real frames as images**: duplicates frames of `video_data/`; `frames.csv` gives their Video frame index.

## Consequences

The dataset grows to about 1,059 GB, slightly over the 1 TB guideline. The example SSIM map's mean equals `ssim_full` of its pair (0.542), which ties the image to `pair_metrics.csv`.

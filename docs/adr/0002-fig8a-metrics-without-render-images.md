# Publish Fig. 8a metrics without the render images

Status: superseded by ADR-0003 (2026-11-25): the Aligned-pose renders of the Detailed model with fur are released.

The Published dataset carries only the Fig. 8a frame list and the per-pair PSNR/SSIM JSON/CSV (Full image and Animal region); the 2,400 render images, their scenes and the environment-region metrics are not released. The released renders had been posed with Raw keypoints while every other released scene uses Aligned keypoints, and the images are fully reproducible from the released per-session `.blend` files, so the code repository (`scripts/06_validation/consistency/`) provides the render → real frames → metrics → figure pipeline plus a `verify` step against the released JSON instead.

## Considered Options

- **Re-render and release the images (one or three body models)**: 8–24 GB of derived data that users can regenerate, and the dataset is close to the 1 TB guideline.
- **Keep the old renders, labelled as Raw-pose**: contradicts ADR-0001 (only Aligned keypoints are released) and the numbers in the paper.

## Consequences

Reproduction depends on Blender 5.2.x: renders from other versions differ slightly (0.45 dB PSNR between 5.1 and 5.2), so `verify` checks per-pair tolerances rather than bit equality.

# Release only Aligned keypoints, as NPZ

A reviewer asked for the 3D keypoints in an open array format instead of pickle, so the Published dataset ships one `{session}.npz` per Session holding the Aligned keypoints (plus the Raw keypoints, rotation and per-session affine needed to recompute them) and nothing else: no CSV, no pickle, no Raw-only release. The NPZ loads without `allow_pickle`, and a single format avoids users picking the pre-alignment coordinates by mistake. The CSV route (`dataset_work`) is frozen as a superseded build and the marmo3Dpose submodule keeps writing pickle as an internal intermediate.

## Considered Options

- **HDF5**: also acceptable to the reviewer, but adds a dependency for a small, flat set of arrays.
- **Keep CSV alongside NPZ**: the per-session Blender scenes are already posed, so users have no need for the CSV that our Blender import uses internally; two formats would double the files to keep in sync.

## Consequences

NPZ key names (`kp3d`, `kp3d_mm`, `i_frame`, `frame_num`) are fixed by the upload and are mapped to GLOSSARY terms in `docs/data_format.md` rather than renamed.

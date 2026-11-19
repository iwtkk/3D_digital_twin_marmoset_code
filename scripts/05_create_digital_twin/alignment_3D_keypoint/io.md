"""
I/O ledger: 05_create_digital_twin/alignment_3D_keypoint shared modules

The scripts in run/ and plot/ are listed in .agent_docs/docs/tool_registry.md. This ledger covers the shared
modules they import (moved from scripts/dev on 2026-11-18; scripts/dev is not part of the public repository).

================================================================================
Tool: _config   (run/_config.py, shared module)
================================================================================
Purpose:
    Constants of the alignment: coordinate transform (kp3d mm -> Blender m), physical heights, kp3d inlier bbox,
    keypoint names / bones, frame rates, animal colours.
Inputs:
    method.yaml (this directory)
Outputs: none (module constants; METHOD_YAML_PATH = method.yaml)

================================================================================
Tool: floorz_core   (run/floorz_core.py, shared module)
================================================================================
Purpose:
    Keypoint names, fixed rotation of the Raw keypoints, KDE peak detection, affine regression, kp3d I/O
    (load_kp3d / load_pickle_to_df via src/marmo3d/kp3d_io.py), global scale + affine correction.
Inputs:
    method.yaml (through _config), kp3d pickle / NPZ given by the caller
Outputs:
    none, except get_or_compute_global_norm(write=True): rewrites the coordinate_transform.kp3d_global_norm
    block of method.yaml when the requested percentile is not cached yet
"""

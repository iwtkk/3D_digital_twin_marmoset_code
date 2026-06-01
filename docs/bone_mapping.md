# Blender IK Bone Mapping

This document describes the correspondence between 3D keypoints and Blender IK (Inverse Kinematics) bones in the rigged marmoset model.

## Keypoint → IK bone correspondence

| 3D keypoint(s) | Derivation | Blender IK bone |
|---|---|---|
| nose (0) / L eye (1) / R eye (2) | direct / midpoint | Head_IK |
| L wrist (9) | direct | Forearm_IK.L |
| R wrist (10) | direct | Forearm_IK.R |
| L elbow (7) | direct | Elbow_IK.L |
| R elbow (8) | direct | Elbow_IK.R |
| L ankle (15) | direct | Foot_IK.L |
| R ankle (16) | direct | Foot_IK.R |
| L knee (13) | direct | Knee_IK.L |
| R knee (14) | direct | Knee_IK.R |
| tail_1 (18) | direct | Tail0_IK |
| tail_2 (19) | direct | Tail_IK |
| midpoint of L shoulder (5) + R shoulder (6) | computed | Chest_IK |
| midpoint of L hip (11) + R hip (12) | computed | Body_Control_IK |
| backbone peak (17) | direct | Spine_IK |

## Armature bone hierarchy (simplified)

```
Root
└── Body_Control_IK      ← midpoint of hips
    └── Chest_IK         ← midpoint of shoulders
        ├── Head_IK      ← nose / eyes
        ├── Forearm_IK.L ← L wrist
        │   └── Elbow_IK.L
        ├── Forearm_IK.R ← R wrist
        │   └── Elbow_IK.R
        └── Spine_IK     ← backbone peak
Foot_IK.L                ← L ankle
├── Knee_IK.L
Foot_IK.R                ← R ankle
└── Knee_IK.R
Tail0_IK                 ← tail_1
└── Tail_IK              ← tail_2
```

## Coordinate conventions

- World coordinates from 3D reconstruction are in **millimeters**.
- Blender scene uses **meters** by default — apply a scale factor of `0.001` when setting bone locations.
- Rotation order: `XYZ` Euler (as used in `blender_animation.py`).

## Implementation notes

The bone locations are set using `matrix_world` via `set_object_world_loc_euler()` in `animate/BlenderSync/blender_animation.py`.

Derived midpoints:
```python
chest_pos = (kp3d[frame, 5] + kp3d[frame, 6]) / 2.0   # midpoint of shoulders
body_pos  = (kp3d[frame, 11] + kp3d[frame, 12]) / 2.0  # midpoint of hips
```

## Reference

See Fig. 3 in the associated paper for a visual overview of the IK rig structure.

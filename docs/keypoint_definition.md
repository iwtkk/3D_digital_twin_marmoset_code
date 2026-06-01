# Keypoint Definition

20 body keypoints are defined for each marmoset individual.

## Keypoint list

| Index | Name | Side | Body region |
|---|---|---|---|
| 0 | nose | - | Head |
| 1 | L eye | Left | Head |
| 2 | R eye | Right | Head |
| 3 | L ear | Left | Head |
| 4 | R ear | Right | Head |
| 5 | L shoulder | Left | Torso |
| 6 | R shoulder | Right | Torso |
| 7 | L elbow | Left | Forelimb |
| 8 | R elbow | Right | Forelimb |
| 9 | L wrist | Left | Forelimb |
| 10 | R wrist | Right | Forelimb |
| 11 | L hip | Left | Torso |
| 12 | R hip | Right | Torso |
| 13 | L knee | Left | Hindlimb |
| 14 | R knee | Right | Hindlimb |
| 15 | L ankle | Left | Hindlimb |
| 16 | R ankle | Right | Hindlimb |
| 17 | backbone peak | - | Torso |
| 18 | tail_1 | - | Tail |
| 19 | tail_2 | - | Tail |

## Skeleton connectivity

```
Connections used for visualization:
(0,2),(0,1)       # nose - eyes
(2,4),(1,3)       # eyes - ears
(5,7),(6,8)       # shoulders - elbows
(7,9),(8,10)      # elbows - wrists
(11,13),(12,14)   # hips - knees
(13,15),(14,16)   # knees - ankles
(0,17)            # nose - backbone peak
(17,5),(17,6)     # backbone peak - shoulders
(17,11),(17,12)   # backbone peak - hips
(18,19)           # tail_1 - tail_2
```

## Array shapes

### 2D keypoints (`kp2d_f.pickle`)

```
Shape: (n_frame, n_keypoint, n_individual, 3, n_camera)
  - n_frame:      number of video frames
  - n_keypoint:   20
  - n_individual: number of animals (typically 1 or 2)
  - 3:            [x, y, confidence]
  - n_camera:     number of cameras (8)

Units: pixels
Coordinate origin: top-left of image
Confidence range: 0.0 – 1.0
```

### 3D keypoints (`kp3d.pickle` / `kp3d_fxdJointLen.pickle`)

```
Shape: (n_frame, n_keypoint, 3)   (per individual)
  or   {'kp3d': array of shape (n_individual, n_frame, n_keypoint, 3)}

Units: millimeters (mm)
Coordinate system: world coordinates defined by camera calibration
```

## Score threshold

Keypoints with confidence below `0.3` (default) are masked as NaN before 3D reconstruction.
This threshold is configurable via `--score-threshold` in reconstruction scripts.

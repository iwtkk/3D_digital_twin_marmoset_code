# 01 — Building the 3D marmoset body model from micro-CT

This step produces the rigged 3D marmoset body model used in step 05.

The body of the digital twin is a closed triangular surface derived from a
whole-body micro-CT volume of a common marmoset. The surface is then
retopologized, textured, given fur and rigged in Blender. The model is built
in four stages:

| Stage | What | Tool |
| --- | --- | --- |
| 1 | CT volume → binary body label | MATLAB (or GNU Octave), `ct_body_segmentation.m` |
| 2 | smoothing of the label | 3D Slicer 4.11, Segment Editor |
| 3 | label → triangular surface (STL) | 3D Slicer 4.11, Segmentations module |
| 4 | surface → rigged body model with fur | Blender |

Stage 1 is scripted. Stages 2–4 are carried out through the graphical
interfaces of 3D Slicer and Blender. The settings needed to repeat stages 2
and 3 are given below.

`ct_body_to_stl.py` is an alternative to stages 1–3 that runs as a single
Python command (numpy, scipy, nibabel, vtk). It was not used to build the
released model.

## Data

Files in the released dataset, under
`3D_digital_twin_marmoset_data/3D_marmoset_body_data/` (see step 00 for
download and extraction):

| File | Content |
| --- | --- |
| `CT_scan_data/CT_scan_data.nii` | whole-body micro-CT, 512 × 512 × 1334 voxels, 0.196 × 0.196 × 0.300 mm, int16 Hounsfield units. The original file name was `20221020_173232BFbBs003a003.nii`. |
| `CT_scan_data/Segmentation.nii.gz` | smoothed binary body label (output of stage 2), on the CT grid |
| `CT_scan_data/Segmentation.stl` | body surface (output of stage 3), binary STL, LPS, mm, 3,400,432 triangles |
| `CT_scan_data/Segmentation.seg.nrrd`, `Marmoset_surface.seg.nrrd`, `Marmoset_surface.stl` | 3D Slicer segmentations resampled to about 0.41 mm isotropic voxels, and a 43,791-triangle body surface, in the same LPS frame |
| `3dmarmoset.blend` | rigged body model with hair-curve fur (output of stage 4) |
| `3dmarmoset_simplified.blend` | the same body and rig with mesh fur cards, for faster rendering |

Files used or written by stages 1–3:

| File | Role | In the dataset |
| --- | --- | --- |
| `body_sample.nii.gz` | field-of-view template; one axial slice defines the area to keep | no |
| `tst2Segmentation-Segment_1-label.nii.gz` | label of the cylindrical holder tube; one axial slice is extruded along *z* | no |
| `test_body2.nii.gz` | output of stage 1: binary body label | no |
| `tst_smooth2.nii.gz`, `tst_smooth2.nrrd` | output of stage 2: smoothed body label | as `Segmentation.nii.gz` |
| `Segmentation_1.stl` | output of stage 3: body surface | as `Segmentation.stl` |

All input volumes share the same grid, so no registration is needed.

---

## Environment for stage 1

### MATLAB (used for the paper)

| Requirement | Used for |
|---|---|
| MATLAB (R2023b) | — |
| Image Processing Toolbox | `imfill`, `bwareaopen` |
| [Tools for NIfTI and ANALYZE image](https://www.mathworks.com/matlabcentral/fileexchange/8797) (File Exchange #8797) | `load_untouch_nii`, `save_untouch_nii` |

### GNU Octave (tested with Octave 11.3.0 on macOS)

| Requirement | Install |
|---|---|
| GNU Octave | `brew install octave` |
| `image` package (tested 2.20.1) | in Octave: `pkg install -forge image` |
| NIfTI I/O (`load_untouch_nii`, `save_untouch_nii`) | the File Exchange toolbox above, or any reader/writer with the same interface for single-file `.nii` / `.nii.gz` |

At the start of each session:

```octave
pkg load image
addpath('/path/to/NIfTI_toolbox')
```

Put these lines in `~/.octaverc` to load them automatically.

Notes:
- Octave's `bwareaopen(BW, 10, 8)` on an N × 1 × M array behaves like
  MATLAB's: it removes runs along the first axis, so the *x* / *y* run removal
  is reproduced.
- `octave-cli` cannot `print` figures without OpenGL. Use `octave --gui` to
  look at figures, or save images with `imwrite`.
- Do not run two Octave sessions that read the same `.nii.gz` at the same
  time: both decompress it into the same directory and one of them fails.

### Machine

At least 8 GB of free RAM. The CT is about 700 MB, and the script holds
several logical volumes of about 350 MB each. One run takes about 16 s in
Octave on an Apple Silicon Mac.

---

## Stage 1 — Binary body label (MATLAB / GNU Octave)

```matlab
ct_body_segmentation
```

All parameters are collected at the top of the script. The input and output
paths are relative; edit `ctFile`, `fovFile`, `holderFile` and `outFile` to
match your layout.

The animal was scanned lying inside a cylindrical holder tube. The scanner
reconstructs a circular field of view inside a square matrix and fills the
matrix outside that circle with −2048 HU. Inside the circle, the holder tube
wall (about 400–800 HU) and the support structures around the tube are above
0 HU, so they cannot be separated from the animal by intensity alone. They are
removed geometrically instead: a single axial cross-section of the
field-of-view template and of the holder label is extruded along *z*. This is
valid because the scanner geometry and the tube do not move during the
acquisition.

1. **Crop.** Keep voxels that are inside the field of view, outside the holder
   tube, above 0 HU, and within slices 84–1255.
   * `fovSlice = 675` — the slice of `body_sample.nii.gz` that defines the
     field of view.
   * `holderSlice = 115` — the slice of the holder label that shows the full
     tube cross-section.
2. **Clean.** In every axial plane, `imfill(...,'holes')` closes the air-filled
   lumina (lung, trachea, gut) so that the body becomes solid, and
   `bwareaopen(...,100,8)` removes fur, bedding debris and reconstruction
   noise. Then, over slices 200–1100, `bwareaopen(...,10,8)` removes runs
   shorter than 10 voxels along *x* and then along *y*. These are thin spurs —
   individual hairs and partial-volume filaments at the skin boundary — which
   survive the axial pass because they are large within a single slice but only
   a few voxels wide across slices.
3. **Close.** Re-impose the CT support (`CT > 0`) and fill the holes this
   reopens, slice by slice.

Output: `test_body2.nii.gz`, on the CT grid and with the CT header.

Slice numbers are 1-based. On `CT_scan_data.nii` as released, the body in
`Segmentation.nii.gz` spans slices 80–1251, four slices below the range 84–1255
set in the script.

### Running without the two templates

`body_sample.nii.gz` and the holder label are not in the dataset. A stand-in
can be estimated from the CT: on slices 70–78, only the holder tube wall is
above 0 HU inside the tube. Take the ring-shaped component that encloses the
image centre on each of these slices and intersect them. Dilate the ring by
2 px to get the holder label, and take the inside of the ring as the field of
view. With these stand-ins, the Octave run gives Dice 0.987 against
`Segmentation.nii.gz`. The remaining differences lie along the tube wall and
at the two ends of the animal.

---

## Stage 2 — Smoothing (3D Slicer, Segment Editor)

3D Slicer 4.11.

1. **Load the label.** *File → Add Data…* → `test_body2.nii.gz`. Open
   *Show Options* and tick **LabelMap**, so that the volume is loaded as a
   label map rather than as a scalar volume.
2. **Convert it to a segmentation.** *Data* module → right-click the loaded
   node → *Convert labelmap to segmentation node*. (Equivalently:
   *Segmentations* module → *Import/Export nodes* → import a labelmap.)
3. **Smooth.** *Segment Editor* module → select the segmentation and its single
   segment → **Smoothing** effect → **Method: Gaussian** →
   **Standard deviation: 0.25 mm** → **Apply**.

   The effect acts on the whole segment in 3D. It replaces the binary label
   with a Gaussian-blurred version re-thresholded at 0.5. This rounds off the
   stair-steps left by the slice-wise processing in stage 1 without moving the
   skin boundary appreciably.
4. **Save.** *File → Save* → `tst_smooth2.nrrd` (Slicer's native format) and
   `tst_smooth2.nii.gz`.

---

## Stage 3 — Surface extraction (3D Slicer, Segmentations module)

Same Slicer session.

1. *Segmentations* module → select the smoothed segmentation.
2. **Export to files…**
   * **File format:** `STL`
   * **Coordinate system:** `LPS`
   * **Merge into single file:** off (there is only one segment)
   * **Size scale:** 1.0
3. Export.

Output: `Segmentation_1.stl`, a binary STL. Its header reads
`3D Slicer output. SPACE=LPS`.

Slicer extracts the surface from the segment's binary label map with discrete
marching cubes. It then applies the segmentation's own surface smoothing
(*Segmentations → Representations → Closed surface → Surface smoothing*,
default 0.5, which is the value stored in the dataset's `.seg.nrrd` files).
Decimation is 0. The vertices of the released STL do not lie on the voxel
grid, which shows that surface smoothing was applied.

---

## Stage 4 — Rigged body model (Blender)

This stage is manual. The steps were:

1. Import the stage 3 STL into Blender.
2. Retopologize the surface into a quad mesh.
3. Add a Subdivision Surface modifier.
4. Add skin materials.
5. Add fur.
6. Build an armature with IK controls and bind the mesh to it.

Below is what the released files contain. `3dmarmoset.blend` was saved with
Blender 5.0, and `3dmarmoset_simplified.blend` with Blender 5.1.

| Part | `3dmarmoset.blend` | `3dmarmoset_simplified.blend` |
| --- | --- | --- |
| Body mesh | `NF_Body`: 7,836 vertices, 7,850 faces; Armature + Subdivision Surface modifiers; 98 vertex groups | `Marmoset_body`: the same mesh and modifiers |
| Eyes, mouth | `NF_Eyes`, `NF_Mouth_Top`, `NF_Mouth_Bottom` | `NF_Eyes.004`, `NF_Mouth_Top.004`, `NF_Mouth_Bottom.004` |
| Skin material | base colour, roughness and normal texture maps | same |
| Fur | hair-curve objects (`Hair_*`, `N_Hair_*`) | textured mesh fur cards (`fur_*`, `N_Hair_*_mesh`) |
| Armature | `NF_Armature`, 113 bones | `NF_Armature_lowpoly`, 113 bones |

Both files also contain duplicates of these objects (suffix `.001` / `_001`).

The armature is driven by these constraints:

| Bone | Constraint | Target | Pole |
| --- | --- | --- | --- |
| `Chest` | IK | `Chest_IK` | `Chest_Control_IK` |
| `Head` | Damped Track | `Head_IK` | — |
| `Forearm.L` / `Forearm.R` | IK, chain 2 | `Forearm_IK.L` / `Forearm_IK.R` | `Elbow_IK_L` / `Elbow_IK_R` |
| `LowerLeg.L` / `LowerLeg.R` | IK, chain 3 | `Foot_IK.L` / `Foot_IK.R` | `Knee_IK.L` / `Knee_IK.R` |
| `Bendy_Bone.001` / `.002` (tail) | Stretch To | `Tail0_IK` / `Tail_IK` | — |
| `Tail0_IK`, `Tail_IK` | Track To | `Base` | — |

The IK target bones are placed at the 3D keypoints in step 05. See
[docs/bone_mapping.md](../../docs/bone_mapping.md) for the keypoint → bone
mapping.

## Output

- `3D_marmoset_body_data/3dmarmoset.blend` — rigged marmoset model
- `3D_marmoset_body_data/3dmarmoset_simplified.blend` — the same model with
  mesh fur cards, for fast rendering

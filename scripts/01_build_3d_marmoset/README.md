# 01 – Build 3D Marmoset Model

This step produces the rigged 3D marmoset body model used in step 05.
It involves CT imaging, 3D Slicer segmentation, and Blender rigging.
**No Python scripts are required — this step uses external software.**

The resulting Blender files are provided in `assets/marmoset_3d/` (download via step 00).

## Workflow

### 1. CT image acquisition
Micro-CT scans were acquired of the marmoset body.
See paper Section 2.1 for acquisition parameters.

### 2. Segmentation (3D Slicer 4.11 + SlicerRT)
1. Open the CT DICOM data in 3D Slicer 4.11
2. Use the **Segment Editor** module to segment bone and soft tissue
3. Export segmentation as `.stl` surface mesh

### 3. Surface model creation (Blender 4.5.0)
1. Import `.stl` into Blender
2. Retopologize and clean up the mesh
3. Apply subdivision surface modifier for smooth geometry

### 4. Fur and skin texture
1. Apply particle system for fur simulation
2. Apply diffuse/specular texture maps for skin

### 5. Rigging
1. Create an Armature with bones matching the 20 keypoint positions
2. Add IK constraints to target bones
   (see [docs/bone_mapping.md](../../docs/bone_mapping.md) for bone names)
3. Bind mesh to armature with automatic weights

## Output

- `assets/marmoset_3d/marmoset_body.blend` — rigged marmoset model
- `assets/simplified_marmoset/marmoset_simplified.blend` — low-poly version for fast rendering

# 04 – Building the 3D Environment Model

This step builds the 3D model of the experimental cage in Blender.  
Modeling itself uses **3D Slicer** and **Blender**, and the Python scripts are run from inside Blender.

The finished Blender files are stored under `3D_digital_twin_marmoset_data/`.

---

## Blender scripts

The scripts that measure vertex coordinates, edge lengths, and object heights are shared with the validation step.

**-> [`scripts/06_validation/environment_accuracy/`](../06_validation/environment_accuracy/README.md)**

The same scripts are also used during model building to check and record the dimensions of each part of the cage.

---

## Workflow overview

1. Measure the actual dimensions of the cage in **3D Slicer**
2. Model the cage geometry in **Blender**
3. Check and record the coordinates of vertices, edges, and objects with the scripts in `06_validation/environment_accuracy/`
4. Place the cameras (for extrinsic parameters, see `config/marmo_cj425m/cam_extrinsic_optim.h5`)

For camera serial IDs and placement, see [docs/camera_mapping.md](../../docs/camera_mapping.md).

---

## Output

- `3D_digital_twin_marmoset_data/environment_model/environment_measurement.blend` — environment model for measurement
- `3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/digital_twin_marmoset.blend` — combined marmoset + environment scene

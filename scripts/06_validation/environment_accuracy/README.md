# Validation – Geometric Accuracy of the 3D Environment Model

Compares the dimensions of the 3D model of the experimental cage against measured values to evaluate the model's geometric accuracy.  
Corresponding paper section: **Technical Validation – Geometric accuracy of the 3D environment model**

## Overview

The measurement scripts live here (`06_validation/environment_accuracy/`), but  
the same scripts are also used in the model-building step ([`scripts/04_build_3d_environment/`](../../04_build_3d_environment/README.md)).  
The `04` README also points here.

## Scripts

All scripts are run by pasting them into Blender's **Scripting workspace** (or Text Editor).  
They can also be run from the CLI with `blender --background <file.blend> --python <script.py>`.

| Script | Purpose |
|---|---|
| `vertex.py` | Prints the local and world coordinates of the vertices selected in Edit mode to the console |
| `vertex.csv.py` | Writes the coordinates of the selected vertices to `selected_vertices.csv` (in mm) |
| `length.py` | Prints the local and world lengths of the edges selected in Edit mode to the console |
| `object_dimensions.py` | Writes MaxZ / MinZ of the selected objects to `height.csv` (in mm) |

## Automatic generation of Table 3 (`make_table3.py`)

Instead of the manual workflow (select vertices, export CSV, subtract in xlsx), the measurement definitions are
written in `config/environment_table3.yaml` and Table 3 is produced from the .blend in one step.
**The default measurement target is the published digital twin** (`3D_digital_twin_data/3D_digitaltwin_marmoset_blender.blend`),
so that the environment actually used for rendering is what gets measured.

### Procedure

```bash
# 1. Measure (default = published digital twin). Takes about 30 s
.venv/bin/python scripts/06_validation/environment_accuracy/make_table3.py

# 2. View the results
cat outputs/dev/environment_table3/digitaltwin/table3.md
```

| Output | Contents |
|---|---|
| `table3.csv` | 19 rows. Blueprint value, model value, error, signed ratio, match flag, measured object, warning flags |
| `table3.md` | The table above + summary statistics (number of matches, MAE, max error, mean and range of the ratio) |
| `table3.tex` | LaTeX table (booktabs) in the same format as Table 3 in the manuscript. Ordered by paper No., with avg./max./min. rows at the end. mm values have 2 decimal places (below 10^-3 mm shown as `<10^{-3}`), ratios have 3 decimal places, and ratios are signed (model / blueprint) |
| `summary.json` | Summary statistics + path and md5 of the measured .blend, Blender version |
| `measured_raw.json` | Raw values returned by Blender (with object hidden / modifier / animation info) |

To measure other versions:

```bash
.venv/bin/python scripts/06_validation/environment_accuracy/make_table3.py --blend v250430   # original version used for paper Table 3
.venv/bin/python scripts/06_validation/environment_accuracy/make_table3.py --blend dataset   # environment_measurement.blend (old version)
.venv/bin/python scripts/06_validation/environment_accuracy/make_table3.py --blend /path/to/x.blend --output-dir outputs/dev/environment_table3/x
```

### How it works

- `make_table3.py` (.venv) reads the YAML and runs `measure_blend.py` with `blender -b`
  (two-stage setup because Blender's bundled Python has no yaml). The .blend is only read, never saved
- There are 3 measurement methods (measured the same way as the existing `object_dimensions.py` / `vertex.csv.py`)
  - `bbox_extent`: length of the bounding box in world coordinates. With `geometry: mesh`, measures the mesh before modifiers are applied
    (used to measure a single door out of a MIRROR-ed pair of doors)
  - `vertex_delta`: difference between the coordinates of 2 vertices
  - `bbox_gap`: face-to-face distance between 2 objects. `a: [Cube, min], b: [Cube.003, max]` means
    "bottom face of Cube − top face of Cube.003" (height inside the cage, paper No.13)
  - `bbox_span`: max of the upper object − min of the lower object
- Rows whose object names differ between versions are overridden with `by_blend` (the 4 door rows)
- Rows with `blueprint_mm: null` show only the measured value in the table and are excluded from summary statistics (for provisionally adding items whose blueprint value is not yet fixed)
- The ratio is signed (model / blueprint). Column J of the xlsx (`1 + ABS(1 - ratio)`) is not used
- The world matrix is recomputed from `matrix_basis` instead of `matrix_world`. Hidden objects
  are not evaluated and their `matrix_world` may be stale (Cube.002 in the 250430 version has its scale unapplied and becomes 3707 mm)
- Warning flags in the note column: `hidden` (not rendered), `matrix_world stale`, `animated` (has an action).
  Cube.004 in the digital twin is flagged `animated`, but it has only one keyframe (frame 72) and its value equals the saved value.
  Verified that its position does not change over frames 1-500

### Adding or changing measurement items

Add one row to `rows` in `config/environment_table3.yaml`. Check object names and vertex indices by selecting the target in Blender
and using `vertex.py` (prints vertex indices and coordinates) or `object_dimensions.py`.

```yaml
- {label: 30, paper_no: null, part: "perch length", blueprint_mm: 850, method: bbox_extent, object: Cylinder.019, axis: y}
```

### Differences between versions (checked 2026-10-30)

| Version | Matches | Door height (#12/#21) | Side height (#16) | Notes |
|---|---|---|---|---|
| `digitaltwin` (published digital twin) | 8/19 | 755.0 / 757.6 | 761.0 | Environment is based on the 250430 version |
| `v250430` (original for paper Table 3) | 8/19 | 755.0 / 755.0 | 761.0 | Doors are measured on hidden old objects |
| `dataset` (`environment_measurement.blend` = 250124) | 6/19 | 710.1 / 710.1 | 715.5 | Old version before the floor was lowered |

## Usage (manual scripts)

### Running from Blender's Scripting workspace

1. Open `assets/env/cage_environment.blend` in Blender
2. Select the target mesh and enter Edit mode if needed
3. Select the vertices, edges, or objects to measure
4. Open the script in the Scripting workspace and run it

### Running from the CLI

```bash
blender --background assets/env/cage_environment.blend --python scripts/06_validation/environment_accuracy/object_dimensions.py
```

## Output files

| File | Contents |
|---|---|
| `selected_vertices.csv` | Vertex index, local coordinates, world coordinates (mm) |
| `height.csv` | Object name, MaxZ, MinZ (mm) |

Output is written to the same directory as the `.blend` file.

## Evaluation procedure

1. Measure the dimensions of each part of the 3D model with the scripts above and export them to CSV
2. Compare with values measured on the physical cage in 3D Slicer 4.11 (Markups module)
3. Results are reported in Table X of the paper

## References

- Physical measurement procedure: paper Section 2.X
- Camera placement: [`docs/camera_mapping.md`](../../../docs/camera_mapping.md)

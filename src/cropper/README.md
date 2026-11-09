
# Synchronized Paired Image Cropper (Paired Image Cropper)

## Features
- For each pair of corresponding images, the crop rectangle selected on one image is applied identically to the other.
- **The previous crop "size"** (width x height) is kept when moving to the next image.
- The size is saved to `paired_cropper_state.json` so it can be restored across sessions.

> If the images in a pair differ in size, use the `--relative` option to convert the coordinates on image A to **relative coordinates** and apply them to image B.

## Usage
Requires Python 3 and OpenCV (`pip install opencv-python`).

### 1) Pair two directories (assumes identical file names)
```
python paired_cropper.py --dir-a path/to/A --dir-b path/to/B --out out_dir
```

### 2) Specify arbitrary pairs with a CSV
Write `A image path,B image path` on each line of `pairs.csv` (no header needed).
```
python paired_cropper.py --map pairs.csv --out out_dir
```

### 3) Apply relatively to pairs with different image sizes
```
python paired_cropper.py --dir-a A --dir-b B --out out_dir --relative
```

### Options
- `--exts ".jpg,.png"`: extensions to load when using `--dir-a/--dir-b` (optional)
- `--start N`: start from the N-th pair (0-based)
- `--relative`: convert the rectangle chosen on A to relative coordinates and apply it to B (useful for pairs with different sizes)

Output goes to `out_dir/A/` and `out_dir/B/`. Files are saved as `.png`, named after the original A-side file name (without extension).

## Window controls
- **Left mouse drag**: create/update the rectangle  
- **Enter or 'n'**: crop and save -> next pair  
- **'b'**: go back to the previous pair (does not save)  
- **'c'**: clear the rectangle  
- **'r'**: re-center the previous "size" (position is reset)  
- **'f'**: toggle fixed-size mode (fixed to the previous size; the clicked position becomes the top-left corner)  
- **'h'**: show/hide help  
- **'q' or ESC**: quit  

> The previous crop size is saved to/restored from `paired_cropper_state.json`. The same size can be used as the initial value across multiple launches.

## FAQ
- **The B-side image has a different size, and the same coordinates go out of bounds.**  
  -> Use `--relative`. The rectangle on A is converted to relative coordinates (0-1), recomputed for the B-side size, and then cropped.

- **I want to crop quickly with a fixed size.**  
  -> Press `f` to enter fixed-size mode; a rectangle of the previous size is placed immediately with the clicked position as its top-left corner.

- **Some files are not matched.**  
  -> With `--dir-a/--dir-b`, only files with the "same name" (ignoring extension) are paired. With a CSV, any pairing is possible.

---

A simple tool proposal for Nakae-Lab. Additional features (rotation, zoomed view, shortcut changes, simultaneous mask cropping, etc.) can be added if needed.

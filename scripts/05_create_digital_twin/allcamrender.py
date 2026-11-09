import bpy, pathlib

scene = bpy.context.scene

# Output directory
OUT_DIR = pathlib.Path(bpy.path.abspath("//renders_selected"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Which cameras to target (use one of the options) =====

# A) Specify explicitly by name (recommended: copy-paste names from the screenshot)
TARGET_CAM_NAMES = [
    "Camera.23506214",
    "Camera.23506226",
    "Camera.23506236",
    'Camera.23506237',
    'Camera.23506239',
    'Camera.23511607',
    'Camera.23511613',
    'Camera.23511614'
    # Keep only the camera names you need
]
cams = [bpy.data.objects[name] for name in TARGET_CAM_NAMES if name in bpy.data.objects]

# B) Only the currently selected cameras
# cams = [o for o in bpy.context.selected_objects if o.type == 'CAMERA']

# C) Only cameras under a specific collection
# TARGET_COLLECTION_NAME = "Camera"  # e.g. collection name in the top-left Outliner
# col = bpy.data.collections.get(TARGET_COLLECTION_NAME)
# cams = [o for o in col.all_objects if o.type == 'CAMERA'] if col else []

# ===== Frames to render =====
# Range + step
FRAME_START, FRAME_END, FRAME_STEP =  162, 162, 5
frames = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
# Or specific frames
# frames = [1, 10, 25, 42]

# ===== Adjust format etc. if needed =====
scene.render.image_settings.file_format = 'PNG'  # 'PNG','JPEG','OPEN_EXR', etc.
# scene.render.engine = 'CYCLES'
# scene.render.resolution_x, scene.render.resolution_y = 1920, 1080

# ===== Run =====
if not cams:
    print("No target cameras found. Check TARGET_CAM_NAMES / selection / collection name.")

for cam in cams:
    (OUT_DIR / cam.name).mkdir(parents=True, exist_ok=True)
    scene.camera = cam
    for f in frames:
        scene.frame_set(f)
        scene.render.filepath = str(OUT_DIR / f'frame_{f}' / f"{cam.name}_{f}.png")
        bpy.ops.render.render(write_still=True)

print("Done.")

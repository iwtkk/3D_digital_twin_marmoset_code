import numpy as np
import bpy
from mathutils import *
import mathutils
import math
import time
import sys
import os
from pathlib import Path

# Intended to be run as a Blender script

def set_object_world_loc_euler(obj, loc_xyz, euler_xyz):
    """
    obj      : bpy.types.Object (Camera)
    loc_xyz  : (x,y,z) in world
    euler_xyz: (rx,ry,rz) in radians, order='XYZ'
    """
    loc = Vector(loc_xyz)
    rot = Euler(euler_xyz, 'XYZ')
    # Set the world matrix while keeping the existing scale
    obj.matrix_world = Matrix.LocRotScale(loc, rot, obj.scale.copy())

def get_obj_any(*names):
    """Get an object by any of the candidate names"""
    for n in names:
        o = bpy.data.objects.get(n)
        if o is not None:
            return o
    raise KeyError(f"Not found: {names}")

# Fallback when an eye is missing: use the midpoint of both eyes
def has_xyz(prefix):
    return all((prefix + a) in cols for a in ("x", "y", "z"))

def read_xyz(frame, prefix):
    return (
        pts[frame, cols[prefix + "x"]],
        pts[frame, cols[prefix + "y"]],
        pts[frame, cols[prefix + "z"]],
    )

def set_pose_location_global(amt, bone_name, global_location):
    global_location = Vector(global_location)
    bone = amt.data.bones[bone_name]
    pose_bone = amt.pose.bones[bone_name]
    mtx = bone.matrix_local.inverted()
    local_loc = mtx @ global_location
    pose_bone.location = local_loc


def set_pose_location_local(amt, bone_name, location):
    bone = amt.data.bones[bone_name]
    pose_bone = amt.pose.bones[bone_name]
    pose_bone.location = location

# (Optional) Apply an offset here if head orientation -> camera orientation needs correction
# Blender camera forward is local -Z. If the CSV Euler treats +Z as forward,
# add 180° around the Y axis to match.
CAM_EULER_OFFSET = (0.0, 0.0, 0.0)
#CAM_EULER_OFFSET = (math.pi, math.pi, math.pi)  # Enable if needed

# --- Added: CSV column prefixes (adjust to your CSV) ---
cam_pts = {
    "camera_l": "pts_l_cam_",  # Expected columns: x,y,z,rx,ry,rz
    "camera_r": "pts_r_cam_",
    "camera_c": "pts_center_eye_", 
}

# Get objects (adjust names to your environment)
cam_objs = {
    "camera_l": get_obj_any("camera_l", "Camera_L", "CameraLeft"),
    "camera_r": get_obj_any("camera_r", "Camera_R", "CameraRight"),
    "camera_c": get_obj_any("camera_c", "Camera_C", "CameraCenter"),
}

# Eye column prefixes
EYE_SRC = {
    "camera_l": "pts_l_eye_",
    "camera_r": "pts_r_eye_",
    "camera_c": "pts_center_eye_", 
}

# Replaced here (originally np.loadtxt(...))

def parse_blender_args():
    """
    When running Blender as:
      blender scene.blend -b -P blender_animation.py -- /path/to/file.csv [--armature NF_Armature]
    everything after "--" is passed through to sys.argv as is.

    Returns:
        csv_path (str), armature_name (str)
    """
    argv = sys.argv
    if "--" not in argv:
        raise ValueError("Pass the CSV path as -- <path> (e.g. blender scene.blend -b -P blender_animation.py -- outputs/animate_temp/transformedcsv/session.csv)")

    idx = argv.index("--")
    rest = argv[idx + 1:]

    csv_path = None
    armature_name = "NF_Armature"

    i = 0
    while i < len(rest):
        if rest[i] == "--armature" and i + 1 < len(rest):
            armature_name = rest[i + 1]
            i += 2
        elif not rest[i].startswith("--"):
            if csv_path is None:
                csv_path = rest[i]
            i += 1
        else:
            i += 1

    if csv_path is None:
        raise ValueError("CSV path not specified. Pass it as -- <path>.")

    return csv_path, armature_name


CSV_PATH, ARMATURE_NAME = parse_blender_args()
print("Using CSV:", CSV_PATH)
print("Using Armature:", ARMATURE_NAME)

pts = np.loadtxt(
    CSV_PATH,
    delimiter=",",
    dtype=str,
)
cols_list = pts[0]
cols = {i: k for k, i in enumerate(cols_list)}
pts = pts[1:].astype(float)

bone_pts = {
    "Base": "pts_base_",
    "Forearm_IK.R": "pts_r_wrist_",
    "Forearm_IK.L": "pts_l_wrist_",
    "Foot_IK.R": "pts_r_ankle_",
    "Foot_IK.L": "pts_l_ankle_",
    "Head_IK": "pts_nose_",
#    "Head_Control_IK": "pts_3_",
    "Chest_IK": "pts_shoulder_mid_",
    "Elbow_IK_L": "pts_l_elbow_",#"pts_7_", 
    "Elbow_IK_R": "pts_r_elbow_",#"pts_8_", 
    "Knee_IK.L": "pts_l_knee_",#"pts_13_", 
    "Knee_IK.R": "pts_r_knee_", #"pts_14_", 
    "Chest_Control_IK": "pts_backbone_peak_",
    "Tail0_IK":"pts_tail_1_",
    "Tail_IK":"pts_tail_2_",
}

face_map = {
    "Base":"pts_base_",
    "L.eye":"pts_l_eye_",
    "R.eye":"pts_r_eye_",
    "L.ear":"pts_l_ear_",
    "R.ear":"pts_r_ear_",
    "L.hand":"pts_l_wrist_",
    "R.hand":"pts_r_wrist_",
    "L.leg":"pts_l_ankle_",
    "R.leg":"pts_r_ankle_",
}

names = ["Base", "L.eye", "R.eye", "L.ear", "R.ear", "L.hand", "R.hand", "L.leg", "R.leg"]

# ── 2) Prepare an empty dict
objs = {}
objs_locs = {}

# ── 3) Fetch from bpy.data.objects in a loop
for name in names:
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        objs[name] = obj
        print(f"Loaded object: {name} → {obj}")
    else:
        print(f"Warning: object '{name}' not found in scene.")
        
bone_names = list(bone_pts.keys())

# Get the Armature object
amt = bpy.data.objects[ARMATURE_NAME]

total_frames = pts.shape[0]
start_time = time.time()

for f in range(pts.shape[0]):
    frame = f + 1
    bpy.context.scene.frame_current = frame

        # Progress output (every 50 frames + first and last)
    if frame == 1 or frame == total_frames or frame % 1000 == 0:
        print(f"[INFO] Processing frame {frame}/{total_frames}")
        sys.stdout.flush()
    
    # --- Set bone positions ---
    bone_locs = []
    for bone in bone_names:
        pts_name = bone_pts[bone]
        xyz_vals = []
        for axis in ["x", "y", "z"]:
            col = pts_name + axis
            xyz_vals.append(pts[f, cols[col]])
        bone_locs.append(xyz_vals)

    # Apply computed coordinates to PoseBones
    for i, bone_name in enumerate(bone_names):
        set_pose_location_global(amt, bone_name, bone_locs[i])

    # ★ Call keyframe_insert directly on each PoseBone here ★
    for bone_name in bone_names:
        pbone = amt.pose.bones[bone_name]
        pbone.keyframe_insert(data_path="location", frame=frame)

    # (Camera / face part processing below can stay as is)


    for cam_key, cam_prefix in cam_pts.items():
        # Position comes from the eyes (left camera = left eye, right camera = right eye)
        eye_prefix = EYE_SRC[cam_key]
        if has_xyz(eye_prefix):
            loc = read_xyz(f, eye_prefix)
        elif has_xyz("pts_l_eye_") and has_xyz("pts_r_eye_"):
            # If one is missing, use the midpoint of both eyes
            le = np.array(read_xyz(f, "pts_l_eye_"))
            re = np.array(read_xyz(f, "pts_r_eye_"))
            loc = tuple((le + re) * 0.5)
        elif has_xyz(cam_prefix):
            # Final fallback: use legacy cam_* position columns if present
            loc = read_xyz(f, cam_prefix)
        else:
            # Skip if no position is available
            continue

        # Rotation (use rx/ry/rz from CSV if present, else 0)
        if all((cam_prefix + ax) in cols for ax in ("rx", "ry", "rz")):
            rx = pts[f, cols[cam_prefix + "rx"]]
            ry = pts[f, cols[cam_prefix + "ry"]]
            rz = pts[f, cols[cam_prefix + "rz"]]
        else:
            rx = ry = rz = 0.0

        rx += CAM_EULER_OFFSET[0]
        ry += CAM_EULER_OFFSET[1]
        rz += CAM_EULER_OFFSET[2]

        cam_obj = cam_objs[cam_key]
        set_object_world_loc_euler(cam_obj, loc, (rx, ry, rz))
        cam_obj.keyframe_insert(data_path="location")
        cam_obj.keyframe_insert(data_path="rotation_euler")
        
        # ── (1) Read coordinates
    face_locs = {}
    for part_name, prefix in face_map.items():
        x = pts[f, cols[f"{prefix}x"]]
        y = pts[f, cols[f"{prefix}y"]]
        z = pts[f, cols[f"{prefix}z"]]
        face_locs[part_name] = mathutils.Vector((x, y, z))
    #        print(f"Frame {frame} — {part_name}: {face_locs[part_name]}")

    # ── (2) Vector computation
    for name in names:
        if name not in objs:
            continue
#        objs[name].location = face_locs[name] @ Matrix.Rotation(math.radians(90), 3, 'X')
        objs[name].location = face_locs[name]
        objs_locs[name] = objs[name].location
        objs[name].keyframe_insert(data_path="location", frame=frame)
        
elapsed = time.time() - start_time
print(f"[INFO] Total processing time for all frames: {elapsed:.2f} s")

blend_stem = Path(bpy.data.filepath).stem
out_dir = Path(os.getcwd()) / "outputs" / "blender"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = str(out_dir / f"{blend_stem}.blend")
bpy.ops.wm.save_mainfile(filepath=out_path)
print("Blend file saved:", out_path)

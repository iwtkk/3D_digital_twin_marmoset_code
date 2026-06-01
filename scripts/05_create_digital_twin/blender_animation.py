import numpy as np
import bpy
from mathutils import Vector, Matrix, Euler
import mathutils
import math
import time
import sys
import os
from pathlib import Path


def get_obj_any(*names):
    for n in names:
        o = bpy.data.objects.get(n)
        if o is not None:
            return o
    print(f"Warning: object not found, skipping: {names}")
    return None


def set_object_world_loc_euler(obj, loc_xyz, euler_xyz):
    loc = Vector(loc_xyz)
    rot = Euler(euler_xyz, 'XYZ')
    obj.matrix_world = Matrix.LocRotScale(loc, rot, obj.scale.copy())


def set_pose_location_global(amt, bone_name, global_location):
    global_location = Vector(global_location)
    bone = amt.data.bones[bone_name]
    pose_bone = amt.pose.bones[bone_name]
    mtx = bone.matrix_local.inverted()
    local_loc = mtx @ global_location
    pose_bone.location = local_loc


def set_pose_location_local(amt, bone_name, location):
    amt.pose.bones[bone_name].location = location


CAM_EULER_OFFSET = (0.0, 0.0, 0.0)

cam_pts = {
    "camera_l": "pts_l_cam_",
    "camera_r": "pts_r_cam_",
    "camera_c": "pts_center_eye_",
    "camera_eqdst_70": "pts_center_eye_",
    "camera_eqdst_100": "pts_center_eye_",
    "camera_pinhole_orig": "pts_center_eye_",
    "camera_pinhole_x2": "pts_center_eye_",
}

cam_objs = {
    "camera_l": get_obj_any("camera_l", "Camera_L", "CameraLeft"),
    "camera_r": get_obj_any("camera_r", "Camera_R", "CameraRight"),
    "camera_c": get_obj_any("camera_c", "Camera_C", "CameraCenter"),
    "camera_eqdst_70": get_obj_any("Camera_Eqdst_70"),
    "camera_eqdst_100": get_obj_any("Camera_Eqdst_100"),
    "camera_pinhole_orig": get_obj_any("Camera_Pinhole_orig"),
    "camera_pinhole_x2": get_obj_any("Camera_Pinhole_x2"),
}

EYE_SRC = {
    "camera_l": "pts_l_eye_",
    "camera_r": "pts_r_eye_",
    "camera_c": "pts_center_eye_",
    "camera_eqdst_70": "pts_center_eye_",
    "camera_eqdst_100": "pts_center_eye_",
    "camera_pinhole_orig": "pts_center_eye_",
    "camera_pinhole_x2": "pts_center_eye_",
}


def parse_blender_args():
    argv = sys.argv
    if "--" not in argv:
        raise ValueError("CSVパスを -- <path> で渡してください")
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
        raise ValueError("CSVパスが指定されていません。")
    return csv_path, armature_name


CSV_PATH, ARMATURE_NAME = parse_blender_args()
print("Using CSV:", CSV_PATH)
print("Using Armature:", ARMATURE_NAME)

pts = np.loadtxt(CSV_PATH, delimiter=",", dtype=str)
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
    "Chest_IK": "pts_shoulder_mid_",
    "Elbow_IK_L": "pts_l_elbow_",
    "Elbow_IK_R": "pts_r_elbow_",
    "Knee_IK.L": "pts_l_knee_",
    "Knee_IK.R": "pts_r_knee_",
    "Chest_Control_IK": "pts_backbone_peak_",
    "Tail0_IK": "pts_tail_1_",
    "Tail_IK": "pts_tail_2_",
}

face_map = {
    "Base": "pts_base_",
    "L.eye": "pts_l_eye_",
    "R.eye": "pts_r_eye_",
    "L.ear": "pts_l_ear_",
    "R.ear": "pts_r_ear_",
    "L.hand": "pts_l_wrist_",
    "R.hand": "pts_r_wrist_",
    "L.leg": "pts_l_ankle_",
    "R.leg": "pts_r_ankle_",
}

names = ["Base", "L.eye", "R.eye", "L.ear", "R.ear", "L.hand", "R.hand", "L.leg", "R.leg"]

objs = {}
for name in names:
    if name in bpy.data.objects:
        objs[name] = bpy.data.objects[name]
        print(f"Loaded object: {name}")
    else:
        print(f"Warning: object '{name}' not found in scene.")

bone_names = list(bone_pts.keys())
amt = bpy.data.objects[ARMATURE_NAME]

total_frames = pts.shape[0]
# 1-indexed のフレーム番号配列（float: fcurve の co に使う）
frame_indices = np.arange(1, total_frames + 1, dtype=float)

start_time = time.time()
print(f"[INFO] 全{total_frames}フレームを一括処理します（バッチモード）")


def _get_fcurves_container(obj, action):
    """Blender 4.x / 5.x 両対応でfcurvesコレクションを返す。

    Blender 5.0でAction APIが刷新され action.fcurves が廃止された。
    5.x では layer → strip → channelbag → fcurves の構造を辿る必要がある。
    """
    # Blender 4.x
    if hasattr(action, 'fcurves'):
        return action.fcurves
    # Blender 5.x: layered action 構造
    layer = action.layers[0] if action.layers else action.layers.new(name="Layer")
    strip = layer.strips[0] if layer.strips else layer.strips.new(type='KEYFRAME')
    slot = action.slots[0] if action.slots else action.slots.new(id_type=obj.id_type, name=obj.name)
    obj.animation_data.action_slot = slot
    return strip.channelbag(slot, ensure=True).fcurves


def get_or_create_action(obj, action_name):
    if obj.animation_data is None:
        obj.animation_data_create()
    if obj.animation_data.action is None:
        obj.animation_data.action = bpy.data.actions.new(name=action_name)
    action = obj.animation_data.action
    fcurves = _get_fcurves_container(obj, action)
    return action, fcurves


BATCH_SIZE = 10000


def insert_fcurve_batch(fcurves, data_path, index, frames, values):
    """全フレーム分のキーフレームを一括挿入。per-frame keyframe_insert より大幅に高速。"""
    fc = fcurves.find(data_path, index=index)
    if fc is None:
        fc = fcurves.new(data_path=data_path, index=index)
    else:
        fc.keyframe_points.clear()

    n = len(frames)
    # BATCH_SIZE ずつ add することでBlender内部のメモリ確保を分散
    for chunk_start in range(0, n, BATCH_SIZE):
        fc.keyframe_points.add(min(BATCH_SIZE, n - chunk_start))

    # co は [frame0, val0, frame1, val1, ...] のフラット配列
    co = np.empty(n * 2, dtype=float)
    co[0::2] = frames
    co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)

    # LINEAR補間 (CONSTANT=0, LINEAR=1, BEZIER=2)
    fc.keyframe_points.foreach_set("interpolation", [1] * n)
    fc.update()


def has_xyz(prefix):
    return all((prefix + a) in cols for a in ("x", "y", "z"))


# ===== 1. アーマチュアボーン =====
_, amt_fcurves = get_or_create_action(amt, "ArmatureAnimation")

for bone_name in bone_names:
    prefix = bone_pts[bone_name]
    global_locs = pts[:, [cols[prefix + "x"], cols[prefix + "y"], cols[prefix + "z"]]]  # (N, 3)

    # rest pose の逆行列でグローバル→ボーンローカル座標に変換（全フレーム一括）
    inv_mat = np.array(amt.data.bones[bone_name].matrix_local.inverted())  # (4, 4)
    ones = np.ones((total_frames, 1))
    local_locs = (inv_mat @ np.hstack([global_locs, ones]).T).T[:, :3]  # (N, 3)

    data_path = f'pose.bones["{bone_name}"].location'
    for axis_idx in range(3):
        insert_fcurve_batch(amt_fcurves, data_path, axis_idx, frame_indices, local_locs[:, axis_idx])

print(f"[INFO] ボーン処理完了: {time.time() - start_time:.2f}秒")

# ===== 2. カメラ =====
for cam_key, cam_prefix in cam_pts.items():
    cam_obj = cam_objs[cam_key]
    if cam_obj is None:
        continue
    eye_prefix = EYE_SRC[cam_key]

    if has_xyz(eye_prefix):
        locs = pts[:, [cols[eye_prefix + a] for a in ("x", "y", "z")]]
    elif has_xyz("pts_l_eye_") and has_xyz("pts_r_eye_"):
        le = pts[:, [cols["pts_l_eye_" + a] for a in ("x", "y", "z")]]
        re = pts[:, [cols["pts_r_eye_" + a] for a in ("x", "y", "z")]]
        locs = (le + re) * 0.5
    elif has_xyz(cam_prefix):
        locs = pts[:, [cols[cam_prefix + a] for a in ("x", "y", "z")]]
    else:
        continue

    if all((cam_prefix + ax) in cols for ax in ("rx", "ry", "rz")):
        rots = pts[:, [cols[cam_prefix + ax] for ax in ("rx", "ry", "rz")]].copy()
    else:
        rots = np.zeros((total_frames, 3))

    rots[:, 0] += CAM_EULER_OFFSET[0]
    rots[:, 1] += CAM_EULER_OFFSET[1]
    rots[:, 2] += CAM_EULER_OFFSET[2]

    _, cam_fcurves = get_or_create_action(cam_obj, f"CamAnimation_{cam_key}")
    for axis_idx in range(3):
        insert_fcurve_batch(cam_fcurves, "location", axis_idx, frame_indices, locs[:, axis_idx])
        insert_fcurve_batch(cam_fcurves, "rotation_euler", axis_idx, frame_indices, rots[:, axis_idx])

print(f"[INFO] カメラ処理完了: {time.time() - start_time:.2f}秒")

# ===== 3. 顔パーツオブジェクト =====
for name in names:
    if name not in objs:
        continue
    prefix = face_map[name]
    obj_locs = pts[:, [cols[f"{prefix}{a}"] for a in ("x", "y", "z")]]

    _, obj_fcurves = get_or_create_action(objs[name], f"ObjAnimation_{name}")
    for axis_idx in range(3):
        insert_fcurve_batch(obj_fcurves, "location", axis_idx, frame_indices, obj_locs[:, axis_idx])

print(f"[INFO] 顔パーツ処理完了: {time.time() - start_time:.2f}秒")

elapsed = time.time() - start_time
print(f"[INFO] 全フレーム処理時間: {elapsed:.2f} 秒")

out_path = bpy.data.filepath
bpy.ops.wm.save_mainfile(filepath=out_path)
print("Blend file saved:", out_path)

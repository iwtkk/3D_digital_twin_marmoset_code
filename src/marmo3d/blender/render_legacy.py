from multiprocessing.util import is_exiting
# Legacy script kept for reference; render.py is the active entry point.
import sys
import os
import json
import argparse
import pickle
import csv
import time
import math
import glob
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

ADDON_DIR = Path(__file__).resolve().parent.parent.parent
print(ADDON_DIR)
VENDOR_DIR = os.path.join(ADDON_DIR, "vendor_src/PyYAML-6.0.1/lib/")
if VENDOR_DIR not in sys.path:
    sys.path.append(VENDOR_DIR)
import yaml

# Attempt to import bpy (will work inside Blender)
try:
    import bpy
    import mathutils
except ImportError:
    print("WARNING: 'bpy' not found. This script must be run inside Blender.")
    # For testing outside blender maybe?
    pass

import numpy as np
try:
    import yaml
except ImportError:
    # Fallback if yaml is not installed in Blender python
    # We might need a simple parser or assume json if yaml fails, 
    # but spec says render_spec.yaml
    print("WARNING: 'yaml' not found. Please install pyyaml in Blender's python.")

# =============================================================================
# CONSTANTS & UTILS
# =============================================================================

KP_NAMES = [
    "nose","L eye","R eye","L ear","R ear","L shoulder","R shoulder",
    "L elbow","R elbow","L wrist","R wrist","L hip","R hip",
    "L knee","R knee","L ankle","R ankle","backbone peak","tail_1","tail_2"
]

def sanitize(name: str) -> str:
    import re
    key = name.lower().strip().replace(" ", "_")
    key = re.sub(r"[^a-z0-9_]", "", key)
    return key

def unique_keys(names):
    seen = {}
    out = []
    for n in names:
        base = sanitize(n)
        k = base
        i = 2
        while k in seen:
            k = f"{base}_{i}"
            i += 1
        seen[k] = True
        out.append(k)
    return out

def rot_euler(rx=0.0, ry=0.0, rz=0.0, order="xyz") -> np.ndarray:
    def rx_mat(t): c,s=np.cos(t),np.sin(t); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def ry_mat(t): c,s=np.cos(t),np.sin(t); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    def rz_mat(t): c,s=np.cos(t),np.sin(t); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    
    mats = {'x': rx_mat, 'y': ry_mat, 'z': rz_mat}
    R = np.eye(3)
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    vals = {'x': rx, 'y': ry, 'z': rz}
    
    for ax in order:
        R = mats[ax](vals[ax]) @ R
    return R

def _safe_norm(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def _rotmat_to_euler_xyz(R):
    r00 = R[..., 0, 0]; r01 = R[..., 0, 1]; r02 = R[..., 0, 2]
    r10 = R[..., 1, 0]; r11 = R[..., 1, 1]; r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]; r21 = R[..., 2, 1]; r22 = R[..., 2, 2]
    sy = np.sqrt(r00*r00 + r10*r10)
    singular = sy < 1e-8
    rx = np.where(~singular, np.arctan2(r21, r22), np.arctan2(-r12, r11))
    ry = np.where(~singular, np.arctan2(-r20, sy), np.arctan2(-r20, sy))
    rz = np.where(~singular, np.arctan2(r10, r00), np.zeros_like(r00))
    return np.stack([rx, ry, rz], axis=-1)

# =============================================================================
# CSV CONVERTER
# =============================================================================

class CsvConverter:
    """
    Handles logic from process_dlc.py and transform_dlc.py
    """
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        # Default settings from transform_dlc.py
        # SCALE = 1 / 1000  (mm -> m for Blender) 
        # I should trust render_spec.yaml
        units = self.spec.get("render", {}).get("armature", {}).get("units", {})
        self.scale_factor = eval(str(units.get("scale_factor")))
        
        # init neck_tail from spec or default
        init_conf = self.spec.get("render", {}).get("csv", {}).get("init", {})
        raw_nt = init_conf.get("neck_tail")
        self.blender_neck_tail = [float(eval(str(v))) for v in raw_nt]

    def load_kp3d(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.data = data
        return data

    def _get_init_hash(self) -> str:
        """csv.init設定からハッシュ値を生成（ファイル名に使用）"""
        import hashlib
        init_conf = self.spec.get("render", {}).get("csv", {}).get("init", {})
        init_str = json.dumps(init_conf, sort_keys=True, default=str)
        return hashlib.md5(init_str.encode()).hexdigest()[:8]


    def process(self, output_csv_path: str):
        # 1. Processing (similar to process_dlc.py)
        # Assuming single animal for simplicity or taking first
        X = self.data["kp3d"] # (n_animal, n_frame, n_kp, 3)
        S = self.data["kp3d_score"]
        
        # Take first animal
        x_all = X[0].copy() # (n_frame, n_kp, 3)
        s_all = S[0].copy()
        
        n_frames, n_kp, _ = x_all.shape
        
        rows = []
        
        for i_frame in range(n_frames):
            x = x_all[i_frame].copy()
            s = s_all[i_frame].copy()
            kp_names_now = KP_NAMES[:]

            # Add shoulder_mid (5+6)/2
            # Indices: 5: L shoulder, 6: R shoulder
            if 6 < len(x):
                mid = (x[5] + x[6]) / 2
                x = np.vstack([x, mid])
                s_mid = (s[5] + s[6]) / 2
                s = np.concatenate([s, [s_mid]])
                kp_names_now.append("shoulder_mid")

            # Add base (11+12)/2
            # Indices: 11: L hip, 12: R hip
            if 12 < len(x):
                mid = (x[11] + x[12]) / 2
                x = np.vstack([x, mid])
                s_mid = (s[11] + s[12]) / 2
                s = np.concatenate([s, [s_mid]])
                kp_names_now.append("base")

            # Rotation (Alignment)
            # process_dlc.py: R = rot_euler(rx=-90, ry=0, rz=180)
            R = rot_euler(rx=-90, ry=0, rz=180, order="xyz")
            pivot = np.zeros(3) # process_dlc.py default
            
            x = (x - pivot) @ R.T + pivot
            
            kp_keys = unique_keys(kp_names_now)
            
            # Dictionary for this frame
            row = {"frame": i_frame + 1} # 1-based frame for Blender usually
            
            # 2. Transforming (scale & offset)
            # transform_dlc.py logic
            # Scale
            x = x * self.scale_factor
            
            # Offset
            # Find shoulder_mid index
            try:
                sm_idx = kp_keys.index("pts_shoulder_mid")
            except ValueError:
                try:
                    sm_idx = kp_keys.index("pts_shoulder_mid_") # depending on sanitize
                except:
                    sm_idx = -1
            
            # In process_dlc, "shoulder_mid" -> sanitize -> "shoulder_mid" -> unique -> "shoulder_mid" or "shoulder_mid_"
            # Let's check keys.
            # actually unique_keys uses sanitize which lower cases and replaces space with _.
            # "shoulder_mid" -> "shoulder_mid"
            
            # transform_dlc: neck_diff = points... - bld_neck
            # It seems it aligns shoulder_mid to BLENDER_NECK_TAIL
            # But BLENDER_NECK_TAIL in transform_dlc is [1.16, 0.965, 0.78] calculated.
            
            bld_neck = np.array(self.blender_neck_tail)
            
            # If we found shoulder_mid
            if sm_idx != -1:
                # Calculate shift
                curr_sm = x[sm_idx]
                offset = bld_neck - curr_sm # We want SM to be at bld_neck
                # However, transform_dlc logic is a bit weird:
                # neck_diff = points... - bld_neck
                # then for each col: col += bld_neck ?
                # Wait, transform_dlc.py says:
                # neck_diff = ... - bld_neck
                # then: if col[-1] == 'x': points[col] += bld_neck[0]
                # This looks like it adds bld_neck blindly?
                # Ah, "points" in transform_dlc are previously scaled.
                # Let's assume standard behavior: we want to place the rig at a specific location.
                # If "shoulder_mid" corresponds to Neck, we align it.
                
                # Re-reading transform_dlc.py carefully:
                # neck_diff = points... - bld_neck  <-- variable calculated but NOT USED?
                # Loop: points[col] += bld_neck[0]
                # It seems it JUST ADDS bld_neck to everything.
                # This explicitly assumes inputs are centered at 0 or relative?
                # process_dlc output seems to be just rotated.
                # So if process_dlc result is around 0, then adding bld_neck puts it at bld_neck.
                pass
                
            # Applying offset (just adding bld_neck as per transform_dlc.py)
            x = x + bld_neck

            # Add to row
            for k, pt in zip(kp_keys, x):
                row[f"pts_{k}_x"] = pt[0]
                row[f"pts_{k}_y"] = pt[1]
                row[f"pts_{k}_z"] = pt[2]

            # 3. Add Camera Pose (Pseudo-eyes)
            # transform_dlc.py: add_cam_pose_with_euler
            # Needs L eye, R eye, L ear, R ear
            # keys sanitized: "l_eye", "r_eye", "l_ear", "r_ear"
            
            # helpers
            def get_vec(k): 
                idx = kp_keys.index(k)
                return x[idx]

            try:
                le = get_vec("l_eye")
                re = get_vec("r_eye")
                ler = get_vec("l_ear")
                rer = get_vec("r_ear")
                
                # Logic from add_cam_pose_with_euler
                eye_c = (le + re) * 0.5
                ear_c = (ler + rer) * 0.5
                
                forward_sign = -1
                s_raw = re - le
                f_raw = (eye_c - ear_c) * float(forward_sign)
                
                s_n = _safe_norm(s_raw)
                f_n = _safe_norm(f_raw)
                
                u_raw = np.cross(f_n, s_n)
                # ...checks...
                u_n = _safe_norm(u_raw)
                s_ortho = _safe_norm(np.cross(u_n, f_n))
                
                R_cam = np.stack([s_ortho, u_n, f_n], axis=-1)
                det = np.linalg.det(R_cam)
                if det < 0: R_cam[:, 0] *= -1.0 # only x column ... wait, this is 1 frame (3,3)
                
                eul = _rotmat_to_euler_xyz(R_cam[np.newaxis, ...])[0] # returns (rx,ry,rz)
                
                # Add cam keys
                for prefix in ["pts_l_cam_", "pts_r_cam_"]:
                    # Location: eyes
                    pos = le if "l_cam" in prefix else re
                    row[f"{prefix}x"] = pos[0]
                    row[f"{prefix}y"] = pos[1]
                    row[f"{prefix}z"] = pos[2]
                    row[f"{prefix}rx"] = eul[0]
                    row[f"{prefix}ry"] = eul[1]
                    row[f"{prefix}rz"] = eul[2]
                    
            except ValueError:
                # Missing required keys
                pass

            rows.append(row)
            
        # Write CSV
        if not rows:
            print("WARNING: No rows generated.")
            return

        header = rows[0].keys()
        
        # Ensure directory exists
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"[INFO] CSV generated at {output_csv_path}")

    def process_single_frame(self, output_csv_path: str, frame_index: int):
        """指定されたフレームのみをCSVに保存する（temp_kp3d.csv用）
        
        Args:
            output_csv_path: 出力CSVパス
            frame_index: フレームインデックス（1-based、Blender形式）
        """
        X = self.data["kp3d"]
        S = self.data["kp3d_score"]
        
        x_all = X[0].copy()
        s_all = S[0].copy()
        
        n_frames, n_kp, _ = x_all.shape
        
        # Convert 1-based frame to 0-based index
        i_frame = frame_index - 1
        
        if i_frame < 0 or i_frame >= n_frames:
            print(f"WARNING: Frame {frame_index} out of range (1-{n_frames})")
            return
        
        # Process single frame using existing logic
        rows = []
        x = x_all[i_frame].copy()
        s = s_all[i_frame].copy()
        kp_names_now = KP_NAMES[:]

        # Add shoulder_mid (5+6)/2
        if 6 < len(x):
            mid = (x[5] + x[6]) / 2
            x = np.vstack([x, mid])
            s_mid = (s[5] + s[6]) / 2
            s = np.concatenate([s, [s_mid]])
            kp_names_now.append("shoulder_mid")

        # Add base (11+12)/2
        if 12 < len(x):
            mid = (x[11] + x[12]) / 2
            x = np.vstack([x, mid])
            s_mid = (s[11] + s[12]) / 2
            s = np.concatenate([s, [s_mid]])
            kp_names_now.append("base")

        R = rot_euler(rx=-90, ry=0, rz=180, order="xyz")
        pivot = np.zeros(3)
        x = (x - pivot) @ R.T + pivot
        
        kp_keys = unique_keys(kp_names_now)
        row = {"frame": frame_index}  # Keep 1-based
        
        x = x * self.scale_factor
        bld_neck = np.array(self.blender_neck_tail)
        x = x + bld_neck

        for k, pt in zip(kp_keys, x):
            row[f"pts_{k}_x"] = pt[0]
            row[f"pts_{k}_y"] = pt[1]
            row[f"pts_{k}_z"] = pt[2]

        # Camera pose
        def get_vec(k): 
            idx = kp_keys.index(k)
            return x[idx]

        try:
            le = get_vec("l_eye")
            re = get_vec("r_eye")
            ler = get_vec("l_ear")
            rer = get_vec("r_ear")
            
            eye_c = (le + re) * 0.5
            ear_c = (ler + rer) * 0.5
            
            forward_sign = -1
            s_raw = re - le
            f_raw = (eye_c - ear_c) * float(forward_sign)
            
            s_n = _safe_norm(s_raw)
            f_n = _safe_norm(f_raw)
            
            u_raw = np.cross(f_n, s_n)
            u_n = _safe_norm(u_raw)
            s_ortho = _safe_norm(np.cross(u_n, f_n))
            
            R_cam = np.stack([s_ortho, u_n, f_n], axis=-1)
            det = np.linalg.det(R_cam)
            if det < 0: R_cam[:, 0] *= -1.0
            
            eul = _rotmat_to_euler_xyz(R_cam[np.newaxis, ...])[0]
            
            for prefix in ["pts_l_cam_", "pts_r_cam_"]:
                pos = le if "l_cam" in prefix else re
                row[f"{prefix}x"] = pos[0]
                row[f"{prefix}y"] = pos[1]
                row[f"{prefix}z"] = pos[2]
                row[f"{prefix}rx"] = eul[0]
                row[f"{prefix}ry"] = eul[1]
                row[f"{prefix}rz"] = eul[2]
        except ValueError:
            pass

        rows.append(row)
        
        header = rows[0].keys()
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"[INFO] Single frame {frame_index} CSV generated at {output_csv_path}")

    def process_all_frames(self, output_csv_path: str) -> bool:
        """全フレームをCSVに保存する（all_kp3d_{hash}.csv用）
        
        既にファイルが存在する場合はスキップする。
        
        Args:
            output_csv_path: 出力CSVパス
            
        Returns:
            bool: 処理を実行した場合True、スキップした場合False
        """
        if Path(output_csv_path).exists():
            print(f"[INFO] CSV already exists, skipping: {output_csv_path}")
            return False
        
        # Use existing process() logic
        self.process(output_csv_path)
        return True



# =============================================================================
# BLENDER APP
# =============================================================================

class BlenderRenderApp:
    def __init__(self):
        self.args = self.parse_args()
        # Parse extra args
        self.cmd_args = self.parse_cmd_args()
        
        self.load_config()
        self.manifest = {
            "animal": "unknown", # TODO: get from where? maybe path
            "session_id": self.cmd_args.get("session_id"),
            "frames": [],
            "render_profile": self.cmd_args.get("profile"),
            "view_ids": [],
            "kp3d_source_path": self.cmd_args.get("kp3d_pkl"),
            "status": "started",
            "skip_reason": ""
        }
        
    def parse_args(self):
        # We need to parse args after "--"
        if "--" in sys.argv:
            return sys.argv[sys.argv.index("--") + 1:]
        return []

    def parse_cmd_args(self):
        # Manual parsing because argparse might conflict with blender args if not careful
        # But here we are after "--", so it's safe to use argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--session_id", required=True)
        parser.add_argument("--kp3d_pkl", required=True)
        parser.add_argument("--render_spec", required=True)
        parser.add_argument("--profile", required=True)
        parser.add_argument("--frames", default=None) # Start,End or List
        parser.add_argument("--render", type=int, default=1)
        parser.add_argument("--output_dir", required=True)
        parser.add_argument("--job_id", default="job")
        parser.add_argument("--csv_output_path", default=None)
        
        args = parser.parse_args(self.args)
        return vars(args)

    def load_config(self):
        with open(self.cmd_args["render_spec"], "r") as f:
            self.spec = yaml.safe_load(f)

    def run(self):
        try:
            self._run_verified()
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            self.manifest["status"] = "failed"
            self.manifest["skip_reason"] = str(e)
        finally:
            self.write_manifest()

    def _run_verified(self):
        # Parse frames argument first to determine CSV processing mode
        frames_arg = self.cmd_args.get("frames")
        target_frames = None
        if frames_arg:
            try:
                import ast
                target_frames = ast.literal_eval(frames_arg)
                if isinstance(target_frames, int):
                    target_frames = [target_frames]
                elif isinstance(target_frames, tuple):
                    target_frames = list(target_frames)
            except:
                pass
        
        # 1. Generate CSV - branch based on frame specification
        converter = CsvConverter(self.spec)
        converter.load_kp3d(self.cmd_args["kp3d_pkl"])
        
        if target_frames:
            # フレーム指定あり → 単一フレーム処理
            csv_path = self.cmd_args.get("csv_output_path")
            if not csv_path:
                csv_path = os.path.join(self.cmd_args["output_dir"], "temp_kp3d.csv")
            
            # 複数フレーム指定の場合は最初のフレームのみ処理
            # TODO: 複数フレームサポートが必要な場合は拡張
            converter.process_single_frame(csv_path, target_frames[0])
        else:
            # フレーム指定なし → 全フレーム処理
            init_hash = converter._get_init_hash()
            csv_path = self.cmd_args.get("csv_output_path")
            if not csv_path:
                csv_path = os.path.join(self.cmd_args["output_dir"], f"all_kp3d_{init_hash}.csv")
            
            converter.process_all_frames(csv_path)
        
        # 2. Apply Armature
        self.apply_armature(csv_path, target_frames)

        
        # 3. Save .blend
        blend_path = os.path.join(self.cmd_args["output_dir"], f"{self.cmd_args['job_id']}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print(f"[INFO] Saved blend to {blend_path}")
        
        # 4. Render
        if self.cmd_args["render"]:
            self.render(target_frames)
            
        self.manifest["status"] = "success"
        if target_frames:
             self.manifest["frames"] = target_frames

    def apply_armature(self, csv_path: str, target_frames: Optional[List[int]]):
        # Similar to blender_animation.py
        import numpy as np
        
        print(f"[INFO] Applying armature from {csv_path}")
        if target_frames:
             print(f"[INFO] Target frames: {target_frames}")
        else:
             print(f"[INFO] Applying to ALL frames")

        # Load CSV
        # We use standard reading because we need it as text/float
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [r for r in reader]
        
        data = np.array(rows) # keep strings for now?
        # Actually better to map headers to indices
        cols = {h: i for i, h in enumerate(header)}
        
        # Bone Mapping from spec
        bone_mapping = self.spec["render"]["armature"]["bone_mapping"]
        # "pts_base_": "Base", etc.
        # But CSV columns are "pts_base_x", etc.
        
        csv_bone_map = {} # CSV Prefix -> Bone Name
        # Invert mapping: Bone -> CSV prefix
        # spec: "pts_base_": "Base"
        # We need CSV keys.
        
        # Prepare helper
        def get_xyz(row_idx, prefix):
            # check headers
            try:
                x = float(data[row_idx, cols[prefix + "x"]])
                y = float(data[row_idx, cols[prefix + "y"]])
                z = float(data[row_idx, cols[prefix + "z"]])
                return [x, y, z]
            except KeyError:
                return None

        amt = bpy.data.objects.get(self.spec["render"]["armature"]["object_name"])
        if not amt:
             raise ValueError(f"Armature {self.spec['render']['armature']['object_name']} not found")

        # Switch to Pose Mode
        bpy.context.view_layer.objects.active = amt
        bpy.ops.object.mode_set(mode='POSE')

        frames_to_process = []
        if target_frames:
             # Find row indices matching frames
             # Assumes CSV "frame" matches target_frames
             # CSV frame is 1-based, target_frames is 1-based (from user arg)
             # Let's create a map frame -> row_idx
             frame_col = cols.get("frame")
             if frame_col is None:
                 print("WARNING: 'frame' column not found in CSV. Assuming sequential starting 1.")
                 # default sequential
                 for f_idx, r in enumerate(data):
                     fr = f_idx + 1
                     if fr in target_frames:
                         frames_to_process.append((fr, f_idx))
             else:
                 for f_idx, r in enumerate(data):
                     fr = int(float(r[frame_col]))
                     if fr in target_frames:
                         frames_to_process.append((fr, f_idx))
        else:
             # Process all
             frame_col = cols.get("frame")
             for f_idx, r in enumerate(data):
                 if frame_col is not None:
                     fr = int(float(r[frame_col]))
                 else:
                     fr = f_idx + 1
                 frames_to_process.append((fr, f_idx))
        
        # Apply
        for fr_num, row_idx in frames_to_process:
            bpy.context.scene.frame_set(fr_num)
            
            # Apply Bones
            for kp_prefix, bone_name in bone_mapping.items():
                xyz = get_xyz(row_idx, kp_prefix)
                if xyz:
                     # Set Global Location
                     vec = mathutils.Vector(xyz)
                     # Convert to local space of bone parent? 
                     # blender_animation.py sets global location 
                     # set_pose_location_global(amt, bone_name, vec)
                     
                     if bone_name in amt.pose.bones:
                         pbone = amt.pose.bones[bone_name]
                         # transform world to local
                         # local = bone.matrix_local.inverted() @ world
                         # But wait, pose bone location is relative to rest pose? 
                         # Actually blender_animation.py:
                         # bone = amt.data.bones[bone_name] (Rest bone)
                         # mtx = bone.matrix_local.inverted()
                         # local_loc = mtx @ global_location
                         # pbone.location = local_loc
                         
                         bone = amt.data.bones[bone_name]
                         mtx = bone.matrix_local.inverted()
                         local_loc = mtx @ vec
                         pbone.location = local_loc
                         
                         pbone.keyframe_insert(data_path="location", frame=fr_num)
            
            # Apply Camera/Eyes logic (if mapped to objects)
            # In blender_animation.py, it moves objects "Camera_L", "Camera_R" etc.
            # We should check if we have camera columns
            cam_prefixes = {"pts_l_cam_": "camera_l", "pts_r_cam_": "camera_r"} # default mapping/assumption
            
            for pre, cam_key in cam_prefixes.items():
                xyz = get_xyz(row_idx, pre)
                if xyz:
                     # Find camera object
                     # render_spec.yaml doesn't specify camera object names explicitly, 
                     # but profiles do. "Camera.{camera_id}"
                     # BUT here we are setting up the SCENE matching the capture (pseudo-eyes)
                     # usually for marmo_eyes profile.
                     # If we are in env_camera profile, we might not need to move cameras?
                     # The cameras in env_camera are fixed.
                     # Only pseudo-eyes cameras move.
                     
                     # Check if objects exist
                     # Mapping: "camera_l" -> Object
                     # In blender_animation.py: get_obj_any("camera_l", "Camera_L", ...)
                     # Let's try to find them
                     potential_names = [cam_key, cam_key.title(), cam_key.replace("_", "").title()] 
                     obj = None
                     for n in potential_names:
                         if n in bpy.data.objects:
                             obj = bpy.data.objects[n]
                             break
                     
                     if obj:
                         obj.location = mathutils.Vector(xyz)
                         obj.keyframe_insert(data_path="location", frame=fr_num)
                         
                         # Rotation
                         try:
                             rx = float(data[row_idx, cols[pre + "rx"]])
                             ry = float(data[row_idx, cols[pre + "ry"]])
                             rz = float(data[row_idx, cols[pre + "rz"]])
                             obj.rotation_euler = (rx, ry, rz)
                             obj.keyframe_insert(data_path="rotation_euler", frame=fr_num)
                         except:
                             pass

    def render(self, target_frames: Optional[List[int]]):
        profile_name = self.cmd_args["profile"]
        prof = self.spec["render"]["rendering"]["profiles"][profile_name]
        
        # Set Engine
        engine = prof.get("engine", "CYCLES")
        bpy.context.scene.render.engine = engine
        bpy.context.scene.cycles.samples = int(prof.get("samples", 128))
        
        # Transparent
        bpy.context.scene.render.film_transparent = prof.get("transparent_background", False)
        
        # Resolution
        res = prof.get("resolution")
        if isinstance(res, list):
            bpy.context.scene.render.resolution_x = res[0]
            bpy.context.scene.render.resolution_y = res[1]
        
        # Determine Cameras
        cameras = []
        if "views" in prof:
             views = prof["views"]
             if "camera_pattern" in views:
                 # e.g. Camera.{camera_id}
                 # get IDs from source
                 # Simplified: Just grab all objects matching pattern or use generic valid cameras
                 # For now, let's implement the specific logic for "env_camera" which has 8 cams
                 source = views['view_index']['source']
                 project_root = Path(__file__).resolve().parents[2]
                 source_path = Path(source.replace("{PROJECT_ROOT}", str(project_root)))
                 source_path = Path(glob.glob(str(source_path))[0])
                 if source_path.exists():
                     with open(source_path, "r") as f:
                         src_data = yaml.safe_load(f)
                         print(src_data['camera_id'])
                         cameras = src_data['camera_id']
                 else:
                     raise ValueError("Camera IDs not found in source file: " + str(source_path))
        else:
            raise ValueError("Views not found in profile: " + profile_name)

        # Frames
        render_frames = target_frames if target_frames else []
        if not render_frames:
             pass # Render list empty?
        
        # Output template
        # We need to construct output path.
        # render_spec.yaml: output info is in rendering.output
        out_cfg = self.spec["render"]["rendering"]["output"]
        # image or video
        # Determine mode
        mode = "image" # Default
        if mode == "image":
             img_cfg = out_cfg["image"]
             # path_template: "{PROJECT_ROOT}/.../{frame:08d}/view_{view_index:02d}.png"
             # helper to format path
             pass
             
        print(f"[INFO] Starting Render for {len(render_frames)} frames on {len(cameras)} cameras")
        
        for cam_id in cameras:
            cam_name = "Camera." + str(cam_id)
            cam_obj = bpy.data.objects.get(cam_name)
            if not cam_obj:
                print(f"[WARN] Camera {cam_name} not found, skipping.")
                continue
                
            bpy.context.scene.camera = cam_obj
            
            for fr in render_frames:
                bpy.context.scene.frame_set(fr)
                
                # Construct Path
                # For simplicity, we use the output_dir passed in cmd_args + filename
                # or strictly follow path_template?
                # The user wants automated pipeline.
                # rendering_parallelized.py should probably have passed the EXACT output directory structure?
                # Or we build it here.
                # cmd_args["output_dir"] is likely the session root or specific folder?
                # Plan says: "Output Dir" passed after --.
                # Let's write to output_dir/frame_XXXX/cam.png
                
                # Using simple naming for now, or respect template if possible.
                # To fail-safe, I'll use simple: output_dir/{frame}/{cam_name}.png
                out_path = Path(self.cmd_args["output_dir"]) / f"{fr:08d}" / f"{cam_name}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                bpy.context.scene.render.filepath = str(out_path)
                
                print(f"Rendering frame {fr} cam {cam_name} -> {out_path}")
                bpy.ops.render.render(write_still=True)
                
                self.manifest["view_ids"].append(cam_name)

    def write_manifest(self):
        man_path = Path(self.cmd_args["output_dir"]) / "manifest.json"
        with open(man_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        print(f"[INFO] Manifest written to {man_path}")

if __name__ == "__main__":
    app = BlenderRenderApp()
    app.run()

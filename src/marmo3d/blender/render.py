from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # blender/ → marmo3d/ → src/ → project root
VENDOR_DIR = PROJECT_ROOT / "third_party"
if str(VENDOR_DIR) not in sys.path:
    sys.path.append(str(VENDOR_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

import glob
import render_paths
import yaml

try:
    import bpy
    import mathutils
except ImportError as exc:
    raise RuntimeError("render.py must be executed inside Blender.") from exc

FRAME_DIGITS = 7
DEFAULT_IMAGE_EXT = "png"
CAMERA_POSE_OBJECTS = {"pts_l_cam_": "Camera_L", "pts_r_cam_": "Camera_R"}


def log(message: str) -> None:
    print(f"[render] {message}")


def parse_blender_args() -> List[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", required=True)
    parser.add_argument("--kp3d_pkl", required=True)
    parser.add_argument("--render_spec", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--job_id", default="job")
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--all_frames", type=int, default=None)
    return parser


def parse_frames_arg(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, tuple):
        if len(value) == 2:
            start, end = value
            return list(range(int(start), int(end) + 1))
        return [int(v) for v in value]
    if isinstance(value, list):
        return [int(v) for v in value]
    return None


def load_spec(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml_from_template(path_template: str) -> Dict[str, Any]:
    resolved = path_template.replace("{PROJECT_ROOT}", str(PROJECT_ROOT))
    matches = sorted(glob.glob(resolved))
    if not matches:
        raise FileNotFoundError(f"config not found: {resolved}")
    with Path(matches[0]).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_resolution(profile_cfg: Dict[str, Any]) -> Optional[Sequence[int]]:
    res_cfg = profile_cfg.get("resolution")
    if isinstance(res_cfg, dict):
        payload = load_yaml_from_template(res_cfg["source"])
        value = payload.get(res_cfg["field"])
        return value
    if isinstance(res_cfg, (list, tuple)):
        return res_cfg
    return None


def resolve_cameras(profile_cfg: Dict[str, Any]) -> List[str]:
    views = profile_cfg.get("views", {})
    if "camera_pattern" in views and "view_index" in views:
        payload = load_yaml_from_template(views["view_index"]["source"])
        ids = payload.get(views["view_index"]["field"])
        if not isinstance(ids, list):
            raise ValueError("view_index.field must be a list of camera ids")
        pattern = views["camera_pattern"]
        return [pattern.format(camera_id=cam_id) for cam_id in ids]
    if "cameras" in views:
        return list(views["cameras"])
    if "camera" in views:
        return [views["camera"]]
    raise ValueError("views section is missing in profile config")


def configure_render_settings(profile_cfg: Dict[str, Any]) -> None:
    engine = profile_cfg.get("engine", "CYCLES")
    bpy.context.scene.render.engine = engine
    if engine == "CYCLES":
        bpy.context.scene.cycles.samples = int(profile_cfg.get("samples", 128))
        # GPU (CUDA/OptiX) を優先して有効化
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dtype in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = dtype
                    prefs.get_devices()
                    gpu_devices = [d for d in prefs.devices if d.type in ("OPTIX", "CUDA")]
                    if gpu_devices:
                        for d in prefs.devices:
                            d.use = True
                        bpy.context.scene.cycles.device = "GPU"
                        log(f"GPU rendering enabled ({dtype}, {len(gpu_devices)} device(s))")
                        break
                except Exception:
                    continue
        except Exception as e:
            log(f"GPU setup skipped, using CPU: {e}")
    bpy.context.scene.render.film_transparent = bool(profile_cfg.get("transparent_background", False))

    resolution = resolve_resolution(profile_cfg)
    if resolution and len(resolution) == 2:
        bpy.context.scene.render.resolution_x = int(resolution[0])
        bpy.context.scene.render.resolution_y = int(resolution[1])


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def read_xyz(row: Dict[str, str], prefix: str) -> Optional[List[float]]:
    try:
        return [
            float(row[f"{prefix}x"]),
            float(row[f"{prefix}y"]),
            float(row[f"{prefix}z"]),
        ]
    except (KeyError, ValueError):
        return None


def find_object(names: Sequence[str]) -> Optional[bpy.types.Object]:
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj:
            return obj
    return None


def apply_armature_from_rows(rows: Sequence[Dict[str, str]], spec: Dict[str, Any], target_frames: Optional[Sequence[int]]) -> None:
    bone_mapping = spec["render"]["armature"]["bone_mapping"]
    armature_name = spec["render"]["armature"]["object_name"]
    amt = bpy.data.objects.get(armature_name)
    if not amt:
        raise ValueError(f"armature not found: {armature_name}")

    bpy.context.view_layer.objects.active = amt
    bpy.ops.object.mode_set(mode="POSE")

    frame_filter = set(target_frames) if target_frames else None
    if not rows:
        log("CSV has no rows; skip armature apply")
        return

    for row in rows:
        if "frame" not in row:
            raise ValueError("CSV must include 'frame' column")
        frame_number = int(float(row["frame"]))
        if frame_filter and frame_number not in frame_filter:
            continue

        bpy.context.scene.frame_set(frame_number)

        for prefix, bone_name in bone_mapping.items():
            xyz = read_xyz(row, prefix)
            if not xyz:
                continue
            if bone_name not in amt.pose.bones:
                continue
            vec = mathutils.Vector(xyz)
            bone = amt.data.bones[bone_name]
            local_loc = bone.matrix_local.inverted() @ vec
            pbone = amt.pose.bones[bone_name]
            pbone.location = local_loc
            pbone.keyframe_insert(data_path="location", frame=frame_number)

        for prefix, cam_key in CAMERA_POSE_OBJECTS.items():
            xyz = read_xyz(row, prefix)
            if not xyz:
                continue
            obj = find_object([cam_key, cam_key.title(), cam_key.replace("_", "").title()])
            if not obj:
                continue
            obj.location = mathutils.Vector(xyz)
            obj.keyframe_insert(data_path="location", frame=frame_number)
            try:
                rx = float(row[f"{prefix}rx"])
                ry = float(row[f"{prefix}ry"])
                rz = float(row[f"{prefix}rz"])
                obj.rotation_euler = (rx, ry, rz)
                obj.keyframe_insert(data_path="rotation_euler", frame=frame_number)
            except (KeyError, ValueError):
                pass


def extract_frames_from_rows(rows: Sequence[Dict[str, str]]) -> List[int]:
    frames: List[int] = []
    for row in rows:
        if "frame" not in row:
            continue
        frames.append(int(float(row["frame"])))
    return frames


def save_blend(blend_path: Path) -> None:
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    blend1 = Path(str(blend_path) + "1")
    if blend1.exists():
        blend1.unlink()


def render_images(image_dir: Path, frames: Sequence[int], cameras: Sequence[str], image_ext: str) -> List[str]:
    rendered_views: List[str] = []
    for cam_name in cameras:
        cam = bpy.data.objects.get(cam_name)
        if not cam:
            log(f"camera not found, skipping: {cam_name}")
            continue
        bpy.context.scene.camera = cam
        rendered_views.append(cam_name)

        for frame_number in frames:
            bpy.context.scene.frame_set(frame_number)
            frame_dir = image_dir / cam_name
            frame_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{frame_number:0{FRAME_DIGITS}d}.{image_ext}"
            out_path = frame_dir / filename
            bpy.context.scene.render.filepath = str(out_path)
            bpy.ops.render.render(write_still=True)
    return rendered_views


def load_manifest_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "runs" in data and isinstance(data["runs"], list):
            return data["runs"]
        return [data]
    return []


def write_manifest(path: Path, entry: Dict[str, Any]) -> None:
    entries = load_manifest_entries(path)
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


class BlenderRenderApp:
    def __init__(self) -> None:
        args = build_parser().parse_args(parse_blender_args())
        self.args = args
        self.spec = load_spec(Path(args.render_spec))
        self.render_hash = render_paths.compute_render_hash(self.spec, PROJECT_ROOT)
        self.paths = render_paths.build_output_paths(Path(args.output_dir).resolve(), self.render_hash)
        render_paths.ensure_dirs(self.paths)

        frames = parse_frames_arg(args.frames)
        self.target_frames = render_paths.normalize_frames(frames) if frames else None

        self.csv_path = self._resolve_csv_path()
        if self.args.all_frames is None:
            self.all_frames = self.csv_path.name == render_paths.ALL_CSV_NAME
        else:
            self.all_frames = bool(self.args.all_frames)
        self.blend_path = render_paths.blend_path(
            self.paths, args.job_id, self.all_frames, self.target_frames
        )

    def _resolve_csv_path(self) -> Path:
        if self.args.csv_path:
            return Path(self.args.csv_path)
        all_csv = render_paths.all_csv_path(self.paths)
        if all_csv.exists():
            return all_csv
        if self.target_frames:
            return render_paths.temp_csv_path(self.paths, self.target_frames)
        raise FileNotFoundError("csv_path is required when no all.csv exists")

    def run(self) -> None:
        status = "success"
        skip_reason = ""
        rendered_views: List[str] = []
        rendered_frames: List[int] = []
        rendered_at = datetime.now(timezone.utc).isoformat()
        try:
            log(f"csv: {self.csv_path}")
            rows = load_csv_rows(self.csv_path)
            if self.all_frames and self.csv_path.exists() and self.blend_path.exists():
                # Reuse cached blend to avoid re-applying all frames each run.
                log(f"reuse blend: {self.blend_path}")
                bpy.ops.wm.open_mainfile(filepath=str(self.blend_path))
            else:
                log("apply armature from csv")
                apply_armature_from_rows(rows, self.spec, self.target_frames)
                save_blend(self.blend_path)

            profile_cfg = self.spec["render"]["rendering"]["profiles"][self.args.profile]
            configure_render_settings(profile_cfg)
            cameras = resolve_cameras(profile_cfg)

            render_frames = self.target_frames or extract_frames_from_rows(rows)
            rendered_frames = render_paths.normalize_frames(render_frames)
            if self.args.render:
                log(f"rendering {len(rendered_frames)} frames across {len(cameras)} cameras")
                rendered_views = render_images(
                    self.paths.image_dir,
                    rendered_frames,
                    cameras,
                    DEFAULT_IMAGE_EXT,
                )
        except Exception as exc:
            status = "failed"
            skip_reason = str(exc)
            log(f"error: {exc}")
            traceback.print_exc()

        base_blend_path = (
            self.spec.get("render", {})
                .get("inputs", {})
                .get("base_blend", {})
                .get("path", "")
        )
        manifest_entry = {
            "session_id": self.args.session_id,
            "render_profile": self.args.profile,
            "frames": rendered_frames,
            "view_ids": list(dict.fromkeys(rendered_views)),
            "kp3d_source_path": self.args.kp3d_pkl,
            "render_hash": self.render_hash,
            "base_blend_name": Path(base_blend_path).name,
            "rendered_at": rendered_at,
            "status": status,
            "skip_reason": skip_reason,
        }
        write_manifest(self.paths.manifest_path, manifest_entry)


if __name__ == "__main__":
    BlenderRenderApp().run()

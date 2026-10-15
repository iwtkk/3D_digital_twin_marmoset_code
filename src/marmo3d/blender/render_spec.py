from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import json

try:
    import yaml  # PyYAML
except ImportError as e:
    raise ImportError("PyYAML is required: pip install pyyaml") from e


# -------------------------
# Utilities
# -------------------------

def deep_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def render_template(s: str, ctx: Dict[str, Any]) -> str:
    # Resolve {PROJECT_ROOT} etc. in the spec via format
    return s.format(**ctx)

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def glob_one(root: Union[str, Path], pattern: str) -> Path:
    root = Path(root)
    hits = sorted(root.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"glob not found: {root}/{pattern}")
    if len(hits) > 1:
        # The spec may allow priority_first etc., but this is a minimal implementation
        return hits[0]
    return hits[0]


# -------------------------
# Spec dataclasses
# -------------------------

ProfileName = Literal["env_camera", "marmo_eyes", "marmo_an_eye"]
EngineName = Literal["CYCLES", "EEVEE"]
OutputKind = Literal["image", "video"]

@dataclass(frozen=True)
class BaseBlendInput:
    path: str
    require_blender_version: bool = False
    blender_version: str = ""

@dataclass(frozen=True)
class Temp3DInput:
    format: Literal["npz", "json", "pickle"]
    path_template: str
    schema_ref: str

@dataclass(frozen=True)
class RenderInputs:
    base_blend: BaseBlendInput
    temp3d: Temp3DInput

@dataclass(frozen=True)
class ArmatureUnits:
    length_unit: str
    blender_unit: str
    scale_factor: float

@dataclass(frozen=True)
class ArmatureSpec:
    object_name: str
    bone_mapping: Dict[str, str]
    units: ArmatureUnits

@dataclass(frozen=True)
class ViewInfo:
    # Normalized view representation used at runtime
    profile: ProfileName
    view_index: int                 # 0..N-1 (for {view_index:02d} in image output)
    view_id: str                    # for {view_id} in video output (e.g. "Camera.0" or "camera_l")
    camera_object_name: str         # camera object name in Blender
    camid: Optional[str] = None     # set to the config camera_id only for env_camera

@dataclass(frozen=True)
class RenderingProfileEnvCamera:
    engine: EngineName
    samples: int
    transparent_background: bool
    resolution_source: str          # template path to config.yaml etc.
    resolution_field: str           # e.g. "img_size"
    camera_pattern: str             # e.g. "Camera.{view_index}"

@dataclass(frozen=True)
class RenderingProfileMarmoEyes:
    engine: EngineName
    resolution: Tuple[int, int]
    samples: int
    transparent_background: bool
    cameras: List[str]              # ["camera_l", "camera_r"]

@dataclass(frozen=True)
class RenderingProfileMarmoAnEye:
    engine: EngineName
    resolution: Tuple[int, int]
    samples: int
    transparent_background: bool
    camera: str                     # e.g. "camera_l"

@dataclass(frozen=True)
class OutputVideoSpec:
    video_format: str
    path_template: str
    overwrite: bool

@dataclass(frozen=True)
class OutputImageSpec:
    image_format: str
    path_template: str
    overwrite: bool

@dataclass(frozen=True)
class OutputSpec:
    video: OutputVideoSpec
    image: OutputImageSpec

@dataclass(frozen=True)
class PipelineSpec:
    steps: List[str]

@dataclass(frozen=True)
class ManifestsSpec:
    enabled: bool
    format: str
    path_template: str
    fields: List[str]

@dataclass(frozen=True)
class CsvSpec:
    path_template: str
    init: Dict[str, Any]

@dataclass(frozen=True)
class RenderSpec:
    version: int
    description: str
    inputs: RenderInputs
    csv: CsvSpec
    armature: ArmatureSpec
    profile: ProfileName
    profiles: Dict[ProfileName, Any]  # each Profile dataclass above
    output: OutputSpec
    pipeline: PipelineSpec
    manifests: ManifestsSpec

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RenderSpec":
        raw = load_yaml(path)
        r = raw["render"]

        inputs = RenderInputs(
            base_blend=BaseBlendInput(**r["inputs"]["base_blend"]),
            temp3d=Temp3DInput(**r["inputs"]["temp3d"]),
        )

        units = ArmatureUnits(**r["armature"]["units"])
        armature = ArmatureSpec(
            object_name=r["armature"]["object_name"],
            bone_mapping=dict(r["armature"]["bone_mapping"]),
            units=units,
        )

        rendering = r["rendering"]
        profile: ProfileName = rendering["profile"]

        profiles: Dict[ProfileName, Any] = {}
        prof_raw = rendering["profiles"]

        # env_camera
        env = prof_raw["env_camera"]
        profiles["env_camera"] = RenderingProfileEnvCamera(
            engine=env["engine"],
            samples=env["samples"],
            transparent_background=env["transparent_background"],
            resolution_source=env["resolution"]["source"],
            resolution_field=env["resolution"]["field"],
            camera_pattern=env["views"]["camera_pattern"],
        )
        # marmo_eyes
        me = prof_raw["marmo_eyes"]
        profiles["marmo_eyes"] = RenderingProfileMarmoEyes(
            engine=me["engine"],
            resolution=tuple(me["resolution"]),
            samples=me["samples"],
            transparent_background=me["transparent_background"],
            cameras=list(me["views"]["cameras"]),
        )
        # marmo_an_eye
        mae = prof_raw["marmo_an_eye"]
        profiles["marmo_an_eye"] = RenderingProfileMarmoAnEye(
            engine=mae["engine"],
            resolution=tuple(mae["resolution"]),
            samples=mae["samples"],
            transparent_background=mae["transparent_background"],
            camera=mae["views"]["camera"],
        )

        out = OutputSpec(
            video=OutputVideoSpec(**r["rendering"]["output"]["video"]),
            image=OutputImageSpec(**r["rendering"]["output"]["image"]),
        )

        pipeline = PipelineSpec(steps=list(r["pipeline"]["steps"]))

        manifests = ManifestsSpec(**r["manifests"])
        
        csv_spec = CsvSpec(
            path_template=r["csv"]["path_template"],
            init=r["csv"]["init"]
        )

        return cls(
            version=int(raw.get("version", 1)),
            description=r.get("description", ""),
            inputs=inputs,
            csv=csv_spec,
            armature=armature,
            profile=profile,
            profiles=profiles,
            output=out,
            pipeline=pipeline,
            manifests=manifests,
        )

    def resolve_base_blend(self, project_root: Union[str, Path]) -> Path:
        ctx = {"PROJECT_ROOT": str(project_root)}
        return Path(render_template(self.inputs.base_blend.path, ctx))

    def temp3d_path(self, project_root: Union[str, Path], session_id: str, frame: int, override_source_path: Optional[str] = None) -> Path:
        if override_source_path:
             # If an override is given, trust it (format assumed to be pickle etc.)
             # User request: "specify the pickle file among the processed outputs... in that case pass the session ID and the processed file path"
             # If a path is given, return it as is (converted to Path)
             return Path(override_source_path)

        ctx = {"PROJECT_ROOT": str(project_root), "session_id": session_id, "frame": frame}
        return Path(render_template(self.inputs.temp3d.path_template, ctx))

    def manifest_path(self, project_root: Union[str, Path], animal: str, session_id: str) -> Path:
        ctx = {"PROJECT_ROOT": str(project_root), "animal": animal, "session_id": session_id}
        return Path(render_template(self.manifests.path_template, ctx))

    def csv_path(self, project_root: Union[str, Path], session_id: str, frame: int) -> Path:
        ctx = {"PROJECT_ROOT": str(project_root), "session_id": session_id, "frame": frame}
        return Path(render_template(self.csv.path_template, ctx))

    # --- config lookup (required for env_camera) ---

    def _load_first_marmo_config(self, project_root: Union[str, Path]) -> Dict[str, Any]:
        # render_spec.yaml references "{PROJECT_ROOT}/config/marmo*/config.yaml" :contentReference[oaicite:1]{index=1}
        ctx = {"PROJECT_ROOT": str(project_root)}
        pattern = render_template("{PROJECT_ROOT}/config/marmo*/config.yaml", ctx)
        # pattern may be relative to project_root or absolute; split here for Path.glob
        # e.g. "/.../proj/config/marmo*/config.yaml"
        p = Path(pattern)
        # Walk up parents and glob
        return load_yaml(glob_one(p.parent.parent, f"{p.parent.name}/{p.name}"))

    def get_views(self, project_root: Union[str, Path], profile: Optional[ProfileName] = None) -> List[ViewInfo]:
        prof: ProfileName = profile or self.profile
        if prof == "env_camera":
            cfg = self._load_first_marmo_config(project_root)
            camids: List[str] = list(cfg.get("camera_id", []))
            camera_pattern = self.profiles["env_camera"].camera_pattern
            views: List[ViewInfo] = []
            for i, camid in enumerate(camids):
                cam_obj = camera_pattern.format(view_index=i, camera_id=camid)
                views.append(ViewInfo(profile=prof, view_index=i, view_id=cam_obj, camera_object_name=cam_obj, camid=str(camid)))
            return views

        if prof == "marmo_eyes":
            cams = self.profiles["marmo_eyes"].cameras
            return [
                ViewInfo(profile=prof, view_index=i, view_id=cam, camera_object_name=cam)
                for i, cam in enumerate(cams)
            ]

        if prof == "marmo_an_eye":
            cam = self.profiles["marmo_an_eye"].camera
            return [ViewInfo(profile=prof, view_index=0, view_id=cam, camera_object_name=cam)]

        raise ValueError(f"unknown profile: {prof}")

    def get_resolution(self, project_root: Union[str, Path], profile: Optional[ProfileName] = None) -> Tuple[int, int]:
        prof: ProfileName = profile or self.profile
        if prof == "env_camera":
            cfg = self._load_first_marmo_config(project_root)
            field = self.profiles["env_camera"].resolution_field
            wh = cfg.get(field)
            if not (isinstance(wh, list) and len(wh) == 2):
                raise ValueError(f"config field '{field}' must be [W,H], got: {wh}")
            return int(wh[0]), int(wh[1])
        if prof == "marmo_eyes":
            return tuple(self.profiles["marmo_eyes"].resolution)  # type: ignore[return-value]
        if prof == "marmo_an_eye":
            return tuple(self.profiles["marmo_an_eye"].resolution)  # type: ignore[return-value]
        raise ValueError(f"unknown profile: {prof}")

    # --- output path generation ---

    def image_paths(
        self,
        project_root: Union[str, Path],
        animal: str,
        session_id: str,
        frames: Iterable[int],
        profile: Optional[ProfileName] = None,
    ) -> Dict[ViewInfo, List[Path]]:
        views = self.get_views(project_root, profile)
        tmpl = self.output.image.path_template
        out: Dict[ViewInfo, List[Path]] = {}
        for v in views:
            out[v] = []
            for frame in frames:
                ctx = {
                    "PROJECT_ROOT": str(project_root),
                    "animal": animal,
                    "session_id": session_id,
                    "frame": int(frame),
                    "view_index": int(v.view_index),
                    "view_id": v.view_id,
                }
                out[v].append(Path(render_template(tmpl, ctx)))
        return out

    def video_paths(
        self,
        project_root: Union[str, Path],
        animal: str,
        session_id: str,
        start_frame: int,
        end_frame: int,
        profile: Optional[ProfileName] = None,
    ) -> Dict[ViewInfo, Path]:
        views = self.get_views(project_root, profile)
        tmpl = self.output.video.path_template
        out: Dict[ViewInfo, Path] = {}
        for v in views:
            ctx = {
                "PROJECT_ROOT": str(project_root),
                "animal": animal,
                "session_id": session_id,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "view_index": int(v.view_index),
                "view_id": v.view_id,
            }
            out[v] = Path(render_template(tmpl, ctx))
        return out


# -------------------------
# Usage example
# -------------------------

if __name__ == "__main__":
    # Pass your project root as project_root
    project_root = Path(".").resolve()

    spec = RenderSpec.from_yaml("render_spec.yaml")

    print("base.blend:", spec.resolve_base_blend(project_root))
    print(type(spec.resolve_base_blend(project_root)))
    print("resolution:", spec.get_resolution(project_root))
    views = spec.get_views(project_root)
    print("views:", [v.view_id for v in views])

    animal = "cj611"
    session_id = "dailylife_cj611_20250101_120000"
    frames = [0, 123, 456]

    img_paths = spec.image_paths(project_root, animal, session_id, frames)
    for v, paths in img_paths.items():
        print(v.view_id, "->", paths[0])

    vid_paths = spec.video_paths(project_root, animal, session_id, start_frame=0, end_frame=1000)
    for v, p in vid_paths.items():
        print("video", v.view_id, "->", p)

    

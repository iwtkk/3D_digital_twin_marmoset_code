from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any
import concurrent.futures
import json
import math
import pickle
import subprocess

from src.tracer import (
    SequenceTracer,
    set_trace_enabled,
    reset_trace,
    trace_events,
    trace_to_plantuml,
    trace_timeline,
    set_trace_summaries,
    trace_scope,
)

FrameSpec = Union[None, int, Sequence[int], Tuple[int, int]]
OverridesSpec = Union[bool, Dict[str, object]]

TRACE_SUMMARIES = {
    "CheckRequest.validate": "Validate input parameters",
    "_is_int": "Check whether a value is an integer",
    "_extract_profiles": "Extract the list of profiles from render_spec",
    "_normalize_sessions": "Normalize sessions into a list of Paths",
    "_validate_frame_spec": "Validate the frames specification format",
    "_find_kp3d_pickle": "Search for the kp3d pickle file",
    "_infer_frame_count": "Infer the frame count from kp3d data",
    "_normalize_frame_list": "Normalize the frame number list",
    "split_frames": "Split the frame list into the given number of splits",
    "resolve_frames": "Resolve the frames spec into a frame list",
    "_frames_from_kp3d": "Build the full frame list from kp3d",
    "should_skip_output": "Decide skipping from override and existing outputs",
    "_overrides_flag": "Determine whether overrides are enabled",
    "RenderPlan.from_request": "Create a RenderPlan from a request",
    "SaveJobs.save": "Save the RenderPlan",
    "SaveJobs.load": "Load the RenderPlan",
    "RenderJob.make_commands": "Build the Blender command",
    "Executor._run_command": "Run a command, handling retries/timeouts",
    "Executor.execute": "Run commands sequentially or in parallel",
}

set_trace_summaries(TRACE_SUMMARIES, clear=True)

# -------------------------
# (Optional) Manifest helper
# -------------------------

@dataclass
class RenderManifest:
    animal: str
    session_id: str
    base_blend_path: str
    render_profile: str
    view_ids: List[str]
    frames: Union[List[int], Dict[str, int]]  # image: [..] / video: {"start":..,"end":..}
    kp3d_source_path: Optional[str] = None
    status: str = "success"  # success / skipped / failed
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "animal": self.animal,
            "session_id": self.session_id,
            "frames": self.frames,
            "base_blend_path": self.base_blend_path,
            "render_profile": self.render_profile,
            "view_ids": self.view_ids,
            "kp3d_source_path": self.kp3d_source_path,
            "status": self.status,
            "skip_reason": self.skip_reason,
        }

    def save_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

@dataclass(frozen=True)
class RenderRequest:
    profile: str
    sessions: Sequence[Path]
    frames: FrameSpec
    split: int
    overrides: OverridesSpec
    workers: int
    dry_run: bool

class CheckRequest:
    def validate(self, req: RenderRequest, render_spec: dict) -> None:
        with trace_scope("CheckRequest.validate"):
            profiles = _extract_profiles(render_spec)
            if req.profile not in profiles:
                raise ValueError(f"profile not found: {req.profile}")
            if not req.sessions:
                raise ValueError("sessions must not be empty")
            for session in _normalize_sessions(req.sessions):
                if not session.exists():
                    raise FileNotFoundError(f"session not found: {session}")
            if not _is_int(req.split) or req.split < 1:
                raise ValueError("split must be >= 1")
            if not _is_int(req.workers) or req.workers < 1:
                raise ValueError("workers must be >= 1")
            _validate_frame_spec(req.frames)

@dataclass(frozen=True)
class RenderPlanItem:
    session: Path
    frames: List[int]
    skip: bool
    output_dir: Optional[Path] = None

def _is_int(value: object) -> bool:
    with trace_scope("_is_int"):
        return isinstance(value, int) and not isinstance(value, bool)

def _extract_profiles(render_spec: object) -> Dict[str, Any]:
    with trace_scope("_extract_profiles"):
        if hasattr(render_spec, "profiles"):
            return dict(getattr(render_spec, "profiles"))
        if isinstance(render_spec, dict):
            for keys in (("render", "rendering", "profiles"), ("rendering", "profiles"), ("profiles",)):
                cur: Any = render_spec
                ok = True
                for key in keys:
                    if not isinstance(cur, dict) or key not in cur:
                        ok = False
                        break
                    cur = cur[key]
                if ok and isinstance(cur, dict):
                    return cur
        raise ValueError("render_spec must define profiles")

def _normalize_sessions(sessions: Sequence[Path]) -> List[Path]:
    with trace_scope("_normalize_sessions"):
        if isinstance(sessions, (str, Path)):
            return [Path(sessions)]
        return [Path(s) for s in sessions]

def _validate_frame_spec(frames: FrameSpec) -> None:
    with trace_scope("_validate_frame_spec"):
        if frames is None:
            return
        if _is_int(frames):
            if frames < 0:
                raise ValueError("frames must be >= 0")
            return
        if isinstance(frames, tuple):
            if len(frames) != 2:
                raise ValueError("tuple frames must be (start, end)")
            start, end = frames
            if not _is_int(start) or not _is_int(end):
                raise ValueError("tuple frames must be ints")
            if start < 0 or end < 0 or start > end:
                raise ValueError("invalid tuple frame range")
            return
        if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
            for value in frames:
                if not _is_int(value) or value < 0:
                    raise ValueError("frames must be ints >= 0")
            return
        raise TypeError("frames must be None/int/sequence/tuple")

def _find_kp3d_pickle(session: Path) -> Path:
    with trace_scope("_find_kp3d_pickle"):
        names = ("kp3d_fxdJointLen.pickle", "kp3d.pickle")
        for name in names:
            candidate = session / name
            if candidate.is_file():
                return candidate
        for name in names:
            matches = sorted(session.rglob(name))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"kp3d pickle not found under: {session}")

def _infer_frame_count(payload: object) -> int:
    with trace_scope("_infer_frame_count"):
        if hasattr(payload, "shape"):
            shape = getattr(payload, "shape")
            if shape and shape[0] is not None:
                return int(shape[0])
        if isinstance(payload, dict):
            for key in ("frames", "kp3d", "pose", "data"):
                if key in payload:
                    return _infer_frame_count(payload[key])
            if payload:
                return _infer_frame_count(next(iter(payload.values())))
            raise ValueError("empty dict payload for kp3d")
        if hasattr(payload, "__len__"):
            return int(len(payload))  # type: ignore[arg-type]
        raise ValueError("unsupported kp3d payload type")

def _normalize_frame_list(values: Sequence[int]) -> List[int]:
    with trace_scope("_normalize_frame_list"):
        normalized: List[int] = []
        for value in values:
            if not _is_int(value) or value < 0:
                raise ValueError("frames must be ints >= 0")
            normalized.append(int(value))
        return sorted(set(normalized))

def split_frames(frames: List[int], split: int) -> List[List[int]]:
    with trace_scope("split_frames"):
        if split <= 0:
            raise ValueError("split must be >= 1")
        if not frames:
            return []
        size = math.ceil(len(frames) / split)
        return [frames[i:i+size] for i in range(0, len(frames), size)]

def resolve_frames(session: Path, frames: FrameSpec) -> List[int]:
    with trace_scope("resolve_frames"):
        if frames is None:
            return _frames_from_kp3d(session)
        if _is_int(frames):
            if frames < 0:
                raise ValueError("frame must be >= 0")
            return [int(frames)]
        if isinstance(frames, tuple):
            if len(frames) == 0:
                return _frames_from_kp3d(session)
            if len(frames) == 2:
                start, end = frames
                if not _is_int(start) or not _is_int(end):
                    raise ValueError("tuple frames must be ints")
                if start < 0 or end < 0 or start > end:
                    raise ValueError("invalid tuple frame range")
                return list(range(int(start), int(end) + 1))
            return _normalize_frame_list(frames)
        if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
            if len(frames) == 0:
                return _frames_from_kp3d(session)
            return _normalize_frame_list(frames)
        raise TypeError("frames must be None/int/sequence/tuple")

def _frames_from_kp3d(session: Path) -> List[int]:
    with trace_scope("_frames_from_kp3d"):
        kp3d_path = _find_kp3d_pickle(session)
        with open(kp3d_path, "rb") as f:
            payload = pickle.load(f)
        count = _infer_frame_count(payload)
        if count < 0:
            raise ValueError("frame count must be >= 0")
        return list(range(count))

def should_skip_output(overrides: bool, output_dir: Path) -> bool:
    with trace_scope("should_skip_output"):
        if overrides:
            return False
        return output_dir.exists()

def _overrides_flag(overrides: OverridesSpec) -> bool:
    with trace_scope("_overrides_flag"):
        if isinstance(overrides, bool):
            return overrides
        return True if overrides else False


@dataclass(frozen=True)
class RenderPlan:
    items: List[RenderPlanItem]

    @classmethod
    def from_request(
        cls,
        req: RenderRequest,
        output_dir_resolver: Optional[Callable[[Path, List[int]], Path]] = None,
    ) -> "RenderPlan":
        with trace_scope("RenderPlan.from_request"):
            items: List[RenderPlanItem] = []
            overrides_flag = _overrides_flag(req.overrides)
            for session in _normalize_sessions(req.sessions):
                frames = resolve_frames(session, req.frames)
                chunks = split_frames(frames, req.split)
                for chunk in chunks:
                    output_dir = output_dir_resolver(session, chunk) if output_dir_resolver else None
                    skip = should_skip_output(overrides_flag, output_dir) if output_dir else False
                    items.append(RenderPlanItem(session=session, frames=chunk, skip=skip, output_dir=output_dir))
            return cls(items=items)

class SaveJobs:
    def save(self, plan: RenderPlan, path: Path) -> None:
        with trace_scope("SaveJobs.save"):
            payload = {
                "items": [
                    {
                        "session": str(item.session),
                        "frames": item.frames,
                        "skip": item.skip,
                        "output_dir": str(item.output_dir) if item.output_dir else None,
                    }
                    for item in plan.items
                ]
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    def load(self, path: Path) -> RenderPlan:
        with trace_scope("SaveJobs.load"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = []
            for raw in payload.get("items", []):
                items.append(
                    RenderPlanItem(
                        session=Path(raw["session"]),
                        frames=list(raw.get("frames", [])),
                        skip=bool(raw.get("skip", False)),
                        output_dir=Path(raw["output_dir"]) if raw.get("output_dir") else None,
                    )
                )
            return RenderPlan(items=items)

class RenderJob:
    def __init__(
        self,
        blender_bin: str = "blender",
        base_blend: Optional[Path] = None,
        render_spec_path: Optional[Path] = None,
        script_path: Optional[Path] = None,
        output_root: Optional[Path] = None,
        profile: Optional[str] = None,
    ) -> None:
        self.blender_bin = blender_bin
        self.base_blend = Path(base_blend) if base_blend else None
        self.render_spec_path = Path(render_spec_path) if render_spec_path else None
        self.script_path = Path(script_path) if script_path else Path("render_blender.py")
        self.output_root = Path(output_root) if output_root else None
        self.profile = profile

    def make_commands(
        self,
        plan: RenderPlan,
        extra_args_resolver: Optional[Callable[[RenderPlanItem], Sequence[str]]] = None,
    ) -> List[List[str]]:
        # Build the blender command list
        with trace_scope("RenderJob.make_commands"):
            if not self.base_blend:
                raise ValueError("base_blend is required to build commands")
            if not self.render_spec_path:
                raise ValueError("render_spec_path is required to build commands")
            if not self.profile:
                raise ValueError("profile is required to build commands")
            commands: List[List[str]] = []
            for item in plan.items:
                if item.skip or not item.frames:
                    continue
                output_dir = item.output_dir
                if output_dir is None:
                    if not self.output_root:
                        raise ValueError("output_dir is required to build commands")
                    output_dir = self.output_root / item.session.name
                output_dir.mkdir(parents=True, exist_ok=True)
                kp3d_path = _find_kp3d_pickle(item.session)
                frames_arg = repr(item.frames)
                job_id = f"{item.session.name}_{self.profile}"
                cmd = [
                    self.blender_bin,
                    "-b",
                    str(self.base_blend),
                    "--python",
                    str(self.script_path),
                    "--",
                    "--session_id",
                    item.session.name,
                    "--kp3d_pkl",
                    str(kp3d_path),
                    "--render_spec",
                    str(self.render_spec_path),
                    "--profile",
                    self.profile,
                    "--frames",
                    frames_arg,
                    "--output_dir",
                    str(output_dir),
                    "--job_id",
                    job_id,
                    "--render",
                    "1",
                ]
                if extra_args_resolver:
                    cmd.extend(list(extra_args_resolver(item)))
                commands.append(cmd)
            return commands

class Executor:
    def __init__(self, workers: int = 1, retries: int = 0, timeout: Optional[float] = None) -> None:
        self.workers = workers
        self.retries = retries
        self.timeout = timeout

    def _run_command(self, command: List[str]) -> None:
        with trace_scope("Executor._run_command"):
            attempts = self.retries + 1
            last_exc: Optional[BaseException] = None
            for _ in range(attempts):
                try:
                    subprocess.run(command, check=True, timeout=self.timeout)
                    return
                except subprocess.TimeoutExpired as exc:
                    last_exc = exc
                except subprocess.CalledProcessError as exc:
                    last_exc = exc
            if last_exc:
                raise last_exc

    def execute(self, commands: List[List[str]], dry_run: bool) -> None:
        with trace_scope("Executor.execute"):
            if dry_run:
                return
            if not commands:
                return
            if self.workers <= 1:
                for command in commands:
                    self._run_command(command)
                return
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(self._run_command, command) for command in commands]
                for future in futures:
                    future.result()

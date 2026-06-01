from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Output layout constants.
ROOT_DIR_NAMES = ("render", "session")
IMAGE_DIR_NAME = "image"
CSV_DIR_NAME = "csv"
BLEND_DIR_NAME = "blend"
ALL_CSV_NAME = "all.csv"
TEMP_CSV_NAME = "temp.csv"


def compute_render_hash(spec: Dict[str, Any], project_root: Optional[Path] = None) -> str:
    """Hash render/csv/init (+ .blend content if project_root given) for reproducibility."""
    render_cfg = spec.get("render", {})
    csv_cfg = render_cfg.get("csv", {})
    init_cfg = csv_cfg.get("init", {})
    payload = {"render": render_cfg, "csv": csv_cfg, "init": init_cfg}
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    spec_hash = hashlib.sha256(raw).hexdigest()

    if project_root is not None:
        blend_path_str = (
            spec.get("render", {})
                .get("inputs", {})
                .get("base_blend", {})
                .get("path", "")
        )
        resolved = blend_path_str.replace("{PROJECT_ROOT}", str(project_root))
        blend_file = Path(resolved)
        if blend_file.is_file():
            blend_hash = hashlib.sha256(blend_file.read_bytes()).hexdigest()
            combined = (spec_hash + blend_hash).encode("utf-8")
            return hashlib.sha256(combined).hexdigest()[:8]

    return spec_hash[:8]


def normalize_frames(frames: Iterable[int]) -> List[int]:
    """Deterministic frame list for signatures/manifests."""
    unique = sorted({int(f) for f in frames})
    return unique


def frames_signature(frames: Iterable[int]) -> str:
    """Stable signature for a frame set to avoid temp.csv collisions."""
    normalized = normalize_frames(frames)
    raw = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


@dataclass(frozen=True)
class RenderOutputPaths:
    root: Path
    image_dir: Path
    csv_dir: Path
    blend_dir: Path
    manifest_path: Path


def build_output_paths(output_dir: Path, render_hash: str) -> RenderOutputPaths:
    root = Path(output_dir)
    root = root / render_hash
    image_dir = root / IMAGE_DIR_NAME
    csv_dir = root / CSV_DIR_NAME
    blend_dir = root / BLEND_DIR_NAME
    # Manifest lives with rendered images per render_temp.md.
    manifest_path = image_dir / "manifest.json"
    return RenderOutputPaths(
        root=root,
        image_dir=image_dir,
        csv_dir=csv_dir,
        blend_dir=blend_dir,
        manifest_path=manifest_path,
    )


def ensure_dirs(paths: RenderOutputPaths) -> None:
    paths.image_dir.mkdir(parents=True, exist_ok=True)
    paths.csv_dir.mkdir(parents=True, exist_ok=True)
    paths.blend_dir.mkdir(parents=True, exist_ok=True)


def all_csv_path(paths: RenderOutputPaths) -> Path:
    return paths.csv_dir / ALL_CSV_NAME


def temp_csv_path(paths: RenderOutputPaths, frames: Iterable[int]) -> Path:
    frame_dir = paths.csv_dir / frames_signature(frames)
    return frame_dir / TEMP_CSV_NAME


def blend_path(paths: RenderOutputPaths, job_id: str, all_frames: bool, frames: Optional[Iterable[int]] = None) -> Path:
    suffix = "all" if all_frames else "frame"
    if all_frames or frames is None:
        return paths.blend_dir / f"{job_id}_{suffix}.blend"
    frame_dir = paths.blend_dir / frames_signature(frames)
    return frame_dir / f"{job_id}_{suffix}.blend"

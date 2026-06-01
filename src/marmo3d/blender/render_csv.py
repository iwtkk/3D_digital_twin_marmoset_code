from __future__ import annotations

import argparse
import ast
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
import render_paths

# Frame numbering offset used for kp3d indexing.
FRAME_INDEX_BASE = 0

KP_NAMES = [
    "nose",
    "L eye",
    "R eye",
    "L ear",
    "R ear",
    "L shoulder",
    "R shoulder",
    "L elbow",
    "R elbow",
    "L wrist",
    "R wrist",
    "L hip",
    "R hip",
    "L knee",
    "R knee",
    "L ankle",
    "R ankle",
    "backbone peak",
    "tail_1",
    "tail_2",
]

# Legacy alignment from process_dlc.py.
ROTATION_DEGREES = (-90.0, 0.0, 180.0)
ROTATION_ORDER = "xyz"


def log(message: str) -> None:
    print(f"[render_csv] {message}")


def safe_eval_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"numeric expression must be str/int/float, got {type(value)}")
    expr = value.strip()
    if not expr:
        raise ValueError("numeric expression is empty")
    node = ast.parse(expr, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub)
    for sub in ast.walk(node):
        if not isinstance(sub, allowed):
            raise ValueError(f"unsupported expression: {expr}")
        if isinstance(sub, ast.Constant) and not isinstance(sub.value, (int, float)):
            raise ValueError(f"unsupported constant in expression: {expr}")
    return float(eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}))


def sanitize(name: str) -> str:
    import re

    key = name.lower().strip().replace(" ", "_")
    key = re.sub(r"[^a-z0-9_]", "", key)
    return key


def unique_keys(names: Sequence[str]) -> List[str]:
    seen: Dict[str, bool] = {}
    out: List[str] = []
    for name in names:
        base = sanitize(name)
        key = base
        idx = 2
        while key in seen:
            key = f"{base}_{idx}"
            idx += 1
        seen[key] = True
        out.append(key)
    return out


def rot_euler(rx: float, ry: float, rz: float, order: str = "xyz") -> np.ndarray:
    def rx_mat(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def ry_mat(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rz_mat(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    mats = {"x": rx_mat, "y": ry_mat, "z": rz_mat}
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    vals = {"x": rx, "y": ry, "z": rz}

    r = np.eye(3)
    for axis in order:
        r = mats[axis](vals[axis]) @ r
    return r


def safe_norm(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def rotmat_to_euler_xyz(r: np.ndarray) -> np.ndarray:
    r00 = r[..., 0, 0]
    r01 = r[..., 0, 1]
    r02 = r[..., 0, 2]
    r10 = r[..., 1, 0]
    r11 = r[..., 1, 1]
    r12 = r[..., 1, 2]
    r20 = r[..., 2, 0]
    r21 = r[..., 2, 1]
    r22 = r[..., 2, 2]
    sy = np.sqrt(r00 * r00 + r10 * r10)
    singular = sy < 1e-8
    rx = np.where(~singular, np.arctan2(r21, r22), np.arctan2(-r12, r11))
    ry = np.where(~singular, np.arctan2(-r20, sy), np.arctan2(-r20, sy))
    rz = np.where(~singular, np.arctan2(r10, r00), np.zeros_like(r00))
    return np.stack([rx, ry, rz], axis=-1)


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


def dedupe_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    out = []
    for v in values:
        iv = int(v)
        if iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out


def build_all_frames(frame_count: int) -> List[int]:
    return [FRAME_INDEX_BASE + i for i in range(frame_count)]


def select_frames(frames: Optional[Sequence[int]], frame_count: int) -> List[int]:
    if frames is None or len(frames) == 0:
        return build_all_frames(frame_count)
    return dedupe_preserve_order(frames)


def validate_frames(frames: Sequence[int], frame_count: int) -> None:
    for frame_number in frames:
        idx = frame_number - FRAME_INDEX_BASE
        if idx < 0 or idx >= frame_count:
            raise ValueError(f"frame {frame_number} out of range ({FRAME_INDEX_BASE}-{frame_count - 1 + FRAME_INDEX_BASE})")


def is_all_frames(frames: Sequence[int], frame_count: int) -> bool:
    return set(frames) == set(build_all_frames(frame_count))


def load_spec(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_kp3d(payload: Any) -> np.ndarray:
    if isinstance(payload, dict) and "kp3d" in payload:
        kp3d = payload["kp3d"]
    else:
        kp3d = payload
    if not hasattr(kp3d, "shape"):
        raise ValueError("kp3d payload must be an array-like")
    if kp3d.ndim == 4:
        return kp3d[0]
    if kp3d.ndim == 3:
        return kp3d
    raise ValueError(f"unsupported kp3d shape: {kp3d.shape}")


class CsvConverter:
    def __init__(self, spec: Dict[str, Any]) -> None:
        units = spec.get("render", {}).get("armature", {}).get("units", {})
        self.scale_factor = safe_eval_number(units.get("scale_factor", 1.0))
        init_conf = spec.get("render", {}).get("csv", {}).get("init", {})
        raw_nt = init_conf.get("neck_tail", [0.0, 0.0, 0.0])
        self.blender_neck_tail = [safe_eval_number(v) for v in raw_nt]

    def iter_rows(self, kp3d: np.ndarray, frames: Sequence[int]) -> Iterable[Dict[str, float]]:
        rotation = rot_euler(*ROTATION_DEGREES, order=ROTATION_ORDER)
        pivot = np.zeros(3)
        bld_neck = np.array(self.blender_neck_tail)

        for frame_number in frames:
            idx = frame_number - FRAME_INDEX_BASE
            x = np.array(kp3d[idx], dtype=float)
            kp_names = list(KP_NAMES)

            # Indices mirror KP_NAMES order; update here if KP_NAMES changes.
            if len(x) > 6:
                mid = (x[5] + x[6]) / 2
                x = np.vstack([x, mid])
                kp_names.append("shoulder_mid")

            if len(x) > 12:
                mid = (x[11] + x[12]) / 2
                x = np.vstack([x, mid])
                kp_names.append("base")

            x = (x - pivot) @ rotation.T + pivot
            x = x * self.scale_factor
            x = x + bld_neck

            kp_keys = unique_keys(kp_names)
            row: Dict[str, float] = {"frame": frame_number}

            for key, pt in zip(kp_keys, x):
                row[f"pts_{key}_x"] = float(pt[0])
                row[f"pts_{key}_y"] = float(pt[1])
                row[f"pts_{key}_z"] = float(pt[2])

            row.update(self._camera_pose_fields(kp_keys, x))
            yield row

    def _camera_pose_fields(self, kp_keys: Sequence[str], points: np.ndarray) -> Dict[str, float]:
        def get_vec(k: str) -> np.ndarray:
            idx = kp_keys.index(k)
            return points[idx]

        try:
            le = get_vec("l_eye")
            re = get_vec("r_eye")
            ler = get_vec("l_ear")
            rer = get_vec("r_ear")
        except ValueError:
            return {}

        eye_c = (le + re) * 0.5
        ear_c = (ler + rer) * 0.5

        forward_sign = -1.0
        s_raw = re - le
        f_raw = (eye_c - ear_c) * forward_sign

        s_n = safe_norm(s_raw)
        f_n = safe_norm(f_raw)
        u_raw = np.cross(f_n, s_n)
        u_n = safe_norm(u_raw)
        s_ortho = safe_norm(np.cross(u_n, f_n))

        r_cam = np.stack([s_ortho, u_n, f_n], axis=-1)
        det = np.linalg.det(r_cam)
        if det < 0:
            r_cam[:, 0] *= -1.0

        eul = rotmat_to_euler_xyz(r_cam[np.newaxis, ...])[0]

        out: Dict[str, float] = {}
        for prefix, pos in (("pts_l_cam_", le), ("pts_r_cam_", re)):
            out[f"{prefix}x"] = float(pos[0])
            out[f"{prefix}y"] = float(pos[1])
            out[f"{prefix}z"] = float(pos[2])
            out[f"{prefix}rx"] = float(eul[0])
            out[f"{prefix}ry"] = float(eul[1])
            out[f"{prefix}rz"] = float(eul[2])
        return out

    def write_csv(self, kp3d: np.ndarray, frames: Sequence[int], output_csv_path: Path) -> None:
        rows_iter = iter(self.iter_rows(kp3d, frames))
        first_row = next(rows_iter, None)
        if first_row is None:
            log("no rows generated; skip writing")
            return
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with output_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(first_row.keys()))
            writer.writeheader()
            writer.writerow(first_row)
            for row in rows_iter:
                writer.writerow(row)


@dataclass(frozen=True)
class CsvExportResult:
    csv_path: Path
    all_frames: bool
    frames: List[int]
    render_hash: str
    output_root: Path
    image_dir: Path
    manifest_path: Path


def export_csv(kp3d_path: Path, render_spec_path: Path, output_dir: Path, frames: Optional[Sequence[int]] = None) -> CsvExportResult:
    spec = load_spec(render_spec_path)
    render_hash = render_paths.compute_render_hash(spec, render_spec_path.parent.parent)
    paths = render_paths.build_output_paths(output_dir, render_hash)
    render_paths.ensure_dirs(paths)

    with kp3d_path.open("rb") as f:
        payload = pickle.load(f)
    kp3d = extract_kp3d(payload)
    frame_count = kp3d.shape[0]

    selected_frames = select_frames(frames, frame_count)
    validate_frames(selected_frames, frame_count)
    all_frames = is_all_frames(selected_frames, frame_count)

    if all_frames:
        csv_path = render_paths.all_csv_path(paths)
        if csv_path.exists():
            log(f"all.csv already exists, skip: {csv_path}")
            return CsvExportResult(
                csv_path=csv_path,
                all_frames=all_frames,
                frames=selected_frames,
                render_hash=render_hash,
                output_root=paths.root,
                image_dir=paths.image_dir,
                manifest_path=paths.manifest_path,
            )
    else:
        # Keep temp.csv name, but isolate by frame signature to avoid collisions in parallel runs.
        csv_path = render_paths.temp_csv_path(paths, selected_frames)

    converter = CsvConverter(spec)
    converter.write_csv(kp3d, selected_frames, csv_path)
    log(f"csv generated: {csv_path}")

    return CsvExportResult(
        csv_path=csv_path,
        all_frames=all_frames,
        frames=selected_frames,
        render_hash=render_hash,
        output_root=paths.root,
        image_dir=paths.image_dir,
        manifest_path=paths.manifest_path,
    )


CsvResultsKey = tuple  # (str(session), tuple(frames))
CsvResultsDict = Dict[CsvResultsKey, Dict[str, Any]]


def build_csv_results(
    plan_items: Iterable[Any],
    kp3d_finder: Any,
    render_spec_path: Path,
    output_root: Path,
    request_all_frames: bool = False,
) -> CsvResultsDict:
    """
    RenderPlanのitemsからCSV結果辞書を構築

    Args:
        plan_items: RenderPlan.items（skip, frames, session, output_dir属性を持つ）
        kp3d_finder: session から kp3d_path を取得する callable
        render_spec_path: render_spec.yaml のパス
        output_root: 出力ルートディレクトリ
        request_all_frames: True の場合、全フレーム用CSVを1回だけ生成して共有

    Returns:
        {(str(session), tuple(frames)): {"csv_path": Path, "all_frames": bool}, ...}
    """
    csv_results: CsvResultsDict = {}
    items_list = list(plan_items)

    if request_all_frames and items_list:
        # 全フレーム最適化: 1回だけ export_csv を呼んで全 item で共有
        first_item = items_list[0]
        kp3d_path = kp3d_finder(first_item.session)
        if kp3d_path is None:
            log(f"kp3d not found for session: {first_item.session}")
            return csv_results

        output_dir = first_item.output_dir or (output_root / first_item.session.name)

        all_csv_result = export_csv(
            kp3d_path=kp3d_path,
            render_spec_path=render_spec_path,
            output_dir=output_dir,
            frames=None,
        )
        full_set = set(all_csv_result.frames)

        for item in items_list:
            if item.skip or not item.frames:
                continue

            key = (str(item.session), tuple(item.frames))
            csv_results[key] = {
                "csv_path": all_csv_result.csv_path,
                "all_frames": set(item.frames) == full_set,
            }
    else:
        # 通常: 各 item ごとに個別に export_csv を呼ぶ
        for item in items_list:
            if item.skip or not item.frames:
                continue

            kp3d_path = kp3d_finder(item.session)
            if kp3d_path is None:
                log(f"kp3d not found for session: {item.session}")
                continue

            output_dir = item.output_dir or (output_root / item.session.name)

            csv_result = export_csv(
                kp3d_path=kp3d_path,
                render_spec_path=render_spec_path,
                output_dir=output_dir,
                frames=item.frames,
            )

            key = (str(item.session), tuple(item.frames))
            csv_results[key] = {
                "csv_path": csv_result.csv_path,
                "all_frames": csv_result.all_frames,
            }

    return csv_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kp3d_pkl", required=True)
    parser.add_argument("--render_spec", required=True)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frames = parse_frames_arg(args.frames)
    result = export_csv(
        kp3d_path=Path(args.kp3d_pkl),
        render_spec_path=Path(args.render_spec),
        output_dir=Path(args.output_dir),
        frames=frames,
    )
    log(json.dumps({"csv_path": str(result.csv_path), "all_frames": result.all_frames}))


if __name__ == "__main__":
    main()

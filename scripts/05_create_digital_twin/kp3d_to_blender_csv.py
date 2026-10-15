"""
Convert kp3d pickle files to CSV for Blender

1. pickle → outputs/animate_temp/csv/{session}_points.csv
2. scale  → outputs/animate_temp/transformedcsv/{session}_points.csv_transformedtest.csv

Usage:
  # Single session
  python scripts/05_create_digital_twin/kp3d_to_blender_csv.py dailylife_cj611_20230227_100000

  # All sessions
  python scripts/05_create_digital_twin/kp3d_to_blender_csv.py --all

  # Specify output directory
  python scripts/05_create_digital_twin/kp3d_to_blender_csv.py --all --out-dir outputs/my_csvs/
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────────

KP_NAMES = [
    "nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder",
    "L elbow", "R elbow", "L wrist", "R wrist", "L hip", "R hip",
    "L knee", "R knee", "L ankle", "R ankle", "backbone peak", "tail_1", "tail_2",
]

SCALE = 1 / 1000
BLENDER_NECK_TAIL = [1.50 - 0.340, 0.995 - 0.06, 0.780]


# ── Utilities ─────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """Convert to a key safe for CSV column names: lowercase, spaces -> _, drop chars other than alphanumerics and _"""
    key = name.lower().strip().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", key)


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


def rot_x(deg: float) -> np.ndarray:
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(deg: float) -> np.ndarray:
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(deg: float) -> np.ndarray:
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rot_euler(rx=0.0, ry=0.0, rz=0.0, order="xyz") -> np.ndarray:
    mats = {"x": rot_x(rx), "y": rot_y(ry), "z": rot_z(rz)}
    R = np.eye(3)
    for ax in order:
        R = mats[ax] @ R
    return R


def _safe_norm(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _rotmat_to_euler_xyz(R):
    """R: (..., 3, 3) → Euler (rx, ry, rz) in radians, XYZ order"""
    r00 = R[..., 0, 0]; r10 = R[..., 1, 0]; r20 = R[..., 2, 0]
    r11 = R[..., 1, 1]; r12 = R[..., 1, 2]
    r21 = R[..., 2, 1]; r22 = R[..., 2, 2]

    sy = np.sqrt(r00 * r00 + r10 * r10)
    singular = sy < 1e-8

    rx = np.where(~singular, np.arctan2(r21, r22), np.arctan2(-r12, r11))
    ry = np.where(~singular, np.arctan2(-r20, sy),  np.arctan2(-r20, sy))
    rz = np.where(~singular, np.arctan2(r10, r00),  np.zeros_like(r00))
    return np.stack([rx, ry, rz], axis=-1)


# ── Head pose estimation ───────────────────────────────────────────────────────────────

def add_cam_pose_with_euler(
    points: pd.DataFrame,
    l_eye="pts_l_eye_",
    r_eye="pts_r_eye_",
    l_ear="pts_l_ear_",
    r_ear="pts_r_ear_",
    out_prefixes=("pts_l_cam_", "pts_r_cam_"),
    center_eye_prefix="pts_center_eye_",
    forward_sign=-1,
):
    """
    Estimate head orientation (XYZ Euler, rad) from both eyes and ears and add columns.
    Write {rx,ry,rz} for each prefix in out_prefixes and for center_eye_prefix.
    """
    need = (
        [f"{l_eye}{a}" for a in "xyz"]
        + [f"{r_eye}{a}" for a in "xyz"]
        + [f"{l_ear}{a}" for a in "xyz"]
        + [f"{r_ear}{a}" for a in "xyz"]
    )
    for c in need:
        if c not in points.columns:
            raise ValueError(f"Missing required column: {c}")

    LE  = points[[f"{l_eye}x", f"{l_eye}y", f"{l_eye}z"]].to_numpy(float)
    RE  = points[[f"{r_eye}x", f"{r_eye}y", f"{r_eye}z"]].to_numpy(float)
    LEr = points[[f"{l_ear}x", f"{l_ear}y", f"{l_ear}z"]].to_numpy(float)
    REr = points[[f"{r_ear}x", f"{r_ear}y", f"{r_ear}z"]].to_numpy(float)

    eye_c = (LE + RE) * 0.5
    ear_c = (LEr + REr) * 0.5

    s_n = _safe_norm(RE - LE)
    f_n = _safe_norm((eye_c - ear_c) * float(forward_sign))

    u_raw = np.cross(f_n, s_n)
    mask_bad = (
        (np.linalg.norm(u_raw, axis=1) < 1e-8)
        | np.isnan(u_raw).any(axis=1)
        | np.isnan(s_n).any(axis=1)
        | np.isnan(f_n).any(axis=1)
    )
    if np.any(mask_bad):
        up_world = np.tile(np.array([0.0, 0.0, 1.0])[None, :], (len(points), 1))
        u_raw[mask_bad] = np.cross(f_n[mask_bad], up_world[mask_bad])

    u_n = _safe_norm(u_raw)
    s_ortho = _safe_norm(np.cross(u_n, f_n))

    R = np.stack([s_ortho, u_n, f_n], axis=-1)
    det = np.linalg.det(R)
    if np.any(det < 0):
        R[det < 0, :, 0] *= -1.0

    eul = _rotmat_to_euler_xyz(R)

    for out_prefix in out_prefixes:
        points[f"{out_prefix}rx"] = eul[:, 0]
        points[f"{out_prefix}ry"] = eul[:, 1]
        points[f"{out_prefix}rz"] = eul[:, 2]

    if center_eye_prefix:
        points[f"{center_eye_prefix}rx"] = eul[:, 0]
        points[f"{center_eye_prefix}ry"] = eul[:, 1]
        points[f"{center_eye_prefix}rz"] = eul[:, 2]

    return points


# ── pickle → intermediate CSV ─────────────────────────────────────────────────────────

def pickletocsv(
    result_dir: Path,
    n_frame2draw: int = -1,
    csv_dir: Path = Path("./csv"),
):
    result_dir = Path(result_dir)
    csv_dir    = Path(csv_dir)

    print("result dir:", result_dir)

    pickleFile = result_dir / "kp3d_fxdJointLen.pickle"
    if not pickleFile.exists():
        pickleFile = result_dir / "kp3d.pickle"
    if not pickleFile.exists():
        raise FileNotFoundError(f"pickle file is not exist: {pickleFile}")
    print("using:", pickleFile)

    with open(pickleFile, "rb") as f:
        data = pickle.load(f)

    data_name = os.path.basename(str(result_dir).rstrip("/"))

    X = data["kp3d"]        # (n_animal, n_frame, n_kp, 3)
    S = data["kp3d_score"]  # (n_animal, n_frame, n_kp)

    n_animal, n_frame, n_kp, _ = X.shape

    if n_frame2draw < 0 or n_frame2draw > n_frame:
        n_frame2draw = n_frame

    csv_dir.mkdir(parents=True, exist_ok=True)

    i_animal = 0
    Xa = X[i_animal, :n_frame2draw, :, :].copy()
    Sa = S[i_animal, :n_frame2draw, :].copy()

    kp_names_now = KP_NAMES[:]

    # Midpoint of index 5 and 6 (shoulder_mid)
    if max(5, 6) < Xa.shape[1]:
        mid56 = (Xa[:, 5, :] + Xa[:, 6, :]) / 2
        Xa = np.concatenate([Xa, mid56[:, np.newaxis, :]], axis=1)
        Sa = np.concatenate([Sa, ((Sa[:, 5] + Sa[:, 6]) / 2)[:, np.newaxis]], axis=1)
        kp_names_now.append("shoulder_mid")

    # Midpoint of index 11 and 12 (base)
    if max(11, 12) < Xa.shape[1]:
        mid1112 = (Xa[:, 11, :] + Xa[:, 12, :]) / 2
        Xa = np.concatenate([Xa, mid1112[:, np.newaxis, :]], axis=1)
        Sa = np.concatenate([Sa, ((Sa[:, 11] + Sa[:, 12]) / 2)[:, np.newaxis]], axis=1)
        kp_names_now.append("base")

    # Midpoint of index 1 and 2 (center_eye)
    if max(1, 2) < Xa.shape[1]:
        mid12 = (Xa[:, 1, :] + Xa[:, 2, :]) / 2
        Xa = np.concatenate([Xa, mid12[:, np.newaxis, :]], axis=1)
        Sa = np.concatenate([Sa, ((Sa[:, 1] + Sa[:, 2]) / 2)[:, np.newaxis]], axis=1)
        kp_names_now.append("center_eye")

    R = rot_euler(rx=-90, ry=0, rz=180)
    Xa = Xa @ R.T

    kp_keys = unique_keys(kp_names_now)
    assert len(kp_keys) == Xa.shape[1] == Sa.shape[1], "Number of key names does not match data"

    n_frames = Xa.shape[0]
    data_dict: dict = {
        "i_frame":  np.arange(n_frames, dtype=np.int64),
        "i_animal": np.full(n_frames, i_animal, dtype=np.int64),
    }
    for kpi, key in enumerate(kp_keys):
        data_dict[f"pts_{key}_x"] = Xa[:, kpi, 0]
        data_dict[f"pts_{key}_y"] = Xa[:, kpi, 1]
        data_dict[f"pts_{key}_z"] = Xa[:, kpi, 2]

    df = pd.DataFrame(data_dict)
    out_csv = csv_dir / f"{data_name}_points.csv"
    df.to_csv(out_csv, index=False)
    print("[OK ] wrote:", out_csv)


# ── Scaling / coordinate transform → final CSV ──────────────────────────────────────────────

def scale(points_path, BNT=None, out_dir: Path = None):
    print("transform and scale DLC points.")
    print("SCALE", SCALE)
    print("TARGET", BLENDER_NECK_TAIL)

    points = pd.read_csv(points_path)
    points.iloc[:, 2:] = points.iloc[:, 2:] * SCALE

    bld_neck = np.array(BNT) if BNT is not None else np.array(BLENDER_NECK_TAIL)
    print(points.filter(regex="pts_shoulder_mid_*").iloc[0])

    for col in points.columns:
        if col.endswith("_x"):
            points[col] += bld_neck[0]
        elif col.endswith("_y"):
            points[col] += bld_neck[1]
        elif col.endswith("_z"):
            points[col] += bld_neck[2]

    points = add_cam_pose_with_euler(points)

    out_dir = Path(out_dir) if out_dir is not None else Path("outputs/animate_temp/transformedcsv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (os.path.basename(str(points_path)) + "_transformedtest.csv")
    points.to_csv(out_path, index=None)
    print("done")


# ── Pipeline ───────────────────────────────────────────────────────────────

def process_one(
    data_name: str,
    base_dir: Path,
    csv_dir: Path,
    out_dir: Path,
    n_frame2draw: int,
):
    result_dir = base_dir / data_name
    csv_path = csv_dir / f"{data_name}_points.csv"

    print(f"\n=== {data_name} ===")
    print(f"  Input: {result_dir}")
    print(f"  Intermediate CSV: {csv_path}")
    print(f"  Output CSV: {out_dir}")

    pickletocsv(result_dir=result_dir, n_frame2draw=n_frame2draw, csv_dir=csv_dir)
    scale(csv_path, out_dir=out_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert kp3d pickle to CSV for Blender",
        epilog=(
            "Examples:\n"
            "  python scripts/05_create_digital_twin/kp3d_to_blender_csv.py dailylife_cj611_20230227_100000\n"
            "  python scripts/05_create_digital_twin/kp3d_to_blender_csv.py --all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_name",
        nargs="?",
        help="Session name (e.g. dailylife_cj611_20230227_100000). Cannot be combined with --all.",
    )
    parser.add_argument("--all", action="store_true", help="Process all sessions under --base-dir")
    parser.add_argument("--n-frame2draw", type=int, default=-1, help="Number of frames to convert (-1 for all frames)")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/3d"),
        help="Root directory of 3D results (default: outputs/3d)",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("outputs/animate_temp/csv"),
        help="Output directory for intermediate CSV (default: outputs/animate_temp/csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/animate_temp/transformedcsv"),
        help="Output directory for transformed CSV (default: outputs/animate_temp/transformedcsv)",
    )
    args = parser.parse_args()

    if args.all and args.data_name:
        parser.error("data_name and --all cannot be specified together")
    if not args.all and not args.data_name:
        parser.error("Specify either data_name or --all")

    if args.all:
        sessions = sorted(d.name for d in args.base_dir.iterdir() if d.is_dir())
        if not sessions:
            print(f"No sessions found: {args.base_dir}")
            sys.exit(1)
        print(f"Processing {len(sessions)} sessions")
        skipped = []
        for session in sessions:
            try:
                process_one(session, args.base_dir, args.csv_dir, args.out_dir, args.n_frame2draw)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}", file=sys.stderr)
                skipped.append(session)
        if skipped:
            print(f"\nSkipped {len(skipped)} sessions: {skipped}", file=sys.stderr)
    else:
        process_one(args.data_name, args.base_dir, args.csv_dir, args.out_dir, args.n_frame2draw)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-frame validation script that uses calibration.toml + kp2d_f.pickle
to recompute triangulation and reprojection error with aniposelib's CameraGroup

Example usage:
  python scripts/06_validation/keypoints/test_projection_error.py \
    --base ../marmo3Dpose/results/3d_v0p8_dark_fix_20/dailylife_cj611_20230226_110000 \
    --config_yaml ./config/marmo_cj425m/config.yaml \
    --frame 70 --animal 0 --score-threshold 0.30
"""

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import argparse
import os
import pickle

import numpy as np
import yaml
from utils.aniposelib.cameras import CameraGroup
from marmo3d.validation.utils import load_ids_from_yaml, load_kp2d_f, build_points2d_for_frame

def base_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="Results folder (e.g. ../marmo3Dpose/results/3d_v0p8_dark_fix_20/dailylife_...)")
    ap.add_argument("--config_yaml", required=True,
                    help="YAML that defines the camera ID order (e.g. ./config/marmo_cj425m/config.yaml)")
    ap.add_argument("--frame", type=int, default=70)
    ap.add_argument("--animal", type=int, default=0)
    ap.add_argument("--score-threshold", type=float, default=0.3)
    args = ap.parse_args()

    base = args.base.rstrip("/")

    calib_toml = os.path.join(base, "calibration.toml")
    if not os.path.isfile(calib_toml):
        raise FileNotFoundError(f"not found: {calib_toml}")

    # Load CameraGroup from calibration.toml (intrinsics scaled, extrinsics optimized)
    cgroup = CameraGroup.load(calib_toml)

    # Fix the camera order to match reconst
    ids = load_ids_from_yaml(args.config_yaml)
    cgroup = cgroup.subset_cameras_names(ids)

    # Load 2D keypoints (after Viterbi filtering)
    kp2d_f = load_kp2d_f(base)
    p2d, scores = build_points2d_for_frame(
        kp2d_f, frame=args.frame, animal=args.animal, score_thresh=args.score_threshold
    )

    n_cam, n_kp, _ = p2d.shape
    print(f"frame={args.frame}, animal={args.animal}")
    print(f"n_cam={n_cam}, n_kp={n_kp}")
    print("camera order:", ids)

    # Reshape to match aniposelib (n_cam, N, 2)
    p2d_flat = p2d.reshape(n_cam, -1, 2)

    # Triangulation (no optimization) -> reprojection error
    pts3d = cgroup.triangulate(p2d_flat, progress=False)        # shape: (N,3)
    err_mean = cgroup.reprojection_error(pts3d, p2d_flat, mean=True)  # (N,)

    print("\n[recomputed mean reprojection error per keypoint (px)]")
    print(err_mean)

    # For comparison: also show the saved kp3d_err ("ground truth")
    kp3d_pkl = os.path.join(base, "kp3d_fxdJointLen.pickle")
    if os.path.isfile(kp3d_pkl):
        with open(kp3d_pkl, "rb") as f:
            kp3d_raw = pickle.load(f)
        saved_err = kp3d_raw["kp3d_err"][args.animal][args.frame]  # (n_kp,)
        print("\n[saved kp3d_err from reconst_single_step3.py (px)]")
        print(saved_err)
    else:
        print("\nNOTE: saved kp3d_err not found (skip comparison).")

if __name__ == “__main__”:
    base_main()

"""
Read kp3d (3D keypoints) from either pickle or npz.

Both formats are returned as the same dict as the marmo3Dpose pickle:
  kp3d       (n_animal, T, 20, 3) float64  triangulated coordinates [mm] (calibration frame)
  kp3d_score (n_animal, T, 20)    float64
  kp3d_err   (n_animal, T, 20)    float64  reprojection error [px]
  joint_len  list[(30,) float64]  one per animal

Supported npz files:
  - the released {session}.npz written by export_aligned_npz.py (has kp3d_mm; single animal, n_animal=1)
  - an npz saved with the same keys as the pickle (4-D kp3d)

resolve_kp3d only looks inside the given directory, so an npz from another directory never
silently replaces the data. To use npz files, pass their directory (e.g. with --base).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

PICKLE_NAMES = ("kp3d_fxdJointLen.pickle", "kp3d.pickle")
NPZ_NAMES = ("kp3d_fxdJointLen.npz", "kp3d.npz")


def resolve_kp3d(base_dir, session: str | None = None) -> Path | None:
    """Find the kp3d file. Returns None if there is none.

    With session, the search order is:
      {base}/{session}.npz (flat released layout) -> {base}/{session}/{NPZ_NAMES} -> {base}/{session}/{PICKLE_NAMES}
    With session=None, base_dir is the session directory itself: NPZ_NAMES -> PICKLE_NAMES.
    """
    base = Path(base_dir)
    if session is not None:
        flat = base / f"{session}.npz"
        if flat.exists():
            return flat
        base = base / session
    for name in NPZ_NAMES + PICKLE_NAMES:
        p = base / name
        if p.exists():
            return p
    return None


def load_kp3d(path) -> dict:
    """Load kp3d as the same dict as the pickle (.pickle or .npz)."""
    path = Path(path)
    if path.suffix != ".npz":
        with open(path, "rb") as f:
            return pickle.load(f)
    with np.load(path) as z:
        if "kp3d_mm" in z.files:                      # released format from export_aligned_npz
            return {"kp3d": z["kp3d_mm"][None], "kp3d_score": z["kp3d_score"][None],
                    "kp3d_err": z["kp3d_err"][None], "joint_len": [z["joint_len"]]}
        if z["kp3d"].ndim == 4:                       # saved with the pickle's keys
            jl = z["joint_len"] if "joint_len" in z.files else np.empty((z["kp3d"].shape[0], 0))
            return {"kp3d": z["kp3d"], "kp3d_score": z["kp3d_score"],
                    "kp3d_err": z["kp3d_err"], "joint_len": list(jl)}
    raise ValueError(f"not a kp3d npz: {path}")

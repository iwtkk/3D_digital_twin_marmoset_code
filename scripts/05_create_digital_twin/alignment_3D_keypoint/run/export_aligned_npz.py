"""
Write the 3D keypoints aligned by the reprojection affine (affine_log.npz) to one NPZ per session.

It goes through the same computation as aligned_csv/<session>_affine.csv
(affine_align.build_aligned_df), so the coordinates match the CSV. It also stores what the CSV
lacks: score / err, the raw triangulated coordinates, the transform parameters and the mapping
from kp3d index to video frame. No object arrays are used, so np.load works without allow_pickle.

NPZ contents (T = number of kp3d frames, K = 23 = 20 measured + 3 derived keypoints):
  kp3d            (T, K, 3)   float64  Blender world coordinates [m]; CSV pts_<kp>_{x,y,z}
  kp_names        (K,)        <U       keypoint names (CSV columns pts_<name>_*); first 20 are measured
  head_euler_xyz  (T, 3)      float64  head orientation, Euler XYZ [rad]; CSV pts_{l_cam,r_cam,center_eye}_r{x,y,z}
  kp3d_mm         (T, 20, 3)  float64  raw triangulated coordinates [mm] (calibration frame, before rotation)
  kp3d_score      (T, 20)     float64  minimum 2D score over the contributing cameras
  kp3d_err        (T, 20)     float64  reprojection error [px]
  joint_len       (30,)       float64  fixed bone lengths [mm]
  rot             (3, 3)      float64  kp3d_mm -> rotated mm
  affine_A        (3, 3)      float64  rotated [m] -> Blender [m]
  affine_t        (3,)        float64      kp3d[:, :20] = ((kp3d_mm @ rot.T) / 1000) @ affine_A.T + affine_t
  i_frame         (T,)        int64    kp3d frame index (0..T-1)
  camera_id       (C,)        <U       camera IDs (order of config.camera_id)
  frame_num       (C, T)      int64    imgstore frame_number of each camera for kp3d[i]; -1 if unknown
  frame_min       (C,)        int64    first frame_number of each camera (index in video = frame_num - frame_min)
  frame_num_source ()         <U       "step1" (frame_num.npy saved at triangulation) / "reconstructed" / "none"
  frame_ref_camera ()         <U       time reference camera when reconstructed ("" for step1)
  fps             ()          int64    sampling rate of kp3d (24)

frame_num prefers <cam>/frame_num.npy saved by step1. Only when it is missing, the step1
resampling (from t[0] + 1 s in 1/fps steps) is reproduced; the step1 reference camera (first in
glob order) cannot be reproduced, so the result can be off by up to 0.5 s when several cameras
give the same T length.

Run:
  .venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/run/export_aligned_npz.py --session cj357_20221009_090000
  .venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/run/export_aligned_npz.py --all --jobs 8 --resume
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/run"))
sys.path.insert(0, str(ROOT / "scripts/dev/floorZ/run"))
import affine_align as aa                    # noqa: E402
import floorz_core as fz                     # noqa: E402

CFG = ROOT / "config/marmo_cj425m/config.yaml"
# path_registry.data.marmo3dpose_step1_results (same run as outputs/3d; frame_num.npy only survives here)
STEP1_RESULTS_DIR = "/data/iwata/iwtkmarmo/results/3d_v0p8_dark_fix_20_all_det"
FPS = 24
N_MEASURED = 20


def _pick_ref_camera(vid_dir, session, cam_ids, n_frame):
    """Choose the time reference camera. Returns (ref_cam, note).

    step1 builds T from the first camera in glob order, which depends on the file system at run
    time and cannot be reproduced. T changes by a few frames with the reference camera, so take a
    camera whose T length equals the number of kp3d frames (the first in config order if several).
    """
    from frame_map import _store_meta
    lens = {}
    for cam in cam_ids:
        _, t, _ = _store_meta(Path(vid_dir), session, cam)
        lens[cam] = len(np.arange(t[0] + 1, t[-1] - 1, 1.0 / FPS))
    match = [c for c in cam_ids if lens[c] == n_frame]
    if not match:
        return cam_ids[0], f"no reference camera with matching T length {sorted(set(lens.values()))} vs kp3d {n_frame}"
    return match[0], ("" if len(match) == 1 else f"{len(match)} candidate reference cameras")


def _load_step1_frame_num(frame_num_dir, session, cam_ids, n_frame):
    """<cam>/frame_num.npy saved by step1 as (C, T). None if missing or of a different length."""
    if not frame_num_dir:
        return None
    for prefix in ("dailylife_", ""):
        d = Path(frame_num_dir) / f"{prefix}{session}"
        fs = [d / cam / "frame_num.npy" for cam in cam_ids]
        if all(f.exists() for f in fs):
            F = np.stack([np.load(f) for f in fs]).astype(np.int64)
            return F if F.shape[1] == n_frame else None
    return None


def _frame_num(vid_dir, session, cam_ids, n_frame, frame_num_dir=None):
    """Returns (frame_num (C,T), frame_min (C,), ref_cam, source, note).

    Uses step1's frame_num.npy when present, otherwise reconstructs it from the video metadata.
    Filled with -1 when neither is available.
    """
    fn = np.full((len(cam_ids), n_frame), -1, dtype=np.int64)
    fmin = np.full(len(cam_ids), -1, dtype=np.int64)
    F = _load_step1_frame_num(frame_num_dir, session, cam_ids, n_frame)
    if F is not None:
        from frame_map import _store_meta
        for ci, cam in enumerate(cam_ids):
            fmin[ci] = _store_meta(Path(vid_dir), session, cam)[2][0]
        return F, fmin, "", "step1", ""
    try:
        from frame_map import build_frame_map
        ref, note = _pick_ref_camera(vid_dir, session, cam_ids, n_frame)
        order = [ref] + [c for c in cam_ids if c != ref]   # build_frame_map uses the first camera as reference
        fmap, nT = build_frame_map(vid_dir, session, order, fps=FPS)
    except Exception as e:
        return fn, fmin, "", "none", f"no frame_num ({type(e).__name__}: {e})"
    n = min(n_frame, nT)
    for ci, cam in enumerate(cam_ids):
        fn[ci, :n] = fmap[cam][0][:n]
        fmin[ci] = fmap[cam][1]
    return fn, fmin, ref, "reconstructed", ("reconstructed " + note).strip()


def build_npz_dict(session, A, t, input_dir, vid_dir, cam_ids, with_frame_num=True,
                   frame_num_dir=None):
    pp = fz.resolve_pickle(Path(input_dir) / session)
    if pp is None:
        raise FileNotFoundError(f"kp3d not found: {session}")
    raw = fz.load_kp3d(pp)
    df = aa.build_aligned_df(session, A, t, input_dir)

    coord = [c for c in df.columns if c.startswith("pts_") and c.endswith("_x")]
    names = [c[len("pts_"):-len("_x")] for c in coord]
    kp3d = np.stack([df[[f"pts_{k}_{a}" for a in "xyz"]].to_numpy(float) for k in names], axis=1)
    head = df[[f"pts_center_eye_r{a}" for a in "xyz"]].to_numpy(float)

    n = len(df)
    d = {
        "kp3d": kp3d,
        "kp_names": np.array(names, dtype=str),
        "head_euler_xyz": head,
        "kp3d_mm": np.asarray(raw["kp3d"][0], float),
        "kp3d_score": np.asarray(raw["kp3d_score"][0], float),
        "kp3d_err": np.asarray(raw["kp3d_err"][0], float),
        "joint_len": np.asarray(raw["joint_len"][0], float).ravel(),
        "rot": np.asarray(fz.KP3D_ROT, float),
        "affine_A": np.asarray(A, float),
        "affine_t": np.asarray(t, float),
        "i_frame": df["i_frame"].to_numpy(np.int64),
        "camera_id": np.array(cam_ids, dtype=str),
        "fps": np.int64(FPS),
    }
    note = ""
    if with_frame_num:
        d["frame_num"], d["frame_min"], ref, src, note = _frame_num(
            vid_dir, session, cam_ids, n, frame_num_dir)
        d["frame_num_source"] = np.array(src, dtype=str)
        d["frame_ref_camera"] = np.array(ref, dtype=str)
    return d, note


def verify(d, csv_path=None):
    """Check the NPZ for internal consistency (and against the CSV if given). Returns a list of problems."""
    errs = []
    if [str(k) for k in d["kp_names"][:N_MEASURED]] != fz.unique_keys(fz.KP_NAMES)[:N_MEASURED]:
        errs.append("first 20 kp_names differ from KP_NAMES")
    P = ((d["kp3d_mm"] @ d["rot"].T) / 1000.0) @ d["affine_A"].T + d["affine_t"]
    diff = np.nanmax(np.abs(P - d["kp3d"][:, :N_MEASURED]))
    if not diff < 1e-9:
        errs.append(f"kp3d does not match the transformed kp3d_mm, max|diff|={diff:.3g}")
    if csv_path is not None and Path(csv_path).exists():
        import pandas as pd
        c = pd.read_csv(csv_path)
        cols = [f"pts_{k}_{a}" for k in d["kp_names"] for a in "xyz"]
        X = c[cols].to_numpy(float).reshape(len(c), -1, 3)
        if X.shape != d["kp3d"].shape:
            errs.append(f"CSV shape mismatch {X.shape} vs {d['kp3d'].shape}")
        else:
            dc = np.nanmax(np.abs(X - d["kp3d"]))
            nan_eq = np.array_equal(np.isnan(X), np.isnan(d["kp3d"]))
            if not (dc < 1e-9 and nan_eq):
                errs.append(f"differs from CSV, max|diff|={dc:.3g}, same NaN={nan_eq}")
    return errs


_W: dict = {}


def _init_worker(a):
    _W.update(a)


def _process(session):
    try:
        p = _W["params"][session]
        A, t = p[:9].reshape(3, 3), p[9:12]
        d, note = build_npz_dict(session, A, t, _W["input_dir"], _W["vid_dir"],
                                 _W["cam_ids"], _W["with_frame_num"], _W["frame_num_dir"])
        csv = Path(_W["csv_dir"]) / f"{session}_affine.csv" if _W["check_csv"] else None
        errs = verify(d, csv)
        if errs:
            return f"{session}: NG {'; '.join(errs)}"
        out = Path(_W["output_dir"]) / f"{session}.npz"
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **d)
        tmp.replace(out)
        return f"{session}: ok T={len(d['i_frame'])}" + (f" ({note})" if note else "")
    except Exception as e:
        return f"{session}: ERROR {type(e).__name__}: {e}"


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--all", action="store_true", help="process every session in affine_log.npz")
    ap.add_argument("--input-dir", default=str(ROOT / "outputs/3d"))
    ap.add_argument("--affine-log", default=str(ROOT / "outputs/reproj_eval/affine_log.npz"))
    ap.add_argument("--vid-dir", default=str(ROOT / "vid"))
    ap.add_argument("--csv-dir", default=str(ROOT / "outputs/reproj_eval/aligned_csv"),
                    help="CSVs compared by --check-csv")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/reproj_eval/aligned_npz"))
    ap.add_argument("--frame-num-dir", default=STEP1_RESULTS_DIR,
                    help="3D results root holding step1's {session}/<cam>/frame_num.npy; '' to only reconstruct")
    ap.add_argument("--no-frame-num", action="store_true", help="do not write frame_num")
    ap.add_argument("--check-csv", action="store_true", help="check that coordinates match the existing CSV")
    ap.add_argument("--resume", action="store_true", help="skip sessions that are already written")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    with np.load(args.affine_log) as z:
        params = {k: z[k] for k in z.files}
    if args.all:
        sessions = sorted(params)
    elif args.session:
        if args.session not in params:
            ap.error(f"{args.session} is not in affine_log; run affine_align.py first")
        sessions = [args.session]
    else:
        ap.error("give --session or --all")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        sessions = [s for s in sessions if not (out_dir / f"{s}.npz").exists()]
    if not sessions:
        print("nothing to do"); return

    cam_ids = [str(c) for c in yaml.safe_load(open(CFG))["camera_id"]]
    w = dict(params=params, input_dir=args.input_dir, vid_dir=args.vid_dir, cam_ids=cam_ids,
             with_frame_num=not args.no_frame_num, frame_num_dir=args.frame_num_dir,
             csv_dir=args.csv_dir,
             check_csv=args.check_csv, output_dir=str(out_dir))
    log = open(out_dir / "export_aligned_npz.log", "a")
    n_bad = 0
    if args.jobs > 1:
        with mp.get_context("fork").Pool(args.jobs, _init_worker, (w,)) as pool:
            results = pool.imap_unordered(_process, sessions)
            for msg in results:
                n_bad += (": ok" not in msg); print(msg); log.write(msg + "\n"); log.flush()
    else:
        _init_worker(w)
        for s in sessions:
            msg = _process(s)
            n_bad += (": ok" not in msg); print(msg); log.write(msg + "\n"); log.flush()
    print(f"done: {len(sessions) - n_bad}/{len(sessions)} ok -> {out_dir}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()

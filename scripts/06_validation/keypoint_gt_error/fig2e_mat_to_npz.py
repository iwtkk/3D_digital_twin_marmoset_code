#!/usr/bin/env python
"""Convert the Fig. 2E matched keypoint file SingleAnimal_20.mat to a self-describing .npz.

SingleAnimal_20.mat was written by s20230220_cperror_single.py (original_code/) with
    annotKp3ds, decodedKp3ds : (20 keypoints, 3 xyz, n_sample) float64, mm
for the 100 single-animal labeled frames, without the session / frame of each sample. This script recovers them:
  - the GT frame of each sample: the labeled JSON whose keypoints_3d[0, :20] equals annotKp3ds (exactly, NaN-aware)
  - the prediction frame: 0-based index round(mean(framedata row)) - 24 used by s20230220_cperror_single.py
    (Python round(), i.e. halves to even) in {results}/3d_v0p8_dark_fix_20_for_3d_eval/{session}/kp3d_fxdJointLen.pickle
    (that prediction is no longer available, so the index is recorded but not re-checked)
It also checks that the sample order is the one of the original script (sessions in framedata.csv order,
rows in file order), and writes {out_dir}/SingleAnimal_20.npz (the .mat file is not modified).

Usage:
  .venv/bin/python scripts/06_validation/keypoint_gt_error/fig2e_mat_to_npz.py \
      --gt-dir <dataset>/marmoset_keypoint_data/training_data/3dkp_gt/labeled_frames \
      --mat-dir <dataset>/marmoset_keypoint_data/training_data/3dkp_gt/matched_keypoints \
      --out-dir <same as mat-dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

STEM = "SingleAnimal_20"
LABELED = "valbin1_nanim1_nfrm100"
KP_NAMES = ["nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder", "L elbow", "R elbow",
            "L wrist", "R wrist", "L hip", "R hip", "L knee", "R knee", "L ankle", "R ankle", "backbone",
            "tail 1", "tail 2"]
N_KP = 20
FRAME_OFFSET = 24  # s20230220_cperror_single.py: currfrm = round(mean(framedata row)) - 24 (0-based)


def same(a, b):
    """Exact equality with NaN == NaN."""
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True, type=Path, help="folder holding valbin1_nanim1_nfrm100/")
    ap.add_argument("--mat-dir", required=True, type=Path, help="folder holding SingleAnimal_20.mat")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    m = loadmat(args.mat_dir / f"{STEM}.mat")
    annot = m["annotKp3ds"].transpose(2, 0, 1)       # (n, 20, 3)
    pred = m["decodedKp3ds"].transpose(2, 0, 1)
    n = annot.shape[0]
    tbl = pd.read_csv(args.gt_dir / LABELED / "framedata.csv", index_col=0)
    gts = {fn: np.asarray(json.load(open(args.gt_dir / LABELED / f"{fn}.json"))["keypoints_3d"], float)[0, :N_KP]
           for fn in tbl.index}

    frame_name = []
    for i in range(n):
        hits = [fn for fn, g in gts.items() if same(g, annot[i])]
        if len(hits) != 1:
            raise RuntimeError(f"sample {i}: GT match is not unique ({len(hits)} hits)")
        frame_name.append(hits[0])
    expected = [fn for s in tbl["VideoSession"].unique()[:10] for fn in tbl.index[tbl["VideoSession"] == s]]
    if frame_name != expected:
        raise RuntimeError("sample order differs from s20230220_cperror_single.py")

    session = tbl.loc[frame_name, "VideoSession"].to_numpy(str)
    cams = [c for c in tbl.columns if c != "VideoSession"]
    cam_frames = tbl.loc[frame_name, cams].to_numpy(np.int64)        # (n, 8) labeled video frame per camera
    pred_frame = np.array([round(float(v)) for v in cam_frames.mean(axis=1)], np.int64) - FRAME_OFFSET

    out = dict(
        gt_kp3d=annot, pred_kp3d=pred, error_mm=np.linalg.norm(annot - pred, axis=-1),
        session=session, frame_name=np.array(frame_name),
        camera_id=np.array(cams), labeled_frame_per_camera=cam_frames,
        pred_frame=pred_frame, kp_names=np.array(KP_NAMES), source_mat=np.array(f"{STEM}.mat"),
    )
    # round trip: the npz must reproduce the .mat arrays exactly
    assert np.array_equal(out["gt_kp3d"].transpose(1, 2, 0), m["annotKp3ds"], equal_nan=True)
    assert np.array_equal(out["pred_kp3d"].transpose(1, 2, 0), m["decodedKp3ds"], equal_nan=True)

    np.savez_compressed(args.out_dir / f"{STEM}.npz", **out)
    e = out["error_mm"]
    print(f"{STEM}: n={n} sessions={len(set(session))} labeled keypoints={np.isfinite(e).sum()} "
          f"median error={np.nanmedian(e):.3f} mm  -> {args.out_dir / (STEM + '.npz')}")


if __name__ == "__main__":
    main()

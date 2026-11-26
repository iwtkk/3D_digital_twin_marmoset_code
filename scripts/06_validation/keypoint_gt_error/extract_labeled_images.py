#!/usr/bin/env python
"""Re-extract the camera images of the labeled 3D ground-truth frames from the dataset videos.

The PNG files the labeling tool worked on were not kept; only the labels (labeled_frames/{set}/{frame_name}.json)
and the frame of each camera (framedata.csv) were. This script cuts the same frames out of the dataset videos
(marmoset_keypoint_data/video_data) and writes them with the labeling tool's file names, so the folder can be opened
with the labeling GUI next to the JSON files.

  framedata.csv value = imgstore frame_number - first frame_number of that camera (index within the video),
  as written by multicam_toolbox.extract_frames_for_3dannotation(); the image is store.get_image(frame_min + value).

Outputs ({output_dir}/labeled_images/):
  {set}/{frame_name}.{camera_id}.png   (1536, 2048, 3) uint8 BGR, lossless, as the labeling tool saved them
  qc/{frame_name}.jpg                  8 cameras tiled (2 x 4, 512 x 384 each), keypoints_2d drawn on top
  extract_log.csv                      one row per image: frame_name, session, camera_id, frame_index,
                                       frame_number, timestamp, n_labeled_kp

Usage:
  .venv/bin/python scripts/06_validation/keypoint_gt_error/extract_labeled_images.py [--dataset-root <dataset>]
      [--set valbin1_nanim1_nfrm100] [--out-dir <dir>] [--no-qc] [--limit N]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import imgstore
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "config" / "keypoint_gt_error.yaml"
SESSION_RE = re.compile(r"(cj\d+m?_\d{8}_\d{6})$")   # dailylife_cj425_20221106_100000 -> cj425_20221106_100000
TILE = (512, 384)


def session_name(video_session):
    m = SESSION_RE.search(video_session)
    if m is None:
        raise ValueError(f"cannot parse the session name from {video_session!r}")
    return m.group(1)


def qc_tile(img, kp2d, label):
    """Downscaled copy of one camera image with the labeled 2D keypoints (kp2d: (20, 2), NaN = not labeled)."""
    out = img.copy()
    for x, y in kp2d:
        if np.isfinite(x):
            cv2.circle(out, (int(round(x)), int(round(y))), 9, (0, 0, 255), -1)
    out = cv2.resize(out, TILE, interpolation=cv2.INTER_AREA)
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--dataset-root", help="overrides dataset_root in the config")
    p.add_argument("--set", help="labeled-frame folder; overrides labeled_images.set in the config")
    p.add_argument("--out-dir", help="default: {output_dir}/labeled_images")
    p.add_argument("--no-qc", action="store_true", help="do not write the qc/ overlays")
    p.add_argument("--limit", type=int, help="only the first N frames (for testing)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.dataset_root or cfg["dataset_root"])
    li = cfg["labeled_images"]
    set_name = args.set or li["set"]
    gt_dir = root / cfg["gt_dir"] / "labeled_frames" / set_name
    video_dir = root / li["video_dir"]
    out_dir = Path(args.out_dir) if args.out_dir else REPO / cfg["output_dir"] / "labeled_images"
    img_dir = out_dir / set_name
    qc_dir = out_dir / "qc"
    img_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_qc:
        qc_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(gt_dir / "framedata.csv", index_col=0, dtype={"VideoSession": str})
    cams = [str(c) for c in table.columns[1:]]
    if args.limit:
        table = table.iloc[:args.limit]
    print(f"{len(table)} frames x {len(cams)} cameras from {gt_dir}")
    print(f"videos: {video_dir}\nout:    {img_dir}")

    rows, stores, n_fail = [], {}, 0
    for i, (frame_name, rec) in enumerate(table.iterrows(), start=1):
        sess = session_name(rec["VideoSession"])
        kp2d = np.array(json.loads((gt_dir / f"{frame_name}.json").read_text())["keypoints_2d"], float)[0]  # (8,20,2)
        tiles = []
        for ic, cam in enumerate(cams):
            key = (sess, cam)
            if key not in stores:
                stores[key] = imgstore.new_for_filename(str(video_dir / f"{sess}.{cam}" / "metadata.yaml"))
            st = stores[key]
            idx = int(rec[cam])
            img, (fnum, ts) = st.get_image(st.frame_min + idx)
            if fnum != st.frame_min + idx:
                print(f"  NG {frame_name} {cam}: asked frame {st.frame_min + idx}, got {fnum}")
                n_fail += 1
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(img_dir / f"{frame_name}.{cam}.png"), img)
            n_lab = int(np.isfinite(kp2d[ic, :, 0]).sum())
            rows.append(dict(frame_name=frame_name, session=sess, camera_id=cam, frame_index=idx,
                             frame_number=int(fnum), timestamp=float(ts), n_labeled_kp=n_lab))
            if not args.no_qc:
                tiles.append(qc_tile(img, kp2d[ic], f"{cam} f={idx} n={n_lab}"))
        if not args.no_qc:
            grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
            cv2.putText(grid, f"{frame_name}  {sess}", (8, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            cv2.imwrite(str(qc_dir / f"{frame_name}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  [{i}/{len(table)}] {frame_name} {sess}")

    log = pd.DataFrame(rows)
    log.to_csv(out_dir / "extract_log.csv", index=False)
    # camera sync: spread of the timestamps of the 8 cameras within each frame
    spread = log.groupby("frame_name")["timestamp"].agg(lambda t: t.max() - t.min())
    print(f"images: {len(log)}  frame-number mismatch: {n_fail}")
    print(f"camera timestamp spread per frame [s]: median {spread.median():.4f}, max {spread.max():.4f}")
    print("RESULT:", "OK" if n_fail == 0 else "NG")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

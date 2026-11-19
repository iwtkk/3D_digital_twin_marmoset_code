"""
Extract the real camera frames of the Fig. 8a pairs (frames.csv) from the published videos.

I/O ledger: scripts/06_validation/consistency/io.md#extract_real_frames

The frame of each pair is the `video_frame_index` column of frames.csv: the Video frame index that the published
metrics were computed with. `--frame-source npz` uses `npz_video_frame_index` instead (frame_num - frame_min of the
3D keypoint NPZ, the correspondence saved at triangulation); the two differ by -2..+1 frames in 816 of 2400 pairs
(README.md, "Real frames"), so only `published` reproduces the released values.

Inputs (read only, --dataset-root = extracted Published dataset):
  3D_digital_twin_data/rendered_environmental_images/frames.csv
  marmoset_keypoint_data/video_data/{session}.{camera_id}/   (imgstore: metadata.yaml, {chunk}.mp4, {chunk}.npz)
  marmoset_keypoint_data/3D_temporal_keypoint_data/{session}.npz   (frame_min)
Outputs (--work-dir):
  real/{session}/Camera.{camera_id}/{frame_3d:07d}.png   [1536, 2048, 3] uint8 BGR (as stored by imgstore)

Example:
  .venv/bin/python scripts/06_validation/consistency/extract_real_frames.py \\
      --dataset-root /path/to/3D_digital_twin_marmoset_data --work-dir outputs/validation/consistency_fig8a --workers 8
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imgstore
import numpy as np

import consistency_core as cc


def extract_session(job):
    root, work, session, rows, column = job
    z = np.load(Path(root) / cc.NPZ.format(session=session))
    fmin = {str(c): int(v) for c, v in zip(z["camera_id"], z["frame_min"])}
    n = 0
    for cam in sorted({r["camera_id"] for r in rows}):
        store = imgstore.new_for_filename(str(Path(root) / cc.VIDEO_DIR.format(session=session, cam=cam)
                                              / "metadata.yaml"))
        for r in sorted((r for r in rows if r["camera_id"] == cam), key=lambda r: r[column]):
            out = cc.real_path(work, session, cam, int(r["frame_3d"]))
            if out.exists():
                continue
            img, _ = store.get_image(frame_number=int(r[column]) + fmin[cam])
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), img); n += 1
        store.close()
    return session, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--frames-csv", type=Path, default=None)
    ap.add_argument("--frame-source", choices=["published", "npz"], default="published")
    ap.add_argument("--session", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    column = "video_frame_index" if a.frame_source == "published" else "npz_video_frame_index"
    pairs = cc.load_pairs(a.dataset_root, a.frames_csv, a.session)
    jobs = [(str(a.dataset_root), str(a.work_dir), s, g.to_dict("records"), column)
            for s, g in pairs.groupby("session_id")]
    total = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (s, n) in enumerate(ex.map(extract_session, jobs), 1):
            total += n
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] sessions, {total} frames written", flush=True)


if __name__ == "__main__":
    main()

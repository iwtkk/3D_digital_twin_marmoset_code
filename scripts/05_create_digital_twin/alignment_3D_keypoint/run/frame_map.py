"""
kp3d フレーム index → 各カメラの実フレーム frame_number の対応を再生成する。

marmo3Dpose の reconst_single_step1.py と同一ロジック:
  - 基準カメラ camera_id[0] の frame_time から t_start=t[0]+1, t_end=t[-1]-1
  - T = arange(t_start, t_end, 1/fps)   (fps=24)
  - 各カメラで frame_num[i] = T[i] に最も近い実フレームの frame_number
これにより kp3d[i] は「生フレーム index = frame_num[cam][i] - frame_min[cam]」に対応する。
outputs/3d/<session>/<cam>/frame_num.npy が消えているため metadata から作り直す。
"""
from __future__ import annotations

from pathlib import Path

import imgstore
import numpy as np


def _store_meta(vid_dir: Path, session: str, cam_id: str):
    st = imgstore.new_for_filename(str(vid_dir / f"{session}.{cam_id}/metadata.yaml"))
    m = st.get_frame_metadata()
    return st, np.asarray(m["frame_time"]), np.asarray(m["frame_number"], dtype=np.int64)


def build_frame_map(vid_dir, session, cam_ids, fps: int = 24, t_intv=None):
    """戻り値: dict cam_id -> (frame_number[i], frame_min, store)。i は kp3d フレーム index。

    cam_ids[0] を時間基準に T を作る（step1 と同じ stores[0]）。
    """
    vid_dir = Path(vid_dir)
    ref_store, t_ref, _ = _store_meta(vid_dir, session, cam_ids[0])
    if t_intv is None:
        t_start, t_end = t_ref[0] + 1, t_ref[-1] - 1
    else:
        t_start, t_end = t_ref[0] + t_intv[0], t_ref[0] + t_intv[1]
    T = np.arange(t_start, t_end, 1.0 / fps)

    out = {}
    for cam in cam_ids:
        st, t_cam, fn_cam = _store_meta(vid_dir, session, cam)
        idx = np.clip(np.searchsorted(t_cam, T), 1, len(t_cam) - 1)
        idx = np.where(np.abs(t_cam[idx - 1] - T) < np.abs(t_cam[idx] - T), idx - 1, idx)
        out[cam] = (fn_cam[idx], int(fn_cam[0]), st)
    return out, len(T)

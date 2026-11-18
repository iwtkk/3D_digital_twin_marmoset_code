"""
Shared definitions of the Fig. 8a reproduction: Published dataset layout, pair list, Animal region and metrics.

I/O ledger: scripts/06_validation/consistency/io.md#consistency_core

Animal region of a pair (same definition as the published values):
  render side: the render background is identical for every Session at a given camera; the empty cage is rendered
               once per camera (render_pairs.py, animals excluded). |render - background| > 25
               (max over channels, uint8), opened with a 3x3 kernel, components < 100 px dropped
               = render silhouette; its bbox is the render bbox.
  real side  : the 2D detection bbox in the real frame (2D_temporal_keypoint_data); with several detections,
               the one with the highest mean keypoint score.
  region     : union of the two bboxes padded by 10% of the long side (>= 16 px), clipped to the image.
               The region of the Detailed model with fur is used for every body model of a pair.
Metrics: RGB, skimage peak_signal_noise_ratio (data_range=255) and structural_similarity (7x7, channel_axis=2,
data_range=255), on the full image and on the Animal-region crop.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

MODELS = ("detailed_fur", "detailed_nofur", "simplified_fur")
CHUNK_FRAMES = 10000        # frames per imgstore chunk (video_data and 2D_temporal_keypoint_data)
DIFF_THR = 25
MIN_COMPONENT_PX = 100
PAD_FRAC = 0.10
PAD_MIN_PX = 16

# Published dataset layout (relative to the extracted dataset root)
FIG8A_DIR = "3D_digital_twin_data/rendered_environmental_images"
DETAILED_BLEND = "3D_digital_twin_data/3D_digitaltwin_marmoset_blender_files/{session}.blend"
SIMPLIFIED_BLEND = "3D_digital_twin_data/3D_digitaltwin_simplified_marmoset/{session}.blend"
NPZ = "marmoset_keypoint_data/3D_temporal_keypoint_data/{session}.npz"
VIDEO_DIR = "marmoset_keypoint_data/video_data/{session}.{cam}"
LABEL2D = "marmoset_keypoint_data/2D_temporal_keypoint_data/{session}/{session}_{cam}_{chunk:06d}.json"


def load_pairs(dataset_root: Path, frames_csv: Path | None = None, sessions=None) -> pd.DataFrame:
    """frames.csv (one row per Session x Camera ID x 3D frame)."""
    p = Path(frames_csv) if frames_csv else Path(dataset_root) / FIG8A_DIR / "frames.csv"
    df = pd.read_csv(p, dtype={"camera_id": str})
    if sessions:
        df = df[df.session_id.isin(sessions)]
    return df.reset_index(drop=True)


def render_path(work_dir: Path, model: str, session: str, cam: str, frame_3d: int) -> Path:
    return Path(work_dir) / "render" / model / session / f"Camera.{cam}" / f"{frame_3d:07d}.png"


def background_path(work_dir: Path, cam: str) -> Path:
    return Path(work_dir) / "render" / "background" / f"Camera.{cam}.png"


def real_path(work_dir: Path, session: str, cam: str, frame_3d: int) -> Path:
    return Path(work_dir) / "real" / session / f"Camera.{cam}" / f"{frame_3d:07d}.png"


def animal_of(session: str) -> str:
    return session.split("_", 1)[0]


# ── render side ────────────────────────────────────────────────────────────────
def render_silhouette(render_bgr: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    diff = np.abs(render_bgr.astype(np.int16) - bg_bgr.astype(np.int16)).max(-1)
    m = cv2.morphologyEx((diff > DIFF_THR).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX
    return keep[lab]


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


# ── real side ──────────────────────────────────────────────────────────────────
@lru_cache(maxsize=16)
def _label2d(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def detection_bbox(dataset_root: Path, session: str, cam: str, video_frame_index: int):
    """2D detection bbox (x1, y1, x2, y2) at a Video frame index, or None."""
    dets = _label2d(str(Path(dataset_root) / LABEL2D.format(session=session, cam=cam,
                                                             chunk=video_frame_index // CHUNK_FRAMES)))
    local = video_frame_index % CHUNK_FRAMES
    if dets is None or local >= len(dets) or not dets[local]:
        return None
    best = max(dets[local], key=lambda d: float(np.mean(np.asarray(d[5])[:, 2])))
    return tuple(int(round(v)) for v in best[1:5])


# ── region and metrics ─────────────────────────────────────────────────────────
def union_roi(boxes, W: int, H: int):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    pad = max(PAD_MIN_PX, int(round(PAD_FRAC * max(x2 - x1, y2 - y1))))
    return (max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad))


def box_iou(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else float("nan")


def psnr_ssim(render_bgr: np.ndarray, real_bgr: np.ndarray, roi=None):
    if roi is not None:
        x1, y1, x2, y2 = roi
        render_bgr, real_bgr = render_bgr[y1:y2, x1:x2], real_bgr[y1:y2, x1:x2]
    a = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2RGB)
    return (float(peak_signal_noise_ratio(a, b, data_range=255)),
            float(structural_similarity(a, b, channel_axis=2, data_range=255)))


def fmt_box(b) -> str:
    return "" if b is None else " ".join(map(str, b))

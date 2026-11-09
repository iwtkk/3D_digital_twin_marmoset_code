"""
scripts/dev 共通設定ローダ。

docs/method.yaml を読み、物理キャリブレーション・骨格・座標変換・フレームレート
(A〜D・F) を提供する。各 target の run/plot からは次のいずれかで参照する:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/dev
    import _config as cfg

アルゴリズムのチューニング値(KDE/affine=E、検出しきい値=G)はここには置かず、
各スクリプト側に残す。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

_YAML_PATH = Path(__file__).resolve().parent / "docs" / "method.yaml"
METHOD_YAML_PATH = _YAML_PATH   # 書き戻し (global norm キャッシュ等) で参照する public パス
with open(_YAML_PATH) as _f:
    _CFG = yaml.safe_load(_f)


def sanitize(name: str) -> str:
    """表示名 → 列キー ('L eye' -> 'l_eye')。floorz_core 旧実装と同一。"""
    key = name.lower().strip().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", key)


# ── A. 座標変換 ────────────────────────────────────────────────────────────────
_ct = _CFG["coordinate_transform"]
BLENDER_BBOX_SCALE = np.array(_ct["blender_bbox_scale"], dtype=float)
BLENDER_CENTER     = np.array(_ct["blender_center"], dtype=float)
KP3D_CENTER        = np.array(_ct["kp3d_center"], dtype=float)
KP3D_NORM_SCALE    = np.array(_ct["kp3d_norm_scale"], dtype=float)
ROTATION_DEG       = _ct["rotation_deg"]   # {'rx': -90, 'ry': 0, 'rz': 180}

# ── B. 物理高さ ────────────────────────────────────────────────────────────────
PHYSICAL_HEIGHTS_M = np.array(_CFG["physical_heights_m"], dtype=float)
PHYS_LABELS        = list(_CFG["physical_labels"])

# ── C. kp3d 空間 inlier bbox (mm) ──────────────────────────────────────────────
_bb = _CFG["kp3d_inlier_bbox_mm"]
BBOX_X = tuple(_bb["x"])
BBOX_Y = tuple(_bb["y"])
BBOX_Z = tuple(_bb["z"])
FLOOR_Y_MM = float(_bb["y"][1])   # 床レベル (= 旧 GLOBAL_FLOOR_Y 92.27)

# ── D. 骨格 ────────────────────────────────────────────────────────────────────
_kp = _CFG["keypoints"]
KP_NAMES_CANONICAL = list(_kp["names"])                       # "L eye" 形
KP_NAMES = [n.replace(" ", "_") for n in KP_NAMES_CANONICAL]  # "L_eye" 形 (表示)
KP_KEYS  = [sanitize(n) for n in KP_NAMES_CANONICAL]          # "l_eye" 形 (列キー)
N_KP = len(KP_NAMES_CANONICAL)

FLOOR_KP_KEYWORDS = tuple(_kp["floor_kp_keywords"])
FLOOR_KP_IDX = [i for i, k in enumerate(KP_KEYS)
                if any(w in k for w in FLOOR_KP_KEYWORDS)]    # [9, 10, 15, 16]
FLOOR_KP_KEYS = [KP_KEYS[i] for i in FLOOR_KP_IDX]            # ['l_wrist', ...]

NEIGHBORS = {int(k): list(v) for k, v in _kp["neighbors"].items()}
BONES = sorted({tuple(sorted((a, b))) for a, nbrs in NEIGHBORS.items() for b in nbrs})
BONE_NAMES = [f"{KP_NAMES[a]}-{KP_NAMES[b]}" for a, b in BONES]
N_BONE = len(BONES)

# ── F. フレームレート ──────────────────────────────────────────────────────────
FPS_3D = float(_CFG["fps"]["fps_3d"])
FPS_2D = float(_CFG["fps"]["fps_2d"])

# ── 描画共通 ────────────────────────────────────────────────────────────────────
ANIMAL_COLORS = dict(_CFG["animal_colors"])

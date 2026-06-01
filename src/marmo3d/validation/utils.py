"""
共通ユーティリティ関数
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import yaml

# パス設定
current_dir = Path(__file__).resolve().parent
marmo3d_dir = current_dir.parent          # src/marmo3d/
src_dir = marmo3d_dir.parent              # src/
project_root = src_dir.parent             # project root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import marmo3d.marmo_dataset as marmo_dataset


def load_valid_config(config_path: Path) -> Dict[str, Any]:
    """consistency.yamlを読み込み"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def expand_path_template(template: str, project_root: Path, **kwargs) -> Path:
    """パステンプレートを展開"""
    expanded = template.replace("{PROJECT_ROOT}", str(project_root))
    for key, value in kwargs.items():
        expanded = expanded.replace(f"{{{key}}}", str(value))
    return Path(expanded)


def discover_camera_ids(dataset_info: marmo_dataset.MarmoDataset) -> List[str]:
    """
    カメラIDを取得（allowed_idsが空の場合はvideo_rootから自動検出）
    """
    # まずallowed_idsを試す
    camera_ids = dataset_info.spec.camera_allowed_ids
    if camera_ids:
        return [str(cid) for cid in camera_ids]

    # 空の場合は自動検出
    discovered = set()
    if dataset_info.video_root.exists():
        for p in dataset_info.video_root.iterdir():
            if p.is_dir():
                m = dataset_info.spec.video_folder_id_re.match(p.name)
                if m:
                    discovered.add(m.group("camid"))

    camera_ids = sorted(list(discovered))
    if not camera_ids:
        raise ValueError("カメラIDが見つかりません")

    return camera_ids


def get_project_root() -> Path:
    """プロジェクトルートを取得"""
    return project_root


# ---- キーポイント検証ユーティリティ ----------------------------------------

# カメラハードウェアID -> 表示ラベルのマッピング
cam_map: Dict[int, str] = {
    23506226: "Cam 1",
    23511607: "Cam 2",
    23506214: "Cam 3",
    23506236: "Cam 4",
    23511614: "Cam 5",
    23506237: "Cam 6",
    23511613: "Cam 7",
    23506239: "Cam 8",
}


def load_ids_from_yaml(config_yaml: str | Path) -> List[str]:
    """カメラ設定yamlからカメラIDリストを読み込む"""
    with open(config_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [str(x) for x in cfg["camera_id"]]


def load_kp2d_f(base_dir: str | Path) -> np.ndarray:
    """kp2d_f.pickle を読み込む。期待形状: (n_frame, n_kp, n_animal, 3, n_cam)"""
    path = os.path.join(base_dir, "kp2d_f.pickle")
    with open(path, "rb") as f:
        kp2d_f = pickle.load(f)
    assert kp2d_f.ndim == 5 and kp2d_f.shape[2] >= 1
    return kp2d_f


def build_points2d_for_frame(
    kp2d_f: np.ndarray, frame: int, animal: int, score_thresh: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    指定フレーム・個体の2Dキーポイントを抽出する。

    Returns:
        points_2d: (n_cam, n_kp, 2)  score_thresh未満は NaN
        scores:    (n_cam, n_kp)
    """
    n_frame, n_kp, n_animal, dim, n_cam = kp2d_f.shape
    if frame >= n_frame:
        raise IndexError(f"frame {frame} is out of range (n_frame={n_frame})")
    if animal >= n_animal:
        raise IndexError(f"animal {animal} is out of range (n_animal={n_animal})")
    assert dim == 3, f"unexpected dim={dim}"

    # (n_kp, 3, n_cam) -> (n_cam, n_kp, 3)
    slice_f = kp2d_f[frame, :, animal, :, :]
    pts = np.transpose(slice_f, (2, 0, 1))
    points_2d = pts[:, :, :2].astype(np.float64)
    scores = pts[:, :, 2].astype(np.float64)

    bad = scores < score_thresh
    points_2d[bad] = np.nan
    return points_2d, scores


def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues ベクトル -> 回転行列 (OpenCV 規約)"""
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float)
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

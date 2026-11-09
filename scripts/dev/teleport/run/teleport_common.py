"""
teleport 解析の共有部品（run/plot 共通）。

定数・キーポイント名・2D チャンクローダ ChunkCache・フレーム変換を提供する。
CLI は持たない。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/dev
import _config as cfg  # noqa: E402

FPS_3D = cfg.FPS_3D
FPS_2D = cfg.FPS_2D
DT = 1.0 / FPS_3D
VIS_SCORE_THR = 0.5    # 2D keypoint "visible" if score >= this (検出しきい値; config 外)

KP_NAMES = cfg.KP_NAMES
N_KP = cfg.N_KP

# 解剖学的な隣接関節（v2 の孤立ジャンプ判定で使用）
NEIGHBORS = cfg.NEIGHBORS


def t3d_to_t2d(t3d: int) -> int:
    return round(t3d * FPS_2D / FPS_3D)


def resolve_pickle(session_dir: Path) -> Path | None:
    for name in ("kp3d_fxdJointLen.pickle", "kp3d.pickle"):
        p = session_dir / name
        if p.exists():
            return p
    return None


class ChunkCache:
    """セッションの 2D JSON チャンクを必要時に読み、メモリにキャッシュする。"""

    def __init__(self, session: str, d2d_dir: Path):
        self.session = session
        self.d2d_dir = d2d_dir / session
        self._cache: dict[tuple, list] = {}
        files = sorted(self.d2d_dir.glob("*.json")) if self.d2d_dir.exists() else []
        pat = re.compile(rf"{re.escape(session)}_(\d+)_(\d+)\.json")
        self.cameras: list[str] = []
        for f in files:
            m = pat.match(f.name)
            if m and m.group(1) not in self.cameras:
                self.cameras.append(m.group(1))

    def _load(self, cam: str, chunk: int) -> list:
        key = (cam, chunk)
        if key not in self._cache:
            path = self.d2d_dir / f"{self.session}_{cam}_{chunk:06d}.json"
            self._cache[key] = json.load(open(path)) if path.exists() else []
        return self._cache[key]

    def frame(self, cam: str, t2d: int) -> list:
        chunk, idx = t2d // 10000, t2d % 10000
        data = self._load(cam, chunk)
        return data[idx] if data and idx < len(data) else []

    def kp_score(self, cam: str, t2d: int, ki: int) -> float | None:
        dets = self.frame(cam, t2d)
        if not dets:
            return None
        best = max(dets, key=lambda d: d[7])
        return float(best[5][ki][2])

    def clear(self):
        self._cache.clear()

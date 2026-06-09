"""
pkl_to_corrected_csv.py の KDE affine 補正結果を可視化する。

pickle から床 KP (wrist/ankle) の bZ 分布を直接計算し、
補正前 / 補正後 を Blender 物理高さ (床/足場/棒/天井) と重ねてプロットする。
※ 142MB の corrected CSV を全て書き出さなくても可視化できるよう、
   床 KP の z 座標だけを in-memory で計算する（pkl_to_corrected_csv と同一ロジック）。

注意: 高さ分布の比較は Blender メートル空間で行う。
      pkl_to_corrected_csv の OUTPUT_SCALE は最終 CSV の単位にのみ影響し、
      ここでのプロット（物理高さとの比較）には掛けない。

使い方:
  # 全セッション → 大きな after-correction グリッド
  OPENBLAS_NUM_THREADS=1 python scripts/dev/affine_pkl_csv/plot_corrected.py --all

  # before/after 2 段グリッド
  OPENBLAS_NUM_THREADS=1 python scripts/dev/affine_pkl_csv/plot_corrected.py --all --before-after

  # 特定セッションのみ
  OPENBLAS_NUM_THREADS=1 python scripts/dev/affine_pkl_csv/plot_corrected.py \\
      --sessions cj425_20210427_120000 cj611_20210902_090000

出力:
  outputs/dev/affine_pkl_csv/plots/grid_after_correction.png
  outputs/dev/affine_pkl_csv/plots/grid_before_after.png   (--before-after 指定時)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ── 同ディレクトリの pkl_to_corrected_csv を import ────────────────────────────
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "pkl_to_corrected_csv", _HERE / "pkl_to_corrected_csv.py"
)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

PHYS = P.PHYSICAL_HEIGHTS_M
PHYS_LABELS = ["floor", "low", "bar", "ceil"]
ANIMAL_COLORS = {"cj425": "steelblue", "cj611": "darkorange", "cj357": "seagreen"}

OUT_DIR = P.OUT_ROOT / "plots"


# ── 床 KP の bZ 分布 + affine を pickle から計算 ───────────────────────────────

def floor_indices() -> list[int]:
    """KP_NAMES から wrist/ankle のインデックスを取得。"""
    return [
        i for i, name in enumerate(P.KP_NAMES)
        if any(k in P.sanitize(name) for k in P.FLOOR_KP_KEYWORDS)
    ]


def compute_session(session: str, base_dir: Path, n_frames: int = -1) -> dict | None:
    """
    1 セッションの床 KP bZ 分布と affine 係数を計算する。

    Returns dict(vals, peaks, a, b, status) ／ pickle が無ければ None
    """
    import pickle

    pkl = base_dir / session / "kp3d_fxdJointLen.pickle"
    if not pkl.exists():
        pkl = base_dir / session / "kp3d.pickle"
    if not pkl.exists():
        return None

    with open(pkl, "rb") as f:
        data = pickle.load(f)

    X = data["kp3d"][0]   # (n_frame, n_kp, 3)
    if n_frames > 0:
        X = X[:n_frames]

    # 回転（pkl_to_corrected_csv と同じ）
    R = P.rot_euler(rx=-90, ry=0, rz=180)
    X = X @ R.T

    # 床 KP の rotated_z → Blender bZ (m) へグローバルスケール
    fidx = floor_indices()
    z_rot = X[:, fidx, 2]                                  # (n_frame, n_floor_kp)
    half_span_z = P.BLENDER_BBOX_SCALE[2] / 2
    bz = (z_rot - P.KP3D_CENTER[2]) / P.KP3D_NORM_SCALE[2] * half_span_z + P.BLENDER_CENTER[2]

    floor_z = bz.min(axis=1)                               # 各フレーム最も床に近い肢端
    lo, hi  = np.percentile(floor_z, 1), np.percentile(floor_z, 99)
    vals    = floor_z[(floor_z >= lo) & (floor_z <= hi)]

    peaks = P.detect_peaks_kde(vals)
    a, b, status = P.fit_affine(peaks)

    return {
        "session": session, "animal": session.split("_")[0],
        "vals": vals, "peaks": peaks, "a": a, "b": b, "status": status,
    }


# ── プロット ────────────────────────────────────────────────────────────────────

def _draw_dist(ax, vals, color, peaks=None, title="", xlabel=""):
    """ヒストグラム + KDE + ピーク(赤) + 物理高さ(緑) を 1 軸に描く。"""
    ax.hist(vals, bins=100, color=color, alpha=0.45, density=True)
    if len(vals) > 10:
        kde = gaussian_kde(vals, bw_method=P.KDE_BW)
        xg = np.linspace(vals.min(), vals.max(), 500)
        ax.plot(xg, kde(xg), "k-", lw=1.0, alpha=0.7)
    if peaks is not None:
        for p in peaks:
            ax.axvline(p, color="red", lw=1.3, alpha=0.9)
    ymax = ax.get_ylim()[1]
    for h, lbl in zip(PHYS, PHYS_LABELS):
        ax.axvline(h, color="green", lw=1.0, ls=":", alpha=0.8)
        ax.text(h + 0.004, ymax * 0.96, lbl, fontsize=6,
                color="darkgreen", rotation=90, va="top")
    ax.set_title(title, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_xlim(0.65, 1.62)
    ax.tick_params(labelsize=6)


def plot_after_grid(recs: list[dict], out_path: Path, ncol: int = 6) -> None:
    """全セッションの after-correction 分布グリッド（大きいまま保存）。"""
    n = len(recs)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 3.0))
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, recs):
        color = ANIMAL_COLORS.get(r["animal"], "gray")
        vals_c  = r["a"] * r["vals"] + r["b"]
        peaks_c = r["a"] * r["peaks"] + r["b"]
        warn = " !" if abs(r["a"] - 1.0) > P.MAX_AFFINE_SLOPE_DEV else ""
        _draw_dist(
            ax, vals_c, color, peaks=peaks_c,
            title=f'{r["session"][3:]}\n[{r["status"]}] a={r["a"]:.3f} b={r["b"]:+.3f}{warn}',
            xlabel="bZ corrected (m)",
        )

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.legend(handles=[
        Line2D([0], [0], color="red",   lw=1.3, label="KDE peak (corrected)"),
        Line2D([0], [0], color="green", lw=1.0, ls=":", label="physical height"),
    ], loc="lower right", fontsize=9)
    fig.suptitle(
        "AFTER affine correction — floor KP bZ distribution (all sessions)\n"
        "red=KDE peak  green=physical heights [0.7337, 1.0179, 1.3035, 1.5016]",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.005, 1, 0.985])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path}  ({nrow}x{ncol} cells, {n} sessions)")


def plot_before_after_grid(recs: list[dict], out_path: Path) -> None:
    """各セッション 2 列（左=before, 右=after）の縦長グリッド。"""
    n = len(recs)
    fig, axes = plt.subplots(n, 2, figsize=(9, n * 2.6))
    axes = np.atleast_2d(axes)

    for i, r in enumerate(recs):
        color = ANIMAL_COLORS.get(r["animal"], "gray")
        warn = " !" if abs(r["a"] - 1.0) > P.MAX_AFFINE_SLOPE_DEV else ""
        _draw_dist(
            axes[i, 0], r["vals"], color, peaks=r["peaks"],
            title=f'{r["session"][3:]}  BEFORE',
            xlabel="bZ (m)",
        )
        vals_c  = r["a"] * r["vals"] + r["b"]
        peaks_c = r["a"] * r["peaks"] + r["b"]
        _draw_dist(
            axes[i, 1], vals_c, color, peaks=peaks_c,
            title=f'AFTER  [{r["status"]}] a={r["a"]:.3f} b={r["b"]:+.3f}{warn}',
            xlabel="bZ corrected (m)",
        )

    fig.suptitle(
        "BEFORE vs AFTER affine correction (floor KP bZ)\n"
        "red=KDE peak  green=physical heights",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.003, 1, 0.99])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {out_path}  ({n} sessions, before/after)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="pkl_to_corrected_csv の補正結果を可視化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--all", action="store_true", help="--base-dir 以下の全セッション")
    ap.add_argument("--sessions", nargs="+", default=None, help="特定セッション名を列挙")
    ap.add_argument("--before-after", action="store_true",
                    help="before/after 2 列グリッドも出力")
    ap.add_argument("--n-frames", type=int, default=-1, help="処理フレーム数 (-1=全フレーム)")
    ap.add_argument("--base-dir", type=Path, default=P.BASE_DIR)
    args = ap.parse_args()

    if not args.all and not args.sessions:
        ap.error("--all か --sessions のどちらかを指定してください")

    sessions = (
        sorted(d.name for d in args.base_dir.iterdir() if d.is_dir())
        if args.all else args.sessions
    )
    print(f"{len(sessions)} セッションを処理します")

    recs = []
    for i, s in enumerate(sessions, 1):
        try:
            r = compute_session(s, args.base_dir, args.n_frames)
        except Exception as e:
            print(f"  [ERROR] {s}: {e}", file=sys.stderr)
            r = None
        if r is None:
            print(f"  [SKIP] {s}", file=sys.stderr)
            continue
        recs.append(r)
        if i % 20 == 0 or i == len(sessions):
            print(f"  {i}/{len(sessions)} done")

    if not recs:
        print("有効なセッションがありません", file=sys.stderr)
        sys.exit(1)

    plot_after_grid(recs, OUT_DIR / "grid_after_correction.png")
    if args.before_after:
        plot_before_after_grid(recs, OUT_DIR / "grid_before_after.png")


if __name__ == "__main__":
    main()

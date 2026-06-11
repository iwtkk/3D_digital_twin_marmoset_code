"""
what is happening 図 (plot) — 1 セッション = 1 行で補正手順を並べる。

各行 4 列:
  ① 補正前の分布(density hist + KDE) + 検出ピーク(赤) + 物理高さ(緑点線)
  ② 線形回帰: x=検出ピーク → y=物理高さ (y=x が無補正基準)
  ③ 補正後の分布: bZ_corrected = a·bZ + b
  ④ floor bZ 時系列: 補正前(青) vs 補正後(黒)

入力 (--input-dir, 既定 outputs/3d):  {input-dir}/{session}/kp3d_*.pickle
出力 (--output-dir, 既定 scripts/dev/floorZ_plot): what_is_happening_kde.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import floorz_core as core  # noqa: E402

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_OUTPUT = Path("scripts/dev/floorZ_plot")

_CJK = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
if Path(_CJK).exists():
    fm.fontManager.addfont(_CJK)
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

DEFAULT_SESSIONS = [
    "cj425_20210427_120000",   # full_affine
    "cj357_20221009_090000",   # partial_affine_3peaks
    "cj611_20210902_090000",   # partial_affine_2peaks
    "cj425_20210428_130000",   # offset_only
]
XLIM = (0.60, 1.75)
GREEN = "#2ca02c"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", nargs="+", default=None,
                    help="セッション名を列挙 (省略時は代表4件)")
    args = ap.parse_args()

    sessions = args.session or DEFAULT_SESSIONS
    fig, axes = plt.subplots(len(sessions), 4, figsize=(19, 3.4 * len(sessions)))
    axes = np.atleast_2d(axes)

    for r, session in enumerate(sessions):
        pkl = core.resolve_pickle(args.input_dir / session)
        if pkl is None:
            axes[r, 0].set_title(f"{session}\n(pickle not found)", fontsize=9)
            continue
        raw = core.floor_bz_series(pkl)          # 時系列 (NaN 込み)
        vals = core.trim_1_99(raw)
        xg, dens = core.kde_curve(vals)
        peaks = core.detect_peaks_kde(vals)
        a, b, status, mp_v, mh_v = core.fit_affine(peaks, return_pairs=True)

        ax1, ax2, ax3, ax4 = axes[r]

        ax1.hist(vals, bins=120, density=True, color="0.80", zorder=1)
        ax1.plot(xg, dens, color="0.25", lw=1.4, zorder=2)
        for ph in core.PHYSICAL_HEIGHTS_M:
            ax1.axvline(ph, color=GREEN, ls=":", lw=1.3, alpha=0.8, zorder=3)
        for pk in peaks:
            ax1.axvline(pk, color="red", lw=1.6, zorder=4)
        ax1.set_xlim(*XLIM); ax1.set_ylabel(f"{session}\n[{status}]", fontsize=9)
        if r == 0:
            ax1.set_title("① 補正前の分布\n赤=検出ピーク, 緑点線=物理高さ", fontsize=10)

        xs = np.linspace(XLIM[0], XLIM[1], 100)
        ax2.plot(xs, xs, color="0.6", ls=":", lw=1.3, zorder=1)
        ax2.plot(xs, a * xs + b, color="crimson", lw=2.0, zorder=2)
        if len(mp_v):
            ax2.scatter(mp_v, mh_v, s=70, color="red", edgecolor="k", lw=0.5, zorder=4)
        ax2.set_xlim(*XLIM); ax2.set_ylim(*XLIM); ax2.set_aspect("equal"); ax2.grid(alpha=0.25)
        ax2.text(0.04, 0.92, f"a={a:.4f}\nb={b:+.4f}", transform=ax2.transAxes,
                 va="top", fontsize=9, bbox=dict(fc="white", ec="0.7", alpha=0.8))
        if r == 0:
            ax2.set_title("② 線形回帰\nx=ピーク → y=物理高さ", fontsize=10)

        vals_c = a * vals + b
        peaks_c = a * peaks + b
        xg_c, dens_c = core.kde_curve(vals_c)
        ax3.hist(vals_c, bins=120, density=True, color="0.80", zorder=1)
        ax3.plot(xg_c, dens_c, color="0.25", lw=1.4, zorder=2)
        for ph in core.PHYSICAL_HEIGHTS_M:
            ax3.axvline(ph, color=GREEN, ls=":", lw=1.3, alpha=0.8, zorder=3)
        for pk in peaks_c:
            ax3.axvline(pk, color="red", lw=1.6, zorder=4)
        ax3.set_xlim(*XLIM)
        if r == 0:
            ax3.set_title("③ 補正後の分布\n赤ピークが緑(物理高さ)に乗る", fontsize=10)

        T = len(raw)
        step = max(1, T // 4000)
        t = np.arange(0, T, step)
        before = raw[::step]
        after = a * before + b
        ax4.plot(t, before, color="tab:blue", lw=0.5, alpha=0.8, label="補正前")
        ax4.plot(t, after, color="black", lw=0.5, alpha=0.8, label="補正後")
        for ph in core.PHYSICAL_HEIGHTS_M:
            ax4.axhline(ph, color=GREEN, ls=":", lw=1.0, alpha=0.7)
        ax4.set_ylim(*XLIM); ax4.set_xlabel("frame")
        if r == 0:
            ax4.set_title("④ floor bZ 時系列\n青=補正前, 黒=補正後", fontsize=10)
            ax4.legend(fontsize=8, loc="upper right")
        print(f"{session:28} [{status:22}] a={a:.4f} b={b:+.4f} peaks={np.round(peaks, 3)}")

    fig.suptitle("what is happening — 各 session で「補正前分布→線形回帰→補正後→時系列」を同手順で処理",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "what_is_happening_kde.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

"""
Raw distribution の KDE ピーク確認図 (plot)。

density ヒストグラム + KDE 曲線 + KDE 検出ピーク(赤) + 物理高さ(緑) を重ね、
ピークが分布のモードに乗ることを示す（旧 GMM 版がモードを外した問題の確認用）。

入力 (--input-dir, 既定 outputs/3d):  {input-dir}/{session}/kp3d_*.pickle
出力 (--output-dir, 既定 scripts/dev/floorZ_plot): what_is_happening_kde_fixed.png
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

# status 別の代表 4 セッション
DEFAULT_SESSIONS = [
    ("cj425_20210427_120000", "full_affine"),
    ("cj357_20221009_090000", "partial_affine_3peaks"),
    ("cj611_20210902_090000", "partial_affine_2peaks"),
    ("cj425_20210428_130000", "offset_only"),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", nargs="+", default=None,
                    help="セッション名を列挙 (省略時は代表4件)")
    args = ap.parse_args()

    sessions = ([(s, "") for s in args.session] if args.session else DEFAULT_SESSIONS)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (session, status) in zip(axes.ravel(), sessions):
        pkl = core.resolve_pickle(args.input_dir / session)
        if pkl is None:
            ax.set_title(f"{session}\n(pickle not found)", fontsize=9)
            continue
        vals = core.trim_1_99(core.floor_bz_series(pkl))
        xg, dens = core.kde_curve(vals)
        peaks = core.detect_peaks_kde(vals)
        _, _, _, mp, _ = core.fit_affine(peaks, return_pairs=True)

        ax.hist(vals, bins=120, density=True, color="0.80", edgecolor="none", zorder=1,
                label="floor KP bZ ヒストグラム (density)")
        ax.plot(xg, dens, color="0.25", lw=1.6, zorder=2, label="KDE 曲線")
        for ph in core.PHYSICAL_HEIGHTS_M:
            ax.axvline(ph, color="green", ls=":", lw=1.4, alpha=0.7, zorder=3)
        for pk in peaks:
            ax.axvline(pk, color="red", lw=1.8, zorder=4)
        for pkx in mp:
            yk = dens[np.argmin(np.abs(xg - pkx))]
            ax.plot(pkx, yk, "o", color="red", ms=7, mec="k", mew=0.5, zorder=5)
        ax.set_title(f"{session}  [{status}]\n赤=KDEピーク  緑点線=物理高さ", fontsize=10)
        ax.set_xlabel("補正前 bZ (m)"); ax.set_ylabel("density")
        ax.set_xlim(0.6, 1.7)

    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Raw distribution — KDE ピーク(赤)はヒストグラム/KDE曲線のモードに乗る",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "what_is_happening_kde_fixed.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

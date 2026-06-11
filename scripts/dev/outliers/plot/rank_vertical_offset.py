"""
補正後 CSV の「垂直方向のずれ」をセッション別にランク付けする。

各セッションについて Blender bbox に対する垂直 (Z) のはみ出しと、
床接地キーポイント (wrist/ankle) の高さオフセットを計算し、
ずれの大きい順に表示・CSV 保存・プロットする。

指標:
  down_rate / up_rate : 床より下 / 天井より上に出た (frame×kp) の割合
  vert_out_rate       : down_rate + up_rate（垂直方向の総はみ出し率）
  down_p50 / up_p50   : はみ出し量の中央値 (mm)
  z_med_off_mm        : 全 kp の Z 中央値 − bbox 中心 (mm)。+で上寄り
  floor_off_mm        : 床接地 kp の高さ中央値 − 床(0.7337m) (mm)。+で浮き気味

使い方:
  python3 scripts/dev/outliers/blender_bbox/rank_vertical_offset.py
  python3 scripts/dev/outliers/blender_bbox/rank_vertical_offset.py --csv-dir <dir> --glob '<pat>'

出力 → <csv-dir>/../blender_bbox_outliers/
  vertical_offset_by_session.csv
  vertical_offset_by_session.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_blender_bbox_outlier_dist import (   # noqa: E402
    KP_KEYS, BBOX_LO, BBOX_HI, BLENDER_CENTER, MM, detect_scale, session_label,
    position_edges,
)

EZ = position_edges()[2]                 # 垂直 Z のビン端 (position_dist と共通)
CZ = 0.5 * (EZ[:-1] + EZ[1:])

FLOOR_KP = ["l_wrist", "r_wrist", "l_ankle", "r_ankle"]
FLOOR_Z  = BBOX_LO[2]    # 0.7337 m
CEIL_Z   = BBOX_HI[2]    # 1.5016 m

DEF_CSV_DIR = Path("scripts/dev/floorZ_files/corrected")
DEF_OUTPUT  = Path("scripts/dev/outliers_plot")
DEF_GLOB    = "*_corrected.csv"


def lowest_floor_peak(floor_val: np.ndarray) -> float:
    """床接地 z 分布の最下 KDE ピーク (m)。validate_affine_correction.py と同じ。

    これが「床較正」の指標。median は滞在時間に引きずられて足場高さに寄るため
    較正のずれを測るには不適。
    """
    lo, hi = np.percentile(floor_val, [1, 99])
    v = floor_val[(floor_val >= lo) & (floor_val <= hi)]
    if len(v) < 10:
        return float(np.median(v))
    kde = gaussian_kde(v, bw_method=0.03)
    xg = np.linspace(v.min(), v.max(), 2000)
    d = kde(xg)
    md = max(1, int(0.10 / (xg[1] - xg[0])))
    idx, _ = find_peaks(d, height=d.max() * 0.03, distance=md)
    return float(xg[idx[0]]) if len(idx) else float(xg[np.argmax(d)])


def analyze_vertical(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    scale = detect_scale(df)

    z_cols = [f"pts_{kp}_z" for kp in KP_KEYS if f"pts_{kp}_z" in df.columns]
    Z = df[z_cols].to_numpy(float) * scale            # (T, n_kp) m
    n_eval = Z.size

    down = np.clip(FLOOR_Z - Z, 0.0, None) * MM        # 床より下 (mm)
    up   = np.clip(Z - CEIL_Z, 0.0, None) * MM         # 天井より上 (mm)
    down_ev = down[down > 0]; up_ev = up[up > 0]

    # 床接地 kp の高さ (フレームごとの最小 Z) の中央値
    fcols = [f"pts_{kp}_z" for kp in FLOOR_KP if f"pts_{kp}_z" in df.columns]
    floor_val = (df[fcols].to_numpy(float) * scale).min(axis=1) if fcols else Z.min(axis=1)
    floor_med  = float(np.median(floor_val))            # 滞在時間依存 (occupancy)
    floor_peak = lowest_floor_peak(floor_val)           # 床較正 (calibration)
    pct_near_floor = float(np.mean(np.abs(floor_val - FLOOR_Z) < 0.06))

    # 垂直 Z 分布 (全 kp, 正規化済み density) — position_dist と同じビン
    zhist, _ = np.histogram(Z.ravel(), bins=EZ)
    zhist = zhist / max(1, zhist.sum())

    return {
        "zhist": zhist,
        "session":       session_label(csv_path),
        "animal":        session_label(csv_path).split("_")[0],
        "scale":         scale,
        "n_frames":      len(df),
        "down_rate":     len(down_ev) / max(1, n_eval),
        "up_rate":       len(up_ev)   / max(1, n_eval),
        "vert_out_rate": (len(down_ev) + len(up_ev)) / max(1, n_eval),
        "down_p50":      float(np.median(down_ev)) if len(down_ev) else 0.0,
        "up_p50":        float(np.median(up_ev))   if len(up_ev)   else 0.0,
        "z_med_off_mm":  (float(np.median(Z)) - BLENDER_CENTER[2]) * MM,
        # 床較正のずれ (最下 KDE ピーク基準, validate と同義)。これが本来の指標。
        "floor_peak_off_mm": (floor_peak - FLOOR_Z) * MM,
        # 足の高さ中央値のずれ (occupancy: 滞在時間に依存, 較正指標ではない)
        "floor_med_off_mm":  (floor_med - FLOOR_Z) * MM,
        "pct_near_floor":    pct_near_floor,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", "--csv-dir", dest="input_dir", type=Path, default=DEF_CSV_DIR,
                    help=f"補正済み CSV のディレクトリ (default: {DEF_CSV_DIR})")
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=DEF_OUTPUT,
                    help=f"出力先 (CSV+PNG) (default: {DEF_OUTPUT})")
    ap.add_argument("--session", default=None, help="セッション名 (前方一致) で絞る")
    ap.add_argument("--glob", default=DEF_GLOB)
    ap.add_argument("--sort-by", default="abs_floor_peak",
                    choices=["abs_floor_peak", "vert_out_rate", "down_rate",
                             "up_rate", "abs_floor_med", "abs_z_off"],
                    help="ランキングのソートキー (既定=床較正ずれ abs_floor_peak)")
    ap.add_argument("--top", type=int, default=30, help="表示する上位件数")
    args = ap.parse_args()
    args.csv_dir = args.input_dir
    args.out_dir = args.output_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.csv_dir.glob(args.glob))
    if args.session:
        files = [f for f in files if f.name.startswith(args.session)]
    if not files:
        raise SystemExit(f"CSV が見つかりません: {args.csv_dir}/{args.glob}")
    print(f"Analyzing {len(files)} session(s) …")

    rows = []
    zhists = {}
    for i, fp in enumerate(files):
        r = analyze_vertical(fp)
        zhists[r["session"]] = r.pop("zhist")
        rows.append(r)
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(files)}]")
    dfm = pd.DataFrame(rows)

    sort_col = {"abs_floor_peak": "floor_peak_off_mm",
                "abs_floor_med":  "floor_med_off_mm",
                "abs_z_off":      "z_med_off_mm"}.get(args.sort_by, args.sort_by)
    key = dfm[sort_col].abs() if args.sort_by.startswith("abs_") else dfm[sort_col]
    dfm = dfm.assign(_key=key).sort_values("_key", ascending=False).drop(columns="_key")

    csv_out = args.out_dir / "vertical_offset_by_session.csv"
    dfm.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    n_miscal = int((dfm["floor_peak_off_mm"].abs() > 30).sum())
    print(f"\n床較正ずれ |floor_peak_off| > 30mm のセッション: {n_miscal} / {len(dfm)}")

    # ── print top table ───────────────────────────────────────────────────────
    print(f"\nWorst {args.top} sessions by {args.sort_by}:")
    print(f"  {'session':28} {'floor_peak':>10} {'floor_med':>10} "
          f"{'%nearFlr':>8} {'vert_out%':>9} {'up%':>7}")
    for _, r in dfm.head(args.top).iterrows():
        print(f"  {r['session']:28} {r['floor_peak_off_mm']:>+9.1f} "
              f"{r['floor_med_off_mm']:>+9.1f} {r['pct_near_floor']*100:>7.1f}% "
              f"{r['vert_out_rate']*100:>8.2f}% {r['up_rate']*100:>6.2f}%")

    # ── 垂直分布の重ね描き図 (position_dist と同じ Z ヒストグラム形式) ─────────
    plot_vertical_overlay(dfm, zhists, args.sort_by, len(files), args.out_dir)

    # ── figure ────────────────────────────────────────────────────────────────
    inds = sorted(dfm["animal"].unique())
    cmap = plt.get_cmap("tab10")
    icolor = {a: cmap(i % 10) for i, a in enumerate(inds)}

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f"Vertical (Z) misalignment by session  ·  sorted by {args.sort_by}  "
                 f"·  {len(files)} sessions", fontsize=13, y=0.995)

    # panel 1: stacked down/up rate, sorted (worst → best)
    d = dfm.sort_values("vert_out_rate", ascending=False).reset_index(drop=True)
    x = np.arange(len(d))
    ax = axes[0]
    ax.bar(x, d["down_rate"] * 100, color="#1f77b4", label="down (below floor)")
    ax.bar(x, d["up_rate"] * 100, bottom=d["down_rate"] * 100,
           color="#d62728", label="up (above ceiling)")
    ax.set_ylabel("vertical-outside rate (%)")
    ax.set_title("Per-session vertical outside-bbox rate (sorted, worst left)")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    # annotate worst few
    for i in range(min(8, len(d))):
        ax.text(x[i], d["vert_out_rate"].iloc[i] * 100, d["session"].iloc[i],
                rotation=90, fontsize=6, ha="center", va="bottom")
    ax.set_xlim(-1, len(d))

    # panel 2: 床較正ずれ (最下KDEピーク基準) — これが本来の calibration 指標
    d2 = dfm.sort_values("floor_peak_off_mm").reset_index(drop=True)
    x2 = np.arange(len(d2))
    ax = axes[1]
    ax.bar(x2, d2["floor_peak_off_mm"],
           color=[icolor[a] for a in d2["animal"]])
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-30, 30, color="green", alpha=0.12, label="±30mm (aligned)")
    ax.set_ylabel("floor calibration offset (mm)\n(lowest KDE peak − floor)")
    ax.set_xlabel("session (sorted by calibration offset)")
    ax.set_title("Floor CALIBRATION offset = lowest floor-KDE-peak − 0.7337 m  "
                 "(validate と同義; ほぼ全 session が ±30mm 内なら較正OK)")
    ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=icolor[a]) for a in inds]
    ax.legend(handles, inds, fontsize=8, title="animal", ncol=min(len(inds), 6))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:+.0f}"))

    fig.tight_layout(pad=2.0)
    png_out = args.out_dir / "vertical_offset_by_session.png"
    fig.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_out}")


def plot_vertical_overlay(dfm, zhists, sort_by, n_sessions, out_dir, n_high=8):
    """各セッションの垂直 Z 分布を重ね描きし、ずれの大きい上位を強調する。

    position_dist.png の Vertical 分布パネルと同じ形式 (Z ヒスト + 床/天井ライン)。
    """
    key_col = {"abs_floor_peak": "floor_peak_off_mm",
               "abs_floor_med":  "floor_med_off_mm",
               "abs_z_off":      "z_med_off_mm"}.get(sort_by, sort_by)
    ranked = dfm.copy()
    ranked["_k"] = ranked[key_col].abs() if sort_by.startswith("abs_") else ranked[key_col]
    ranked = ranked.sort_values("_k", ascending=False)
    worst = ranked.head(n_high)["session"].tolist()

    # 全セッション平均 (参考線)
    mean_h = np.mean([zhists[s] for s in zhists], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Per-session vertical (Z) distribution  ·  "
                 f"{n_sessions} sessions  ·  worst {n_high} by {sort_by} highlighted",
                 fontsize=13, y=1.0)

    cmap = plt.get_cmap("autumn")
    for ax, (title, logy) in zip(axes, [("linear", False), ("log y", True)]):
        # 全セッションを薄いグレーで
        for s, h in zhists.items():
            ax.plot(CZ, h, color="0.8", lw=0.5, alpha=0.5, zorder=1)
        ax.plot(CZ, mean_h, color="black", lw=1.8, label="all-session mean", zorder=3)
        # 上位を強調
        for j, s in enumerate(worst):
            r = dfm[dfm["session"] == s].iloc[0]
            ax.plot(CZ, zhists[s], color=cmap(j / max(1, n_high - 1)), lw=1.6,
                    zorder=4,
                    label=f"{s}  (cal{r['floor_peak_off_mm']:+.0f}mm, "
                          f"med{r['floor_med_off_mm']:+.0f}mm)")
        ax.axvline(FLOOR_Z, color="orange", ls="--", lw=1.4,
                   label=f"floor {FLOOR_Z:.3f} / ceiling {CEIL_Z:.3f} m")
        ax.axvline(CEIL_Z, color="orange", ls="--", lw=1.4)
        ax.set_xlabel("Z (m, vertical)"); ax.set_ylabel("fraction per bin")
        ax.set_title(f"Z distribution ({title})")
        ax.grid(alpha=0.3)
        if logy:
            ax.set_yscale("log")
        else:
            ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout(pad=1.5)
    out = out_dir / "vertical_dist_by_session.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

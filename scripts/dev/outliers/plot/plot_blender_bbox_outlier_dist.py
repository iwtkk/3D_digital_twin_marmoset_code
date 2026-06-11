"""
Plot distribution of Blender-bbox outlier deviations across corrected CSVs.

pkl_to_corrected_csv.py で生成した補正済み CSV
(outputs/dev/affine_pkl_csv/corrected/{session}_corrected.csv, Blender m 空間)
の各 (frame, kp) が Blender bbox の外にどれだけはみ出すかを調べ、
plot_outlier_deviation_dist.py と同様の分布図を出力する。

Blender bbox (pkl_to_corrected_csv.py と同一):
  center = BLENDER_CENTER, full extent = BLENDER_BBOX_SCALE
  → bounds = center ± scale/2
    X: [-0.4333, 0.4333]  Y: [-0.5554, 0.3325]  Z: [0.7337, 1.5016] (m)

deviation は Blender m 空間で計算し、読みやすさのため mm (×1000) に変換して描画する。

使い方:
  python scripts/dev/outliers/blender_bbox/plot_blender_bbox_outlier_dist.py
  python scripts/dev/outliers/blender_bbox/plot_blender_bbox_outlier_dist.py --session cj425_20210427_120000
  # 別ディレクトリの補正済み CSV を対象にする
  python scripts/dev/outliers/blender_bbox/plot_blender_bbox_outlier_dist.py --csv-dir path/to/corrected

出力 → <csv-dir>/../blender_bbox_outliers/  (--out-dir で上書き可)
  deviation_dist.png

図の各パネルの説明は同ディレクトリの README.md を参照。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

# ── Blender bbox (pkl_to_corrected_csv.py と同一) ─────────────────────────────
BLENDER_BBOX_SCALE = np.array([0.86658, 0.88791, 0.7679])
BLENDER_CENTER     = np.array([0.0, -0.11142, 1.11765])
HALF = BLENDER_BBOX_SCALE / 2
BBOX_LO = BLENDER_CENTER - HALF   # (3,)  m
BBOX_HI = BLENDER_CENTER + HALF   # (3,)  m

MM = 1000.0   # m → mm 変換 (描画用)

# 方向別はみ出し: up=天井より上(+Z), down=床より下(-Z), horizontal=X/Y 面内
DIRECTIONS = ["up", "down", "horizontal"]
DIR_LABELS = {
    "up":         "up (above ceiling, +Z)",
    "down":       "down (below floor, -Z)",
    "horizontal": "horizontal (X/Y)",
}
DIR_COLORS = {"up": "#d62728", "down": "#1f77b4", "horizontal": "#2ca02c"}

# 位置分布プロット用 (bbox とは無関係に全 kp 座標を集計)。
# 表示レンジは bbox を POS_PAD_M だけ広げた範囲 (範囲外の点はヒストグラムに入らない)。
POS_PAD_M = 0.4
POS_NBINS = 150

# 元の 20 キーポイントのみ評価 (派生点 shoulder_mid/base/center_eye は除外)
KP_KEYS = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
    "backbone_peak", "tail_1", "tail_2",
]

CSV_DIR = Path("scripts/dev/floorZ_files/corrected")
DEFAULT_OUTPUT = Path("scripts/dev/outliers_plot")


def excess(vals, lo, hi):
    """bbox 境界からのはみ出し量 (内側は 0)。"""
    below = np.minimum(vals - lo, 0.0)   # lo を下回ると負
    above = np.maximum(vals - hi, 0.0)   # hi を上回ると正
    return below + above


def detect_scale(df: pd.DataFrame) -> float:
    """
    補正済み CSV は pkl_to_corrected_csv.py の OUTPUT_SCALE (1/1 または 1/1000) で
    書き出されている。CSV 座標を Blender m に戻す係数を z 分布から自動推定する。
    z の中央値が BLENDER_CENTER[z] に最も近づく候補係数を返す。
    """
    z_cols = [f"pts_{kp}_z" for kp in KP_KEYS if f"pts_{kp}_z" in df.columns]
    z_med = float(np.median(df[z_cols].to_numpy(float)))
    candidates = [1.0, 1000.0]   # m そのまま / ×1000 (mm 相当で保存) の逆変換
    factor = min(candidates, key=lambda f: abs(z_med * f - BLENDER_CENTER[2]))
    return factor


def session_label(path: Path) -> str:
    """CSV ファイル名からセッション名を取り出す (既知のサフィックスを除去)。"""
    name = path.name
    for suf in ("_corrected.csv", "_points.csv_transformedtest.csv",
                "_points.csv", ".csv"):
        if name.endswith(suf):
            return name[:-len(suf)]
    return path.stem


# ── 位置ヒストグラム (bbox 無関係, 全点を逐次集計) ────────────────────────────
def position_edges():
    """各軸のビン端 (Blender m)。bbox を POS_PAD_M 広げた範囲。"""
    return [np.linspace(BBOX_LO[j] - POS_PAD_M, BBOX_HI[j] + POS_PAD_M,
                        POS_NBINS + 1) for j in range(3)]


def new_position_hists():
    z2 = (POS_NBINS, POS_NBINS)
    return {"x": np.zeros(POS_NBINS), "y": np.zeros(POS_NBINS), "z": np.zeros(POS_NBINS),
            "xy": np.zeros(z2), "xz": np.zeros(z2)}


def add_positions(H, pos, edges):
    """位置配列 (M,3) をヒストグラム dict H に加算する。"""
    ex, ey, ez = edges
    H["x"]  += np.histogram(pos[:, 0], bins=ex)[0]
    H["y"]  += np.histogram(pos[:, 1], bins=ey)[0]
    H["z"]  += np.histogram(pos[:, 2], bins=ez)[0]
    H["xy"] += np.histogram2d(pos[:, 0], pos[:, 1], bins=[ex, ey])[0]
    H["xz"] += np.histogram2d(pos[:, 0], pos[:, 2], bins=[ex, ez])[0]


def load_positions(csv_path: Path) -> np.ndarray:
    """CSV を読み 20 kp の全座標 (M,3) を Blender m で返す (位置プロット専用)。"""
    df = pd.read_csv(csv_path)
    scale = detect_scale(df)
    arrs = []
    for kp in KP_KEYS:
        c = [f"pts_{kp}_{a}" for a in "xyz"]
        if all(x in df.columns for x in c):
            arrs.append(df[c].to_numpy(float))
    return np.vstack(arrs) * scale if arrs else np.empty((0, 3))


def accumulate_position_hists(files):
    """ファイル群から位置ヒストグラムを集計して (H, n_pos) を返す。"""
    edges = position_edges()
    H = new_position_hists()
    n_pos = 0
    for fp in files:
        pos = load_positions(fp)
        add_positions(H, pos, edges)
        n_pos += len(pos)
    return H, n_pos


def analyze_csv(csv_path: Path):
    """1 セッション分の解析。

    Returns
    -------
    devs_all : (N,)        全 kp のユークリッドはみ出し量 (mm, 外れイベントのみ)
    dev_kp   : {kp: (n,)}  kp ごとのユークリッドはみ出し量 (mm)
    dir_kp   : {direction: {kp: (n,)}}  方向別はみ出し量 (mm, その方向に出たイベントのみ)
    pos      : (M, 3)      全 kp の座標 (Blender m, bbox 内外を問わず全点)
    stats    : dict
    """
    df = pd.read_csv(csv_path)
    T = len(df)
    scale = detect_scale(df)   # CSV 座標 → Blender m への係数

    dev_kp = {}          # kp_key -> deviation_mm 配列 (外れフレームのみ)
    dir_kp = {d: {} for d in DIRECTIONS}   # direction -> kp -> deviation_mm 配列
    devs_all = []
    pos_list = []        # 全 kp 座標 (Blender m)
    n_events = 0
    for kp in KP_KEYS:
        cols = [f"pts_{kp}_{a}" for a in "xyz"]
        if not all(c in df.columns for c in cols):
            continue
        xyz = df[cols].to_numpy(float) * scale      # (T, 3) Blender m
        pos_list.append(xyz)
        dev_axis = np.stack([
            excess(xyz[:, j], BBOX_LO[j], BBOX_HI[j]) for j in range(3)
        ], axis=1)                                  # (T, 3) 符号付きはみ出し (m)
        out_mask = np.any(dev_axis != 0.0, axis=1)  # (T,)
        dev_mag_mm = np.linalg.norm(dev_axis, axis=1) * MM
        ev = dev_mag_mm[out_mask]
        dev_kp[kp] = ev
        devs_all.append(ev)
        n_events += int(out_mask.sum())

        # 方向別: dev_z<0 → 床より下(down), dev_z>0 → 天井より上(up)
        up_mm    = np.clip(dev_axis[:, 2],  0.0, None) * MM
        down_mm  = np.clip(-dev_axis[:, 2], 0.0, None) * MM
        horiz_mm = np.linalg.norm(dev_axis[:, :2], axis=1) * MM
        for d, arr in (("up", up_mm), ("down", down_mm), ("horizontal", horiz_mm)):
            dir_kp[d][kp] = arr[arr > 0.0]

    devs_all = np.concatenate(devs_all) if devs_all else np.array([])
    stats = {
        "session": session_label(csv_path),
        "scale": scale,
        "n_frames": T,
        "n_kp_evaluated": len(KP_KEYS) * T,
        "n_outlier_events": n_events,
        "outlier_rate": n_events / max(1, len(KP_KEYS) * T),
        "n_dir_events": {d: int(sum(len(v) for v in dir_kp[d].values()))
                         for d in DIRECTIONS},
    }
    pos = np.vstack(pos_list) if pos_list else np.empty((0, 3))
    return devs_all, dev_kp, dir_kp, pos, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", "--csv-dir", dest="input_dir", type=Path, default=CSV_DIR,
                    help=f"補正済み CSV のディレクトリ (default: {CSV_DIR})")
    ap.add_argument("--output-dir", "--out-dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"図の出力先 (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None,
                    help="単一セッション名 (省略時は --input-dir 内の全 CSV)")
    ap.add_argument("--glob", default="*_corrected.csv",
                    help="対象 CSV のグロブパターン (default: *_corrected.csv). "
                         "補正前は '*_transformedtest.csv' を指定")
    args = ap.parse_args()
    args.csv_dir = args.input_dir
    args.out_dir = args.output_dir

    files = sorted(args.csv_dir.glob(args.glob))
    if args.session:
        files = [f for f in files if f.name.startswith(args.session)]
    files = [f for f in files if f.exists()]
    if not files:
        raise SystemExit(f"補正済み CSV が見つかりません: {args.csv_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(files)} corrected CSV(s) …")
    devs_all_list = []
    devs_kp = defaultdict(list)
    devs_ind = defaultdict(list)
    dir_all = {d: [] for d in DIRECTIONS}              # direction -> [arrays]
    dir_all_kp = {d: defaultdict(list) for d in DIRECTIONS}  # direction -> kp -> [arrays]
    # 位置ヒストグラム (bbox 無関係, 全点を逐次集計)
    pos_edges = position_edges()
    H = new_position_hists()
    n_pos_total = 0
    n_frames_total = 0
    n_events_total = 0
    for fp in files:
        ev_all, ev_kp, ev_dir_kp, pos, st = analyze_csv(fp)
        ind = st["session"].split("_")[0]
        devs_all_list.append(ev_all)
        for kp, arr in ev_kp.items():
            devs_kp[kp].append(arr)
        devs_ind[ind].append(ev_all)
        for d in DIRECTIONS:
            for kp, arr in ev_dir_kp[d].items():
                dir_all[d].append(arr)
                dir_all_kp[d][kp].append(arr)
        # 位置ヒストグラムに加算してから pos は破棄 (メモリ節約)
        add_positions(H, pos, pos_edges)
        n_pos_total += len(pos)
        n_frames_total += st["n_frames"]
        n_events_total += st["n_outlier_events"]
        nd = st["n_dir_events"]
        print(f"  {st['session']:30}  scale×{st['scale']:>6.0f}  "
              f"frames={st['n_frames']:>7,}  "
              f"outlier_events={st['n_outlier_events']:>8,}  "
              f"rate={st['outlier_rate']*100:6.2f}%  "
              f"[up={nd['up']:,} down={nd['down']:,} horiz={nd['horizontal']:,}]")

    devs_all = np.concatenate(devs_all_list) if devs_all_list else np.array([])
    devs_kp = {k: (np.concatenate(v) if v else np.array([])) for k, v in devs_kp.items()}
    devs_ind = {k: (np.concatenate(v) if v else np.array([])) for k, v in devs_ind.items()}
    dir_all = {d: (np.concatenate(v) if v else np.array([])) for d, v in dir_all.items()}
    dir_all_kp = {d: {k: (np.concatenate(v) if v else np.array([]))
                      for k, v in dkp.items()}
                  for d, dkp in dir_all_kp.items()}

    n_eval_total = len(KP_KEYS) * n_frames_total
    print(f"\nTotal evaluated (frame×kp): {n_eval_total:,}")
    print(f"Total outlier events:       {n_events_total:,} "
          f"({n_events_total/max(1,n_eval_total)*100:.2f}% of all frame×kp)")

    if len(devs_all) == 0:
        print("外れ値イベントなし。図は生成しません。")
        return

    # ── stats ────────────────────────────────────────────────────────────────
    print("\nDeviation stats over outlier events (mm):")
    for p in [25, 50, 75, 90, 95, 99, 99.9, 100]:
        print(f"  p{p:5.1f}: {np.percentile(devs_all, p):.2f}")

    bins_lo = [0, 10, 20, 50, 100, 200, 500, 1000, 5000, 20000]
    counts, _ = np.histogram(devs_all, bins=bins_lo)
    print("\nBin counts (mm):")
    for lo, hi, cnt in zip(bins_lo, bins_lo[1:], counts):
        print(f"  [{lo:>6}, {hi:>6}) : {cnt:>10,}  ({cnt/len(devs_all)*100:.2f}%)")

    # ── directional stats ─────────────────────────────────────────────────────
    print("\nDirectional excess (events outside in that direction, mm):")
    n_dir_total = sum(len(dir_all[d]) for d in DIRECTIONS)
    print(f"  {'direction':12} {'N events':>12} {'% of dir':>9} "
          f"{'p50':>7} {'p95':>7} {'p99':>7} {'max':>8}")
    for d in DIRECTIONS:
        a = dir_all[d]
        if len(a) == 0:
            print(f"  {DIR_LABELS[d]:12} {0:>12,}")
            continue
        print(f"  {d:12} {len(a):>12,} {len(a)/max(1,n_dir_total)*100:>8.1f}% "
              f"{np.percentile(a,50):>7.1f} {np.percentile(a,95):>7.1f} "
              f"{np.percentile(a,99):>7.1f} {a.max():>8.1f}")

    p50 = float(np.percentile(devs_all, 50))
    p95 = float(np.percentile(devs_all, 95))
    p99 = float(np.percentile(devs_all, 99))
    near = devs_all[devs_all <= 200]
    far  = devs_all[devs_all > 200]

    inds = sorted(devs_ind.keys())
    cmap = plt.get_cmap("tab10")
    colors_ind = {ind: cmap(i % 10) for i, ind in enumerate(inds)}

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(18, 20),
                             gridspec_kw={"height_ratios": [2, 2, 1.5, 2.5]})
    fig.suptitle("kp deviation from Blender bbox boundary "
                 f"(corrected CSV, {len(files)} session(s))",
                 fontsize=14, y=0.995)

    # row 0 left: log-x hist (all)
    ax = axes[0, 0]
    log_bins = np.logspace(np.log10(max(devs_all.min(), 0.1)),
                           np.log10(devs_all.max() + 1), 100)
    ax.hist(devs_all, bins=log_bins, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlabel("Deviation from Blender bbox (mm) — log scale")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"All outlier events  ·  N = {len(devs_all):,}")
    for v, color, label in [(p50, "green", f"p50 = {p50:.0f} mm"),
                            (p95, "orange", f"p95 = {p95:.0f} mm"),
                            (p99, "red",    f"p99 = {p99:.0f} mm")]:
        ax.axvline(v, color=color, lw=1.4, ls="--", label=label)
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)

    # row 0 right: per-individual log-x hist
    ax = axes[0, 1]
    log_bins_ind = np.logspace(np.log10(max(devs_all.min(), 0.1)),
                               np.log10(devs_all.max() + 1), 80)
    for ind in inds:
        d = devs_ind[ind]
        if len(d) == 0:
            continue
        ax.hist(d, bins=log_bins_ind, color=colors_ind[ind], alpha=0.55,
                label=f"{ind}  (N={len(d):,})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlabel("Deviation (mm) — log scale")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Per individual")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)

    # row 1 left: zoom 0–200 mm
    ax = axes[1, 0]
    ax.hist(near, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Deviation (mm)"); ax.set_ylabel("Count")
    ax.set_title(f"Zoom: 0–200 mm  ·  N = {len(near):,}  "
                 f"({len(near)/len(devs_all)*100:.1f}%)")
    ax.axvline(p50, color="green",  lw=1.4, ls="--", label=f"p50 = {p50:.0f} mm")
    ax.axvline(p95, color="orange", lw=1.4, ls="--", label=f"p95 = {p95:.0f} mm")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # row 1 right: far > 200 mm
    ax = axes[1, 1]
    if len(far) > 0:
        ax.hist(far, bins=100, color="tomato", edgecolor="none", alpha=0.85)
        ax.set_xlabel("Deviation (mm)"); ax.set_ylabel("Count")
        ax.set_title(f"Far outliers: >200 mm  ·  N = {len(far):,}  "
                     f"({len(far)/len(devs_all)*100:.2f}%)")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.set_visible(False)

    # row 2 left: per-individual near vs far bar
    ax = axes[2, 0]
    x = np.arange(len(inds)); w = 0.35
    ind_near = [int((devs_ind[i] <= 200).sum()) for i in inds]
    ind_far  = [int((devs_ind[i] >  200).sum()) for i in inds]
    bars_n = ax.bar(x - w/2, ind_near, w, label="near (0–200 mm)",
                    color=[colors_ind[i] for i in inds], alpha=0.8)
    bars_f = ax.bar(x + w/2, ind_far, w, label="far (>200 mm)",
                    color=[colors_ind[i] for i in inds], alpha=0.4, hatch="///")
    ax.set_xticks(x); ax.set_xticklabels(inds)
    ax.set_ylabel("N outlier events")
    ax.set_title("Near vs far outlier events per individual")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    for bar in list(bars_n) + list(bars_f):
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7)

    # row 2 right: CDF of near outliers
    ax = axes[2, 1]
    near_sorted = np.sort(near)
    cdf = np.arange(1, len(near_sorted) + 1) / len(devs_all) * 100
    ax.plot(near_sorted, cdf, color="steelblue", lw=1.5)
    ax.set_xlabel("Deviation (mm)")
    ax.set_ylabel("Cumulative % of all outlier events")
    ax.set_title("CDF — near outliers (as % of all outlier events)")
    ax.axhline(95, color="orange", ls="--", lw=1, label=f"95% → {p95:.0f} mm")
    ax.axhline(97.5, color="red", ls="--", lw=1,
               label=f"97.5% → {np.percentile(devs_all,97.5):.0f} mm")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 200)

    # row 3: per-kp boxplot (capped 200 mm)
    ax = fig.add_subplot(4, 1, 4)
    axes[3, 0].set_visible(False); axes[3, 1].set_visible(False)
    kp_data = [devs_kp.get(k, np.array([])) for k in KP_KEYS]
    kp_data_cap = [np.minimum(d, 200) if len(d) else np.array([0.0]) for d in kp_data]
    bp = ax.boxplot(kp_data_cap, patch_artist=True, showfliers=False,
                    medianprops=dict(color="red", lw=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue"); patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(KP_KEYS) + 1))
    ax.set_xticklabels(KP_KEYS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Deviation (mm, capped at 200)")
    ax.set_title("Per-keypoint deviation distribution "
                 "(fliers hidden, capped at 200 mm)")
    for i, d in enumerate(kp_data):
        if len(d):
            m = float(np.median(d))
            ax.text(i + 1, min(m, 200) + 1.5, f"{m:.0f}",
                    ha="center", va="bottom", fontsize=7, color="darkred")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=2.0)
    out_path = args.out_dir / "deviation_dist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # ── 方向別の図 ────────────────────────────────────────────────────────────
    plot_by_direction(dir_all, dir_all_kp, len(files), args.out_dir)

    # ── 位置分布の図 (bbox 無関係) ────────────────────────────────────────────
    ex, ey, ez = pos_edges
    plot_positions(H["x"], H["y"], H["z"], H["xy"], H["xz"], pos_edges,
                   n_pos_total, len(files), args.out_dir)
    # 補正前後比較などに再利用できるよう位置ヒストグラムを保存
    npz_path = args.out_dir / "position_hist.npz"
    np.savez_compressed(npz_path, x=H["x"], y=H["y"], z=H["z"],
                        xy=H["xy"], xz=H["xz"], ex=ex, ey=ey, ez=ez,
                        n_pos=n_pos_total, n_sessions=len(files))
    print(f"Saved: {npz_path}")


def plot_by_direction(dir_all, dir_all_kp, n_sessions, out_dir):
    """up / down / horizontal のはみ出しを分けて比較する図を出力する。"""
    active = [d for d in DIRECTIONS if len(dir_all[d]) > 0]
    if not active:
        print("方向別イベントなし。direction 図はスキップ。")
        return
    vmax = max(dir_all[d].max() for d in active)
    log_bins = np.logspace(np.log10(0.1), np.log10(vmax + 1), 80)
    n_dir_total = sum(len(dir_all[d]) for d in active)

    fig, axes = plt.subplots(3, 2, figsize=(18, 16),
                             gridspec_kw={"height_ratios": [2, 2, 2.5]})
    fig.suptitle("kp deviation from Blender bbox — by direction "
                 f"(corrected CSV, {n_sessions} session(s))", fontsize=14, y=0.995)

    # row 0 left: log-log overlay histogram
    ax = axes[0, 0]
    for d in active:
        a = dir_all[d]
        ax.hist(a, bins=log_bins, histtype="step", lw=1.8,
                color=DIR_COLORS[d],
                label=f"{DIR_LABELS[d]}  (N={len(a):,}, p50={np.median(a):.1f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlabel("Deviation (mm) — log scale"); ax.set_ylabel("Count (log scale)")
    ax.set_title("Deviation distribution by direction")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)

    # row 0 right: event-count bars (with % of all directional events)
    ax = axes[0, 1]
    counts = [len(dir_all[d]) for d in DIRECTIONS]
    bars = ax.bar(range(len(DIRECTIONS)), counts,
                  color=[DIR_COLORS[d] for d in DIRECTIONS], alpha=0.85)
    ax.set_xticks(range(len(DIRECTIONS)))
    ax.set_xticklabels([DIR_LABELS[d] for d in DIRECTIONS], fontsize=9)
    ax.set_ylabel("N outlier events")
    ax.set_title("Outlier event count per direction")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{c:,}\n({c/max(1,n_dir_total)*100:.1f}%)",
                ha="center", va="bottom", fontsize=8)

    # row 1 left: zoom 0–200 mm overlay (linear)
    ax = axes[1, 0]
    for d in active:
        a = dir_all[d]
        ax.hist(a[a <= 200], bins=np.linspace(0, 200, 80), histtype="step",
                lw=1.8, color=DIR_COLORS[d], label=DIR_LABELS[d])
    ax.set_xlabel("Deviation (mm)"); ax.set_ylabel("Count")
    ax.set_title("Zoom: 0–200 mm")
    ax.set_xlim(0, 200); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # row 1 right: CDF overlay (each normalized to its own count)
    ax = axes[1, 1]
    for d in active:
        a = np.sort(dir_all[d])
        cdf = np.arange(1, len(a) + 1) / len(a) * 100
        ax.plot(a, cdf, lw=1.8, color=DIR_COLORS[d], label=DIR_LABELS[d])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_xlabel("Deviation (mm) — log scale")
    ax.set_ylabel("Cumulative % (within direction)")
    ax.set_title("CDF per direction"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # row 2: per-keypoint stacked event counts by direction
    ax = fig.add_subplot(3, 1, 3)
    axes[2, 0].set_visible(False); axes[2, 1].set_visible(False)
    x = np.arange(len(KP_KEYS))
    bottom = np.zeros(len(KP_KEYS))
    for d in DIRECTIONS:
        vals = np.array([len(dir_all_kp[d].get(k, [])) for k in KP_KEYS])
        ax.bar(x, vals, bottom=bottom, color=DIR_COLORS[d],
               alpha=0.85, label=DIR_LABELS[d])
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(KP_KEYS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("N outlier events")
    ax.set_title("Per-keypoint outlier event count, stacked by direction")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=2.0)
    out_path = out_dir / "deviation_dist_by_direction.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_positions(hx, hy, hz, h_xy, h_xz, edges, n_pos, n_sessions, out_dir):
    """bbox 無関係に全 kp 座標の水平/垂直分布を描き、bbox 範囲を重ねる。

    座標は Blender m。表示レンジ外の点はヒストグラムに含まれない。
    """
    ex, ey, ez = edges
    cx = 0.5 * (ex[:-1] + ex[1:])
    cy = 0.5 * (ey[:-1] + ey[1:])
    cz = 0.5 * (ez[:-1] + ez[1:])

    def bbox_rect(j_horiz, j_vert):
        lo_h, hi_h = BBOX_LO[j_horiz], BBOX_HI[j_horiz]
        lo_v, hi_v = BBOX_LO[j_vert], BBOX_HI[j_vert]
        return Rectangle((lo_h, lo_v), hi_h - lo_h, hi_v - lo_v,
                         fill=False, ec="red", lw=2.0, ls="--", label="Blender bbox")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("kp position distribution (all points, bbox-independent)  "
                 f"·  {n_sessions} session(s), N = {n_pos:,}", fontsize=14, y=0.995)

    # (0,0) 上面図 (X–Y): 水平面
    ax = axes[0, 0]
    pc = ax.pcolormesh(ex, ey, h_xy.T, norm=LogNorm(vmin=1), cmap="viridis")
    fig.colorbar(pc, ax=ax, label="count (log)")
    ax.add_patch(bbox_rect(0, 1))
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Top view  X–Y (horizontal plane)")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)

    # (0,1) 側面図 (X–Z): 水平 X vs 垂直 Z
    ax = axes[0, 1]
    pc = ax.pcolormesh(ex, ez, h_xz.T, norm=LogNorm(vmin=1), cmap="viridis")
    fig.colorbar(pc, ax=ax, label="count (log)")
    ax.add_patch(bbox_rect(0, 2))
    ax.axhline(BBOX_LO[2], color="orange", ls=":", lw=1.2, label="floor / ceiling")
    ax.axhline(BBOX_HI[2], color="orange", ls=":", lw=1.2)
    ax.set_xlabel("X (m, horizontal)"); ax.set_ylabel("Z (m, vertical)")
    ax.set_title("Side view  X–Z (horizontal vs vertical)")
    ax.legend(loc="upper right", fontsize=8)

    # (1,0) 垂直方向 Z ヒストグラム
    ax = axes[1, 0]
    ax.fill_between(cz, hz, step="mid", color="steelblue", alpha=0.85)
    ax.axvline(BBOX_LO[2], color="red", ls="--", lw=1.6,
               label=f"floor = {BBOX_LO[2]:.3f} m")
    ax.axvline(BBOX_HI[2], color="red", ls="--", lw=1.6,
               label=f"ceiling = {BBOX_HI[2]:.3f} m")
    ax.set_xlabel("Z (m, vertical)"); ax.set_ylabel("count")
    ax.set_title("Vertical distribution (Z)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1) 水平方向 X / Y ヒストグラム
    ax = axes[1, 1]
    ax.fill_between(cx, hx, step="mid", color="#2ca02c", alpha=0.5, label="X")
    ax.fill_between(cy, hy, step="mid", color="#9467bd", alpha=0.5, label="Y")
    for j, color in [(0, "#2ca02c"), (1, "#9467bd")]:
        ax.axvline(BBOX_LO[j], color=color, ls="--", lw=1.4)
        ax.axvline(BBOX_HI[j], color=color, ls="--", lw=1.4)
    ax.set_xlabel("Horizontal position (m)"); ax.set_ylabel("count")
    ax.set_title("Horizontal distribution (X, Y)  — dashed = bbox range")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(pad=2.0)
    out_path = out_dir / "position_dist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

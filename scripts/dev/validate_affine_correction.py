"""
Phase 1: CSV ベースで垂直方向 affine 補正を検証する。

入力 : outputs/animate_temp/test_scale/*_scaletest.csv  (bZ in m)
出力 :
  outputs/dev/affine_validation/gmm_peaks_all.csv   -- session × peak 一覧
  outputs/dev/affine_validation/affine_summary.csv  -- session × (a, b, status)
  outputs/dev/affine_validation/height_dist_grid.png -- 全 session 分布グリッド
  outputs/dev/affine_validation/affine_scatter.png   -- affine 係数の分布

物理高さ (Blender bZ, m):
  床=0.7337  足場=1.0179  棒=1.3035  天井=1.5016

ピーク検出: KDE (gaussian_kde) + scipy.signal.find_peaks
  旧実装の GMM は BIC で過剰な k を選び、min_gap フィルタが
  本来のモードではなく端の成分を残すケースがあったため置き換え。
"""
from __future__ import annotations
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# ── 定数 ─────────────────────────────────────────────────────────────────────
PHYSICAL_HEIGHTS_M = np.array([0.7337, 1.0179, 1.3035, 1.5016])  # 床/足場/棒/天井
FLOOR_KP_KEYWORDS  = ("wrist", "ankle")
MIN_PEAK_GAP_M     = 0.10    # 隣接ピーク間の最小距離 (m)
MAX_MATCH_DIST_M   = 0.10    # この距離を超える peak-physical ペアは除外
KDE_BW             = 0.03    # KDE bandwidth (m)
KDE_GRID_N         = 2000    # KDE 評価点数
MAX_AFFINE_SLOPE_DEV = 0.15  # |slope - 1| がこれを超えたら警告

CSV_DIR = Path("outputs/animate_temp/test_scale")
OUT_DIR = Path("outputs/dev/affine_validation")
ANIMAL_COLORS = {"cj425": "steelblue", "cj611": "darkorange", "cj357": "seagreen"}


# ── KDE ピーク検出 ─────────────────────────────────────────────────────────────

def detect_peaks_kde(vals: np.ndarray) -> np.ndarray:
    """
    KDE のローカル最大値をピークとして返す。

    旧 GMM 実装の問題点:
      - BIC が過剰な k を選択して偽ピークが発生
      - min_gap フィルタがクラスタの端の成分を残し本来のモードとずれる
    KDE は分布のモードを直接検出するためこれらの問題が生じない。

    Returns
    -------
    peaks : sorted peak positions (m), min_gap フィルタ適用済み
    """
    if len(vals) < 10:
        return np.array([float(np.median(vals))])

    kde = gaussian_kde(vals, bw_method=KDE_BW)
    xgrid = np.linspace(vals.min(), vals.max(), KDE_GRID_N)
    density = kde(xgrid)

    min_dist_pts = max(1, int(MIN_PEAK_GAP_M / (xgrid[1] - xgrid[0])))
    peak_idx, _ = find_peaks(
        density,
        height=density.max() * 0.03,   # 最大密度の3%以上
        distance=min_dist_pts,
    )

    if len(peak_idx) == 0:
        return np.array([float(xgrid[np.argmax(density)])])

    return np.sort(xgrid[peak_idx])


def fit_affine(peaks: np.ndarray) -> tuple[float, float, str]:
    """
    検出ピークを物理高さにマッピングして affine (slope, intercept) を計算。

    Hungarian algorithm で最小コスト割り当て後、
    MAX_MATCH_DIST_M を超えるペアは除外して回帰する。
    これにより偽ピークが物理高さに強制マッチされるのを防ぐ。

    Returns
    -------
    slope, intercept, status
    """
    from scipy.optimize import linear_sum_assignment

    n = len(peaks)
    if n == 0:
        return 1.0, 0.0, "no_peaks"

    if n == 1:
        nearest_phys = PHYSICAL_HEIGHTS_M[np.argmin(np.abs(PHYSICAL_HEIGHTS_M - peaks[0]))]
        if abs(peaks[0] - nearest_phys) > MAX_MATCH_DIST_M:
            return 1.0, 0.0, "no_peaks"
        return 1.0, nearest_phys - peaks[0], "offset_only"

    # 矩形コスト行列: peaks(n) × physical(4)
    use_peaks = peaks[:4]
    cost      = np.abs(use_peaks[:, None] - PHYSICAL_HEIGHTS_M[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_peaks = use_peaks[row_ind]
    matched_phys  = PHYSICAL_HEIGHTS_M[col_ind]

    # 距離閾値フィルタ
    valid = np.abs(matched_peaks - matched_phys) <= MAX_MATCH_DIST_M
    matched_peaks = matched_peaks[valid]
    matched_phys  = matched_phys[valid]

    if len(matched_peaks) == 0:
        return 1.0, 0.0, "no_peaks"
    if len(matched_peaks) == 1:
        return 1.0, float(matched_phys[0] - matched_peaks[0]), "offset_only"

    slope, intercept = np.polyfit(matched_peaks, matched_phys, 1)
    status = "full_affine" if len(matched_peaks) >= 4 else f"partial_affine_{len(matched_peaks)}peaks"
    return float(slope), float(intercept), status


# ── メイン処理 ─────────────────────────────────────────────────────────────────

def process_all() -> pd.DataFrame:
    files = sorted(CSV_DIR.glob("*_scaletest.csv"))
    print(f"Processing {len(files)} CSV files ...")

    records = []
    for path in files:
        session = path.name.split("_points.csv")[0]
        animal  = session.split("_")[0]

        df = pd.read_csv(path)
        z_cols = [c for c in df.columns
                  if c.endswith("_z") and any(k in c for k in FLOOR_KP_KEYWORDS)]
        if not z_cols:
            records.append({"session": session, "animal": animal, "status": "no_floor_cols"})
            continue

        floor_z_per_frame = df[z_cols].min(axis=1).values
        lo = np.percentile(floor_z_per_frame, 1)
        hi = np.percentile(floor_z_per_frame, 99)
        vals = floor_z_per_frame[(floor_z_per_frame >= lo) & (floor_z_per_frame <= hi)]

        peaks                     = detect_peaks_kde(vals)
        slope, intercept, status  = fit_affine(peaks)

        # 床ピーク残差: 補正後に最も床に近いピークと物理床の差
        floor_residual = None
        if len(peaks) and status != "no_peaks":
            floor_peak_corr = slope * peaks[0] + intercept
            floor_residual  = round((floor_peak_corr - PHYSICAL_HEIGHTS_M[0]) * 1000, 1)

        rec = {
            "session":       session,
            "animal":        animal,
            "n_frames":      len(df),
            "n_peaks":       len(peaks),
            "peaks_m":       peaks.tolist(),
            "slope":         round(slope, 4),
            "intercept":     round(intercept, 4),
            "status":        status,
            "slope_warn":    abs(slope - 1.0) > MAX_AFFINE_SLOPE_DEV,
            "floor_residual_mm": floor_residual,
            # 生データ保持
            "_vals":    vals,
            "_peaks":   peaks,
        }
        records.append(rec)

    return records


# ── 可視化 ─────────────────────────────────────────────────────────────────────

def plot_grid(records: list[dict], out_dir: Path) -> None:
    from scipy.stats import norm as sp_norm

    n    = len(records)
    ncol = 6
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.5, nrow * 2.8))
    axes = axes.ravel()

    for ax, r in zip(axes, records):
        session = r["session"]
        animal  = r["animal"]
        color   = ANIMAL_COLORS.get(animal, "gray")

        if "_vals" not in r:
            ax.set_title(f'{session[3:]}\n{r["status"]}', fontsize=6)
            ax.set_visible(True)
            continue

        vals  = r["_vals"]
        peaks = r["_peaks"]

        ax.hist(vals, bins=100, color=color, alpha=0.5, density=True)

        # KDE curve
        kde = gaussian_kde(vals, bw_method=KDE_BW)
        x = np.linspace(vals.min(), vals.max(), 400)
        ax.plot(x, kde(x), "k-", lw=1, alpha=0.6)
        for mu in peaks:
            ax.axvline(mu, color="red", lw=1.2, alpha=0.8)

        # 物理高さを縦線で表示
        for h, label in zip(PHYSICAL_HEIGHTS_M, ["bed", "low", "bar", "ceil"]):
            ax.axvline(h, color="green", lw=0.8, ls=":", alpha=0.7)

        slope_str = f"a={r['slope']:.3f}" + (" !" if r["slope_warn"] else "")
        ax.set_title(
            f'{session[3:]}\npeaks={r["n_peaks"]} {slope_str}',
            fontsize=6,
        )
        ax.set_xlabel("bZ (m)", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_xlim(0.65, 1.60)

    for ax in axes[n:]:
        ax.set_visible(False)

    # 凡例パネル
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="red",   lw=1.5, label="GMM peak"),
        Line2D([0], [0], color="green", lw=0.8, ls=":", label="physical height"),
    ]
    fig.legend(handles=legend_elems, loc="lower right", fontsize=8)
    fig.suptitle(
        "Height dist per session (bZ m)\n"
        "red=KDE peak  green=physical heights [0.7337, 1.0179, 1.3035, 1.5016]",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = out_dir / "height_dist_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_affine_summary(records: list[dict], out_dir: Path) -> None:
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in records if "slope" in r])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # slope 分布
    for animal, grp in df.groupby("animal"):
        axes[0].hist(grp["slope"], bins=30, alpha=0.6,
                     color=ANIMAL_COLORS.get(animal, "gray"), label=animal)
    axes[0].axvline(1.0, color="k", lw=1.5, ls="--", label="ideal (a=1)")
    axes[0].set_xlabel("slope (a)")
    axes[0].set_title("Affine slope distribution\n(1.0 = no scale error)")
    axes[0].legend(fontsize=8)

    # intercept 分布
    for animal, grp in df.groupby("animal"):
        axes[1].hist(grp["intercept"], bins=30, alpha=0.6,
                     color=ANIMAL_COLORS.get(animal, "gray"), label=animal)
    axes[1].axvline(0.0, color="k", lw=1.5, ls="--", label="ideal (b=0)")
    axes[1].set_xlabel("intercept (b) m")
    axes[1].set_title("Affine intercept distribution\n(0 = no offset)")
    axes[1].legend(fontsize=8)

    # floor 残差
    df_valid = df.dropna(subset=["floor_residual_mm"])
    for animal, grp in df_valid.groupby("animal"):
        axes[2].hist(grp["floor_residual_mm"], bins=30, alpha=0.6,
                     color=ANIMAL_COLORS.get(animal, "gray"), label=animal)
    axes[2].axvline(0.0, color="k", lw=1.5, ls="--")
    axes[2].set_xlabel("floor residual (mm)")
    axes[2].set_title("Floor peak residual after affine\n(0 = perfect floor alignment)")
    axes[2].legend(fontsize=8)

    fig.suptitle("Affine correction summary (all sessions)", fontsize=11)
    plt.tight_layout()
    out = out_dir / "affine_summary_plot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = process_all()

    # CSV 保存
    df_out = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                            for r in records])
    df_out.to_csv(OUT_DIR / "affine_summary.csv", index=False)
    print(f"Saved: {OUT_DIR / 'affine_summary.csv'}")

    # サマリー表示
    df = df_out[df_out["status"].notna() & df_out["status"].str.contains("affine|offset")]
    print(f"\n{'Status':25} {'Count':>6}")
    for status, grp in df.groupby("status"):
        print(f"  {status:23} {len(grp):>6}")

    n_warn = int(df["slope_warn"].sum()) if "slope_warn" in df.columns else 0
    print(f"\nSlope warnings (|a-1|>{MAX_AFFINE_SLOPE_DEV}): {n_warn} sessions ({100*n_warn/len(df):.0f}%)")
    print(f"Slope    mean={df['slope'].mean():.4f}  std={df['slope'].std():.4f}  "
          f"min={df['slope'].min():.4f}  max={df['slope'].max():.4f}")
    print(f"Intercept mean={df['intercept'].mean():.4f}  std={df['intercept'].std():.4f}")
    fr = df["floor_residual_mm"].dropna()
    if len(fr):
        print(f"Floor residual (mm) mean={fr.abs().mean():.2f}  max={fr.abs().max():.2f}")

    plot_grid(records, OUT_DIR)
    plot_affine_summary(records, OUT_DIR)


if __name__ == "__main__":
    main()

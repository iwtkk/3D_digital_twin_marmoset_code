"""
Per-session Y-offset correction based on floor-touching estimation.

設計:
  kp3d Y軸は 大きい=床側 / 小さい=天井側 (Y_floor≈92mm, Y_ceil≈-665mm)
  各sessionで「床に最も近かった瞬間」= p99(max(floor_KP_Y per frame)) を床高さの推定値とする。

  - 正常session: 動物がほぼ床に居る → median≈75mm, p99≈97mm
  - 浮きsession : 動物がほぼ棚にいるが時々床に降りる → median≈-507mm, p99≈83mm

  → p99が近い=どちらも床への接触がある → 床レベルはsessionを通じて一定

  判定ルール:
    p99 >= FLOOR_Y_CONTACT_THR (-50mm) → 床接触あり → 通常補正
    p99 <  FLOOR_Y_CONTACT_THR          → 床に一度も接触なし → 要フラグ、全体オフセット使用

  「ずっと上にいるとき」の対処:
    floor_Y_est がグローバル床レベルより大幅に低い場合、
    フラグを立てた上でグローバル中央値補正を適用(フォールバック)。

使い方:
  python scripts/dev/correct_y_offset.py              # 全session分析+プロット
  python scripts/dev/correct_y_offset.py --apply      # 補正値をCSVに保存
  python scripts/dev/correct_y_offset.py --session cj425_20210426_141115  # 1session
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── 定数 ─────────────────────────────────────────────────────────────────────
FLOOR_KP = [9, 10, 15, 16]          # L/R wrist, L/R ankle
FLOOR_KP_NAMES = ["L_wri", "R_wri", "L_ank", "R_ank"]

# kp3d Y: p99 がこの値より低ければ床接触なしと判定
FLOOR_Y_CONTACT_THR = -50.0          # mm

# グローバル床レベル (bbox inlier 上限 ≈ 実際の床)
GLOBAL_FLOOR_Y = 92.27               # mm

BASE_DIR  = Path("outputs/3d")
OUT_DIR   = Path("outputs/dev/correct_y_offset")


# ── 関数 ─────────────────────────────────────────────────────────────────────

def estimate_floor_y(kp3d: np.ndarray, percentile: float = 99.0) -> float:
    """
    1sessionのkp3d (T,20,3) から床高さ(Y)を推定。
    各フレームで floor KP の max Y をとり、その p99 を床レベルとする。
    """
    max_y_per_frame = kp3d[:, FLOOR_KP, 1].max(axis=1)   # (T,)
    return float(np.percentile(max_y_per_frame, percentile))


def classify_session(floor_y_est: float) -> str:
    """床接触の有無でsessionを分類"""
    if floor_y_est >= FLOOR_Y_CONTACT_THR:
        return "contact"       # 床接触あり → 通常補正可能
    else:
        return "no_contact"    # 床接触なし → フォールバック補正


def compute_correction(floor_y_est: float, global_floor_y: float) -> float:
    """Y補正量 = global_floor_y - session_floor_y_est"""
    return global_floor_y - floor_y_est


def analyze_all_sessions(base_dir: Path) -> pd.DataFrame:
    pickles = sorted(base_dir.glob("*/kp3d_fxdJointLen.pickle"))
    records = []
    for path in pickles:
        session = path.parent.name
        animal  = session.split("_")[0]
        with open(path, "rb") as f:
            data = pickle.load(f)
        kp3d = data["kp3d"][0]   # (T,20,3)

        floor_y_est   = estimate_floor_y(kp3d, percentile=99.0)
        median_floor  = float(np.median(kp3d[:, FLOOR_KP, 1].max(axis=1)))
        classification = classify_session(floor_y_est)
        correction    = compute_correction(floor_y_est, GLOBAL_FLOOR_Y)

        records.append({
            "session":        session,
            "animal":         animal,
            "n_frames":       kp3d.shape[0],
            "floor_y_p99":    round(floor_y_est, 2),
            "floor_y_median": round(median_floor, 2),
            "classification": classification,
            "y_correction_mm": round(correction, 2),
        })
    return pd.DataFrame(records)


# ── 可視化 ────────────────────────────────────────────────────────────────────

def plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    animal_colors = {"cj425": "steelblue", "cj611": "darkorange", "cj357": "seagreen"}

    # ① per-session floor_y_p99 と floor_y_median
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    for ax, col, title in [
        (axes[0], "floor_y_p99",    "Floor Y estimate (p99 of max floor-KP Y)\n→ 床到達レベル"),
        (axes[1], "floor_y_median", "Floor Y median (typical height of floor KPs)\n→ 個体の常用高さ"),
    ]:
        colors = [animal_colors.get(a, "gray") for a in df["animal"]]
        bars = ax.bar(range(len(df)), df[col], color=colors, alpha=0.8)
        ax.axhline(GLOBAL_FLOOR_Y, color="brown", lw=2, ls="--",
                   label=f"global floor ({GLOBAL_FLOOR_Y}mm)")
        ax.axhline(FLOOR_Y_CONTACT_THR, color="red", lw=1.5, ls=":",
                   label=f"no-contact threshold ({FLOOR_Y_CONTACT_THR}mm)")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["session"].str[3:], rotation=60, ha="right", fontsize=6)
        ax.set_ylabel("kp3d Y (mm)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        # no_contact sessionをマーク
        for i, row in df.iterrows():
            if row["classification"] == "no_contact":
                ax.annotate("!", (i, df[col].iloc[i]), ha="center", va="bottom",
                            fontsize=8, color="red", fontweight="bold")

    patches = [plt.Rectangle((0,0),1,1, color=c, alpha=0.8) for c in animal_colors.values()]
    axes[0].legend(handles=patches + axes[0].get_lines(),
                   labels=list(animal_colors.keys()) + [l.get_label() for l in axes[0].get_lines()],
                   fontsize=8)

    fig.suptitle("Per-session floor estimation & Y-correction", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "floor_estimate_summary.png", dpi=150)
    plt.close(fig)

    # ② 補正量の分布
    fig, ax = plt.subplots(figsize=(16, 4))
    colors = [animal_colors.get(a, "gray") for a in df["animal"]]
    ax.bar(range(len(df)), df["y_correction_mm"], color=colors, alpha=0.8)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["session"].str[3:], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Y correction (mm)\n正値=上方シフト必要 / 負値=下方シフト必要")
    ax.set_title("Per-session Y correction (= GLOBAL_FLOOR_Y - session floor_y_p99)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / "y_correction_per_session.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}")


def plot_before_after(sessions: list[str], df: pd.DataFrame, out_dir: Path) -> None:
    """指定sessionについてBefore/Afterの床KP Y分布を比較"""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(sessions)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    if n == 1:
        axes = [axes]

    for ax, session in zip(axes, sessions):
        row = df[df["session"] == session].iloc[0]
        path = BASE_DIR / session / "kp3d_fxdJointLen.pickle"
        with open(path, "rb") as f:
            data = pickle.load(f)
        kp3d = data["kp3d"][0]

        max_y = kp3d[:, FLOOR_KP, 1].max(axis=1)
        max_y_corrected = max_y + row["y_correction_mm"]

        ax.hist(max_y,           bins=100, alpha=0.5, color="tomato",
                label=f"before  median={np.median(max_y):.0f}mm", density=True)
        ax.hist(max_y_corrected, bins=100, alpha=0.5, color="steelblue",
                label=f"after   median={np.median(max_y_corrected):.0f}mm", density=True)
        ax.axvline(GLOBAL_FLOOR_Y, color="brown", ls="--", lw=2,
                   label=f"floor target ({GLOBAL_FLOOR_Y}mm)")
        ax.set_title(f"{session}  correction={row['y_correction_mm']:.1f}mm  [{row['classification']}]")
        ax.set_xlabel("max floor-KP Y per frame (mm)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "before_after.png", dpi=150)
    plt.close(fig)
    print(f"Before/after plot saved to {out_dir}/before_after.png")


# ── メイン ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default=None, help="単一sessionを指定")
    parser.add_argument("--apply",   action="store_true", help="補正値をCSVとして保存")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--out-dir",  type=Path, default=OUT_DIR)
    args = parser.parse_args()

    print("Analysing sessions ...")
    df = analyze_all_sessions(args.base_dir)

    # 統計サマリー
    n_contact    = (df["classification"] == "contact").sum()
    n_no_contact = (df["classification"] == "no_contact").sum()
    print(f"\n{'Sessions analysed':25}: {len(df)}")
    print(f"{'  floor contact':25}: {n_contact}")
    print(f"{'  no floor contact':25}: {n_no_contact}")
    print(f"\n{'':10} {'floor_y_p99':>12} {'floor_y_med':>12} {'correction':>12} {'class'}")
    print("-" * 65)
    for _, row in df.iterrows():
        flag = " !" if row["classification"] == "no_contact" else "  "
        print(f"{row['session'][3:]:35} {row['floor_y_p99']:>10.1f}mm "
              f"{row['floor_y_median']:>10.1f}mm "
              f"{row['y_correction_mm']:>10.1f}mm "
              f"{row['classification']}{flag}")

    plot_summary(df, args.out_dir)

    # Before/After: 代表的な4 session
    showcase = []
    for label in ["cj425_20210427_190000", "cj425_20210426_141115",
                  "cj611_20230217_170000", "cj611_20210831_080000"]:
        if (args.base_dir / label).exists():
            showcase.append(label)
    if showcase:
        plot_before_after(showcase, df, args.out_dir)

    if args.apply:
        out_csv = args.out_dir / "y_corrections.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nCorrections saved to {out_csv}")

    # no_contact sessionの警告
    no_contact = df[df["classification"] == "no_contact"]
    if len(no_contact):
        print(f"\n[WARNING] {len(no_contact)} sessions flagged as 'no floor contact':")
        for _, row in no_contact.iterrows():
            print(f"  {row['session']}  floor_y_p99={row['floor_y_p99']:.1f}mm")
        print("  → 全体オフセット(global median)で補正するか手動確認を推奨")


if __name__ == "__main__":
    main()

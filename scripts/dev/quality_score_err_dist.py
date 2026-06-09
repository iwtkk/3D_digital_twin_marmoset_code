"""
3D keypoint quality — distribution exploration (NO thresholds yet).

Goal: understand kp3d_score / kp3d_err before deciding QC criteria.
  - kp3d_score : triangulation/detection confidence (higher = better, clamped at 0.5)
  - kp3d_err   : reprojection error in px        (lower  = better)

These two axes are complementary:
  low score  -> uncertain detection (occlusion / faint)
  high err   -> geometric inconsistency (ID switch / calibration / mismatch)
They can disagree, so we look at the JOINT distribution as well as each axis.

This script ONLY describes the data (percentiles, clamp/NaN rates, per-kp and
per-session breakdowns) and plots distributions. No good/bad classification.

Outputs -> outputs/analysis_quality_score_err/
  quality_dist_summary.json
  dist_score.png            global score histogram (log-y, clamp annotated)
  dist_err.png              global err histogram
  joint_score_err.png       hexbin score vs err (reveals clamp layer)
  perkp_score.png           per-keypoint score box/percentiles
  perkp_err.png             per-keypoint err  box/percentiles
  heatmap_score.png         session x keypoint mean score
  heatmap_err.png           session x keypoint mean err
  clamp_nan_rate.png        per-keypoint clamp(score) / nan(err) rates
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────
KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
    "backbone_peak", "tail_1", "tail_2",
]
N_KP = 20

SCORE_CLAMP = 0.5         # observed lower bound of kp3d_score
CLAMP_EPS   = 0.05        # "near clamp" band: score <= SCORE_CLAMP + CLAMP_EPS
FRAME_STEP  = 15          # subsample for global scatter/hist (memory)
PCTS        = [1, 5, 25, 50, 75, 95, 99]

BASE_3D = Path("/data/iwata/3D_digital_twin_marmoset_code/outputs/3d")
OUT_DIR = Path("/data/iwata/3D_digital_twin_marmoset_code/outputs/analysis_quality_score_err")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── per-session descriptive stats ────────────────────────────────────────────────
def describe_session(session: str):
    pkl_path = BASE_3D / session / "kp3d_fxdJointLen.pickle"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    sc = d["kp3d_score"][0]   # (T, 20)
    er = d["kp3d_err"][0]     # (T, 20)
    T  = sc.shape[0]

    per_kp = []
    for ki in range(N_KP):
        s = sc[:, ki]
        e = er[:, ki]
        e_valid = e[~np.isnan(e)]
        per_kp.append({
            "kp_idx": ki,
            "kp_name": KP_NAMES[ki],
            "score_mean": round(float(np.nanmean(s)), 4),
            "score_pcts": [round(float(x), 4) for x in np.nanpercentile(s, PCTS)],
            "clamp_rate": round(float((s <= SCORE_CLAMP + CLAMP_EPS).mean()), 4),
            "err_mean": round(float(np.nanmean(e)), 3) if e_valid.size else None,
            "err_pcts": [round(float(x), 3) for x in np.nanpercentile(e, PCTS)] if e_valid.size else None,
            "err_nan_rate": round(float(np.isnan(e).mean()), 5),
        })

    # subsampled raw values for global plots
    s_sub = sc[::FRAME_STEP].ravel()
    e_sub = er[::FRAME_STEP].ravel()

    summary = {
        "session": session,
        "n_frames": int(T),
        "score_mean": round(float(np.nanmean(sc)), 4),
        "score_pcts": [round(float(x), 4) for x in np.nanpercentile(sc, PCTS)],
        "err_mean": round(float(np.nanmean(er)), 3),
        "err_pcts": [round(float(x), 3) for x in np.nanpercentile(er, PCTS)],
        "clamp_rate": round(float((sc <= SCORE_CLAMP + CLAMP_EPS).mean()), 4),
        "err_nan_rate": round(float(np.isnan(er).mean()), 5),
        "per_kp": per_kp,
    }
    return summary, s_sub, e_sub


# ── plots ──────────────────────────────────────────────────────────────────────
def plot_global_hist(score_all, err_all):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(score_all, bins=np.linspace(0.5, 1.05, 120), color="steelblue")
    ax.axvline(SCORE_CLAMP, color="red", ls="--", lw=1, label=f"clamp ({SCORE_CLAMP})")
    ax.axvline(SCORE_CLAMP + CLAMP_EPS, color="orange", ls=":", lw=1,
               label=f"near-clamp band (+{CLAMP_EPS})")
    ax.set_yscale("log"); ax.set_xlabel("kp3d_score"); ax.set_ylabel("count (log)")
    ax.set_title("Global kp3d_score distribution (all sessions, subsampled)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "dist_score.png", dpi=150); plt.close(fig)
    print("  saved dist_score.png")

    err_valid = err_all[~np.isnan(err_all)]
    p995 = np.percentile(err_valid, 99.5)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(err_valid, bins=np.linspace(0, p995, 120), color="seagreen")
    for q in (50, 95, 99):
        v = np.percentile(err_valid, q)
        ax.axvline(v, ls="--", lw=1, color="black", alpha=0.6)
        ax.text(v, ax.get_ylim()[1]*0.9, f"p{q}={v:.0f}", rotation=90,
                va="top", ha="right", fontsize=8)
    ax.set_xlabel("kp3d_err (px)"); ax.set_ylabel("count")
    ax.set_title("Global kp3d_err distribution (all sessions, subsampled)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "dist_err.png", dpi=150); plt.close(fig)
    print("  saved dist_err.png")


def plot_joint(score_all, err_all):
    m = ~np.isnan(err_all)
    s = score_all[m]; e = np.clip(err_all[m], 0, np.percentile(err_all[m], 99.5))
    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(s, e, gridsize=60, cmap="viridis", bins="log", mincnt=1)
    ax.axvline(SCORE_CLAMP + CLAMP_EPS, color="orange", ls=":", lw=1)
    ax.set_xlabel("kp3d_score (higher better)")
    ax.set_ylabel("kp3d_err px (lower better)")
    ax.set_title("Joint distribution: score vs err\n"
                 "(top-left = low score & high err = worst; clamp layer at left)")
    fig.colorbar(hb, ax=ax, label="log10 count")
    fig.tight_layout(); fig.savefig(OUT_DIR / "joint_score_err.png", dpi=150); plt.close(fig)
    print("  saved joint_score_err.png")


def plot_perkp(summaries):
    # aggregate per-kp percentile bands across sessions (median of session medians etc.)
    def gather(field, pct_field, idx):
        return np.array([[s["per_kp"][ki][pct_field][idx] if s["per_kp"][ki][pct_field] else np.nan
                          for s in summaries] for ki in range(N_KP)])
    # PCTS = [1,5,25,50,75,95,99] -> indices for 5,25,50,75,95
    i5, i25, i50, i75, i95 = 1, 2, 3, 4, 5

    for field, pct_field, fname, ylab, invert in [
        ("score", "score_pcts", "perkp_score.png", "kp3d_score", False),
        ("err",   "err_pcts",   "perkp_err.png",   "kp3d_err (px)", True),
    ]:
        med = np.nanmedian(gather(field, pct_field, i50), axis=1)
        lo  = np.nanmedian(gather(field, pct_field, i25), axis=1)
        hi  = np.nanmedian(gather(field, pct_field, i75), axis=1)
        w5  = np.nanmedian(gather(field, pct_field, i5),  axis=1)
        w95 = np.nanmedian(gather(field, pct_field, i95), axis=1)
        order = np.argsort(med)[::-1] if not invert else np.argsort(med)

        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(N_KP)
        ax.vlines(x, w5[order], w95[order], color="lightgray", lw=4, label="p5–p95")
        ax.vlines(x, lo[order], hi[order], color="steelblue", lw=8, label="p25–p75")
        ax.plot(x, med[order], "o", color="black", ms=5, label="median")
        ax.set_xticks(x); ax.set_xticklabels([KP_NAMES[i] for i in order], rotation=45, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(f"Per-keypoint {field} distribution (median of session percentiles)")
        ax.legend()
        fig.tight_layout(); fig.savefig(OUT_DIR / fname, dpi=150); plt.close(fig)
        print(f"  saved {fname}")


def plot_heatmaps(summaries):
    sessions = [s["session"] for s in summaries]
    sc_mat = np.array([[s["per_kp"][ki]["score_mean"] for ki in range(N_KP)] for s in summaries])
    er_mat = np.array([[(s["per_kp"][ki]["err_mean"] or np.nan) for ki in range(N_KP)] for s in summaries])

    for mat, fname, cmap, title in [
        (sc_mat, "heatmap_score.png", "RdYlGn",   "Mean kp3d_score (green=good)"),
        (er_mat, "heatmap_err.png",   "RdYlGn_r", "Mean kp3d_err px (green=good)"),
    ]:
        fig, ax = plt.subplots(figsize=(22, max(6, len(sessions) * 0.28 + 2)))
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(N_KP)); ax.set_xticklabels(KP_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(sessions))); ax.set_yticklabels(sessions, fontsize=6)
        ax.set_title(f"{title} — session x keypoint")
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(OUT_DIR / fname, dpi=150); plt.close(fig)
        print(f"  saved {fname}")


def plot_clamp_nan(summaries):
    clamp = np.array([[s["per_kp"][ki]["clamp_rate"]   for ki in range(N_KP)] for s in summaries]).mean(0)
    nan   = np.array([[s["per_kp"][ki]["err_nan_rate"] for ki in range(N_KP)] for s in summaries]).mean(0)
    x = np.arange(N_KP)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 7))
    a1.bar(x, clamp, color="tomato")
    a1.set_xticks(x); a1.set_xticklabels(KP_NAMES, rotation=45, ha="right")
    a1.set_ylabel(f"score <= {SCORE_CLAMP+CLAMP_EPS} rate")
    a1.set_title("Per-keypoint near-clamp score rate (mean over sessions) — effectively no info")
    a2.bar(x, nan, color="slateblue")
    a2.set_xticks(x); a2.set_xticklabels(KP_NAMES, rotation=45, ha="right")
    a2.set_ylabel("err NaN rate")
    a2.set_title("Per-keypoint reprojection-failure (NaN err) rate")
    fig.tight_layout(); fig.savefig(OUT_DIR / "clamp_nan_rate.png", dpi=150); plt.close(fig)
    print("  saved clamp_nan_rate.png")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    sessions = sorted(p.name for p in BASE_3D.iterdir() if p.is_dir())
    print(f"Describing {len(sessions)} sessions …")

    summaries = []
    score_chunks, err_chunks = [], []
    for i, session in enumerate(sessions):
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(sessions)}] {session}")
        out = describe_session(session)
        if out is None:
            continue
        summary, s_sub, e_sub = out
        summaries.append(summary)
        score_chunks.append(s_sub)
        err_chunks.append(e_sub)

    score_all = np.concatenate(score_chunks)
    err_all   = np.concatenate(err_chunks)

    # global descriptive numbers
    ev = err_all[~np.isnan(err_all)]
    sv = score_all[~np.isnan(score_all)]
    global_stats = {
        "n_sessions": len(summaries),
        "score_clamp_ref": SCORE_CLAMP,
        "near_clamp_band": SCORE_CLAMP + CLAMP_EPS,
        "frame_step_subsample": FRAME_STEP,
        "pcts_labels": PCTS,
        "score_pcts": [round(float(x), 4) for x in np.percentile(sv, PCTS)],
        "score_nan_rate": round(float(np.isnan(score_all).mean()), 5),
        "score_clamp_rate": round(float((sv <= SCORE_CLAMP + CLAMP_EPS).mean()), 4),
        "err_pcts": [round(float(x), 3) for x in np.percentile(ev, PCTS)],
        "err_nan_rate": round(float(np.isnan(err_all).mean()), 5),
    }
    print("\nGlobal score pcts", PCTS, "=", global_stats["score_pcts"])
    print("Global err   pcts", PCTS, "=", global_stats["err_pcts"])
    print("Global near-clamp rate:", global_stats["score_clamp_rate"])

    with open(OUT_DIR / "quality_dist_summary.json", "w") as f:
        json.dump({"global": global_stats, "sessions": summaries}, f, indent=2, ensure_ascii=False)
    print("  saved quality_dist_summary.json")

    print("Generating plots …")
    plot_global_hist(score_all, err_all)
    plot_joint(score_all, err_all)
    plot_perkp(summaries)
    plot_heatmaps(summaries)
    plot_clamp_nan(summaries)
    print("Done.")


if __name__ == "__main__":
    main()

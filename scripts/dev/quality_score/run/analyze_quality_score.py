"""
3D キーポイント品質 (kp3d_score / kp3d_err) の分布把握 (run)。

しきい値判定はせず、score（検出確信度, 高いほど良, 0.5 でクランプ）と
err（再投影誤差 px, 低いほど良）の分布を記述する。
全セッションを集計し、要約 JSON と全体プロット用のサブサンプル NPZ を出力する。
可視化は plot/plot_quality_score.py。

入力 (--input-dir, 既定 outputs/3d):  {input-dir}/{session}/kp3d_fxdJointLen.pickle
出力 (--output-dir, 既定 scripts/dev/quality_score_files):
  quality_dist_summary.json   global + per-session per-kp 記述統計
  quality_subsample.npz       全体ヒスト/散布図用のサブサンプル score/err
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_OUTPUT = Path("scripts/dev/quality_score_files")

KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
    "backbone_peak", "tail_1", "tail_2",
]
N_KP = 20
SCORE_CLAMP = 0.5
CLAMP_EPS = 0.05
FRAME_STEP = 15
PCTS = [1, 5, 25, 50, 75, 95, 99]


def resolve_pickle(session_dir: Path) -> Path | None:
    for name in ("kp3d_fxdJointLen.pickle", "kp3d.pickle"):
        p = session_dir / name
        if p.exists():
            return p
    return None


def describe_session(session: str, input_dir: Path):
    pkl = resolve_pickle(input_dir / session)
    if pkl is None:
        return None
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    sc = d["kp3d_score"][0]
    er = d["kp3d_err"][0]
    T = sc.shape[0]

    per_kp = []
    for ki in range(N_KP):
        s, e = sc[:, ki], er[:, ki]
        e_valid = e[~np.isnan(e)]
        per_kp.append({
            "kp_idx": ki, "kp_name": KP_NAMES[ki],
            "score_mean": round(float(np.nanmean(s)), 4),
            "score_pcts": [round(float(x), 4) for x in np.nanpercentile(s, PCTS)],
            "clamp_rate": round(float((s <= SCORE_CLAMP + CLAMP_EPS).mean()), 4),
            "err_mean": round(float(np.nanmean(e)), 3) if e_valid.size else None,
            "err_pcts": [round(float(x), 3) for x in np.nanpercentile(e, PCTS)] if e_valid.size else None,
            "err_nan_rate": round(float(np.isnan(e).mean()), 5),
        })

    summary = {
        "session": session, "n_frames": int(T),
        "score_mean": round(float(np.nanmean(sc)), 4),
        "score_pcts": [round(float(x), 4) for x in np.nanpercentile(sc, PCTS)],
        "err_mean": round(float(np.nanmean(er)), 3),
        "err_pcts": [round(float(x), 3) for x in np.nanpercentile(er, PCTS)],
        "clamp_rate": round(float((sc <= SCORE_CLAMP + CLAMP_EPS).mean()), 4),
        "err_nan_rate": round(float(np.isnan(er).mean()), 5),
        "per_kp": per_kp,
    }
    return summary, sc[::FRAME_STEP].ravel(), er[::FRAME_STEP].ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--session", default=None, help="セッション名 (省略時は全セッション)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(p.name for p in args.input_dir.iterdir() if p.is_dir())
    print(f"Describing {len(sessions)} sessions …")

    summaries, score_chunks, err_chunks = [], [], []
    for i, session in enumerate(sessions):
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(sessions)}] {session}")
        out = describe_session(session, args.input_dir)
        if out is None:
            continue
        summary, s_sub, e_sub = out
        summaries.append(summary)
        score_chunks.append(s_sub)
        err_chunks.append(e_sub)

    if not summaries:
        print("有効なセッションがありません", file=sys.stderr)
        sys.exit(1)

    score_all = np.concatenate(score_chunks)
    err_all = np.concatenate(err_chunks)
    sv = score_all[~np.isnan(score_all)]
    ev = err_all[~np.isnan(err_all)]
    global_stats = {
        "n_sessions": len(summaries), "score_clamp_ref": SCORE_CLAMP,
        "near_clamp_band": SCORE_CLAMP + CLAMP_EPS, "frame_step_subsample": FRAME_STEP,
        "pcts_labels": PCTS,
        "score_pcts": [round(float(x), 4) for x in np.percentile(sv, PCTS)],
        "score_nan_rate": round(float(np.isnan(score_all).mean()), 5),
        "score_clamp_rate": round(float((sv <= SCORE_CLAMP + CLAMP_EPS).mean()), 4),
        "err_pcts": [round(float(x), 3) for x in np.percentile(ev, PCTS)],
        "err_nan_rate": round(float(np.isnan(err_all).mean()), 5),
    }
    print("\nGlobal score pcts", PCTS, "=", global_stats["score_pcts"])
    print("Global err   pcts", PCTS, "=", global_stats["err_pcts"])

    with open(args.output_dir / "quality_dist_summary.json", "w") as f:
        json.dump({"global": global_stats, "sessions": summaries}, f, indent=2, ensure_ascii=False)
    print(f"  saved {args.output_dir / 'quality_dist_summary.json'}")
    np.savez_compressed(args.output_dir / "quality_subsample.npz",
                        score_all=score_all, err_all=err_all)
    print(f"  saved {args.output_dir / 'quality_subsample.npz'}")


if __name__ == "__main__":
    main()

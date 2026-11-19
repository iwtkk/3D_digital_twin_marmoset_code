"""
Compute the Fig. 8a metrics (Full image and Animal region PSNR / SSIM, bbox IoU) and optionally verify them
against the published files.

I/O ledger: scripts/06_validation/consistency/io.md#compute_metrics

Definitions: consistency_core.py. The silhouette background of each camera is the empty-cage render of
render_pairs.py (render/background/Camera.{camera_id}.png); the Animal region of a pair comes from its detailed_fur silhouette and is shared by every
body model, while bbox_iou uses each model's own silhouette bbox.

Inputs (read only):
  --dataset-root   extracted Published dataset: frames.csv, 2D_temporal_keypoint_data/
  --work-dir       render/{model}/..., real/... from render_pairs.py and extract_real_frames.py
  --verify         directory with the published pair_metrics.csv (default: {dataset-root}/.../rendered_environmental_images)
Outputs (--work-dir/metrics/):
  pair_metrics.csv, camera_summary.csv, metrics_animal_{model}.json   (same formats as the published files)
  verify.json (with --verify)   per metric: max |difference|, share of pairs within the tolerance

Example:
  .venv/bin/python scripts/06_validation/consistency/compute_metrics.py \\
      --dataset-root /path/to/3D_digital_twin_marmoset_data --work-dir outputs/validation/consistency_fig8a --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import consistency_core as cc  # noqa: E402

PAIR_COLS = ["session_id", "animal_id", "camera_id", "frame_3d", "model", "roi", "render_bbox", "real_bbox",
             "bbox_iou", "psnr_full", "ssim_full", "psnr_animal", "ssim_animal"]
SUMMARY_METRICS = ["bbox_iou", "psnr_full", "ssim_full", "psnr_animal", "ssim_animal"]
# --verify passes when, for every metric, >= MIN_SHARE of the pairs are within TOLERANCE and the means differ by
# <= MEAN_TOLERANCE (README.md, "Tolerance")
TOLERANCE = {"psnr_full": 0.05, "ssim_full": 0.002, "psnr_animal": 0.1, "ssim_animal": 0.005, "bbox_iou": 0.05}
MEAN_TOLERANCE = {"psnr_full": 0.01, "ssim_full": 0.0005, "psnr_animal": 0.02, "ssim_animal": 0.001, "bbox_iou": 0.005}
MIN_SHARE = 0.95


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--frames-csv", type=Path, default=None)
    p.add_argument("--models", nargs="+", default=list(cc.MODELS), choices=cc.MODELS)
    p.add_argument("--session", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--verify", nargs="?", const="", default=None, metavar="DIR",
                   help="compare with the published pair_metrics.csv (DIR defaults to the dataset copy)")
    return p.parse_args()


def process_session(job):
    root, work, session, rows, models, bg_dir = job
    out, bgs = [], {}
    for r in rows:
        cam, f = r["camera_id"], int(r["frame_3d"])
        if cam not in bgs:
            bgs[cam] = cv2.imread(str(Path(bg_dir) / f"Camera.{cam}.png"))
        real = cv2.imread(str(cc.real_path(work, session, cam, f)))
        ren = {m: cv2.imread(str(cc.render_path(work, m, session, cam, f))) for m in models}
        if real is None or any(v is None for v in ren.values()):
            continue
        H, W = real.shape[:2]
        dbox = cc.detection_bbox(root, session, cam, int(r["video_frame_index"]))
        rbox = {m: cc.mask_bbox(cc.render_silhouette(ren[m], bgs[cam])) for m in models}
        roi = cc.union_roi([rbox["detailed_fur"], dbox], W, H)
        for m in models:
            pf, sf = cc.psnr_ssim(ren[m], real)
            pa, sa = cc.psnr_ssim(ren[m], real, roi) if roi else (np.nan, np.nan)
            out.append(dict(session_id=session, animal_id=cc.animal_of(session), camera_id=cam, frame_3d=f,
                            model=m, roi=cc.fmt_box(roi), render_bbox=cc.fmt_box(rbox[m]), real_bbox=cc.fmt_box(dbox),
                            bbox_iou=cc.box_iou(rbox[m], dbox), psnr_full=pf, ssim_full=sf,
                            psnr_animal=pa, ssim_animal=sa))
    return out


def stats(rs):
    ps = np.array([r["psnr"] for r in rs]); ss = np.array([r["ssim"] for r in rs])
    return {"count": len(rs), "psnr_mean": float(ps.mean()), "psnr_std": float(ps.std()),
            "psnr_min": float(ps.min()), "psnr_max": float(ps.max()),
            "ssim_mean": float(ss.mean()), "ssim_std": float(ss.std()),
            "ssim_min": float(ss.min()), "ssim_max": float(ss.max())}


def metrics_json(g: pd.DataFrame, model: str) -> dict:
    det = [{"session_id": r.session_id, "camera_id": r.camera_id, "frame_3d": int(r.frame_3d),
            "psnr": float(r.psnr_animal), "ssim": float(r.ssim_animal)} for r in g.itertuples()]
    by_s, by_c = defaultdict(list), defaultdict(list)
    for d in det:
        by_s[d["session_id"]].append(d); by_c[d["camera_id"]].append(d)
    return {"region": "animal", "model": model, "pose": "aligned", "total_pairs": len(det),
            "summary": {"total": stats(det), "by_session": {k: stats(v) for k, v in sorted(by_s.items())},
                        "by_camera": {k: stats(v) for k, v in sorted(by_c.items())}},
            "details": det}


def write_outputs(df: pd.DataFrame, out: Path, labels: dict):
    df.to_csv(out / "pair_metrics.csv", index=False)
    cs = df.groupby(["camera_id", "model"])[SUMMARY_METRICS].agg(["mean", "std"])
    cs.columns = [f"{m}_{s}" for m, s in cs.columns]
    cs.insert(0, "n_pairs", df.groupby(["camera_id", "model"]).size())
    cs = cs.reset_index()
    cs.insert(1, "camera_label", cs.camera_id.map(labels))
    cs.to_csv(out / "camera_summary.csv", index=False)
    for m, g in df.groupby("model"):
        (out / f"metrics_animal_{m}.json").write_text(json.dumps(metrics_json(g, m), indent=1))


def verify(df: pd.DataFrame, published: Path, out: Path) -> bool:
    keys = ["session_id", "camera_id", "frame_3d", "model"]
    pub = pd.read_csv(published, dtype={"camera_id": str})
    pub = pub[pub.model.isin(df.model.unique()) & pub.session_id.isin(df.session_id.unique())]
    j = df.merge(pub, on=keys, suffixes=("", "_pub"), how="inner")
    rep = {"pairs_compared": int(len(j)), "pairs_published": int(len(pub)),
           "tolerance": TOLERANCE, "mean_tolerance": MEAN_TOLERANCE, "min_share": MIN_SHARE, "metrics": {}}
    ok = 0 < rep["pairs_compared"] == rep["pairs_published"]
    for m, tol in TOLERANCE.items():
        a, b = j[m].to_numpy(float), j[f"{m}_pub"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(a - b))
        d = np.where(np.isnan(d), np.inf, d)
        share = float((d <= tol).mean()) if len(d) else 0.0
        rep["metrics"][m] = {"max_abs_diff": float(d.max()) if len(d) else None,
                             "mean_abs_diff": float(d[np.isfinite(d)].mean()) if np.isfinite(d).any() else None,
                             "share_within_tolerance": share,
                             "mean_reproduced": float(np.nanmean(a)), "mean_published": float(np.nanmean(b))}
        mean_d = abs(rep["metrics"][m]["mean_reproduced"] - rep["metrics"][m]["mean_published"])
        rep["metrics"][m]["mean_abs_diff_of_means"] = mean_d
        ok &= share >= MIN_SHARE and mean_d <= MEAN_TOLERANCE[m]
    rep["result"] = "OK" if ok else "FAILED"
    (out / "verify.json").write_text(json.dumps(rep, indent=1))
    print(f"verify: {rep['pairs_compared']}/{rep['pairs_published']} pairs compared")
    for m, v in rep["metrics"].items():
        if v["max_abs_diff"] is None:
            continue
        print(f"  {m:12s} max|d| {v['max_abs_diff']:.4g}  within {TOLERANCE[m]}: {100 * v['share_within_tolerance']:.1f}%"
              f"  mean {v['mean_reproduced']:.4f} vs {v['mean_published']:.4f}")
    print(f"verify: {rep['result']}")
    return ok


def main():
    a = parse_args()
    import yaml
    root_repo = Path(__file__).resolve().parents[3]
    labels = {str(k): f"Cam{v}" for k, v in
              yaml.safe_load(open(root_repo / "config/consistency.yaml"))["figure"]["mapping_table"].items()}
    pairs = cc.load_pairs(a.dataset_root, a.frames_csv)
    out = a.work_dir / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    bg_dir = a.work_dir / "render" / "background"
    missing = [c for c in sorted(pairs.camera_id.unique()) if not cc.background_path(a.work_dir, c).exists()]
    if missing:
        raise SystemExit(f"no empty-cage background for cameras {missing}; run render_pairs.py first")

    if a.session:
        pairs = pairs[pairs.session_id.isin(a.session)]
    models = [m for m in cc.MODELS if m in a.models]
    if "detailed_fur" not in models:
        models.insert(0, "detailed_fur")        # needed for the Animal region
    jobs = [(str(a.dataset_root), str(a.work_dir), s, g.to_dict("records"), models, str(bg_dir))
            for s, g in pairs.groupby("session_id")]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(process_session, jobs), 1):
            rows.extend(r)
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] sessions", flush=True)
    df = pd.DataFrame(rows, columns=PAIR_COLS)
    df = df[df.model.isin(a.models)].sort_values(["session_id", "camera_id", "frame_3d", "model"])
    df = df.reset_index(drop=True)
    print(f"{len(df) // max(1, len(a.models))} of {len(pairs)} pairs computed")
    write_outputs(df, out, labels)
    print(df.groupby("model")[SUMMARY_METRICS].mean().round(4).to_string())
    print(f"wrote {out}")
    if a.verify is not None:
        d = Path(a.verify) if a.verify else a.dataset_root / cc.FIG8A_DIR
        sys.exit(0 if verify(df, d / "pair_metrics.csv", out) else 1)


if __name__ == "__main__":
    main()

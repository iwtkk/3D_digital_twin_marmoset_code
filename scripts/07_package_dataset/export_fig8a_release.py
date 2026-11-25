"""
Write the Fig. 8a files of the Release tree (rendered_environmental_images/) from the validation outputs.

The published numbers are the Aligned-keypoint renders of the Fig. 8a run (consistency_variants pose x model
study, rows with pose == "affine"), compared with the real frames that run used. The metrics of all three body
models, the 2,400 renders of the Detailed model with fur and one example full-image SSIM map are released
(docs/adr/0003-release-fig8a-detailed-fur-renders.md); scripts/06_validation/consistency reproduces them from the
Published dataset and checks them with `compute_metrics.py --verify`.

I/O ledger: scripts/07_package_dataset/io.md#export_fig8a_release

Inputs (read only):
  --roi-csv       consistency_roi_files/{run}/metrics_roi.csv  (rows real_source == "mapped": raw_frame_index,
                  i.e. the Video frame index of the real image that was compared)
  --pose-fur-csv  consistency_affine_files/{run}/metrics_pose_fur.csv  (rows pose == "affine")
  --npz-dir       Release tree marmoset_keypoint_data/3D_temporal_keypoint_data/{session}.npz  (frame_num, frame_min)
  --render-dir    consistency_affine_files/render/detailed_fur/{run session_id}/Camera.{cam}/{frame_3d:07d}.png
                  (optional; the run session_id may carry the `dailylife_` prefix)
  --example       SSIM-map images to copy into example/ (optional; export_ssim_map_gray.py output, the
                  `ssim_map_gray_affine_` prefix is shortened to `ssim_map_`)
Outputs (--out-dir):
  frames.csv               one row per pair (Session x Camera ID x 3D frame)
  pair_metrics.csv         one row per pair x model
  camera_summary.csv       one row per Camera ID x model (mean / std over the pairs)
  metrics_animal_{model}.json  per-pair Animal-region PSNR / SSIM with summaries (visualize.py --metrics format)
  render/detailed_fur/{session}/Camera.{camera_id}/{frame_3d:07d}.png  hard links to --render-dir (every pair of
                  frames.csv must exist)
  example/ssim_map_*.png   copies of --example

Example:
  .venv/bin/python scripts/07_package_dataset/export_fig8a_release.py \\
      --roi-csv outputs/dev/consistency_roi_files/run_20260527_185342/metrics_roi.csv \\
      --pose-fur-csv outputs/dev/consistency_affine_files/run_20260527_185342/metrics_pose_fur.csv \\
      --npz-dir /data/iwata/3D_digital_twin_marmoset_data_work_npz/marmoset_keypoint_data/3D_temporal_keypoint_data \\
      --render-dir outputs/dev/consistency_affine_files/render/detailed_fur \\
      --example outputs/dev/consistency_affine_plot/ssim_map_gray_affine_detailed_fur_full_Cam1_cj611_20230211_150000_f0022467{,_colorbar}.png \\
      --out-dir outputs/fig8a_release
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONSISTENCY_YAML = ROOT / "config/consistency.yaml"
MODELS = ("detailed_fur", "detailed_nofur", "simplified_fur")
PAIR_COLS = ["session_id", "animal_id", "camera_id", "frame_3d", "model", "roi", "render_bbox", "real_bbox",
             "bbox_iou", "psnr_full", "ssim_full", "psnr_animal", "ssim_animal"]
SUMMARY_METRICS = ["bbox_iou", "psnr_full", "ssim_full", "psnr_animal", "ssim_animal"]


def session_of(s: str) -> str:
    """Validation-run session_id -> Session name (the run kept a `dailylife_` prefix)."""
    return s.replace("dailylife_", "", 1)


def build_frames(roi: pd.DataFrame, npz_dir: Path) -> pd.DataFrame:
    rows = []
    for s, g in roi.groupby("session_id"):
        z = np.load(npz_dir / f"{s}.npz")
        cams = [str(c) for c in z["camera_id"]]
        for r in g.itertuples():
            c = cams.index(r.camera_id)
            npz_idx = int(z["frame_num"][c, r.frame_3d] - z["frame_min"][c])
            rows.append(dict(session_id=s, camera_id=r.camera_id, frame_3d=int(r.frame_3d),
                             blend_frame=int(r.frame_3d) + 1, video_frame_index=int(r.raw_frame_index),
                             npz_video_frame_index=npz_idx, frame_offset=int(r.raw_frame_index) - npz_idx))
    return pd.DataFrame(rows).sort_values(["session_id", "camera_id", "frame_3d"]).reset_index(drop=True)


def link_renders(frames: pd.DataFrame, render_dir: Path, out_dir: Path) -> int:
    """Hard-link the render of every pair to out_dir/render/detailed_fur/ (copy if linking fails)."""
    n = 0
    for r in frames.itertuples():
        name = Path(f"Camera.{r.camera_id}") / f"{r.frame_3d:07d}.png"
        src = next((render_dir / s / name for s in (r.session_id, f"dailylife_{r.session_id}")
                    if (render_dir / s / name).exists()), None)
        if src is None:
            raise SystemExit(f"render missing: {r.session_id} {name}")
        dst = out_dir / "render" / "detailed_fur" / r.session_id / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        n += 1
    return n


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--roi-csv", type=Path, required=True)
    ap.add_argument("--pose-fur-csv", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--render-dir", type=Path)
    ap.add_argument("--example", type=Path, nargs="*", default=[])
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    labels = {str(k): f"Cam{v}" for k, v in yaml.safe_load(open(CONSISTENCY_YAML))["figure"]["mapping_table"].items()}

    roi = pd.read_csv(a.roi_csv, dtype={"camera_id": str, "real_bbox": str})
    roi = roi[roi.real_source == "mapped"].rename(columns={"frame": "frame_3d"})
    roi["session_id"] = roi.session_id.map(session_of)
    frames = build_frames(roi, a.npz_dir)
    frames.to_csv(a.out_dir / "frames.csv", index=False)

    pf = pd.read_csv(a.pose_fur_csv, dtype={"camera_id": str, "roi": str, "render_bbox": str, "real_bbox": str})
    pf = pf[pf.pose == "affine"].rename(columns={"frame": "frame_3d", "animal": "animal_id"})
    pf["session_id"] = pf.session_id.map(session_of)
    pf = pf[PAIR_COLS].sort_values(["session_id", "camera_id", "frame_3d", "model"]).reset_index(drop=True)
    keys = ["session_id", "camera_id", "frame_3d"]
    assert len(pf) == len(frames) * len(MODELS), (len(pf), len(frames))
    assert set(map(tuple, pf[keys].drop_duplicates().to_numpy())) == set(map(tuple, frames[keys].to_numpy()))
    pf.to_csv(a.out_dir / "pair_metrics.csv", index=False)

    cs = pf.groupby(["camera_id", "model"])[SUMMARY_METRICS].agg(["mean", "std"])
    cs.columns = [f"{m}_{s}" for m, s in cs.columns]
    cs.insert(0, "n_pairs", pf.groupby(["camera_id", "model"]).size())
    cs = cs.reset_index()
    cs.insert(1, "camera_label", cs.camera_id.map(labels))
    cs.to_csv(a.out_dir / "camera_summary.csv", index=False)

    for m in MODELS:
        (a.out_dir / f"metrics_animal_{m}.json").write_text(json.dumps(metrics_json(pf[pf.model == m], m), indent=1))

    if a.render_dir:
        print(f"render/detailed_fur: {link_renders(frames, a.render_dir, a.out_dir)} images")
    for e in a.example:
        (a.out_dir / "example").mkdir(exist_ok=True)
        shutil.copy2(e, a.out_dir / "example" / e.name.replace("ssim_map_gray_affine_", "ssim_map_"))

    off = frames.frame_offset.value_counts().sort_index().to_dict()
    print(f"frames.csv: {len(frames)} pairs, {frames.session_id.nunique()} sessions, frame_offset counts {off}")
    print(pf.groupby("model")[SUMMARY_METRICS].mean().round(4).to_string())
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()

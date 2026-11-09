"""
pickle → KDE affine 補正済み Blender CSV (run)

各フレームの床 KP (wrist/ankle) min(bZ) 分布から KDE ピークを検出し、
物理高さ PHYSICAL_HEIGHTS_M に affine (a, b) でフィットして bZ を補正する。
アルゴリズム本体は floorz_core.py に集約。

入力 (--input-dir, 既定 outputs/3d):
  {input-dir}/{session}/kp3d_fxdJointLen.pickle  (無ければ kp3d.pickle)

出力 (--output-dir, 既定 outputs/dev/floorZ_files):
  {output-dir}/csv/{session}_points.csv          中間 CSV (rotated mm)
  {output-dir}/corrected/{session}_corrected.csv 補正済み CSV
  {output-dir}/affine_log.csv                    session ごとの (a, b, status, peaks_m,
                                                 norm_mode)

norm_mode:
  global     全セッション共通で XYZ を規格化。--norm-pct の percentile を method.yaml の
             kp3d_global_norm キャッシュから引き、未計算なら全 pickle から算出して追記する
  per_animal 個体ごとの --norm-pct パーセンタイル（既定 2-98）で XYZ を規格化（新動作）

使い方:
  # 全セッション・per_animal モード（既定）
  OPENBLAS_NUM_THREADS=1 python pkl_to_corrected_csv.py

  # legacy モードで別ディレクトリに出力して比較
  OPENBLAS_NUM_THREADS=1 python pkl_to_corrected_csv.py \\
      --norm-mode global --output-dir outputs/dev/floorZ_files_global

  # 単一セッション
  OPENBLAS_NUM_THREADS=1 python pkl_to_corrected_csv.py --session cj425_20210427_120000

  # 既定は progress bar のみ。各 session の a,b を逐次見たいときは --verbose
  OPENBLAS_NUM_THREADS=1 python pkl_to_corrected_csv.py --verbose
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

_RUN = Path(__file__).resolve().parent
if _RUN.name == "plot":
    _RUN = _RUN.parent / "run"
sys.path.insert(0, str(_RUN))
import floorz_core as core  # noqa: E402

DEFAULT_INPUT  = Path("outputs/3d")
DEFAULT_OUTPUT = Path("outputs/dev/floorZ_files")

# 出力 CSV の座標スケール（全軸に乗算）。1.0=Blenderメートル, 1/1000=mm相当
OUTPUT_SCALE = 1.0


def process_session(session: str, input_dir: Path, csv_dir: Path,
                    corr_dir: Path, n_frames: int = -1,
                    center=None, scale=None,
                    legacy: bool = False,
                    fit_mode: str = "kde", max_pct: float = 10.0,
                    verbose: bool = False, writer=print) -> dict:
    pkl = core.resolve_pickle(input_dir / session)
    if pkl is None:
        raise FileNotFoundError(f"pickle not found: {input_dir / session}")

    df_raw = core.load_pickle_to_df(pkl, n_frames)
    csv_dir.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(csv_dir / f"{session}_points.csv", index=False)

    # core 由来の warning は捕捉し、progress bar を壊さず writer 経由で出す
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        df_corr, a, b, status, peaks, info = core.scale_and_correct(
            df_raw, output_scale=OUTPUT_SCALE, return_peaks=True,
            center=center, scale=scale, legacy=legacy,
            fit_mode=fit_mode, max_pct=max_pct)
    corr_dir.mkdir(parents=True, exist_ok=True)
    df_corr.to_csv(corr_dir / f"{session}_corrected.csv", index=False)

    warn = core.slope_warn(a)
    extra = (f" max_peak={info['max_peak']:.4f} merge={info['merge_label']}"
             if fit_mode == "kde_max" else "")
    detail = f"[{status}] a={a:.4f} b={b:.4f}{extra}"
    wmsgs = [str(w.message) for w in wlist]
    if warn:
        wmsgs.append("|a-1|>0.15")
    if verbose:
        writer(f"  {detail}" + (f"  WARN: {'; '.join(wmsgs)}" if wmsgs else ""))
    elif wmsgs:   # 非verbose でも warning だけは出す
        writer(f"  [WARN] {session}: {detail} :: {'; '.join(wmsgs)}")
    return {
        "session": session, "animal": session.split("_")[0], "n_frames": len(df_raw),
        "slope": round(a, 4), "intercept": round(b, 4),
        "status": status, "slope_warn": warn,
        # 補正前の検出ピーク (Blender m, 昇順)。plot はこれを読めば pickle 再計算が不要。
        "peaks_m": ";".join(f"{p:.4f}" for p in peaks),
        "max_peak": round(info["max_peak"], 4),
        "merge_label": info["merge_label"],
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                    help=f"pickle のルート (既定: {DEFAULT_INPUT})")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                    help=f"出力ルート (既定: {DEFAULT_OUTPUT})")
    ap.add_argument("--session", default=None,
                    help="セッション名 (省略時は input-dir 以下の全セッション)")
    ap.add_argument("--n-frames", type=int, default=-1, help="処理フレーム数 (-1=全)")
    ap.add_argument("--norm-mode", choices=["global", "per_animal"], default="per_animal",
                    help="XYZ 規格化モード: global=全体共通(従来), per_animal=個体別(新)")
    ap.add_argument("--norm-pct", type=float, nargs=2, default=(2.0, 98.0),
                    metavar=("LO", "HI"),
                    help="規格化のパーセンタイル下限/上限 (既定: 2 98)。例: --norm-pct 1 99。"
                         "global モードでは method.yaml の kp3d_global_norm を percentile "
                         "キャッシュとして使い、未計算なら全 pickle から算出して追記する")
    ap.add_argument("--recompute", action="store_true",
                    help="global モードでキャッシュ済み percentile でも強制再計算して追記")
    ap.add_argument("--fit-mode", choices=["kde", "kde_max"], default="kde",
                    help="垂直アンカー: kde=床KP min の KDE のみ(従来), "
                         "kde_max=全KP max の上位 --max-pct%% を KDE に統合(新)")
    ap.add_argument("--max-pct", type=float, default=10.0,
                    help="kde_max 時の上位パーセンタイル (既定 10。5/1 も可、<=0 で max のみ)")
    ap.add_argument("--legacy", action="store_true",
                    help="旧動作 (Hungarian マッチング) で実行 (比較用)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="各セッションの a,b ログを逐次表示する。"
                         "未指定時は progress bar のみ (warning/error は表示)")
    args = ap.parse_args()

    csv_dir  = args.output_dir / "csv"
    corr_dir = args.output_dir / "corrected"
    log_path = args.output_dir / "affine_log.csv"

    if args.session:
        sessions = [args.session]
    else:
        sessions = sorted(d.name for d in args.input_dir.iterdir() if d.is_dir())
    print(f"{len(sessions)} セッションを処理します  [norm-mode: {args.norm_mode}]")

    # global XYZ 規格化パラメータ（global モード時のみ。percentile キャッシュ参照 or 計算）
    global_norm: tuple | None = None
    if args.norm_mode == "global":
        global_norm = core.get_or_compute_global_norm(
            tuple(args.norm_pct), args.input_dir, args.n_frames,
            recompute=args.recompute)

    # 個体別 XYZ 規格化パラメータの事前計算（per_animal モード時のみ）
    animal_norm: dict[str, tuple] = {}
    if args.norm_mode == "per_animal":
        animal_sessions: dict[str, list[str]] = defaultdict(list)
        for s in sessions:
            animal_sessions[s.split("_")[0]].append(s)

        for animal, asessions in sorted(animal_sessions.items()):
            pkl_paths = [p for s in asessions
                         if (p := core.resolve_pickle(args.input_dir / s)) is not None]
            if not pkl_paths:
                continue
            center, scale = core.compute_animal_norm_stats(
                pkl_paths, args.n_frames, pct=tuple(args.norm_pct))
            animal_norm[animal] = (center, scale)
            print(f"[norm stats] {animal}: "
                  f"center=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) "
                  f"scale=({scale[0]:.1f}, {scale[1]:.1f}, {scale[2]:.1f}) mm "
                  f"({len(pkl_paths)} sessions)")

    # 非 verbose は progress bar 表示。verbose は従来通り逐次ログ。
    bar = None
    iterator = sessions
    if not args.verbose:
        try:
            from tqdm import tqdm
            bar = tqdm(sessions, unit="sess", dynamic_ncols=True)
            iterator = bar
        except ImportError:
            bar = None
    writer = bar.write if bar is not None else print

    records, skipped = [], []
    for session in iterator:
        if args.verbose:
            print(f"\n=== {session} ===")
        elif bar is not None:
            bar.set_description(session)
        animal = session.split("_")[0]
        if args.norm_mode == "global":
            center, scale = global_norm
        else:
            center, scale = animal_norm.get(animal, (None, None))
        try:
            rec = process_session(
                session, args.input_dir, csv_dir, corr_dir, args.n_frames,
                center=center, scale=scale, legacy=args.legacy,
                fit_mode=args.fit_mode, max_pct=args.max_pct,
                verbose=args.verbose, writer=writer)
            rec["norm_mode"] = args.norm_mode
            rec["norm_pct"] = f"{args.norm_pct[0]:g}-{args.norm_pct[1]:g}"
            rec["fit_mode"] = args.fit_mode
            if args.fit_mode == "kde_max":
                rec["max_pct"] = args.max_pct
            records.append(rec)
        except Exception as e:
            writer(f"  [SKIP] {session}: {e}")   # error は常に表示
            skipped.append(session)
    if bar is not None:
        bar.close()

    if records:
        df = pd.DataFrame(records)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(log_path, index=False)
        print(f"\naffine log: {log_path}")
        print(f"\n{'Status':25} {'Count':>6}")
        for s, grp in df.groupby("status"):
            print(f"  {s:23} {len(grp):>6}")
        n_warn = int(df["slope_warn"].sum())
        print(f"\nSlope warnings: {n_warn} ({100 * n_warn / len(df):.0f}%)")
        print(f"Slope  mean={df['slope'].mean():.4f} std={df['slope'].std():.4f} "
              f"min={df['slope'].min():.4f} max={df['slope'].max():.4f}")
    if skipped:
        print(f"\n{len(skipped)} セッションをスキップ: {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()

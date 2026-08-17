#!/usr/bin/env python3
"""全セッション × 全カメラ × フレームチャンク を1本のグローバルキューにして
GPUを動的スケジューリングでレンダリングするオーケストレータ。

従来の run_render.sh はセッション・カメラを逐次ループし、各 (session, camera) の
全チャンクが終わるのを待ってから次へ進んでいた。そのため
  1. 1チャンクが長引くと他GPUが末尾で遊ぶ（チャンクが粗い + 全chunk join）
  2. セッションの末尾でGPUが空いても次セッションが始まらない
という2つのバリアでGPUが余っていた。

ここでは
  - 全セッション×全カメラを細かめのフレームチャンクに割って1リストにまとめ、
  - GPUスロット数だけの warm ワーカー（.blend を読み込んだまま常駐する blender）で
    1本のキューから動的に消化する
ことで、空いたGPUに即座に次のチャンクが入る。

warm 運用のポイント（--cold で従来の1チャンク1プロセス方式に戻せる）:
  - ワーカーは自分がロード済みの .blend のジョブを優先的に取るので、
    セッションをまたぐときだけ blender を起動し直す。
    カメラ切り替えとフレーム範囲の変更はプロセス内で完結する。
  - .blend ロード / GPU コンテキスト初期化 / BVH 構築が「チャンクごと」から
    「セッションごと」に減る。use_persistent_data もチャンクをまたいで効く。
"""
import argparse
import glob
import math
import os
import subprocess
import sys
import threading

# 同ディレクトリのモジュールを再利用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_parallel import (  # noqa: E402
    RENDER_SCRIPT, add_pool_args, add_worker_args, build_blender_cmd, run_one,
    worker_extra_args,
)
from render_worker import resolve_gpu_list, run_warm_pool  # noqa: E402


def count_frames(csv_path):
    """CSVのデータ行数（=全フレーム数）を返す。ヘッダー1行を除く。"""
    with open(csv_path) as f:
        n = sum(1 for _ in f)
    return n - 1


def parse_args():
    p = argparse.ArgumentParser(
        description="全セッション×全カメラをグローバルキューでGPU並列レンダリング"
                    "（デフォルトは blender 常駐の warm 運用）"
    )
    p.add_argument("--cam", action="append", default=None,
                   help="カメラ名（複数指定可、デフォルト: Camera_C Camera_L Camera_R）")
    p.add_argument("--count", type=int, default=1, help="処理するCSV数の上限（デフォルト: 1）")
    p.add_argument("--all-csv", action="store_true", help="すべてのCSVを処理")
    p.add_argument("--frame-start", type=int, default=1, help="開始フレーム（デフォルト: 1）")
    p.add_argument("--frame-end", type=int, default=10, help="終了フレーム（デフォルト: 10）")
    p.add_argument("--all-frames", action="store_true",
                   help="CSVの全フレームを処理（--frame-start/--frame-end を上書き）")
    p.add_argument("--csv-dir", default="outputs/animate_temp/transformedcsv")
    p.add_argument("--blend-dir", default="outputs/blender")
    p.add_argument("--output-dir", default="outputs/render")
    p.add_argument("--resolution", default=None, help="解像度 WxH（例: 1920x1080）")
    p.add_argument("--gpu", default=None,
                   help="使用するGPU ID（カンマ区切り）。省略時は nvidia-smi で自動検出")
    p.add_argument("--workers-per-gpu", type=int, default=1,
                   help="GPU 1台あたりに同時実行する blender 数（デフォルト: 1）。"
                        "増やしすぎるとVRAM不足で逆に遅くなるので実測してから上げること")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="1チャンクのフレーム数（省略時は --chunks-per-slot から自動計算）")
    p.add_argument("--chunks-per-slot", type=int, default=8,
                   help="GPUスロットあたりの目標チャンク数。大きいほど負荷分散が細かくなる。"
                        "warm では blender 再起動が起きないので細かくしても損しない"
                        "（デフォルト: 8）")
    add_worker_args(p)
    add_pool_args(p)
    p.add_argument("--verbose", action="store_true", help="Blender の詳細ログを表示")
    p.add_argument("--dry-run", action="store_true",
                   help="blender を起動せずジョブ分解の計画だけ表示")
    return p.parse_args()


def build_targets(args):
    """選択されたセッションごとに (name, blend, frame_start, frame_end) を返す。"""
    csvs = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not args.all_csv:
        csvs = csvs[: args.count]

    targets = []
    for csv in csvs:
        name = os.path.basename(csv)[:-4]  # 末尾 .csv を除去
        blend = os.path.join(args.blend_dir, f"work_{name}.blend")
        if not os.path.exists(blend):
            print(f"[SKIP] blend が見つかりません: {blend}")
            continue
        if args.all_frames:
            fs, fe = 1, count_frames(csv)
        else:
            fs, fe = args.frame_start, args.frame_end
        if fe < fs:
            print(f"[SKIP] フレーム範囲が不正: {name} ({fs}-{fe})")
            continue
        targets.append((name, blend, fs, fe))
    return targets


def build_jobs(args, targets, cams, chunk_size):
    """全セッション×全カメラ×チャンクのフラットなジョブ辞書リストを作る。

    warm プールは job["blend"] でワーカーの割り当てをまとめるので、同じ .blend の
    ジョブ（全カメラ・全チャンク）は同じ blender プロセスで処理される。
    """
    jobs = []
    for name, blend, fs, fe in targets:
        for cam in cams:
            out = os.path.join(args.output_dir, name, cam)
            prefix = f"{name}_{cam}_"
            s = fs
            while s <= fe:
                e = min(s + chunk_size - 1, fe)
                jobs.append({
                    "blend": blend,
                    "label": f"{name}/{cam} {s}-{e}",
                    "out": out,
                    "prefix": prefix,
                    "start": s,
                    "end": e,
                    "payload": {
                        "camera": cam, "start": s, "end": e,
                        "out": out, "prefix": prefix,
                        "resolution": args.resolution,
                    },
                })
                s = e + 1
    return jobs


def missing_frames(job):
    """チャンク内で出力PNGが存在しないフレーム番号を返す。
    blender 側が例外を投げても exit 0 になることがあるため、ステータスだけでなく
    実際の出力ファイルの有無で成否を判定する。"""
    miss = []
    for frame in range(job["start"], job["end"] + 1):
        path = os.path.join(job["out"], f"{job['prefix']}{frame:05d}.png")
        if not os.path.exists(path):
            miss.append(frame)
    return miss


def main():
    args = parse_args()
    cams = args.cam if args.cam else ["Camera_C", "Camera_L", "Camera_R"]

    gpu_list = resolve_gpu_list(args.gpu)
    gpu_count = len(gpu_list)
    if gpu_count == 0:
        print("GPU が見つかりません。")
        sys.exit(1)
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu は 1 以上にしてください")
    slots = gpu_count * args.workers_per_gpu

    targets = build_targets(args)
    if not targets:
        print("レンダリング対象がありません。")
        return

    if args.chunk_size:
        chunk_size = args.chunk_size
    else:
        total = sum((fe - fs + 1) for _, _, fs, fe in targets) * len(cams)
        chunk_size = max(1, math.ceil(total / (slots * args.chunks_per_slot)))

    jobs = build_jobs(args, targets, cams, chunk_size)
    total_jobs = len(jobs)

    print(
        f"mode={'cold' if args.cold else 'warm'}, sessions={len(targets)}, "
        f"cameras={len(cams)}, GPUs={gpu_count}({','.join(gpu_list)}), "
        f"workers_per_gpu={args.workers_per_gpu}, slots={slots}"
    )
    print(f"chunk_size={chunk_size} frames, total_jobs={total_jobs}")

    if args.dry_run:
        for job in jobs:
            print("  ", job["label"])
        print(f"[dry-run] {total_jobs} jobs（実行はしていません）")
        return

    extra = worker_extra_args(args)
    lock = threading.Lock()
    state = {"done": 0, "failed": []}

    def report(job, ok, reason, gpu):
        with lock:
            state["done"] += 1
            n = state["done"]
            if not ok:
                state["failed"].append(f"{job['label']} ({reason})")
        status = "done" if ok else "FAILED"
        suffix = f" -- {reason}" if reason else ""
        print(f"[{n}/{total_jobs}] {status}: {job['label']} (GPU={gpu}){suffix}",
              flush=True)

    def on_result(job, reply, gpu):
        # 出力ファイルの有無で最終判定する（warm でも cold でも同じ基準）
        miss = missing_frames(job)
        if reply.get("status") == "ok" and not miss:
            report(job, True,
                   f"{reply.get('rendered', 0)}枚 / skip {reply.get('skipped', 0)} "
                   f"({reply.get('sec', 0)}s)", gpu)
            return
        if miss:
            reason = f"出力なし frames={miss[0]}..{miss[-1]} ({len(miss)}枚)"
        else:
            reason = reply.get("message", reply.get("status", "unknown"))
        report(job, False, reason, gpu)

    if args.cold:
        # 従来方式（1チャンク=1プロセス）。warm で問題が出たときの退避用。
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import queue as _queue
        slot_q = _queue.Queue()
        for g in gpu_list:
            for _ in range(args.workers_per_gpu):
                slot_q.put(g)

        def cold_worker(job):
            gpu = slot_q.get()
            cmd = build_blender_cmd(
                job["blend"], job["payload"]["camera"], job["start"], job["end"],
                job["out"], job["prefix"], args.resolution, args.backend,
                args.persistent_data, RENDER_SCRIPT, extra,
            )
            reply = {"status": "ok"}
            try:
                run_one(cmd, gpu, args.verbose, args.cuda_path)
            except subprocess.CalledProcessError as e:
                reply = {"status": "error",
                         "message": f"blender returncode={e.returncode}"}
            finally:
                slot_q.put(gpu)
            on_result(job, reply, gpu)

        with ThreadPoolExecutor(max_workers=slots) as ex:
            futures = [ex.submit(cold_worker, job) for job in jobs]
            for f in as_completed(futures):
                f.result()
    else:
        run_warm_pool(
            jobs, slots, gpu_list, RENDER_SCRIPT, extra, args.cuda_path,
            verbose=args.verbose, progress=args.progress, on_result=on_result,
            restart_every=args.restart_every, job_timeout=args.job_timeout,
        )

    if state["failed"]:
        print(f"\n失敗したジョブ {len(state['failed'])} 件:")
        for label in state["failed"]:
            print("  ", label)
        sys.exit(1)
    print("\nすべてのレンダリングジョブが終了しました")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""warm レンダリングのディスパッチャ側ユーティリティ。

従来（cold）は 1チャンク = blender 1プロセスだったため、チャンクごとに
.blend ロード + GPU コンテキスト初期化 + BVH 構築を払い直していた。
ここでは .blend を読み込んだまま常駐する blender プロセス（warm ワーカー）を
GPU スロット数だけ立ち上げ、render_ipc.py の行JSONプロトコルでジョブを流し込む。

  - ワーカー1本は1つの .blend に紐づく（カメラ・フレーム範囲はジョブごとに変えられる）
  - ジョブは .blend 単位でまとめてキューに入れ、ワーカーは「今ロード済みの .blend」の
    ジョブを優先的に取る。無くなったら別の .blend に移る（そこで初めて再起動）
  - ワーカーが死んだ・応答しない場合はプロセスを作り直して1回だけ再試行する

このモジュール自体は bpy を使わない（システムの python で動く）。
"""
import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_ipc import decode  # noqa: E402

DEFAULT_CUDA_PATH = "/usr/local/cuda-12"
# ワーカー起動（.blend ロード + GPU 初期化）の待ち時間。重いシーンでも足りる程度に長め
START_TIMEOUT = 900
# 起動に連続で失敗したらそのスロットを諦める回数（blender が無い等の設定ミス対策）
MAX_START_FAILS = 3


def make_env(gpu_id=None, cuda_path=None):
    """この blender 子プロセスにだけ効く環境変数を作る。
    グローバルの CUDA 設定は変更しない。"""
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if cuda_path and os.path.isdir(cuda_path):
        env["CUDA_HOME"] = cuda_path
        env["CUDA_PATH"] = cuda_path
        env["PATH"] = os.path.join(cuda_path, "bin") + os.pathsep + env.get("PATH", "")
        lib64 = os.path.join(cuda_path, "lib64")
        env["LD_LIBRARY_PATH"] = lib64 + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def _descendants(pid):
    """pid とその子孫プロセスの集合。snap 版 blender はラッパー経由で起動するため、
    実際に CUDA コンテキストを持つのは子プロセスになることがある。"""
    pids = {int(pid)}
    try:
        r = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True,
                           timeout=10)
        for child in r.stdout.split():
            pids |= _descendants(child)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return pids


def gpu_memory_of(pid):
    """このプロセス（またはその子孫）が GPU を実際に使っているかを nvidia-smi で確認する。

    戻り値: 使用中なら使用メモリ文字列、使っていなければ False、確認不能なら None。
    Cycles は compute_device_type が未設定だと無言で CPU にフォールバックするため、
    「GPUで回っているつもりで実は CPU」を検出するのに使う。
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    family = _descendants(pid)
    for line in r.stdout.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if fields and fields[0].isdigit() and int(fields[0]) in family:
            return fields[1] if len(fields) > 1 else "?"
    return False


def resolve_gpu_list(gpu_arg=None):
    """--gpu の文字列、または nvidia-smi から GPU ID のリストを作る。"""
    if gpu_arg:
        return [g.strip() for g in gpu_arg.split(",") if g.strip()]
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    except FileNotFoundError:
        return []
    return [str(i) for i in range(len(result.stdout.strip().splitlines()))]


class WarmWorker:
    """.blend を読み込んだまま常駐する blender プロセス1本。"""

    def __init__(self, blend, script, extra_args=(), gpu=None,
                 cuda_path=DEFAULT_CUDA_PATH, verbose=False, label="", progress=False,
                 threads=None, cpu_set=None):
        self.blend = blend
        self.script = script
        self.extra_args = list(extra_args)
        self.gpu = gpu
        self.threads = threads
        self.cpu_set = cpu_set
        self.cuda_path = cuda_path
        self.verbose = verbose
        self.progress = progress
        self.label = label or f"GPU={gpu}"
        self.proc = None
        self._replies = queue.Queue()
        self._tail = deque(maxlen=60)  # 失敗時に見せる直近のログ
        self._next_id = 0

    def cmd(self):
        # -t は Blender 全体のタスクスケジューラのスレッド数。指定しないと
        # 1プロセスが全コアぶんのスレッドプールを作り、ワーカーを並べたときに
        # CPU が激しく取り合いになる（実測: 1プロセスで約17コア相当を消費）。
        thr = ["-t", str(self.threads)] if self.threads else []
        # taskset で使用コアを固定する。-t だけでは Blender は全コアぶんの
        # スレッドプールを作り続け（実測 775 スレッド / 1フレームあたり 600 CPU秒、
        # うち半分が sys = スピン待ち）、ワーカーを並べると CPU を食い潰す。
        pin = ["taskset", "-c", self.cpu_set] if self.cpu_set else []
        return pin + ["blender", self.blend, "-b"] + thr \
            + ["-P", self.script, "--", "--serve"] + self.extra_args

    def start(self):
        self.proc = subprocess.Popen(
            self.cmd(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env=make_env(self.gpu, self.cuda_path),
        )
        threading.Thread(target=self._read_loop, daemon=True).start()
        try:
            msg = self._replies.get(timeout=START_TIMEOUT)
        except queue.Empty:
            self.stop()
            raise RuntimeError(f"[{self.label}] ワーカーの起動がタイムアウトしました")
        if msg.get("event") != "ready":
            self.stop()
            raise RuntimeError(f"[{self.label}] ワーカーの起動に失敗しました\n"
                               + self.tail_text())
        return msg

    def _read_loop(self):
        """blender の stdout を読み続ける。制御メッセージだけキューに流す。

        universal-newline 読みなので、Cycles の `\\r` 終端の進捗行も1行として届く。
        drain し続けないと blender 側が stdout の書き込みでブロックする。"""
        try:
            for raw in self.proc.stdout:
                msg = decode(raw)
                if msg is None:
                    self._tail.append(raw.rstrip())
                    if self.verbose:
                        print(f"[{self.label}] {raw.rstrip()}", flush=True)
                    continue
                if msg.get("event") == "frame":
                    if self.progress:
                        print(f"[{self.label}] frame {msg['frame']} "
                              f"({msg['sec']}s)", flush=True)
                    continue
                self._replies.put(msg)
        except Exception as e:  # パイプが壊れた等
            self._tail.append(f"(stdout 読み取り終了: {e})")
        finally:
            self._replies.put({"status": "dead"})

    def submit(self, job, timeout=None):
        """1ジョブを投げて完了を待つ。戻り値は status を持つ dict。"""
        self._next_id += 1
        payload = dict(job, id=self._next_id, cmd="render")
        try:
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as e:
            return {"status": "dead", "message": f"stdin への書き込み失敗: {e}"}
        try:
            msg = self._replies.get(timeout=timeout)
        except queue.Empty:
            return {"status": "timeout",
                    "message": f"ジョブが {timeout}s 以内に終わりませんでした"}
        if msg.get("status") == "dead":
            msg = dict(msg, message=msg.get("message", "blender が終了しました"),
                       tail=self.tail_text())
        return msg

    def tail_text(self):
        return "\n".join(f"    | {line}" for line in self._tail)

    def stop(self, timeout=30):
        if not self.proc:
            return
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc = None


class BlendJobQueue:
    """.blend 単位でまとめたジョブキュー。

    ワーカーは今ロード済みの .blend のジョブを優先して取り、無くなったら
    「担当ワーカーが最も少なく、残ジョブが最も多い .blend」へ移る。
    こうすると blender の再起動回数が最小に近づきつつ、末尾でスロットが遊ばない。
    """

    def __init__(self, jobs):
        self._groups = {}
        self._order = []
        for j in jobs:
            b = j["blend"]
            if b not in self._groups:
                self._groups[b] = deque()
                self._order.append(b)
            self._groups[b].append(j)
        self._active = {b: 0 for b in self._order}
        self._lock = threading.Lock()

    def __len__(self):
        return sum(len(g) for g in self._groups.values())

    def take(self, current=None):
        """次のジョブを返す。もう無ければ None。"""
        with self._lock:
            g = self._groups.get(current)
            if g:
                return g.popleft()
            if current is not None:
                self._active[current] = max(0, self._active[current] - 1)
            candidates = [b for b in self._order if self._groups[b]]
            if not candidates:
                return None
            best = min(candidates, key=lambda b: (self._active[b], -len(self._groups[b])))
            self._active[best] += 1
            return self._groups[best].popleft()

    def release(self, current):
        with self._lock:
            if current in self._active:
                self._active[current] = max(0, self._active[current] - 1)


def auto_threads(slots):
    """ワーカー1本あたりに渡す blender -t の値。

    実測（A100 / 500x500 / samples=32 / 1ワーカー、1枚あたり）:
        -t 無制限 : wall 15.9s, CPU 130.8 core-s (平均 8.2 コア占有)
        -t 64     : wall 14.6s, CPU 145.7 core-s (平均 9.9 コア占有)
        -t 16     : wall 17.4s, CPU  78.6 core-s (平均 4.5 コア占有)
        -t 4      : wall 28.9s, CPU  69.4 core-s (平均 2.4 コア占有)
    -t を絞ると CPU 消費が半分近くまで落ち、そのぶんワーカーを増やせるので
    ノード全体のスループットは上がる。ただし小さすぎると1枚の wall が伸びる
    （4 は 16 の1.7倍）ので 16 を上限とする。
    実占有コアは -t の 1/3 程度なので、公平割り当ての2倍を目安に配る。
    """
    ncpu = os.cpu_count() or 8
    fair = ncpu / max(1, slots)
    return int(max(2, min(16, round(fair * 2))))


def slot_cpu_set(slot, slots, cores_per_slot=None):
    """スロット番号から taskset に渡すコア範囲文字列を作る（重ならないよう山分け）。"""
    ncpu = os.cpu_count() or 8
    k = cores_per_slot or max(1, ncpu // max(1, slots))
    start = (slot * k) % ncpu
    end = min(start + k, ncpu) - 1
    return f"{start}-{end}" if end > start else str(start)


def run_warm_pool(jobs, slots, gpu_list, script, extra_args=(), cuda_path=DEFAULT_CUDA_PATH,
                  verbose=False, progress=False, on_result=None, restart_every=0,
                  job_timeout=None, retries=1, threads=None, pin_cpu=False):
    """jobs を warm ワーカー slots 本で消化する。

    jobs: [{"blend": path, "label": str, "payload": {...ジョブJSON...}}, ...]
    on_result(job, reply, gpu) が各ジョブの完了時に（別スレッドから）呼ばれる。
    戻り値は失敗したジョブの (job, reply) リスト。
    """
    q = BlendJobQueue(jobs)
    failures = []
    lock = threading.Lock()

    def slot_loop(slot):
        gpu = gpu_list[slot % len(gpu_list)] if gpu_list else None
        cpu_set = slot_cpu_set(slot, slots, threads) if pin_cpu else None
        worker = None
        current = None
        since_start = 0
        start_fails = 0
        gpu_verified = False
        while True:
            job = q.take(current)
            if job is None:
                break
            need_restart = (worker is None or current != job["blend"]
                            or (restart_every and since_start >= restart_every))
            attempt = 0
            reply = None
            while True:
                if need_restart:
                    if worker is not None:
                        worker.stop()
                        worker = None
                    label = f"GPU={gpu}" if gpu is not None else f"slot{slot}"
                    try:
                        worker = WarmWorker(
                            job["blend"], script, extra_args, gpu, cuda_path,
                            verbose, label, progress, threads, cpu_set)
                        worker.start()
                    except Exception as e:
                        worker = None
                        reply = {"status": "error", "message": str(e)}
                        start_fails += 1
                        break
                    start_fails = 0
                    current = job["blend"]
                    since_start = 0
                    need_restart = False
                    gpu_verified = False
                reply = worker.submit(job["payload"], timeout=job_timeout)
                if reply.get("status") == "ok":
                    break
                attempt += 1
                # ワーカーが死んだ/固まった場合だけ作り直して再試行する。
                # status=="error" はワーカーが生きたまま返した恒久的な失敗
                # （カメラ名の誤り等）なので、作り直しても同じ結果になる。
                if reply.get("status") not in ("dead", "timeout") or attempt > retries:
                    break
                need_restart = True

            # 最初の1枚が終わった時点で、このプロセスが本当に GPU を使っているか確認する
            # （CUDA コンテキストはレンダー開始時に作られるので起動直後には見えない）
            if reply.get("status") == "ok" and not gpu_verified and worker is not None:
                gpu_verified = True
                mem = gpu_memory_of(worker.proc.pid)
                if mem is False:
                    print(f"[{worker.label}] 警告: このプロセスが GPU を使っている形跡が"
                          f"ありません（nvidia-smi の compute-apps に不在）。"
                          f"CPU レンダリングになっている可能性があります。", flush=True)
                elif mem:
                    print(f"[{worker.label}] GPU 使用を確認 (VRAM {mem})", flush=True)

            since_start += 1
            with lock:
                if reply.get("status") != "ok":
                    failures.append((job, reply))
            if on_result:
                on_result(job, reply, gpu)
            if worker is None:
                # 起動できなかった。回復の見込みが無ければこのスロットは降りる
                q.release(job["blend"])
                current = None
                if start_fails >= MAX_START_FAILS:
                    print(f"[slot{slot}] ワーカーを {MAX_START_FAILS} 回連続で起動できません"
                          f"でした。このスロットを停止します。", flush=True)
                    break
        if worker is not None:
            worker.stop()
        q.release(current)

    # 変数名を threads にすると引数の threads（1ワーカーのCPUスレッド数）を隠す
    slot_threads = [threading.Thread(target=slot_loop, args=(i,)) for i in range(slots)]
    for t in slot_threads:
        t.start()
    for t in slot_threads:
        t.join()

    # スロットが全滅した場合、未処理ジョブが残る
    leftover = []
    while True:
        job = q.take(None)
        if job is None:
            break
        reply = {"status": "error", "message": "未処理（ワーカーを起動できませんでした）"}
        leftover.append((job, reply))
        if on_result:
            on_result(job, reply, None)
    return failures + leftover

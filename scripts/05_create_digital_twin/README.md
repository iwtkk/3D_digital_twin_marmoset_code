# 05 – Create Digital Twin

Apply 3D keypoints to the rigged marmoset model in Blender and render
pseudo-first-person-view videos.

## Requirements

- Blender 4.5.0
- Blender scene file: `3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/digital_twin_marmoset.blend`
- NVIDIA GPU（CUDA対応）— レンダリングは CYCLES エンジン + CUDA で実行されます（動作確認環境: NVIDIA A100-SXM4-40GB × 7, CUDA 12.7）

## Input

- `kp3d_fxdJointLen.pickle` from step 03 (`outputs/3d/{session}/kp3d_fxdJointLen.pickle`)

## Step 1 – Convert pickle to CSV

```bash
# 全セッション一括
python scripts/05_create_digital_twin/kp3d_to_blender_csv.py --all

# 1セッションのみ
python scripts/05_create_digital_twin/kp3d_to_blender_csv.py dailylife_cj611_20230227_100000
```

## Step 2 – Apply keypoints to Blender rig

Blender の Scripting ワークスペースで `blender_animation.py` を開いて実行するか、CLI で実行します:

```bash
blender 3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/digital_twin_marmoset.blend \
  --background \
  --python scripts/05_create_digital_twin/blender_animation.py \
  -- \
  outputs/animate_temp/transformedcsv/{session}.csv
```

`--` 以降のオプション:

| 引数 | デフォルト | 説明 |
|---|---|---|
| `<csv_path>` | （必須） | Step 1 で生成した CSV ファイルのパス |
| `--armature <name>` | `NF_Armature` | Blender シーン内のアーマチュアオブジェクト名 |

別のアーマチュアを使う場合:

```bash
blender scene.blend \
  --background \
  --python scripts/05_create_digital_twin/blender_animation.py \
  -- \
  outputs/animate_temp/transformedcsv/{session}.csv \
  --armature MyArmature
```

## Step 3 – Render pseudo-view

Blender の Scripting ワークスペースで `allcamrender.py` を実行するか、並列レンダリングスクリプトを使います:

```bash
python scripts/05_create_digital_twin/render_parallel.py \
  --blend 3D_digital_twin_marmoset_data/3D_digital_twin_marmoset_data/digital_twin_marmoset.blend \
  --camera Camera_R \
  --entire_start 1 \
  --entire_end 3000 \
  --out outputs/render/{session}/Camera_R \
  --chunk_size 100 \
  --max_jobs 4
```

### warm 運用（デフォルト）

レンダリング系スクリプトは **blender を常駐させる warm 運用** が既定です。
GPU スロット数だけ blender を立ち上げ、チャンクを stdin 経由で流し込みます
（プロトコルは `render_ipc.py`、ディスパッチャは `render_worker.py`）。

- `.blend` のロード / CUDA・OptiX コンテキスト初期化 / BVH 構築 / シェーダコンパイルは
  **ワーカーあたり1回**。従来はチャンクごとに払っていた。
- `use_persistent_data` はプロセス内でしか効かないため、warm ではチャンクをまたいで効く。
- カメラ切り替えとフレーム範囲の変更はプロセス内で完結する。`render_sessions.py` では
  ワーカーは自分がロード済みの `.blend` のジョブを優先して取り、**セッションをまたぐ時だけ**
  blender を起動し直す。
- 実測（A100, 320x320, samples=16）: 1枚目 70s → 2枚目 20s → 以降 12〜15s/枚。
  cold 運用ではチャンクごとにこの初回コストを払い直していた。
- 問題が起きたら `--cold` で従来の「1チャンク=1プロセス」方式に戻せる。

`blender_render/tools/` の検証結果に基づく設定も渡せます:

| オプション | 内容 |
|---|---|
| `--optimize` | デノイザーOptiX・Simplify+テクスチャ上限・indirect clamp・adaptive threshold・ガラスの影オフをまとめて適用（**画が変わる**のでオプトイン） |
| `--samples N` | Cycles サンプル数の上書き。**カラーの既定は 32**（depth は 1）。`0` 以下で .blend の設定をそのまま使う。値は `blender_render/tools/run_sample_sweep.sh` で検証する |
| `--denoiser {OPTIX,OPENIMAGEDENOISE,NONE}` | デノイザー指定 |
| `--allow-cpu-fallback` | GPU を有効化できなくても CPU で続行（既定はエラー停止） |
| `--threads N` | blender 1本あたりの CPU スレッド数（`blender -t`）。**既定は自動（コア数÷スロット数×2、上限16）**。`0` で無制限。下記の実測表のとおりスループットへの影響が最も大きいパラメータ |
| `--restart-every N` | ワーカーが N ジョブ処理したら blender を再起動（長時間運用の保険） |
| `--job-timeout SEC` | チャンクが固まったらワーカーを作り直して再試行 |
| `--progress` | フレーム単位の所要時間を表示 |
| `--blender-arg X` | `render_range*.py` へ任意の引数を素通し |

GPU が有効化できないと Cycles は**無言で CPU にフォールバックして数十倍遅くなる**ため、
既定ではエラーで停止します（`blender_render/tools/README.md` §0）。

### GPU が本当に使われているかの確認方法

Cycles は「GPU を選んだつもりで実は CPU」という状態になりやすい（デスクトップで発生）。
本パイプラインには3段の検出を入れてある:

1. **起動時**: `prefs.get_devices_for_type(backend)` で*実在する*デバイスを取り、
   `prefs.has_active_device()` が False なら起動を中止する（`--allow-cpu-fallback` で続行可）。
   `prefs.devices` は過去にそのマシンで見えたデバイスが userpref に残り続けるので当てにならない
   （GPU1枚しか見えない状況で13件並ぶことを実測）。ログには実在デバイスだけを出す。
2. **1枚目のレンダー後**: ディスパッチャが `nvidia-smi --query-compute-apps` に
   ワーカーPID（およびその子プロセス）が載っているかを確認し、
   `GPU 使用を確認 (VRAM 4272MiB)` か警告を出す。CUDA コンテキストはレンダー開始時に
   作られるので、起動直後では判定できない。
3. **手動での確定診断**: `check_gpu_device.py` がサンプル数を振って所要時間の伸びを見る。
   GPU で回っていればサンプルを増やしても時間はほとんど増えない。

   ```bash
   blender outputs/blender/work_{session}.blend -b \
     -P scripts/05_create_digital_twin/check_gpu_device.py -- \
     --camera Camera_C --res 1000x1000 --frame 3000 --samples 32,256
   ```

   実測（1000×1000・空きGPU・デノイズOFF）:

   | device | samples=32 | samples=256 | 追加224サンプル分 |
   |---|---|---|---|
   | GPU | 27.4s | 22.1s | ほぼ 0（1枚目の初期化コストの方が大きい）|
   | CPU（256コア）| 15.4s | 79.9s | +64.5s（288 ms/サンプル）|

   この差（GPU は samples に対してほぼ平坦）が GPU 稼働の証拠になる。レンダー中は
   `nvidia-smi` の GPU 使用率も 99〜100% に達する。

**注意**: samples=32 のような低サンプルだと、A100 より 256コアEPYC の方が1枚は速い
（15.4s vs 22.1s）。GPU が遊んでいるように見えても異常ではなく、1枚の時間が
GPU パストレースではなく CPU 側処理（シーン同期・転送・film・PNG書き出し）で
決まっているだけ。本番の samples=500 では GPU が圧倒的に速い。

### 性能特性（実測 / A100・1000×1000・本番 blend `work_*.blend`）

| 条件 | 1枚あたり wall | 1枚あたり CPU | 起動コスト(CPU) |
|---|---|---|---|
| samples=500（blend既定）| 42〜45s | – | – |
| samples=128 | 13〜19s | – | – |
| samples=32（現在の既定）| 10〜12s | 153 core-s | 453 core-s |
| samples=32 + `--pin-cpu`(6コア) | 43s | 70 core-s | 30 core-s |

`--threads`（blender `-t`）別（500×500・samples=32・1ワーカー・1枚あたり）:

| 設定 | wall/枚 | CPU/枚 | 平均コア占有 | 起動CPU | 256コアで収容できるワーカー数 | 総スループット |
|---|---|---|---|---|---|---|
| 0（無制限）| 15.9s | 130.8 core-s | 8.2 | 163.8s | 31 | 7,100枚/時 |
| 64 | 14.6s | 145.7 core-s | 9.9 | 48.2s | 26 | 6,400枚/時 |
| **16** | 17.4s | 78.6 core-s | 4.5 | 18.8s | 57 | **11,800枚/時** |
| 4 | 28.9s | 69.4 core-s | 2.4 | 16.4s | 107 | 13,300枚/時 |
| 4 + `--pin-cpu` | 24.3s | 67.2 core-s | 2.8 | 15.7s | 91 | 13,500枚/時 |
| 16 + `--pin-cpu` | 19.5s | 94.1 core-s | 4.8 | 20.6s | 53 | 9,800枚/時 |

- `-t` を絞ると **1枚あたり CPU がほぼ半減**（131→70 core-s）。起動コストも 164→16 core-s。
  そのぶんワーカーを増やせるので、ノード全体のスループットは 1.7〜1.9倍になる。
- 小さすぎると1枚の wall が伸びる（4 は 16 の1.7倍）。**16 前後が最良バランス**。
- 64 や無制限は無制限とほぼ同じで、CPU を無駄に食うだけ。
- `--pin-cpu` の追加効果は `-t` 併用時には小さい（wall −16%, CPU −3%）。効くのは `-t` の方。
- 既定値は「コア数÷スロット数 ×2、上限16」（実占有コアは `-t` の約1/3のため）。

解像度別（samples=32、1000×1000 → 500×500 → 320×320）:

| 解像度 | CPU/枚（無制限）| CPU/枚（6コアpin）| wall/枚（6コアpin）| PNG 1枚 |
|---|---|---|---|---|
| 1000×1000 | 153 core-s | 70 core-s | 43.0s | 約1.2MB |
| 500×500 | 144 core-s | 73 core-s | **25.4s** | 約238KB |
| 320×320 | 132 core-s | – | – | 約157KB |

読み方:

- **samples を 500→32 にしても速くなるのは約4倍が上限**。1枚には samples に依らない
  固定コスト（シーン評価・出力）があるため。
- **CPU が律速**。8ワーカー/GPU で回すと GPU 使用率は 12% しかなく、blender 1本が
  約17コア（775スレッド）を掴む。解像度を 1/9.8（320×320）にしても CPU は
  153→132 core-s にしか下がらない ＝ ピクセルではなく**シーン評価（リグ＋43万ポリゴンの
  depsgraph 同期）が主因**。
- CPU を絞ると 1枚あたり CPU が半減し（153→70 core-s、うち sys 時間は 302→15 秒）、
  無駄なスピン待ちが消える。ワーカー単体の wall は伸びるが、多数のワーカーを並べた
  ときの総スループットは上がる。**効くのは `--threads`（blender -t）で、`--pin-cpu`
  は併用しても追加効果は小さい**（上の表を参照）。
- **解像度を下げる効果はスレッド無制限だとほぼ無い**（1000²→500² で CPU は 6% 減）。
  一方 `--pin-cpu` 併用時は 1枚 43s→25.4s（1.7倍）になる。コアを絞ると CPU 側の
  ピクセル処理（film変換・PNG圧縮・デノイズ）が律速に回るため。
  出力サイズは 1.2MB→238KB（1/5）で、後段の I/O にも効く。
- ワーカーを増やす前に `--pin-cpu` と `--workers_per_gpu` を実測で決めること。
  VRAM（1本 約1.7GB）より先に CPU が先に飽和する。

またはシェルスクリプトを使います:

```bash
# Step 2 + Step 3 を逐次実行（デフォルト: NF_Armature）
bash scripts/05_create_digital_twin/run_all.sh

# Step 2 のみ並列実行（処理済み .blend はスキップ）
bash scripts/05_create_digital_twin/run_animate_parallel.sh

# アーマチュア名を指定する場合
bash scripts/05_create_digital_twin/run_all.sh --armature MyArmature
bash scripts/05_create_digital_twin/run_animate_parallel.sh --armature MyArmature

# blend ファイルも合わせて指定する場合
bash scripts/05_create_digital_twin/run_all.sh --blend custom.blend --armature MyArmature
bash scripts/05_create_digital_twin/run_animate_parallel.sh --blend custom.blend --armature MyArmature
```

## Scripts

| Script | Purpose |
|---|---|
| `blender_animation.py` | Apply 3D keypoints to IK rig |
| `allcamrender.py` | Render from all camera positions (Blender script) |
| `render_sessions.py` | 全セッション×全カメラを1本のキューにして GPU 並列レンダリング（`run_render.sh` の実体） |
| `render_parallel.py` | 1 blend / 1カメラのフレーム範囲を分割して並列レンダリング |
| `render_range.py` | Blender 側ワーカー（PNG）。`--serve` で warm 常駐、無指定で従来の1回実行 |
| `render_parallel_depth.py` | `render_parallel.py` の Depth(Z) 版 |
| `render_range_depth.py` | Blender 側ワーカー（Depth EXR）。`--serve` 対応 |
| `render_worker.py` | warm ワーカーのプロセス管理・blend 単位のジョブスケジューラ（ディスパッチャ側） |
| `render_ipc.py` | warm ワーカーとの行JSONプロトコル定義 |
| `check_gpu_device.py` | Cycles が実際に GPU でレンダリングしているかを判定する診断（Blender スクリプト） |
| `kp3d_to_blender_csv.py` | Convert kp3d pickle to Blender-ready CSV |
| `run_all.sh` | Step 2 + Step 3 を逐次実行（1CSV ずつ順番に） |
| `run_animate_parallel.sh` | Step 2 のみを並列実行（最大 `MAX_JOBS` 同時、処理済みはスキップ） |
| `run_render.sh` | Step 3 のみ実行 |

## See also

- [docs/blender_usage.md](../../docs/blender_usage.md)
- [docs/bone_mapping.md](../../docs/bone_mapping.md)

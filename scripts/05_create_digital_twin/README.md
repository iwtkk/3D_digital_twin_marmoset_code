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
| `--threads N` | blender 1本あたりの CPU スレッド数（`blender -t`）。**既定はコア数÷スロット数の自動配分**。`0` で無制限 |
| `--restart-every N` | ワーカーが N ジョブ処理したら blender を再起動（長時間運用の保険） |
| `--job-timeout SEC` | チャンクが固まったらワーカーを作り直して再試行 |
| `--progress` | フレーム単位の所要時間を表示 |
| `--blender-arg X` | `render_range*.py` へ任意の引数を素通し |

GPU が有効化できないと Cycles は**無言で CPU にフォールバックして数十倍遅くなる**ため、
既定ではエラーで停止します（`blender_render/tools/README.md` §0）。

### 性能特性（実測 / A100・1000×1000・本番 blend `work_*.blend`）

| 条件 | 1枚あたり wall | 1枚あたり CPU | 起動コスト(CPU) |
|---|---|---|---|
| samples=500（blend既定）| 42〜45s | – | – |
| samples=128 | 13〜19s | – | – |
| samples=32（現在の既定）| 10〜12s | 153 core-s | 453 core-s |
| samples=32 + `--pin-cpu`(6コア) | 43s | 70 core-s | 30 core-s |

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
- `--pin-cpu` で使用コアを固定すると 1枚あたり CPU が半減し（153→70 core-s、うち
  sys 時間は 302→15 秒）、無駄なスピン待ちが消える。ワーカー単体の wall は伸びるが、
  多数のワーカーを並べたときの総スループットは上がる。
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
| `kp3d_to_blender_csv.py` | Convert kp3d pickle to Blender-ready CSV |
| `run_all.sh` | Step 2 + Step 3 を逐次実行（1CSV ずつ順番に） |
| `run_animate_parallel.sh` | Step 2 のみを並列実行（最大 `MAX_JOBS` 同時、処理済みはスキップ） |
| `run_render.sh` | Step 3 のみ実行 |

## See also

- [docs/blender_usage.md](../../docs/blender_usage.md)
- [docs/bone_mapping.md](../../docs/bone_mapping.md)

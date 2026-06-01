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
| `render_parallel.py` | Parallel chunk rendering |
| `render_range.py` | Render a specific frame range |
| `kp3d_to_blender_csv.py` | Convert kp3d pickle to Blender-ready CSV |
| `run_all.sh` | Step 2 + Step 3 を逐次実行（1CSV ずつ順番に） |
| `run_animate_parallel.sh` | Step 2 のみを並列実行（最大 `MAX_JOBS` 同時、処理済みはスキップ） |
| `run_render.sh` | Step 3 のみ実行 |

## See also

- [docs/blender_usage.md](../../docs/blender_usage.md)
- [docs/bone_mapping.md](../../docs/bone_mapping.md)

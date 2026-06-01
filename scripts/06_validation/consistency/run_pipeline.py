"""
検証パイプライン実行スクリプト

1. random_frame_selector: ランダムフレーム選択
2. render_from_selection: レンダリング実行
3. extract_frames: 実画像フレーム抽出
4. calculate_metrics: PSNR/SSIM計算
5. visualize: 可視化
6. outlier_detector: 外れ値検出
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# パス設定
current_script = Path(__file__).resolve()
validation_dir = current_script.parent
# consistency/ → 06_validation/ → scripts/ → project root
project_root = validation_dir.parent.parent.parent
marmo3d_dir = project_root / "src" / "marmo3d"


@dataclass
class PipelineConfig:
    """パイプライン設定"""
    # 基本設定
    seed: int = 42
    frames_per_animal: int = 100
    animals: List[str] = field(default_factory=lambda: ["cj611", "cj357", "cj425"])

    # レンダリング設定
    profile: str = "env_camera"
    split: int = 1
    workers: int = 1
    session_process: int = 1

    # 外れ値検出設定
    psnr_multiplier: float = 1.5
    ssim_multiplier: float = 1.5
    max_outlier_images: int = 20

    # パス設定
    output_dir: Optional[Path] = None
    valid_config: Optional[Path] = None

    # 実行制御
    dry_run: bool = False
    skip_frame_selection: bool = False
    skip_render: bool = False
    skip_extract: bool = False
    skip_metrics: bool = False
    render_hash: Optional[str] = None  # 既存のrender結果を使う場合


@dataclass
class PipelineStep:
    """パイプラインステップ"""
    name: str
    command: List[str]
    description: str
    skip: bool = False


def generate_render_hash(config: PipelineConfig) -> str:
    """render_paths.compute_render_hash と同じロジックで hash を生成（.blendも含む）"""
    import yaml
    render_spec_path = project_root / "config" / "render_spec.yaml"
    with open(render_spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    sys.path.insert(0, str(project_root / "src"))
    from marmo3d.blender.render_paths import compute_render_hash
    return compute_render_hash(spec, project_root)


def run_step(step: PipelineStep, verbose: bool = True) -> bool:
    """
    1ステップを実行

    Returns:
        成功したらTrue
    """
    if step.skip:
        print(f"\n{'='*60}")
        print(f"SKIP: {step.name}")
        print(f"  {step.description}")
        print(f"{'='*60}")
        return True

    print(f"\n{'='*60}")
    print(f"STEP: {step.name}")
    print(f"  {step.description}")
    print(f"{'='*60}")

    if verbose:
        print(f"Command: {' '.join(step.command)}")
        print()

    try:
        result = subprocess.run(
            step.command,
            cwd=project_root,
            check=True,
        )
        print(f"\n[OK] {step.name} 完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {step.name} 失敗: return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] {step.name} 失敗: {e}")
        return False


def build_pipeline(config: PipelineConfig) -> List[PipelineStep]:
    """パイプラインを構築"""

    # 出力ディレクトリ
    if config.output_dir:
        output_dir = config.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / "validation" / "consistency" / f"run_{timestamp}"

    output_dir = Path(output_dir)

    # 各種パス
    selection_json = output_dir / "selected_frames.json"
    frames_cache = output_dir / "frames_cache.json"
    render_output = project_root / "outputs" / "validation" / "consistency" / "render"
    frames_output = output_dir / "frames"
    metrics_json = output_dir / "metrics.json"
    figures_dir = output_dir / "figures"
    outliers_dir = output_dir / "outliers"

    # render_hash
    if config.render_hash:
        render_hash = config.render_hash
    else:
        render_hash = generate_render_hash(config)

    # consistency.yaml パス
    valid_config_path = config.valid_config or (project_root / "config" / "consistency.yaml")

    steps = []

    # Step 1: random_frame_selector
    step1_cmd = [
        sys.executable,
        str(marmo3d_dir / "blender" / "random_frame_selector.py"),
        "--animals", *config.animals,
        "--frames-per-animal", str(config.frames_per_animal),
        "--seed", str(config.seed),
        "--output", str(selection_json),
        "--format", "both",
        "--cache", str(frames_cache),
    ]
    steps.append(PipelineStep(
        name="1. Random Frame Selection",
        command=step1_cmd,
        description=f"個体ごとに{config.frames_per_animal}フレームをランダム選択",
        skip=config.skip_frame_selection,
    ))

    # Step 2: render_from_selection
    step2_cmd = [
        sys.executable,
        str(marmo3d_dir / "blender" / "render_from_selection.py"),
        "--input", str(selection_json),
        "--profile", config.profile,
        "--split", str(config.split),
        "--workers", str(config.workers),
        "--session-process", str(config.session_process),
        "--output-root", str(render_output),
    ]
    if config.dry_run:
        step2_cmd.append("--dry-run")

    steps.append(PipelineStep(
        name="2. Render from Selection",
        command=step2_cmd,
        description=f"選択フレームをBlenderでレンダリング (profile: {config.profile})",
        skip=config.skip_render,
    ))

    # Step 3: extract_frames
    step3_cmd = [
        sys.executable,
        str(validation_dir / "extract_frames.py"),
        "--input", str(selection_json),
        "--output-root", str(frames_output),
        "--valid-config", str(valid_config_path),
        "--verbose",
    ]
    steps.append(PipelineStep(
        name="3. Extract Frames",
        command=step3_cmd,
        description="viddataから実画像フレームを抽出",
        skip=config.skip_extract,
    ))

    # Step 4: calculate_metrics
    step4_cmd = [
        sys.executable,
        str(validation_dir / "calculate_metrics.py"),
        "--render-hash", render_hash,
        "--frames-root", str(frames_output),
        "--render-root", str(render_output),
        "--output", str(metrics_json),
        "--valid-config", str(valid_config_path),
        "--verbose",
    ]
    steps.append(PipelineStep(
        name="4. Calculate Metrics",
        command=step4_cmd,
        description="PSNR/SSIMを計算",
        skip=config.skip_metrics,
    ))

    # Step 5: visualize
    step5_cmd = [
        sys.executable,
        str(validation_dir / "visualize.py"),
        "--metrics", str(metrics_json),
        "--output-dir", str(figures_dir),
        "--valid-config", str(valid_config_path),
    ]
    steps.append(PipelineStep(
        name="5. Visualize",
        command=step5_cmd,
        description="分布図・散布図を生成",
    ))

    # Step 6: outlier_detector
    step6_cmd = [
        sys.executable,
        str(validation_dir / "outlier_detector.py"),
        "--metrics", str(metrics_json),
        "--render-hash", render_hash,
        "--output-dir", str(outliers_dir),
        "--valid-config", str(valid_config_path),
        "--psnr-multiplier", str(config.psnr_multiplier),
        "--ssim-multiplier", str(config.ssim_multiplier),
        "--max-images", str(config.max_outlier_images),
        "--frames-root", str(frames_output),
        "--render-root", str(render_output),
    ]
    steps.append(PipelineStep(
        name="6. Outlier Detection",
        command=step6_cmd,
        description="外れ値を検出・可視化",
    ))

    return steps


def main():
    parser = argparse.ArgumentParser(
        description="検証パイプライン実行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
実行例:
  # 基本実行（新規）
  python run_pipeline.py

  # Step 4-6 のみ実行（既存結果を使う）
  python run_pipeline.py \\
    --skip-frame-selection \\
    --skip-render \\
    --skip-extract \\
    --render-hash 1228b6de \\
    --output-dir valid/run_20260108_144800

  # Step 5-6 のみ実行（メトリクス計算済み）
  python run_pipeline.py \\
    --skip-frame-selection \\
    --skip-render \\
    --skip-extract \\
    --skip-metrics \\
    --render-hash 1228b6de \\
    --output-dir valid/run_20260108_144800

  # フレーム数とシードを指定
  python run_pipeline.py --frames-per-animal 50 --seed 123

  # 特定の個体のみ
  python run_pipeline.py --animals cj611 cj357

  # dry-run（レンダリングコマンドのみ表示）
  python run_pipeline.py --dry-run
        """
    )

    # 基本設定
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (デフォルト: 42)",
    )
    parser.add_argument(
        "--frames-per-animal",
        type=int,
        default=100,
        help="各個体から選択するフレーム数 (デフォルト: 100)",
    )
    parser.add_argument(
        "--animals",
        nargs="+",
        default=["cj611", "cj357", "cj425"],
        help="対象の個体ID (デフォルト: cj611 cj357 cj425)",
    )

    # レンダリング設定
    parser.add_argument(
        "--profile",
        type=str,
        default="env_camera",
        help="レンダリングプロファイル (デフォルト: env_camera)",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="分割数 (デフォルト: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="並列ワーカー数 (デフォルト: 1)",
    )
    parser.add_argument(
        "--session-process",
        type=int,
        default=1,
        help="セッション並列数 (デフォルト: 1)",
    )

    # 外れ値検出設定
    parser.add_argument(
        "--psnr-multiplier",
        type=float,
        default=1.5,
        help="PSNR用のIQR倍率 (デフォルト: 1.5)",
    )
    parser.add_argument(
        "--ssim-multiplier",
        type=float,
        default=1.5,
        help="SSIM用のIQR倍率 (デフォルト: 1.5)",
    )
    parser.add_argument(
        "--max-outlier-images",
        type=int,
        default=20,
        help="外れ値可視化の最大数 (デフォルト: 20)",
    )

    # パス設定
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ディレクトリ (デフォルト: valid/run_{timestamp})",
    )
    parser.add_argument(
        "--valid-config",
        type=str,
        default=None,
        help="consistency.yaml のパス (デフォルト: config/consistency.yaml)",
    )

    # 実行制御
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="レンダリングを実行せずコマンドのみ表示",
    )
    parser.add_argument(
        "--skip-frame-selection",
        action="store_true",
        help="フレーム選択をスキップ（既存結果を使用）",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="レンダリングをスキップ（既存結果を使用）",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="フレーム抽出をスキップ（既存結果を使用）",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="メトリクス計算をスキップ（既存結果を使用）",
    )
    parser.add_argument(
        "--render-hash",
        type=str,
        default=None,
        help="既存のrender結果のhash（skip-render時に必要）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細な進捗を表示",
    )

    args = parser.parse_args()

    # 設定を構築
    config = PipelineConfig(
        seed=args.seed,
        frames_per_animal=args.frames_per_animal,
        animals=args.animals,
        profile=args.profile,
        split=args.split,
        workers=args.workers,
        session_process=args.session_process,
        psnr_multiplier=args.psnr_multiplier,
        ssim_multiplier=args.ssim_multiplier,
        max_outlier_images=args.max_outlier_images,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        valid_config=Path(args.valid_config) if args.valid_config else None,
        dry_run=args.dry_run,
        skip_frame_selection=args.skip_frame_selection,
        skip_render=args.skip_render,
        skip_extract=args.skip_extract,
        skip_metrics=args.skip_metrics,
        render_hash=args.render_hash,
    )

    # スキップオプション使用時は --output-dir が必須
    has_skip = config.skip_frame_selection or config.skip_render or config.skip_extract or config.skip_metrics
    if has_skip and not config.output_dir:
        print("エラー: --skip-* オプションを使用する場合は --output-dir も必要です")
        print("  既存の出力ディレクトリを指定してください（例: valid/run_20260108_144800）")
        sys.exit(1)

    # skip-renderの場合はrender-hashが必要
    if config.skip_render and not config.render_hash:
        print("エラー: --skip-render を指定する場合は --render-hash も必要です")
        sys.exit(1)

    # パイプライン構築
    steps = build_pipeline(config)

    print("=" * 60)
    print("検証パイプライン")
    print("=" * 60)
    print(f"個体: {', '.join(config.animals)}")
    print(f"フレーム数/個体: {config.frames_per_animal}")
    print(f"シード: {config.seed}")
    print(f"プロファイル: {config.profile}")
    if config.render_hash:
        print(f"render_hash: {config.render_hash}")
    print(f"dry-run: {config.dry_run}")
    print(f"skip-frame-selection: {config.skip_frame_selection}")
    print(f"skip-render: {config.skip_render}")
    print(f"skip-extract: {config.skip_extract}")
    print(f"skip-metrics: {config.skip_metrics}")

    # パイプライン実行
    success_count = 0
    fail_count = 0

    for step in steps:
        success = run_step(step, verbose=args.verbose)
        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"\n[ABORT] パイプラインを中断します")
            break

    # サマリー
    print("\n" + "=" * 60)
    print("パイプライン完了")
    print("=" * 60)
    print(f"成功: {success_count} / {len(steps)}")
    if fail_count > 0:
        print(f"失敗: {fail_count}")
        sys.exit(1)

    print("\n完了")


if __name__ == "__main__":
    main()

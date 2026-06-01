"""
外れ値検出・可視化スクリプト

統計的手法（IQR法）で外れ値を検出し、CSV出力と画像可視化を行う
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# パス設定 (consistency/ → 06_validation/ → scripts/ → project root)
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from marmo3d.validation.utils import (
    load_valid_config,
    expand_path_template,
    get_project_root,
)

# プロジェクトルート
project_root = get_project_root()


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """メトリクスJSONを読み込み"""
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_outliers_iqr(
    data: List[float],
    multiplier: float = 1.5,
) -> Tuple[float, float, List[int]]:
    """
    IQR法で外れ値を検出

    Args:
        data: データ配列
        multiplier: IQR倍率（1.5が一般的、3.0で極端な外れ値のみ）

    Returns:
        (lower_bound, upper_bound, outlier_indices)
    """
    if not data:
        return 0.0, 0.0, []

    data_array = np.array(data)
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_indices = [
        i for i, val in enumerate(data)
        if val < lower_bound or val > upper_bound
    ]

    return lower_bound, upper_bound, outlier_indices


def detect_all_outliers(
    metrics_data: Dict[str, Any],
    psnr_multiplier: float = 1.5,
    ssim_multiplier: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    全メトリクスから外れ値を検出

    Args:
        metrics_data: メトリクスデータ
        psnr_multiplier: PSNR用のIQR倍率
        ssim_multiplier: SSIM用のIQR倍率

    Returns:
        外れ値のリスト
    """
    details = metrics_data.get("details", [])

    if not details:
        return []

    # データを抽出
    psnr_values = [d["psnr"] for d in details]
    ssim_values = [d["ssim"] for d in details]

    # IQR法で外れ値を検出
    psnr_lower, psnr_upper, psnr_outlier_indices = detect_outliers_iqr(
        psnr_values, psnr_multiplier
    )
    ssim_lower, ssim_upper, ssim_outlier_indices = detect_outliers_iqr(
        ssim_values, ssim_multiplier
    )

    print(f"PSNR 外れ値範囲: < {psnr_lower:.2f} または > {psnr_upper:.2f}")
    print(f"SSIM 外れ値範囲: < {ssim_lower:.4f} または > {ssim_upper:.4f}")

    # 外れ値のインデックスを統合（PSNR or SSIM）
    outlier_indices = set(psnr_outlier_indices) | set(ssim_outlier_indices)

    # 外れ値データを収集
    outliers = []
    for idx in outlier_indices:
        detail = details[idx]
        outlier_type = []
        if idx in psnr_outlier_indices:
            outlier_type.append("PSNR")
        if idx in ssim_outlier_indices:
            outlier_type.append("SSIM")

        outliers.append({
            **detail,
            "outlier_type": ", ".join(outlier_type),
            "psnr_lower": psnr_lower,
            "psnr_upper": psnr_upper,
            "ssim_lower": ssim_lower,
            "ssim_upper": ssim_upper,
        })

    # PSNRの低い順にソート
    outliers.sort(key=lambda x: x["psnr"])

    return outliers


def save_outliers_csv(outliers: List[Dict[str, Any]], output_path: Path):
    """
    外れ値をCSVに保存

    Args:
        outliers: 外れ値リスト
        output_path: 出力CSVパス
    """
    if not outliers:
        print("外れ値が見つかりませんでした")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "session_id",
        "camera_id",
        "frame",
        "psnr",
        "ssim",
        "outlier_type",
        "psnr_lower",
        "psnr_upper",
        "ssim_lower",
        "ssim_upper",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outliers)

    print(f"外れ値CSVを保存しました: {output_path}")


def load_image(image_path: Path) -> np.ndarray:
    """画像を読み込み（RGB）"""
    if not image_path.exists():
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"画像の読み込みに失敗: {image_path}")

    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def visualize_outlier_pairs(
    outliers: List[Dict[str, Any]],
    validation_config: Dict[str, Any],
    render_hash: str,
    output_dir: Path,
    max_pairs: int = 20,
    frames_root: Optional[Path] = None,
    render_root_override: Optional[Path] = None,
):
    """
    外れ値の画像ペアを可視化

    Args:
        outliers: 外れ値リスト
        validation_config: consistency.yamlの設定
        render_hash: レンダリングハッシュ
        output_dir: 出力ディレクトリ
        max_pairs: 最大表示数
        frames_root: 実画像フレームのルート（指定時はconsistency.yamlより優先）
        render_root_override: レンダリング画像のルート（指定時はconsistency.yamlより優先）
    """
    if not outliers:
        print("外れ値が見つかりませんでした")
        return

    if render_root_override is not None:
        render_root = render_root_override
    else:
        render_root_template = validation_config.get("render", {}).get("root", "{PROJECT_ROOT}/outputs/validation/render")
        render_root = expand_path_template(render_root_template, project_root)

    if frames_root is not None:
        valid_root = frames_root
    else:
        valid_root_template = validation_config.get("frame_extraction", {}).get("output_root", "{PROJECT_ROOT}/outputs/validation/frames")
        valid_root = expand_path_template(valid_root_template, project_root)

    render_path_template = validation_config.get("render", {}).get(
        "path_template", "{render_root}/{session_id}/{hash}/image/Camera.{camera_id}/{frame_number:07d}.png"
    )
    valid_path_template = validation_config.get("frame_extraction", {}).get(
        "output_path_template", "{output_root}/{session_id}/Camera.{camera_id}/{frame_number:07d}.png"
    )
    render_dir_base_template = str(Path(render_path_template).parent)
    valid_dir_base_template = str(Path(valid_path_template).parent)

    # 最大件数まで処理
    for i, outlier in enumerate(outliers[:max_pairs], 1):
        session_id = outlier["session_id"]
        camera_id = outlier["camera_id"]
        frame = outlier["frame"]
        psnr = outlier["psnr"]
        ssim = outlier["ssim"]
        outlier_type = outlier["outlier_type"]

        # 画像パスをテンプレートから構築
        render_dir = expand_path_template(
            render_dir_base_template,
            project_root,
            render_root=str(render_root),
            session_id=session_id,
            hash=render_hash,
            camera_id=camera_id,
        )
        render_path = render_dir / f"{frame:07d}.png"

        valid_dir = expand_path_template(
            valid_dir_base_template,
            project_root,
            output_root=str(valid_root),
            session_id=session_id,
            camera_id=camera_id,
        )
        valid_path = valid_dir / f"{frame:07d}.png"

        try:
            # 画像を読み込み
            render_img = load_image(render_path)
            valid_img = load_image(valid_path)

            # 差分画像を計算
            diff_img = np.abs(render_img.astype(float) - valid_img.astype(float))
            diff_img = (diff_img / diff_img.max() * 255).astype(np.uint8) if diff_img.max() > 0 else diff_img.astype(np.uint8)

            # プロット作成
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 実画像
            axes[0].imshow(valid_img)
            axes[0].set_title("Real Image", fontsize=14, fontweight='bold')
            axes[0].axis('off')

            # レンダリング画像
            axes[1].imshow(render_img)
            axes[1].set_title("Rendered Image", fontsize=14, fontweight='bold')
            axes[1].axis('off')

            # 差分画像
            axes[2].imshow(diff_img)
            axes[2].set_title("Difference", fontsize=14, fontweight='bold')
            axes[2].axis('off')

            # タイトル
            fig.suptitle(
                f"Outlier #{i}: {session_id} - Camera {camera_id} - Frame {frame}\n"
                f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f} (Type: {outlier_type})",
                fontsize=16,
                fontweight='bold'
            )

            plt.tight_layout()

            # 保存
            output_path = output_dir / f"outlier_{i:03d}_{session_id}_cam{camera_id}_frame{frame}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"外れ値画像を保存: {output_path.name}")

            plt.close()

        except Exception as e:
            print(f"外れ値 #{i} の可視化に失敗: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="外れ値検出・可視化"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="メトリクスJSONファイルパス (デフォルト: valid/metrics.json)",
    )
    parser.add_argument(
        "--render-hash",
        type=str,
        required=True,
        help="レンダリング結果のhash（例: 1228b6de）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ディレクトリ (デフォルト: valid/outliers)",
    )
    parser.add_argument(
        "--valid-config",
        type=str,
        default=None,
        help="consistency.yaml のパス (デフォルト: config/consistency.yaml)",
    )
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
        "--max-images",
        type=int,
        default=20,
        help="可視化する最大外れ値数 (デフォルト: 20)",
    )
    parser.add_argument(
        "--frames-root",
        type=str,
        default=None,
        help="実画像フレームのルートディレクトリ (consistency.yamlより優先)",
    )
    parser.add_argument(
        "--render-root",
        type=str,
        default=None,
        help="レンダリング画像のルートディレクトリ (consistency.yamlより優先)",
    )

    args = parser.parse_args()

    # consistency.yamlを読み込み
    if args.valid_config:
        valid_config_path = Path(args.valid_config)
    else:
        valid_config_path = project_root / "config" / "consistency.yaml"

    if valid_config_path.exists():
        valid_config = load_valid_config(valid_config_path)
        validation_config = valid_config.get("validation", {})
    else:
        validation_config = {}

    # メトリクスファイルパス
    if args.metrics:
        metrics_path = Path(args.metrics)
    else:
        metrics_template = validation_config.get("metrics", {}).get(
            "output_path", "{PROJECT_ROOT}/valid/metrics.json"
        )
        metrics_path = expand_path_template(metrics_template, project_root)

    if not metrics_path.exists():
        print(f"エラー: メトリクスファイルが見つかりません: {metrics_path}")
        sys.exit(1)

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "outputs" / "validation" / "outliers"

    print(f"外れ値検出開始")
    print(f"  メトリクスファイル: {metrics_path}")
    print(f"  出力ディレクトリ: {output_dir}")
    print(f"  PSNR IQR倍率: {args.psnr_multiplier}")
    print(f"  SSIM IQR倍率: {args.ssim_multiplier}")

    # メトリクスを読み込み
    metrics_data = load_metrics(metrics_path)

    # 外れ値を検出
    print(f"\n外れ値検出中...")
    outliers = detect_all_outliers(
        metrics_data,
        psnr_multiplier=args.psnr_multiplier,
        ssim_multiplier=args.ssim_multiplier,
    )

    print(f"\n検出された外れ値数: {len(outliers)}")

    if not outliers:
        print("外れ値が見つかりませんでした")
        return

    # CSVに保存
    csv_path = output_dir / "outliers.csv"
    save_outliers_csv(outliers, csv_path)

    # 外れ値の統計を表示
    print(f"\n外れ値の統計:")
    print(f"  最小PSNR: {min(o['psnr'] for o in outliers):.2f} dB")
    print(f"  最大PSNR: {max(o['psnr'] for o in outliers):.2f} dB")
    print(f"  最小SSIM: {min(o['ssim'] for o in outliers):.4f}")
    print(f"  最大SSIM: {max(o['ssim'] for o in outliers):.4f}")

    # 画像ペアを可視化
    print(f"\n外れ値画像ペアを可視化中（最大{args.max_images}件）...")
    visualize_outlier_pairs(
        outliers,
        validation_config,
        args.render_hash,
        output_dir,
        max_pairs=args.max_images,
        frames_root=Path(args.frames_root) if args.frames_root else None,
        render_root_override=Path(args.render_root) if args.render_root else None,
    )

    print(f"\n完了")


if __name__ == "__main__":
    main()

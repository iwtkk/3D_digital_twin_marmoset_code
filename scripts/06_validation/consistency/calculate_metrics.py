"""
レンダリング画像と実画像のPSNR/SSIMを計算するスクリプト

PSNR: RGB画像全体（全ピクセル×全チャンネル）でMSEを計算してPSNRを算出
SSIM: R, G, Bの各チャンネルで個別にSSIMを計算して平均を返す
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# パス設定 (consistency/ → 06_validation/ → scripts/ → project root)
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import marmo3d.marmo_dataset as marmo_dataset
from marmo3d.validation.utils import (
    load_valid_config,
    expand_path_template,
    discover_camera_ids,
    get_project_root,
)

# プロジェクトルート
project_root = get_project_root()


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """画像を読み込み（RGB）"""
    if not image_path.exists():
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    PSNRを計算
    RGB画像全体（全ピクセル×全チャンネル）でMSEを計算
    """
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    SSIMを計算
    R, G, Bの各チャンネルで個別にSSIMを計算して平均を返す
    """
    return ssim(img1, img2, channel_axis=2, data_range=255)


def find_image_pairs(
    validation_config: Dict[str, Any],
    project_root: Path,
    session_id: str,
    render_hash: str,
    camera_ids: List[str],
    frames_root_override: Optional[Path] = None,
    render_root_override: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    レンダリング画像と実画像のペアを探す

    Returns:
        [{"session_id": str, "camera_id": str, "frame": int,
          "render_path": Path, "valid_path": Path}, ...]
    """
    pairs = []

    # render_root_override が指定されていればそちらを使用（consistency.yamlより優先）
    if render_root_override is not None:
        render_root = render_root_override
    else:
        render_root_template = validation_config.get("render", {}).get("root", "{PROJECT_ROOT}/outputs/validation/render")
        render_root = expand_path_template(render_root_template, project_root)

    # frames_root_override が指定されていればそちらを使用
    if frames_root_override is not None:
        valid_root = frames_root_override
    else:
        valid_root_template = validation_config.get("frame_extraction", {}).get("output_root", "{PROJECT_ROOT}/outputs/validation/frames")
        valid_root = expand_path_template(valid_root_template, project_root)

    # パステンプレートを取得
    render_path_template = validation_config.get("render", {}).get(
        "path_template", "{render_root}/{session_id}/{hash}/image/Camera.{camera_id}/{frame_number:07d}.png"
    )
    valid_path_template = validation_config.get("frame_extraction", {}).get(
        "output_path_template", "{output_root}/{session_id}/Camera.{camera_id}/{frame_number:07d}.png"
    )

    for camera_id in camera_ids:
        # テンプレートからディレクトリパスを構築
        # ファイル名部分（{frame_number:07d}.png）を削除してディレクトリを取得
        render_dir_template = str(Path(render_path_template).parent)
        render_dir = expand_path_template(
            render_dir_template,
            project_root,
            render_root=str(render_root),
            session_id=session_id,
            hash=render_hash,
            camera_id=camera_id,
        )

        valid_dir_template = str(Path(valid_path_template).parent)
        valid_dir = expand_path_template(
            valid_dir_template,
            project_root,
            output_root=str(valid_root),
            session_id=session_id,
            camera_id=camera_id,
        )

        if not render_dir.exists():
            continue
        if not valid_dir.exists():
            continue

        # valid側のファイルを基準にする
        for valid_file in valid_dir.glob("*.png"):
            frame_name = valid_file.name
            render_file = render_dir / frame_name

            if render_file.exists():
                # フレーム番号を取得（拡張子なし）
                frame_number = int(valid_file.stem)
                pairs.append({
                    "session_id": session_id,
                    "camera_id": camera_id,
                    "frame": frame_number,
                    "render_path": render_file,
                    "valid_path": valid_file,
                })

    return pairs


def calculate_metrics_for_pair(pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """画像ペアのメトリクスを計算"""
    render_img = load_image(pair["render_path"])
    valid_img = load_image(pair["valid_path"])

    if render_img is None or valid_img is None:
        return None

    # サイズチェック
    if render_img.shape != valid_img.shape:
        print(f"  WARN: サイズ不一致 {pair['session_id']} Camera.{pair['camera_id']} frame {pair['frame']}")
        return None

    try:
        psnr_value = calculate_psnr(render_img, valid_img)
        ssim_value = calculate_ssim(render_img, valid_img)

        return {
            "session_id": pair["session_id"],
            "camera_id": pair["camera_id"],
            "frame": pair["frame"],
            "psnr": float(psnr_value),
            "ssim": float(ssim_value),
        }
    except Exception as e:
        print(f"  ERROR: メトリクス計算失敗 {pair['session_id']} Camera.{pair['camera_id']} frame {pair['frame']}: {e}")
        return None


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """メトリクスを集計"""
    if not results:
        return {}

    psnr_values = [r["psnr"] for r in results]
    ssim_values = [r["ssim"] for r in results]

    # セッションごと
    by_session = defaultdict(list)
    for r in results:
        by_session[r["session_id"]].append(r)

    session_stats = {}
    for session_id, session_results in by_session.items():
        session_psnr = [r["psnr"] for r in session_results]
        session_ssim = [r["ssim"] for r in session_results]
        session_stats[session_id] = {
            "count": len(session_results),
            "psnr_mean": float(np.mean(session_psnr)),
            "psnr_std": float(np.std(session_psnr)),
            "ssim_mean": float(np.mean(session_ssim)),
            "ssim_std": float(np.std(session_ssim)),
        }

    # カメラごと
    by_camera = defaultdict(list)
    for r in results:
        by_camera[r["camera_id"]].append(r)

    camera_stats = {}
    for camera_id, camera_results in by_camera.items():
        camera_psnr = [r["psnr"] for r in camera_results]
        camera_ssim = [r["ssim"] for r in camera_results]
        camera_stats[camera_id] = {
            "count": len(camera_results),
            "psnr_mean": float(np.mean(camera_psnr)),
            "psnr_std": float(np.std(camera_psnr)),
            "ssim_mean": float(np.mean(camera_ssim)),
            "ssim_std": float(np.std(camera_ssim)),
        }

    return {
        "total": {
            "count": len(results),
            "psnr_mean": float(np.mean(psnr_values)),
            "psnr_std": float(np.std(psnr_values)),
            "psnr_min": float(np.min(psnr_values)),
            "psnr_max": float(np.max(psnr_values)),
            "ssim_mean": float(np.mean(ssim_values)),
            "ssim_std": float(np.std(ssim_values)),
            "ssim_min": float(np.min(ssim_values)),
            "ssim_max": float(np.max(ssim_values)),
        },
        "by_session": session_stats,
        "by_camera": camera_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="レンダリング画像と実画像のPSNR/SSIMを計算"
    )
    parser.add_argument(
        "--render-hash",
        type=str,
        required=True,
        help="レンダリング結果のhash（例: 1228b6de）",
    )
    parser.add_argument(
        "--valid-config",
        type=str,
        default=None,
        help="consistency.yaml のパス (デフォルト: config/consistency.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="結果の出力先JSONファイル (consistency.yamlより優先)",
    )
    parser.add_argument(
        "--session-ids",
        nargs="+",
        default=None,
        help="計算対象のセッションID（指定しない場合は全セッション）",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細な進捗を表示",
    )

    args = parser.parse_args()

    # consistency.yamlを読み込み
    if args.valid_config:
        valid_config_path = Path(args.valid_config)
    else:
        valid_config_path = project_root / "config" / "consistency.yaml"

    if not valid_config_path.exists():
        print(f"エラー: consistency.yaml が見つかりません: {valid_config_path}")
        sys.exit(1)

    valid_config = load_valid_config(valid_config_path)
    validation_config = valid_config.get("validation", {})

    # dataset_spec.yamlのパスを取得
    dataset_spec_template = validation_config.get("dataset_spec", "{PROJECT_ROOT}/config/dataset_spec.yaml")
    spec_path = expand_path_template(dataset_spec_template, project_root)

    if not spec_path.exists():
        print(f"エラー: dataset_spec.yaml が見つかりません: {spec_path}")
        sys.exit(1)

    dataset_info = marmo_dataset.MarmoDataset.from_yaml(
        spec_path, project_root=project_root
    )

    # カメラIDを取得
    camera_ids = discover_camera_ids(dataset_info)

    # 出力パス
    if args.output:
        output_path = Path(args.output)
    else:
        output_template = validation_config.get("metrics", {}).get(
            "output_path", "{PROJECT_ROOT}/valid/metrics.json"
        )
        output_path = expand_path_template(output_template, project_root)

    # frames_root を決定
    if args.frames_root:
        frames_root = Path(args.frames_root)
    else:
        frames_root_template = validation_config.get("frame_extraction", {}).get(
            "output_root", "{PROJECT_ROOT}/outputs/validation/frames"
        )
        frames_root = expand_path_template(frames_root_template, project_root)

    # セッションIDを取得
    if args.session_ids:
        session_ids = args.session_ids
    else:
        # frames_rootディレクトリから取得
        if not frames_root.exists():
            print(f"エラー: フレームディレクトリが見つかりません: {frames_root}")
            sys.exit(1)
        session_ids = [d.name for d in frames_root.iterdir() if d.is_dir()]

    print(f"PSNR/SSIM計算開始")
    print(f"  hash: {args.render_hash}")
    print(f"  フレームディレクトリ: {frames_root}")
    print(f"  セッション数: {len(session_ids)}")
    print(f"  カメラ数: {len(camera_ids)}")
    print(f"  出力先: {output_path}")

    # 画像ペアを収集
    all_pairs = []
    for session_id in session_ids:
        pairs = find_image_pairs(
            validation_config=validation_config,
            project_root=project_root,
            session_id=session_id,
            render_hash=args.render_hash,
            camera_ids=camera_ids,
            frames_root_override=Path(args.frames_root) if args.frames_root else None,
            render_root_override=Path(args.render_root) if args.render_root else None,
        )
        all_pairs.extend(pairs)
        if args.verbose and pairs:
            print(f"  {session_id}: {len(pairs)} ペア")

    print(f"\n合計画像ペア数: {len(all_pairs)}")

    if len(all_pairs) == 0:
        print("エラー: 画像ペアが見つかりませんでした")
        sys.exit(1)

    # メトリクスを計算
    results = []
    for i, pair in enumerate(all_pairs, 1):
        if args.verbose and i % 100 == 0:
            print(f"  進捗: {i}/{len(all_pairs)}")

        result = calculate_metrics_for_pair(pair)
        if result:
            results.append(result)

    print(f"計算成功: {len(results)}/{len(all_pairs)}")

    # 集計
    aggregated = aggregate_metrics(results)

    # 結果を保存
    output_data = {
        "render_hash": args.render_hash,
        "total_pairs": len(all_pairs),
        "successful_calculations": len(results),
        "summary": aggregated,
        "details": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n結果を保存しました: {output_path}")

    # サマリーを表示
    if aggregated:
        total = aggregated["total"]
        print(f"\n=== サマリー ===")
        print(f"PSNR: {total['psnr_mean']:.2f} ± {total['psnr_std']:.2f} (range: {total['psnr_min']:.2f} - {total['psnr_max']:.2f})")
        print(f"SSIM: {total['ssim_mean']:.4f} ± {total['ssim_std']:.4f} (range: {total['ssim_min']:.4f} - {total['ssim_max']:.4f})")


if __name__ == "__main__":
    main()

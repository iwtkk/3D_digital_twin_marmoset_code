"""
selected_frames.jsonを使ってviddataから画像フレームを抽出するスクリプト
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

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
from utils.frame_to_imgs import resolve_video_frame

# プロジェクトルート
project_root = get_project_root()


def load_selection(selection_path: Path) -> Dict:
    """選択結果JSONを読み込み"""
    with open(selection_path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_session(selections: List[Dict]) -> Dict[str, List[Dict]]:
    """
    セッションIDごとにフレームをグループ化

    Returns:
        {session_id: [{selection...}, ...], ...}
    """
    grouped = defaultdict(list)
    for sel in selections:
        session_id = sel["session_id"]
        grouped[session_id].append(sel)

    return {session_id: sorted(items, key=lambda x: int(x["frame"])) for session_id, items in grouped.items()}


def extract_frame(
    mp4_path: Path,
    frame_in_chunk: int,
    output_path: Path,
) -> bool:
    """
    mp4から指定フレームを抽出してPNGで保存

    Args:
        mp4_path: 入力mp4ファイル
        frame_in_chunk: mp4チャンク内のフレーム番号（0始まり）
        output_path: 出力PNGファイルパス

    Returns:
        成功したらTrue
    """
    if not mp4_path.exists():
        print(f"  WARN: mp4が見つかりません: {mp4_path}")
        return False

    # 出力ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpegでフレームを抽出
    filter_expr = f"select=eq(n\\,{frame_in_chunk})"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(mp4_path),
                "-vf",
                filter_expr,
                "-vsync",
                "vfr",
                "-frames:v",
                "1",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: ffmpegが失敗しました: {e.stderr.decode()}")
        return False


def extract_session_frames(
    session_id: str,
    selections: List[Dict],
    video_root: Path,
    output_root: Path,
    camera_ids: List[str],
) -> None:
    """
    1セッションの全カメラから指定フレームを抽出

    Args:
        session_id: セッションID
        selections: selected_frames.json の選択行リスト
        video_root: viddataディレクトリ
        output_root: 出力ルートディレクトリ
        camera_ids: 抽出するカメラIDリスト
    """
    print(f"\n{session_id}:")
    print(f"  フレーム数: {len(selections)}")
    print(f"  カメラ数: {len(camera_ids)}")

    total_extracted = 0
    total_skipped = 0
    total_failed = 0

    for camera_id in camera_ids:
        for selection in selections:
            frame_number = int(selection["frame"])
            result_dir = Path(selection["kp3d_path"]).parent if selection.get("kp3d_path") else None
            stored_frame = frame_number
            if result_dir is not None:
                frame_num_path = result_dir / str(camera_id) / "frame_num.npy"
                if frame_num_path.exists():
                    frame_nums = np.load(frame_num_path)
                    if frame_number >= len(frame_nums):
                        print(f"  WARN: frame out of range: {frame_num_path} frame={frame_number}")
                        total_failed += 1
                        continue
                    stored_frame = int(frame_nums[frame_number])

            # 出力パス
            output_path = (
                output_root
                / session_id
                / f"Camera.{camera_id}"
                / f"{frame_number:07d}.png"
            )

            # 既に存在する場合はスキップ
            if output_path.exists():
                total_skipped += 1
                continue

            # mp4パスとチャンク内フレームを取得
            video_frame = resolve_video_frame(video_root, session_id, camera_id, stored_frame)
            if video_frame is None:
                print(f"  WARN: mp4が見つかりません: session={session_id} camera={camera_id} stored_frame={stored_frame}")
                total_failed += 1
                continue
            mp4_path, frame_in_chunk = video_frame

            # フレームを抽出
            success = extract_frame(mp4_path, frame_in_chunk, output_path)

            if success:
                total_extracted += 1
            else:
                total_failed += 1

    print(f"  抽出成功: {total_extracted}")
    if total_skipped > 0:
        print(f"  スキップ（既存）: {total_skipped}")
    if total_failed > 0:
        print(f"  抽出失敗: {total_failed}")


def main():
    parser = argparse.ArgumentParser(
        description="selected_frames.jsonを使ってviddataから画像フレームを抽出"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="選択結果JSONファイルパス (selected_frames.json)",
    )
    parser.add_argument(
        "--valid-config",
        type=str,
        default=None,
        help="consistency.yaml のパス (デフォルト: config/consistency.yaml)",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default=None,
        help="viddataディレクトリパス (consistency.yamlより優先)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="出力ルートディレクトリ (consistency.yamlより優先)",
    )
    parser.add_argument(
        "--camera-ids",
        nargs="+",
        default=None,
        help="抽出するカメラIDリスト（指定しない場合はdataset_specから読み込み）",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=None,
        help="処理するセッション数の上限（テスト用）",
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
    if args.camera_ids:
        camera_ids = args.camera_ids
    else:
        camera_ids = discover_camera_ids(dataset_info)

    # 選択結果を読み込み
    selection_path = Path(args.input)
    if not selection_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {selection_path}")
        sys.exit(1)

    selection_data = load_selection(selection_path)
    print(f"選択結果を読み込みました: {selection_path}")
    print(f"  シード: {selection_data['seed']}")
    print(f"  合計フレーム数: {selection_data['total_frames']}")

    # セッションごとにグループ化
    grouped = group_by_session(selection_data["selections"])
    print(f"  セッション数: {len(grouped)}")

    # パス設定（コマンドライン引数 > consistency.yaml > dataset_info）
    if args.video_root:
        video_root = Path(args.video_root)
    else:
        # dataset_infoからvideo_rootを取得
        video_root = dataset_info.video_root

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        # consistency.yamlから取得
        output_root_template = validation_config.get("frame_extraction", {}).get(
            "output_root", "{PROJECT_ROOT}/outputs/validation/frames"
        )
        output_root = expand_path_template(output_root_template, project_root)

    if not video_root.exists():
        print(f"エラー: viddataディレクトリが見つかりません: {video_root}")
        sys.exit(1)

    print(f"\n抽出開始")
    print(f"  入力: {video_root}")
    print(f"  出力: {output_root}")
    print(f"  カメラ数: {len(camera_ids)}")
    print(f"  カメラID: {', '.join(camera_ids)}")

    # セッションごとに抽出
    sessions_to_process = list(sorted(grouped.items()))
    if args.limit_sessions:
        sessions_to_process = sessions_to_process[:args.limit_sessions]
        print(f"  制限: 最初の{args.limit_sessions}セッションのみ処理")

    for i, (session_id, selections) in enumerate(sessions_to_process, 1):
        if args.verbose:
            print(f"\n[{i}/{len(sessions_to_process)}] 処理中: {session_id}")
        extract_session_frames(
            session_id=session_id,
            selections=selections,
            video_root=video_root,
            output_root=output_root,
            camera_ids=camera_ids,
        )

    print("\n完了")


if __name__ == "__main__":
    main()

"""
selected_frames.json を読み込んでレンダリングを実行するスクリプト
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# パス設定
current_script = Path(__file__).resolve()
marmo3d_dir = current_script.parent.parent  # src/marmo3d/
src_dir = marmo3d_dir.parent                # src/
project_root = src_dir.parent               # project root

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import marmo3d.marmo_dataset as marmo_dataset
from marmo3d.blender import render_setting
from marmo3d.blender import render_csv
from marmo3d.blender import render_spec


def load_selection(selection_path: Path) -> Dict[str, Any]:
    """選択結果JSONを読み込み"""
    with open(selection_path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_session(selections: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    セッションIDごとにフレームをグループ化

    Returns:
        {session_id: [frame1, frame2, ...], ...}
    """
    grouped = defaultdict(list)
    for sel in selections:
        session_id = sel["session_id"]
        frame = sel["frame"]
        grouped[session_id].append(frame)

    # フレームをソートして重複除去
    return {
        session_id: sorted(set(frames))
        for session_id, frames in grouped.items()
    }


def render_session(
    session_id: str,
    frames: List[int],
    dataset_info: marmo_dataset.MarmoDataset,
    render_info: render_spec.RenderSpec,
    profile: str,
    output_root: Path,
    render_spec_path: Path,
    split: int = 1,
    overrides: bool = True,
    workers: int = 1,
    dry_run: bool = False,
) -> None:
    """1セッションをレンダリング"""
    session_path = os.path.join(
        dataset_info.results_root,
        dataset_info.results_subdir,
        session_id,
    )

    req = render_setting.RenderRequest(
        profile=profile,
        sessions=session_path,
        frames=frames,
        split=split,
        overrides=overrides,
        workers=workers,
        dry_run=dry_run,
    )

    render_setting.CheckRequest().validate(req, render_info)

    plan = render_setting.RenderPlan.from_request(
        req,
        output_dir_resolver=lambda session, _frames: output_root / session.name,
    )

    # CSV結果を構築（再利用可能な関数を使用）
    csv_results = render_csv.build_csv_results(
        plan_items=plan.items,
        kp3d_finder=render_setting._find_kp3d_pickle,
        render_spec_path=render_spec_path,
        output_root=output_root,
    )

    # Blenderコマンド生成
    base_blend = render_info.resolve_base_blend(project_root)
    script_path = current_script.parent / "render.py"

    commands = render_setting.RenderJob(
        base_blend=base_blend,
        render_spec_path=render_spec_path,
        script_path=script_path,
        output_root=output_root,
        profile=req.profile,
    ).make_commands(
        plan,
        extra_args_resolver=lambda item: [
            "--csv_path",
            str(csv_results[(str(item.session), tuple(item.frames))]["csv_path"]),
            "--all_frames",
            "1" if csv_results[(str(item.session), tuple(item.frames))]["all_frames"] else "0",
        ]
        if (str(item.session), tuple(item.frames)) in csv_results
        else [],
    )
    # print(commands)
    # print(f"  コマンド数: {len(commands)}")
    # exit()
    # 実行
    render_setting.Executor(workers=req.workers, retries=1, timeout=None).execute(
        commands, dry_run=req.dry_run
    )


def main():
    parser = argparse.ArgumentParser(
        description="selected_frames.json を使ってレンダリング"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="選択結果JSONファイルパス",
    )
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
        "--dry-run",
        action="store_true",
        help="実行せずコマンドのみ表示",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="出力ルートディレクトリ (デフォルト: {project_root}/render)",
    )
    parser.add_argument(
        "--overrides",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--session-process",
        type=int,
        default=1,
    )

    args = parser.parse_args()

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

    # データセット・レンダー設定を読み込み
    dataset_spec_path = project_root / "config" / "dataset_spec.yaml"
    render_spec_path = project_root / "config" / "render_spec.yaml"

    if not dataset_spec_path.exists():
        print(f"エラー: dataset_spec.yaml が見つかりません: {dataset_spec_path}")
        sys.exit(1)

    dataset_info = marmo_dataset.MarmoDataset.from_yaml(
        dataset_spec_path, project_root=project_root
    )
    render_info = render_spec.RenderSpec.from_yaml(render_spec_path)

    # 出力ディレクトリ
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = project_root / "outputs" / "validation" / "render"

    print(f"\nレンダリング開始")
    print(f"  プロファイル: {args.profile}")
    print(f"  出力先: {output_root}")
    print(f"  dry-run: {args.dry_run}")
    print()

    # セッションごとにレンダリング
    sessions = list(grouped.items())
    total_sessions = len(sessions)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.session_process) as pool:
        futures = []
        for i, (session_id, frames) in enumerate(sessions, 1):
            # print(f"[{i}/{total_sessions}] {session_id} ({len(frames)} frames)")
            future = pool.submit(
                render_session,
                session_id=session_id,
                frames=frames,
                dataset_info=dataset_info,
                render_info=render_info,
                profile=args.profile,
                output_root=output_root,
                render_spec_path=render_spec_path,
                split=args.split,
                overrides=args.overrides,
                workers=args.workers,
                dry_run=args.dry_run,
            )
            futures.append(future)

        for future in futures:
            future.result()

    print("\n完了")


if __name__ == "__main__":
    main()

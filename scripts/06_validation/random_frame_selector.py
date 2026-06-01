"""
個体ごとにランダムなフレームを選択してJSON/CSVに保存するスクリプト
"""
from __future__ import annotations

import argparse
import json
import csv
import pickle
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# パス設定
current_script = Path(__file__).resolve()
# random_frame_selector.py → 06_validation/ → scripts/ → project root
project_root = current_script.parent.parent.parent
src_dir = project_root / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import marmo3d.marmo_dataset as marmo_dataset


@dataclass
class FrameSelection:
    """選択されたフレームの情報"""
    animal: str
    session_id: str
    frame: int
    kp3d_path: str


@dataclass
class SelectionResult:
    """選択結果全体"""
    seed: int
    frames_per_animal: int
    animals: List[str]
    total_frames: int
    selections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FramesCache:
    """全フレーム情報のキャッシュ"""
    animals: List[str]
    frames_by_animal: Dict[str, List[Dict[str, Any]]]


def load_kp3d_frame_count(kp3d_path: Path) -> int:
    """kp3d.pickleからフレーム数を取得（全animal_idの最小値）"""
    with open(kp3d_path, "rb") as f:
        data = pickle.load(f)

    if "kp3d" not in data:
        raise KeyError(f"'kp3d' キーが見つかりません: {kp3d_path}")

    return min(len(x) for x in data["kp3d"])


def collect_frames(session: marmo_dataset.Session) -> List[Dict[str, Any]]:
    """
    1セッションからフレーム情報を収集

    Args:
        session: 対象セッション

    Returns:
        [{"session_id": ..., "frame": ..., "kp3d_path": ...}, ...]
    """
    kp3d_path = session.kp3d_path()
    if kp3d_path is None or not kp3d_path.exists():
        return []

    try:
        frame_count = load_kp3d_frame_count(kp3d_path)
    except Exception as e:
        print(f"警告: {kp3d_path} の読み込みに失敗: {e}")
        return []

    frames = []
    for frame_idx in range(frame_count):
        frames.append({
            "session_id": session.session_id,
            "frame": frame_idx,
            "kp3d_path": str(kp3d_path),
        })

    return frames


def collect_animal_frames(
    ds: marmo_dataset.MarmoDataset,
    animal: str,
) -> List[Dict[str, Any]]:
    """
    指定した個体の全セッションからフレーム情報を収集

    Args:
        ds: データセット
        animal: 個体ID

    Returns:
        全セッションのフレーム情報リスト
    """
    sessions = list(ds.iter_sessions(valid_only=True, log_skips=False))
    animal_sessions = ds.filter(sessions, animal=animal)

    all_frames = []
    for session in animal_sessions:
        frames = collect_frames(session)
        all_frames.extend(frames)

    return all_frames


def save_frames_cache(cache: FramesCache, cache_path: Path) -> None:
    """フレームキャッシュを保存"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cache), f, ensure_ascii=False, indent=2)
    print(f"フレームキャッシュを保存しました: {cache_path}")


def load_frames_cache(cache_path: Path) -> Optional[FramesCache]:
    """フレームキャッシュを読み込み"""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return FramesCache(
            animals=data["animals"],
            frames_by_animal=data["frames_by_animal"],
        )
    except Exception as e:
        print(f"警告: キャッシュ読み込み失敗: {e}")
        return None


def collect_all_frames(
    ds: marmo_dataset.MarmoDataset,
    animals: List[str],
    cache_path: Optional[Path] = None,
) -> FramesCache:
    """
    全個体のフレーム情報を収集（キャッシュ対応）

    Args:
        ds: データセット
        animals: 対象個体リスト
        cache_path: キャッシュファイルパス（指定時は保存/読込）

    Returns:
        FramesCache
    """
    # キャッシュがあれば読み込み
    if cache_path:
        cache = load_frames_cache(cache_path)
        if cache and set(cache.animals) == set(animals):
            print(f"キャッシュから読み込みました: {cache_path}")
            return cache

    # 収集
    frames_by_animal = {}
    for animal in animals:
        print(f"個体 {animal} のフレームを収集中...")
        frames = collect_animal_frames(ds, animal)
        frames_by_animal[animal] = frames
        print(f"  利用可能フレーム数: {len(frames)}")

    cache = FramesCache(animals=animals, frames_by_animal=frames_by_animal)

    # キャッシュ保存
    if cache_path:
        save_frames_cache(cache, cache_path)

    return cache


def select_random_frames(
    cache: FramesCache,
    frames_per_animal: int,
    seed: int,
) -> SelectionResult:
    """
    キャッシュからランダムにフレームを選択
    """
    random.seed(seed)

    result = SelectionResult(
        seed=seed,
        frames_per_animal=frames_per_animal,
        animals=cache.animals,
        total_frames=0,
    )

    for animal in cache.animals:
        all_frames = cache.frames_by_animal.get(animal, [])

        if len(all_frames) == 0:
            print(f"警告: {animal} には利用可能なフレームがありません")
            continue

        # フレーム数が要求より少ない場合は全て選択
        n_select = min(frames_per_animal, len(all_frames))
        selected = random.sample(all_frames, n_select)

        for frame_info in selected:
            selection = FrameSelection(
                animal=animal,
                session_id=frame_info["session_id"],
                frame=frame_info["frame"],
                kp3d_path=frame_info["kp3d_path"],
            )
            result.selections.append(asdict(selection))

        print(f"{animal}: {n_select} フレーム選択")

    result.total_frames = len(result.selections)
    return result


def save_json(result: SelectionResult, output_path: Path) -> None:
    """結果をJSONに保存"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    print(f"JSONに保存しました: {output_path}")


def save_csv(result: SelectionResult, output_path: Path) -> None:
    """結果をCSVに保存"""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["animal", "session_id", "frame", "kp3d_path"])
        writer.writeheader()
        writer.writerows(result.selections)
    print(f"CSVに保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="個体ごとにランダムなフレームを選択"
    )
    parser.add_argument(
        "--animals",
        nargs="+",
        default=["cj611", "cj357", "cj425"],
        help="対象の個体ID (デフォルト: cj611 cj357 cj425)",
    )
    parser.add_argument(
        "--frames-per-animal",
        type=int,
        default=100,
        help="各個体から選択するフレーム数 (デフォルト: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (デフォルト: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="selected_frames.json",
        help="出力ファイルパス (デフォルト: selected_frames.json)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "both"],
        default="json",
        help="出力形式 (デフォルト: json)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="フレームキャッシュファイルパス (指定時は保存/読込)",
    )
    parser.add_argument(
        "--dataset-spec",
        type=str,
        default=None,
        help="dataset_spec.yaml のパス (デフォルト: config/dataset_spec.yaml)",
    )

    args = parser.parse_args()

    # データセット読み込み
    if args.dataset_spec:
        spec_path = Path(args.dataset_spec)
    else:
        spec_path = project_root / "config" / "dataset_spec.yaml"

    if not spec_path.exists():
        print(f"エラー: dataset_spec.yaml が見つかりません: {spec_path}")
        sys.exit(1)

    print(f"データセット仕様: {spec_path}")
    ds = marmo_dataset.MarmoDataset.from_yaml(spec_path, project_root=project_root)

    # キャッシュパス
    cache_path = Path(args.cache) if args.cache else None

    # フレーム収集（キャッシュ対応）
    cache = collect_all_frames(ds, args.animals, cache_path)

    # ランダム選択
    result = select_random_frames(
        cache=cache,
        frames_per_animal=args.frames_per_animal,
        seed=args.seed,
    )

    print(f"\n合計選択フレーム数: {result.total_frames}")

    # 保存
    output_path = Path(args.output)

    if args.format in ("json", "both"):
        json_path = output_path.with_suffix(".json") if args.format == "both" else output_path
        save_json(result, json_path)

    if args.format in ("csv", "both"):
        csv_path = output_path.with_suffix(".csv") if args.format == "both" else output_path
        save_csv(result, csv_path)


if __name__ == "__main__":
    main()

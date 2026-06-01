
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paired Auto Crop with MMDetection (v2.x)
=======================================

A/B の対応画像ペアに対して、A画像を MMDetection で物体検出し、選んだ1つの矩形（または結合矩形）で
A/B 両方をクロップします。

- MMDetection 2.26.0 を想定（`mmdet.apis.init_detector` / `inference_detector`）。
- 既存のモデル（config と checkpoint）をそのまま読み込み。
- 複数検出がある場合の選び方をオプションで指定：
  - `--select highest`  : 最高スコア
  - `--select largest`  : 面積最大
  - `--select union`    : 全検出を結合した外接矩形（クラス/スコア条件を満たすもの）
- `--classes` で対象クラスを番号指定（例: `0,2,5`）。未指定なら全クラス対象。
- `--score-thr` でスコア閾値。
- `--margin` でパーセント拡張（例: `0.1` なら幅高の10%ずつ拡大）。
- B側には `--relative` で相対座標適用（サイズが違うペア向き）。
- ペアの指定は CSV または `--dir-a/--dir-b`（同名ファイルペア）。

使い方例:
  python paired_auto_crop_mmdet.py \

      --config path/to/config.py \

      --checkpoint path/to/model.pth \

      --dir-a path/to/A --dir-b path/to/B \

      --out out_dir --select highest --score-thr 0.5 --relative

  python paired_auto_crop_mmdet.py \

      --config cfg.py --checkpoint ckpt.pth \

      --map pairs.csv --out out_dir --select union --classes 0,2 --margin 0.08

出力: out_dir/A/ と out_dir/B/ に PNG 保存（A側ファイル名ベース）。
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector


def load_pairs_from_dirs(dir_a: Path, dir_b: Path, exts: List[str]) -> List[Tuple[Path, Path]]:
    def list_images(d: Path) -> Dict[str, Path]:
        table = {}
        lower_exts = tuple([e.lower() for e in exts]) if exts else None
        for p in d.rglob('*'):
            if p.is_file():
                if lower_exts:
                    if p.suffix.lower() in lower_exts:
                        table[p.stem] = p
                else:
                    if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}:
                        table[p.stem] = p
        return table

    a_tbl = list_images(dir_a)
    b_tbl = list_images(dir_b)

    pairs = []
    missing_b = []
    for stem, a_path in sorted(a_tbl.items()):
        b_path = b_tbl.get(stem)
        if b_path is None:
            missing_b.append(stem)
            continue
        pairs.append((a_path, b_path))

    if missing_b:
        print(f"[WARN] {len(missing_b)} files in A had no match in B (by stem). Example: {missing_b[:3]}" )

    if not pairs:
        raise RuntimeError('No pairs found. Check directories and filenames.')
    return pairs


def load_pairs_from_csv(csv_path: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            a, b = row[0].strip(), row[1].strip()
            if not a or not b:
                continue
            pairs.append((Path(a), Path(b)))
    if not pairs:
        raise RuntimeError('No valid pairs in CSV.')
    return pairs


def ensure_out_dirs(out_dir: Path) -> Tuple[Path, Path]:
    a_out = out_dir / 'A'
    b_out = out_dir / 'B'
    a_out.mkdir(parents=True, exist_ok=True)
    b_out.mkdir(parents=True, exist_ok=True)
    return a_out, b_out


def parse_cls_filter(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    res = []
    for t in s.split(','):
        t = t.strip()
        if not t:
            continue
        res.append(int(t))
    return res if res else None


def clamp_rect(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def enlarge_with_margin(x, y, w, h, W, H, margin: float):
    if margin <= 0:
        return clamp_rect(x, y, w, h, W, H)
    dw = int(round(w * margin))
    dh = int(round(h * margin))
    x2 = x + w
    y2 = y + h
    x = x - dw
    y = y - dh
    x2 = x2 + dw
    y2 = y2 + dh
    x = max(0, x)
    y = max(0, y)
    x2 = min(W - 1, x2)
    y2 = min(H - 1, y2)
    w = max(1, x2 - x)
    h = max(1, y2 - y)
    return x, y, w, h


def to_relative(x, y, w, h, W, H):
    return (x / W, y / H, w / W, h / H)


def from_relative(rx, ry, rw, rh, W, H):
    x = int(round(rx * W)); y = int(round(ry * H))
    w = int(round(rw * W)); h = int(round(rh * H))
    return clamp_rect(x, y, w, h, W, H)


def pick_box_from_result(result, cls_filter: Optional[List[int]], score_thr: float, select: str):
    """
    result は inference_detector(model, img) の戻り値。
    - bbox 結果のみ：list[np.ndarray]（クラスごと、 shape=(N,5) で x1,y1,x2,y2,score）
    - segm 結果が含まれる場合：(bbox_results, segm_results)
    ここでは bbox を使う。
    """
    if isinstance(result, tuple):
        bbox_results = result[0]
    else:
        bbox_results = result

    boxes = []  # (x1,y1,x2,y2,score,cls)
    for cls_idx, arr in enumerate(bbox_results):
        if arr is None or len(arr) == 0:
            continue
        if cls_filter is not None and cls_idx not in cls_filter:
            continue
        for b in arr:
            x1, y1, x2, y2, score = b.tolist()
            if score < score_thr:
                continue
            boxes.append((x1, y1, x2, y2, score, cls_idx))

    if not boxes:
        return None

    if select == 'highest':
        boxes.sort(key=lambda t: t[4], reverse=True)
        x1, y1, x2, y2, score, cls_idx = boxes[0]
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    if select == 'largest':
        boxes.sort(key=lambda t: (t[2]-t[0])*(t[3]-t[1]), reverse=True)
        x1, y1, x2, y2, score, cls_idx = boxes[0]
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    if select == 'union':
        xs1 = min(b[0] for b in boxes)
        ys1 = min(b[1] for b in boxes)
        xs2 = max(b[2] for b in boxes)
        ys2 = max(b[3] for b in boxes)
        return int(xs1), int(ys1), int(xs2 - xs1), int(ys2 - ys1)

    x1, y1, x2, y2, _, _ = boxes[0]
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def crop_xywh(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]


def main():
    parser = argparse.ArgumentParser(description='Paired auto cropper using MMDetection 2.x')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--map', type=str, help='CSV with pairs (A_path,B_path)')
    g.add_argument('--dir-a', type=str, help='Directory A (use with --dir-b)')
    parser.add_argument('--dir-b', type=str, help='Directory B')
    parser.add_argument('--exts', type=str, default='', help='Comma-separated extensions, e.g. .jpg,.png')
    parser.add_argument('--out', type=str, required=True, help='Output directory root (creates A/ and B/)')

    parser.add_argument('--config', type=str, required=True, help='MMDet config path')
    parser.add_argument('--checkpoint', type=str, required=True, help='MMDet checkpoint path')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 / cpu')

    parser.add_argument('--select', type=str, default='highest', choices=['highest', 'largest', 'union'],
                        help='Which bbox to use when multiple detections exist')
    parser.add_argument('--classes', type=str, default='', help='Target class indices, comma-separated (e.g., 0,2,5)')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold')
    parser.add_argument('--margin', type=float, default=0.0, help='Margin percent (e.g., 0.08 for +8%%)')

    parser.add_argument('--relative', action='store_true', help='Apply A-rect to B relatively (for different sizes)')

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    a_out, b_out = ensure_out_dirs(out_dir)

    if args.map:
        pairs = load_pairs_from_csv(Path(args.map))
    else:
        if not args.dir_a or not args.dir_b:
            raise RuntimeError('--dir-a と --dir-b はセットで指定してください')
        exts = [e.strip() for e in args.exts.split(',') if e.strip()] if args.exts else []
        pairs = load_pairs_from_dirs(Path(args.dir_a), Path(args.dir_b), exts)

    print('[INFO] Loading model...')
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print('[INFO] Model ready.')

    cls_filter = parse_cls_filter(args.classes)

    for idx, (a_path, b_path) in enumerate(pairs, 1):
        img_a = cv2.imread(str(a_path), cv2.IMREAD_COLOR)
        img_b = cv2.imread(str(b_path), cv2.IMREAD_COLOR)
        if img_a is None or img_b is None:
            print(f'[WARN] Cannot read: {a_path} or {b_path}. Skipping.')
            continue

        H, W = img_a.shape[:2]
        result = inference_detector(model, str(a_path))
        rect = pick_box_from_result(result, cls_filter, args.score_thr, args.select)
        if rect is None:
            print(f'[INFO] No detection passed filters: {a_path.name}. Skipping.')
            continue

        x, y, w, h = rect
        x, y, w, h = clamp_rect(x, y, w, h, W, H)
        x, y, w, h = enlarge_with_margin(x, y, w, h, W, H, args.margin)

        crop_a = crop_xywh(img_a, (x, y, w, h))

        if args.relative:
            rx, ry, rw, rh = to_relative(x, y, w, h, W, H)
            Hb, Wb = img_b.shape[:2]
            bx, by, bw, bh = from_relative(rx, ry, rw, rh, Wb, Hb)
            crop_b = crop_xywh(img_b, (bx, by, bw, bh))
        else:
            Hb, Wb = img_b.shape[:2]
            bx, by, bw, bh = clamp_rect(x, y, w, h, Wb, Hb)
            crop_b = crop_xywh(img_b, (bx, by, bw, bh))

        base = a_path.stem
        out_a_path = a_out / f'{base}.png'
        out_b_path = b_out / f'{base}.png'
        cv2.imwrite(str(out_a_path), crop_a)
        cv2.imwrite(str(out_b_path), crop_b)
        print(f'[SAVED {idx}/{len(pairs)}] {out_a_path.name}, {out_b_path.name}')

    print('Done.')


if __name__ == '__main__':
    main()

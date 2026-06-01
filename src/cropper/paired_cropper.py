
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Paired Image Cropper
# ====================

# 機能:
# - 1枚目の画像でドラッグして矩形を作ると、対応するもう1枚にも同じ範囲でクロップします
# - 次の画像に移っても「直前のクロップサイズ（幅x高さ）」を保持します
# - クロップサイズはセッションを越えて保存/復元されます（state.json）
# - 画像サイズが異なるペアへも、相対スケールで同じ領域を切り出すオプションあり(--relative)

# 操作:
# - マウス左ドラッグ: 矩形を選択/更新
# - 'n' または Enter: 現在のペアをクロップして保存し、次へ進む
# - 'b': 前のペアに戻る（保存はしない）
# - 'c': 矩形をクリア
# - 'r': 直前のクロップ「サイズ」を中央に再配置（位置はリセット）
# - 'q' または ESC: 終了
# - 'h': ヘルプ表示/非表示
# - 'f': 固定サイズモードのON/OFF（直前サイズで固定・クリック位置を左上にして配置）

# 使い方の例:
#     python paired_cropper.py --dir-a path/to/A --dir-b path/to/B --out out_dir
#     python paired_cropper.py --map pairs.csv --out out_dir --relative

# ペア指定の方法:
# - --map CSV: 1列目がA画像パス、2列目がB画像パス（ヘッダ無し）。
# - または --dir-a と --dir-b を与える: 同名(拡張子除く)ファイル同士をペアにします。
#   例) A/abc.jpg と B/abc.png が対応。

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

STATE_FILE_NAME = "paired_cropper_state.json"


def load_pairs_from_dirs(dir_a: Path, dir_b: Path, exts: List[str]) -> List[Tuple[Path, Path]]:
    def list_images(d: Path) -> Dict[str, Path]:
        table = {}
        lower_exts = tuple([e.lower() for e in exts]) if exts else None
        for p in d.rglob("*"):
            if p.is_file():
                if lower_exts:
                    if p.suffix.lower() in lower_exts:
                        table[p.stem] = p
                else:
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
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
        print(f"[WARN] {len(missing_b)} files in A had no match in B (by stem). Example: {missing_b[:3]}")

    if not pairs:
        raise RuntimeError("No pairs found. Check directories and filenames.")
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
        raise RuntimeError("No valid pairs in CSV.")
    return pairs


def ensure_out_dirs(out_dir: Path) -> Tuple[Path, Path]:
    a_out = out_dir / "A"
    b_out = out_dir / "B"
    a_out.mkdir(parents=True, exist_ok=True)
    b_out.mkdir(parents=True, exist_ok=True)
    return a_out, b_out


def load_state(out_dir: Path) -> dict:
    state_path = out_dir / STATE_FILE_NAME
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(out_dir: Path, state: dict):
    state_path = out_dir / STATE_FILE_NAME
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


class CropSession:
    def __init__(self, window_name: str = "Paired Cropper"):
        self.window = window_name
        self.img = None  # type: Optional[np.ndarray]
        self.clone = None  # for drawing

        self.rect_start = None  # (x, y)
        self.rect_end = None  # (x, y)
        self.dragging = False

        self.last_size = None  # (w, h)
        self.fixed_size_mode = False

        self.help_on = True

    def set_image(self, img: np.ndarray):
        self.img = img.copy()
        self.clone = img.copy()
        self.rect_start = None
        self.rect_end = None
        self.dragging = False

    def put_overlay(self):
        if self.clone is None:
            return
        disp = self.clone.copy()

        # draw rect if any
        if self.rect_start and self.rect_end:
            x1, y1 = self.rect_start
            x2, y2 = self.rect_end
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            cv2.putText(disp, f"{w}x{h}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            cv2.putText(disp, f"{w}x{h}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if self.help_on:
            lines = [
                "操作: ドラッグ=矩形作成/更新, n/Enter=保存して次へ, b=戻る, c=クリア, r=直前サイズを中央配置, f=固定サイズ切替, h=ヘルプ, q/ESC=終了",
            ]
            y = 60
            for line in lines:
                cv2.putText(disp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(disp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 25

        cv2.imshow(self.window, disp)

    def mouse_cb(self, event, x, y, flags, param):
        h, w = self.img.shape[:2]

        def clamp(pt):
            px, py = pt
            return (max(0, min(px, w - 1)), max(0, min(py, h - 1)))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            if self.fixed_size_mode and self.last_size is not None:
                # 固定サイズ: クリック位置を左上に
                lw, lh = self.last_size
                x2 = x + lw
                y2 = y + lh
                self.rect_start = clamp((x, y))
                self.rect_end = clamp((x2, y2))
                self.dragging = False  # そのまま確定
            else:
                self.rect_start = clamp((x, y))
                self.rect_end = clamp((x, y))

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.rect_start is None:
                return
            cur = clamp((x, y))
            if self.fixed_size_mode and self.last_size is not None:
                # 固定サイズでドラッグする場合: ドラッグ点を左上としサイズ固定
                lw, lh = self.last_size
                x2 = self.rect_start[0] + lw
                y2 = self.rect_start[1] + lh
                self.rect_end = clamp((x2, y2))
            else:
                self.rect_end = cur

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            if self.rect_start is None:
                return
            cur = clamp((x, y))
            if self.fixed_size_mode and self.last_size is not None:
                lw, lh = self.last_size
                x2 = self.rect_start[0] + lw
                y2 = self.rect_start[1] + lh
                self.rect_end = clamp((x2, y2))
            else:
                self.rect_end = cur

            # 更新されたサイズを記録
            x1, y1 = self.rect_start
            x2, y2 = self.rect_end
            self.last_size = (abs(x2 - x1), abs(y2 - y1))

        self.put_overlay()

    def current_rect_xywh(self) -> Optional[Tuple[int, int, int, int]]:
        if not (self.rect_start and self.rect_end):
            return None
        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w < 1 or h < 1:
            return None
        return (x, y, w, h)

    def clear_rect(self):
        self.rect_start = None
        self.rect_end = None

    def center_last_size(self):
        if self.img is None or self.last_size is None:
            return
        H, W = self.img.shape[:2]
        lw, lh = self.last_size
        x = max(0, (W - lw) // 2)
        y = max(0, (H - lh) // 2)
        self.rect_start = (x, y)
        self.rect_end = (min(W - 1, x + lw), min(H - 1, y + lh))

    def toggle_fixed(self):
        self.fixed_size_mode = not self.fixed_size_mode


def crop_xywh(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    return img[y:y + h, x:x + w]


def crop_xywh_relative(img: np.ndarray, rect_norm: Tuple[float, float, float, float]) -> np.ndarray:
    H, W = img.shape[:2]
    x = int(round(rect_norm[0] * W))
    y = int(round(rect_norm[1] * H))
    w = int(round(rect_norm[2] * W))
    h = int(round(rect_norm[3] * H))
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return img[y:y + h, x:x + w]


def to_relative(rect: Tuple[int, int, int, int], img: np.ndarray) -> Tuple[float, float, float, float]:
    x, y, w, h = rect
    H, W = img.shape[:2]
    return (x / W, y / H, w / W, h / H)


def main():
    parser = argparse.ArgumentParser(description="Paired image cropper")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--map", type=str, help="CSV with pairs (A_path,B_path)")
    g.add_argument("--dir-a", type=str, help="Directory A")
    parser.add_argument("--dir-b", type=str, help="Directory B (required if using --dir-a)")
    parser.add_argument("--exts", type=str, default="", help="Comma-separated extensions to include, e.g. .jpg,.png (optional)")

    parser.add_argument("--out", type=str, required=True, help="Output directory root (will create A/ and B/)")
    parser.add_argument("--relative", action="store_true", help="Apply crop to B with relative coordinates (recommended if sizes differ)")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    a_out, b_out = ensure_out_dirs(out_dir)

    # Load pairs
    if args.map:
        pairs = load_pairs_from_csv(Path(args.map))
    else:
        if not args.dir_a or not args.dir_b:
            raise RuntimeError("--dir-a と --dir-b はセットで指定してください")
        exts = [e.strip() for e in args.exts.split(",") if e.strip()] if args.exts else []
        pairs = load_pairs_from_dirs(Path(args.dir_a), Path(args.dir_b), exts)

    # Load state
    state = load_state(out_dir)
    last_size = None
    if "last_size" in state:
        last_size = tuple(state["last_size"])

    idx = max(0, min(args.start, len(pairs) - 1))

    win = CropSession()
    cv2.namedWindow(win.window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win.window, win.mouse_cb)

    while 0 <= idx < len(pairs):
        a_path, b_path = pairs[idx]
        img_a = cv2.imread(str(a_path), cv2.IMREAD_COLOR)
        img_b = cv2.imread(str(b_path), cv2.IMREAD_COLOR)

        if img_a is None or img_b is None:
            print(f"[WARN] Cannot read: {a_path} or {b_path}. Skipping.")
            idx += 1
            continue

        # update last_size
        if last_size:
            win.last_size = last_size

        # Prepare and show A
        title = f"[{idx+1}/{len(pairs)}] A: {a_path.name}  <->  B: {b_path.name}"
        cv2.setWindowTitle(win.window, title)
        win.set_image(img_a)
        if win.last_size and win.current_rect_xywh() is None:
            # show last size centered
            win.center_last_size()
        win.put_overlay()

        # Interaction loop
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('h'):
                win.help_on = not win.help_on
                win.put_overlay()
            elif key == ord('f'):
                win.toggle_fixed()
                win.put_overlay()
            elif key == ord('c'):
                win.clear_rect()
                win.put_overlay()
            elif key == ord('r'):
                win.center_last_size()
                win.put_overlay()
            elif key in (ord('b'),):
                # back
                idx -= 1
                break
            elif key in (ord('q'), 27):  # ESC
                cv2.destroyAllWindows()
                return
            elif key in (ord('n'), 13):  # Enter
                rect = win.current_rect_xywh()
                if rect is None:
                    print("[INFO] 矩形が未選択です。ドラッグで選択してください。")
                    continue

                # save last size to session & state
                last_size = (rect[2], rect[3])
                state["last_size"] = list(last_size)
                save_state(out_dir, state)

                # Crop A
                crop_a = crop_xywh(img_a, rect)

                # Crop B (relative or absolute)
                if args.relative:
                    rect_rel = to_relative(rect, img_a)
                    crop_b = crop_xywh_relative(img_b, rect_rel)
                else:
                    # absolute coordinates; clamp by B size
                    x, y, w, h = rect
                    Hb, Wb = img_b.shape[:2]
                    x = max(0, min(x, Wb - 1))
                    y = max(0, min(y, Hb - 1))
                    w = max(1, min(w, Wb - x))
                    h = max(1, min(h, Hb - y))
                    crop_b = crop_xywh(img_b, (x, y, w, h))

                # Write
                base = a_path.stem  # use A's name
                out_a_path = a_out / f"{base}.png"
                out_b_path = b_out / f"{base}.png"
                cv2.imwrite(str(out_a_path), crop_a)
                cv2.imwrite(str(out_b_path), crop_b)
                print(f"[SAVED] {out_a_path} , {out_b_path}")

                idx += 1
                break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

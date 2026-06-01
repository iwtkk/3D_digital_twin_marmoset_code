# -*- coding: utf-8 -*-
# This is same as the "visualize_2_simpleSingleVid.py"
# This is kept for compatibility, so please do not use this function
#
import json
import shutil
import os
import cv2
import numpy as np
from tqdm import tqdm
from math import floor, ceil
import tempfile
import copy
import matplotlib.pyplot as plt
import math
import platform
import argparse
import imgstore

def ellipse_line(img, x1, x2, mrksize, bgr):
    if x2[0] - x1[0] == 0:
        ang = 90
    else:
        ang = math.atan((x2[1] - x1[1])/(x2[0] - x1[0])) / math.pi * 180
    cen = ((x1[0] + x2[0])/2, (x1[1] + x2[1])/2)
    d = math.sqrt(math.pow(x2[0] - x1[0], 2) + math.pow(x2[1] - x1[1], 2) )
    # bgr: (B,G,R) in [0..255]
    cv2.ellipse(img, (cen, (d, mrksize), ang),
                (int(bgr[0]), int(bgr[1]), int(bgr[2])),
                thickness=-1)
    return 0


def mid_tone_palette(n, s=0.55, v=0.75):
    import colorsys
    cols = []
    for i in range(n):
        h = (i % n) / float(n)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((r, g, b))
    return cols


def rgb_to_bgr255(rgb):
    r, g, b = rgb
    return (int(b * 255), int(g * 255), int(r * 255))


def clean_kp(kp):
    for i_kp in range(len(kp)):
        if kp[i_kp][2] < 0.3:
            kp[i_kp] = None
        else:
            kp[i_kp] = kp[i_kp][0:2]

def add_neckkp(kp):
    if kp[5] is not None and kp[6] is not None:
        d = [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2]
    else:
        d = None
    kp.append(d)

def draw_kps(img, kp, mrksize, clr=None):

    kp_con = [
        (0, 2), (0, 1), (2, 4), (1, 3), (6, 8), (5, 7), (8, 10), (7, 9),
        (12, 14), (11, 13), (14, 16), (13, 15), (0, 17), (17, 6), (17, 5),
        (17, 12), (17, 11)
    ]

    n_kp = len(kp)
    if n_kp >= 20:
        kp_con.append((18, 17))
        if n_kp > 20:
            kp_con.append((18, 19))

    cols_rgb = mid_tone_palette(n_kp, s=0.55, v=0.75)

    for i in reversed(range(len(kp))):
        if kp[i] is not None:
            if clr is None:
                bgr = rgb_to_bgr255(cols_rgb[i])
            else:
                bgr = rgb_to_bgr255(clr)
            cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), mrksize, bgr, thickness=-1)

    for j1, j2 in kp_con:
        if j1 < len(kp) and j2 < len(kp) and kp[j1] is not None and kp[j2] is not None:
            if clr is None:
                bgr = rgb_to_bgr255(cols_rgb[j1])
            else:
                bgr = rgb_to_bgr255(clr)
            ellipse_line(img, kp[j1], kp[j2], mrksize, bgr)

def parse_hex_color(hex_str):
    """
    '#RRGGBB' or 'RRGGBB' -> (r,g,b) in [0,1]; raises ValueError if invalid
    """
    s = hex_str.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError("color_hex must be 6 hex digits like #FF8800")
    r = int(s[0:2], 16)/255.0
    g = int(s[2:4], 16)/255.0
    b = int(s[4:6], 16)/255.0
    return (r, g, b)

def procSingleVideo(path_vid, path_json, path_output, n_frame_to_Save,
                    redo=False, color_hex=None):
    import os
    if os.path.exists(path_output) and not redo:
        print(f'Skipping as 2D movie already exists: {path_output}')
        return
    with open(path_json, 'r') as f:
        data = json.load(f)

    n_frame = len(data)

    if (n_frame < n_frame_to_Save) or (n_frame_to_Save < 1):
        n_frame_to_Save = n_frame

    # 単色指定：未指定なら青(0,0,1)
    if color_hex is None or color_hex == "":
        clr_override = (0.0, 0.0, 1.0)   # blue
    else:
        try:
            clr_override = parse_hex_color(color_hex)
        except Exception as e:
            print(f"[WARN] Invalid --color_hex '{color_hex}'; fallback to BLUE. ({e})")
            clr_override = (0.0, 0.0, 1.0)

    cap = cv2.VideoCapture(path_vid)
    # 出力サイズは入力動画に合わせる（固定値だと崩れる可能性）
    ok, probe_img = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read video: {path_vid}")
    h, w = probe_img.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    vw = cv2.VideoWriter(path_output, cv2.VideoWriter_fourcc(*'mp4v'), 24, (w, h))

    for i_frame in tqdm(range(n_frame_to_Save)):
        ret, img = cap.read()
        if not ret:
            break

        d = data[i_frame]

        for dd in d:
            trkl_id = dd[0]
            bb = dd[1:5]
            kp = copy.deepcopy(dd[5])
            clrinfo = dd[6:8]

            a = min(bb[2]-bb[0], bb[3]-bb[1])
            mrksize = max(int(a/30), 1)

            clean_kp(kp)
            add_neckkp(kp)

            # 単色で描画（キー点＆骨格）
            draw_kps(img, kp, mrksize, clr=clr_override)

            # バウンディングボックスも同色（cv2はBGR）
            bgr = (int(clr_override[2]*255), int(clr_override[1]*255), int(clr_override[0]*255))
            img = cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), bgr, thickness=3)

            # フレーム番号
            cv2.putText(img, str(i_frame), (10, 150),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2.0,
                        color=(255, 255, 255),
                        thickness=5,
                        lineType=cv2.LINE_4)

            # クラス表示（しきい値）
            if clrinfo[1] > 0.9 and int(clrinfo[0]) != 4:
                cv2.putText(img, f"{int(clrinfo[0])}:{int(clrinfo[1]*100)}",
                            (int(bb[0]), int(bb[1])),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2.0,
                            color=(0, 255, 255),
                            thickness=5,
                            lineType=cv2.LINE_4)

        vw.write(img)

    vw.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_vid', default='', type=str, help='fullpath for video e.g. ./X.mp4')
    parser.add_argument('--path_json', default='', type=str, help='fullpath for keypoints json')
    parser.add_argument('--path_output', default='', type=str, help='fullpath for output video, e.g. ./test.mp4')
    parser.add_argument('--n_frame_to_save', default=10000, type=int, help='Number of frames to draw, -1=all')
    parser.add_argument('--redo', default=False, type=bool, help='redo')
    parser.add_argument('--color_hex', default=None, type=str,
                        help='HEX color like "#0000FF" or "0000FF". If omitted, BLUE is used.')

    args = parser.parse_args()
    path_vid = args.path_vid
    path_json = args.path_json
    path_output = args.path_output
    n_frame_to_Save = int(args.n_frame_to_save)

    procSingleVideo(path_vid, path_json, path_output, n_frame_to_Save,
                    redo=args.redo, color_hex=args.color_hex)

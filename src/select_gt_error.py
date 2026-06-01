import os
import json
import shlex
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ========= カラーパレット =========
def mid_tone_palette(n, s=0.55, v=0.75):
    import colorsys
    cols = []
    for i in range(n):
        h = (i % n) / float(n)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((r, g, b))  # 0-1 RGB
    return cols

# ========= スケルトン（必要なら使う・ここでは点のみ重ね） =========
SKELETON = [
    (0,2),(0,1),(2,4),(1,3),
    (6,8),(5,7),(8,10),(7,9),
    (12,14),(11,13),(14,16),(13,15),
    (0,17),(17,6),(17,5),(17,12),(17,11),(18,19)
]

def extract_frame_ffmpeg(video_path, frame_idx, out_path):
    os.makedirs(Path(out_path).parent, exist_ok=True)
    cmd = f'ffmpeg -y -i {shlex.quote(video_path)} -vf "select=eq(n\\,{frame_idx})" -vframes 1 {shlex.quote(out_path)}'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or (not os.path.exists(out_path)):
        raise RuntimeError(f"ffmpeg frame extraction failed.\ncmd: {cmd}\nstderr:\n{proc.stderr.decode('utf-8', errors='ignore')}")

def load_pred_points(det_json_path, frame_number):
    # あなたの検出JSON: res[frame_number][0][5] が [ [x,y,conf], ... ]
    with open(det_json_path, "rb") as f:
        res = json.load(f)
    frame = res[frame_number]
    det = frame[0]
    pts = np.array(det[5], dtype=float)  # (K,3) [x,y,conf]
    return pts

def interactive_compare(video_path,
                        det_json_path,
                        frame_number,
                        save_dir="./frames",
                        out_compare_json="pred_gt_comparison.json",
                        kp_names=None):
    """
    - 指定フレームを抽出し背景に表示
    - 予測点を色分けで重ね
    - キーポイント順にクリックでGT取得（左=可視, 右=不可視）
    - その場で誤差(px)を表示
    - 最後に comparison JSON に追記保存
    """
    # 1) フレーム抽出
    out_png = os.path.join(save_dir, f"frame_{frame_number:06d}.png")
    extract_frame_ffmpeg(video_path, frame_number, out_png)

    # 2) 予測キーポイント
    pred = load_pred_points(det_json_path, frame_number)  # (K,3)
    K = pred.shape[0]
    if kp_names is None:
        kp_names = [f"KP{i:02d}" for i in range(K)]

    # 3) 背景＋予測点の描画
    img = mpimg.imread(out_png)
    H, W = img.shape[0], img.shape[1]
    kp_colors = mid_tone_palette(K, s=0.55, v=0.75)

    fig, ax = plt.subplots(figsize=(min(10, W/100), min(10, H/100)))
    ax.imshow(img)
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])  # 画像座標
    ax.set_axis_off()
    # 予測点
    for j in range(K):
        x, y, conf = pred[j]
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, s=18, c=[kp_colors[j]], edgecolors='none', alpha=0.95, zorder=3)
        ax.text(x+3, y-3, f"{j}", fontsize=8, color=kp_colors[j], zorder=4)

    # 4) クリックでGT収集
    state = {
        "idx": 0,                       # いま入力するKPのインデックス
        "gt_pts": [None]*K,             # (x,y) or None
        "gt_vis": [0]*K,                # 0=不可視, 1=可視
        "error_artists": [],            # ライン/テキストの参照
    }

    title = ax.set_title(f"Frame {frame_number} | 左クリック=可視点, 右クリック=不可視(スキップ), "
                         f"Backspace=Undo, q=終了 | 入力中: {kp_names[0]}(0/{K})",
                         fontsize=11)

    def draw_error_for(j):
        # 既存のアーティストは残す（Undo時には消す）
        if state["gt_vis"][j] == 0 or state["gt_pts"][j] is None:
            return None, None
        gx, gy = state["gt_pts"][j]
        px, py, _ = pred[j]
        # 距離
        dist = float(np.hypot(gx - px, gy - py))
        # 線とテキスト
        line = ax.plot([px, gx], [py, gy], linewidth=1.4,
                       color=kp_colors[j], alpha=0.9, zorder=2)[0]
        mx, my = (px + gx)/2, (py + gy)/2
        txt = ax.text(mx+3, my-3, f"{dist:.1f}px",
                      fontsize=9, color=kp_colors[j], bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
                      zorder=5)
        return line, txt

    def update_title():
        i = state["idx"]
        done = sum(v is not None for v in state["gt_pts"])
        if i < K:
            title.set_text(f"Frame {frame_number} | 左=可視, 右=不可視, Backspace=Undo, q=終了 | "
                           f"入力中: {kp_names[i]} ({done}/{K})")
        else:
            title.set_text(f"Frame {frame_number} | 入力完了 ({done}/{K}) - qで終了/保存")

    def onclick(event):
        # キャンバス外やボタン無し
        if event.inaxes != ax:
            return
        i = state["idx"]
        if i >= K:
            return  # もう入力済み
        # 左クリック: 可視
        if event.button == 1:
            state["gt_pts"][i] = (float(event.xdata), float(event.ydata))
            state["gt_vis"][i] = 1
            line, txt = draw_error_for(i)
            state["error_artists"].append(("add", i, line, txt))
            state["idx"] += 1
            update_title()
            fig.canvas.draw_idle()
        # 右クリック: 不可視(スキップ)
        elif event.button == 3:
            state["gt_pts"][i] = None
            state["gt_vis"][i] = 0
            state["error_artists"].append(("skip", i, None, None))
            state["idx"] += 1
            update_title()
            fig.canvas.draw_idle()

    def onkey(event):
        # Undo: 直前の入力を取り消し
        if event.key == "backspace":
            # 何も入力がない
            if all(v is None for v in state["gt_pts"]) and state["idx"] == 0:
                return
            # idx をひとつ戻す
            state["idx"] = max(0, state["idx"] - 1)
            j = state["idx"]
            state["gt_pts"][j] = None
            state["gt_vis"][j] = 0
            # 最後に追加したエラー表示を消す（あれば）
            while state["error_artists"]:
                kind, jj, line, txt = state["error_artists"].pop()
                if line is not None:
                    line.remove()
                if txt is not None:
                    txt.remove()
                if jj == j:
                    break
            update_title()
            fig.canvas.draw_idle()
        # q で終了（保存）
        elif event.key == "q":
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    cid_key   = fig.canvas.mpl_connect("key_press_event", onkey)
    plt.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    # 5) 比較結果をまとめて保存
    # エラーは「可視のみ」計算。不可視は gt_visibility=0 として保存（あなたの集計コードはこれをスキップ）。
    records = []
    for j in range(K):
        rec = {
            "frame_number": int(frame_number),
            "image_path": str(out_png),
            "keypoint_index": int(j),
            "pred_x": float(pred[j,0]),
            "pred_y": float(pred[j,1]),
            "gt_visibility": int(state["gt_vis"][j]),
        }
        if state["gt_vis"][j] == 1 and state["gt_pts"][j] is not None:
            gx, gy = state["gt_pts"][j]
            rec["gt_x"] = float(gx)
            rec["gt_y"] = float(gy)
            rec["error"] = float(np.hypot(gx - pred[j,0], gy - pred[j,1]))
        else:
            rec["gt_x"] = None
            rec["gt_y"] = None
            rec["error"] = 0.0  # 使われないが数値で入れておく
        records.append(rec)

    # 既存ファイルがあれば追記
    out_path = Path(out_compare_json)
    if out_path.exists():
        with out_path.open("r") as f:
            prev = json.load(f)
        if isinstance(prev, list):
            prev.extend(records)
            data_to_save = prev
        else:
            data_to_save = records
    else:
        data_to_save = records

    with out_path.open("w") as f:
        json.dump(data_to_save, f, indent=2)
    print(f"[Saved] {out_path} ({len(records)} records)")

if __name__ == "__main__":
    # 使い方をあなたの環境に合わせて調整
    video_path = "vid/dailylife_cj611_20230226_110000.23506214/000000.mp4"
    det_json   = "results/2d_v0p8_Dark_fix_20/dailylife_cj611_20230226_110000/dailylife_cj611_20230226_110000_23506214_000000.json"
    frame_num  = 65
    interactive_compare(video_path, det_json, frame_num,
                        save_dir="./frames",
                        out_compare_json="pred_gt_comparison.json")

import os
import json
import shlex
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ========= Color palette =========
def mid_tone_palette(n, s=0.55, v=0.75):
    import colorsys
    cols = []
    for i in range(n):
        h = (i % n) / float(n)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((r, g, b))  # 0-1 RGB
    return cols

# ========= Skeleton (use if needed; only points are overlaid here) =========
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
    # Detection JSON: res[frame_number][0][5] is [ [x,y,conf], ... ]
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
    - Extract the specified frame and show it as background
    - Overlay predicted points color-coded
    - Collect GT by clicking in keypoint order (left=visible, right=invisible)
    - Show error (px) immediately
    - Finally append results to the comparison JSON
    """
    # 1) Extract frame
    out_png = os.path.join(save_dir, f"frame_{frame_number:06d}.png")
    extract_frame_ffmpeg(video_path, frame_number, out_png)

    # 2) Predicted keypoints
    pred = load_pred_points(det_json_path, frame_number)  # (K,3)
    K = pred.shape[0]
    if kp_names is None:
        kp_names = [f"KP{i:02d}" for i in range(K)]

    # 3) Draw background + predicted points
    img = mpimg.imread(out_png)
    H, W = img.shape[0], img.shape[1]
    kp_colors = mid_tone_palette(K, s=0.55, v=0.75)

    fig, ax = plt.subplots(figsize=(min(10, W/100), min(10, H/100)))
    ax.imshow(img)
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])  # image coordinates
    ax.set_axis_off()
    # Predicted points
    for j in range(K):
        x, y, conf = pred[j]
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, s=18, c=[kp_colors[j]], edgecolors='none', alpha=0.95, zorder=3)
        ax.text(x+3, y-3, f"{j}", fontsize=8, color=kp_colors[j], zorder=4)

    # 4) Collect GT by clicking
    state = {
        "idx": 0,                       # index of the KP currently being entered
        "gt_pts": [None]*K,             # (x,y) or None
        "gt_vis": [0]*K,                # 0=invisible, 1=visible
        "error_artists": [],            # references to lines/texts
    }

    title = ax.set_title(f"Frame {frame_number} | left click=visible, right click=invisible (skip), "
                         f"Backspace=Undo, q=quit | entering: {kp_names[0]}(0/{K})",
                         fontsize=11)

    def draw_error_for(j):
        # Keep existing artists (removed on Undo)
        if state["gt_vis"][j] == 0 or state["gt_pts"][j] is None:
            return None, None
        gx, gy = state["gt_pts"][j]
        px, py, _ = pred[j]
        # Distance
        dist = float(np.hypot(gx - px, gy - py))
        # Line and text
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
            title.set_text(f"Frame {frame_number} | left=visible, right=invisible, Backspace=Undo, q=quit | "
                           f"entering: {kp_names[i]} ({done}/{K})")
        else:
            title.set_text(f"Frame {frame_number} | input complete ({done}/{K}) - press q to quit/save")

    def onclick(event):
        # Outside canvas or no button
        if event.inaxes != ax:
            return
        i = state["idx"]
        if i >= K:
            return  # already entered
        # Left click: visible
        if event.button == 1:
            state["gt_pts"][i] = (float(event.xdata), float(event.ydata))
            state["gt_vis"][i] = 1
            line, txt = draw_error_for(i)
            state["error_artists"].append(("add", i, line, txt))
            state["idx"] += 1
            update_title()
            fig.canvas.draw_idle()
        # Right click: invisible (skip)
        elif event.button == 3:
            state["gt_pts"][i] = None
            state["gt_vis"][i] = 0
            state["error_artists"].append(("skip", i, None, None))
            state["idx"] += 1
            update_title()
            fig.canvas.draw_idle()

    def onkey(event):
        # Undo: cancel the previous input
        if event.key == "backspace":
            # Nothing entered
            if all(v is None for v in state["gt_pts"]) and state["idx"] == 0:
                return
            # Step idx back by one
            state["idx"] = max(0, state["idx"] - 1)
            j = state["idx"]
            state["gt_pts"][j] = None
            state["gt_vis"][j] = 0
            # Remove the last added error display (if any)
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
        # Quit with q (save)
        elif event.key == "q":
            plt.close(fig)

    cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
    cid_key   = fig.canvas.mpl_connect("key_press_event", onkey)
    plt.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    # 5) Save comparison results
    # Errors are computed for visible points only. Invisible ones are saved with gt_visibility=0 (skipped by the aggregation code).
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
            rec["error"] = 0.0  # unused, but stored as a number
        records.append(rec)

    # Append if the file already exists
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
    # Adjust usage to your environment
    video_path = "vid/dailylife_cj611_20230226_110000.23506214/000000.mp4"
    det_json   = "results/2d_v0p8_Dark_fix_20/dailylife_cj611_20230226_110000/dailylife_cj611_20230226_110000_23506214_000000.json"
    frame_num  = 65
    interactive_compare(video_path, det_json, frame_num,
                        save_dir="./frames",
                        out_compare_json="pred_gt_comparison.json")

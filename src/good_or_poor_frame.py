import os
import json
import shlex
import subprocess
import numpy as np
from matplotlib import pyplot as plt

def good_poor(res):
    f_num=0
    good_frame=[]
    for frame in res:
        for kp_data in frame:
            conf=1
            for point in kp_data[5]:
                if(point[2]<0.7):
                    conf=0
                    if(conf==1):
                        good_frame.append(f_num)
                        # print('good')
                    # else:
                        # print('poor')
        f_num=f_num+1
    print(good_frame)

# ========= Color palette =========
def mid_tone_palette(n, s=0.55, v=0.75):
    import colorsys
    cols = []
    for i in range(n):
        h = (i % n) / float(n)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append((r, g, b))  # 0-1 RGB
    return cols

# ========= Skeleton (marmo20 + back) =========
SKELETON = [
    (0,2),(0,1),(2,4),(1,3),
    (6,8),(5,7),(8,10),(7,9),
    (12,14),(11,13),(14,16),(13,15),
    (0,17),(17,6),(17,5),(17,12),(17,11),(18,19)
]

def extract_frame_ffmpeg(video_path, frame_idx, out_path):
    """
    Extract only the frame with n==frame_idx via ffmpeg and save it as PNG.
    - Assumes 0-based frame numbers
    - 'select=eq(n,idx)' works even with variable frame rate (based on input decode order)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Note the escaped comma in eq(n\,idx)
    cmd = f'ffmpeg -y -i {shlex.quote(video_path)} -vf "select=eq(n\\,{frame_idx})" -vframes 1 {shlex.quote(out_path)}'
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or (not os.path.exists(out_path)):
        raise RuntimeError(f"ffmpeg frame extraction failed.\ncmd: {cmd}\nstderr:\n{proc.stderr.decode('utf-8', errors='ignore')}")

def plot_frame_with_keypoints(img_path, points, skeleton=SKELETON, conf_thr=0.0):
    """
    Overlay keypoints and skeleton on top of the frame image.
    points: (K,3) ndarray [x,y,conf]
    """
    import matplotlib.image as mpimg

    img = mpimg.imread(img_path)  # (H,W,3)
    H, W = img.shape[0], img.shape[1]

    K = points.shape[0]
    kp_colors = mid_tone_palette(K, s=0.55, v=0.75)

    fig, ax = plt.subplots(figsize=(min(10, W/100), min(10, H/100)))
    # Background image
    ax.imshow(img)
    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])  # Flip vertically to match image coordinates
    ax.set_axis_off()

    # Points
    for j in range(K):
        x, y, conf = points[j]
        if np.isnan(x) or np.isnan(y) or conf < conf_thr:
            continue
        ax.scatter(x, y, s=12, c=[kp_colors[j]], edgecolors='none', alpha=0.95, zorder=3)

    # # Lines
    # edges = list(skeleton)
    # if K > 19:
    #     edges.append((18,17))
    #     if K > 20:
    #         edges.append((18,19))
    # for a, b in edges:
    #     if a < K and b < K:
    #         xa, ya, ca = points[a]
    #         xb, yb, cb = points[b]
    #         if any(np.isnan(v) for v in (xa, ya, xb, yb)):
    #             continue
    #         if (ca < conf_thr) or (cb < conf_thr):
    #             continue
    #         c = (np.array(kp_colors[a]) + np.array(kp_colors[b])) / 2.0
    #         ax.plot([xa, xb], [ya, yb], linewidth=1.4, color=c, alpha=0.9, zorder=2)

    plt.tight_layout()
    plt.show()

def main(video_path, json_path, frame_number, save_dir="./frames"):
    # 1) Extract the specified frame as an image
    out_png = os.path.join(save_dir, f"frame_{frame_number:06d}.png")
    extract_frame_ffmpeg(video_path, frame_number, out_png)

    # 2) Get keypoints for the same frame from JSON (adapt to your data structure)
    with open(json_path, "rb") as f:
        res = json.load(f)

    # Example using kp of the first individual in res[frame_number] (loop for multiple)
    frame = res[frame_number]
    det = frame[0]                     # first individual
    points = np.array(det[5], float)   # (K,3) [x,y,conf]

    # 3) Use the extracted frame as background and overlay keypoints
    plot_frame_with_keypoints(out_png, points, conf_thr=0.0)

if __name__ == "__main__":
    # Usage example (adjust values to your environment)
    video_path = "vid/dailylife_cj611_20230226_110000.23506214/000000.mp4"
    json_path  = "results/2d_v0p8_Dark_fix_20/dailylife_cj611_20230226_110000/dailylife_cj611_20230226_110000_23506214_000000.json"
    frame_number = 65  # 0-based
    main(video_path, json_path, frame_number)

"""
affine_align.py が何をしているかを 1 枚で図示する説明プロット (plot)。

affine_align は「回転だけ残した計測座標(m)」から Blender座標へ、3Dアフィン P=A·X+t を
**再投影誤差最小化** で per-session に直接当てはめる。floorz の初期値 A0,t0 から least_squares
で A,t を求め、各カメラの 2D 観測 O_c(raw) に Blender投影 B_c(A·X+t) を近づける。

この図は affine_log.npz に保存済みの最適 A,t を読み、affine_align の関数をそのまま使って
「最適化で再投影誤差がどれだけ縮んだか / 何が変わったか」を 3 パネルで示す:

  左   : 再投影誤差(px) の分布   floorz初期(A0,t0) → 最適(A,t)。median 線つき。
  中央 : 1 カメラ・1 フレームの 2D 重畳。緑=実2D観測, 赤=初期投影, 青=最適投影。
         赤→青 が観測(緑)へ寄るのが「当てはめ」の中身。
  右   : 学習した A の中身。floorz初期に対する 残差回転角(deg) と 軸スケール。
         floorz の固定回転 ~3.75° ずれが主因、という主張をそのまま可視化する。

実行:
  python3 scripts/05_create_digital_twin/alignment_3D_keypoint/plot/plot_affine_align_explained.py \
      --session cj357_20221009_090000
出力 (--output-dir, 既定 outputs/reproj_eval/affine_explained): <session>_affine_explained.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/run"))
sys.path.insert(0, str(ROOT / "scripts/dev/floorZ/run"))
import align as _al                            # noqa: E402
import affine_align as aa                      # noqa: E402
from blender_fisheye import load_cameras       # noqa: E402

CFG = ROOT / "config/marmo_cj425m/config.yaml"

_CJK = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
if Path(_CJK).exists():
    fm.fontManager.addfont(_CJK)
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False


def per_kp_errors(items, cam_ids, bcams, A, t):
    """与えた A,t での 全(frame,cam,kp) の再投影誤差(px) を 1D 配列で返す。

    affine_align._residual は誤差ベクトル(dx,dy,...) を連結して返すので、(-1,2) に畳んで
    各点のユークリッド長を取れば「点ごとの px 誤差」になる。
    """
    p = np.concatenate([A.ravel(), t])
    res = aa._residual(p, items, cam_ids, bcams).reshape(-1, 2)
    return np.linalg.norm(res, axis=1)


def overlay_one_view(items, cam_ids, bcams, proj_omni, A0, t0, A, t):
    """有効点が一番多い (frame,cam) を選び、観測/初期投影/最適投影の 2D 点を返す。"""
    best = None
    for fi, (Xm, rd, vd) in enumerate(items):
        for c in cam_ids:
            n = int(vd[c].sum())
            if best is None or n > best[0]:
                best = (n, fi, c)
    _, fi, c = best
    Xm, rd, vd = items[fi]
    v = vd[c]
    obs = rd[c][v]                              # 実2D観測 O_c(raw)
    bef, _ = bcams[c].project((A0 @ Xm[v].T).T + t0, clip_fov=False)
    aft, _ = bcams[c].project((A @ Xm[v].T).T + t, clip_fov=False)
    return c, obs, bef, aft


def residual_rotation_and_scale(A0, A):
    """A を A0 基準で見る: 残差回転角(deg) と 各軸スケール(列ノルム比)。

    M = A0^{-1} A を極分解 M=R·S すると、R が floorz 初期からの残差回転、S が残差スケール。
    回転角は R のトレースから θ=arccos((tr-1)/2)。
    """
    M = np.linalg.solve(A0, A)
    U, s, Vt = np.linalg.svd(M)
    Rrot = U @ Vt
    if np.linalg.det(Rrot) < 0:                 # 反射を除く
        U[:, -1] *= -1
        Rrot = U @ Vt
    ang = np.degrees(np.arccos(np.clip((np.trace(Rrot) - 1) / 2, -1, 1)))
    axis_scale = np.linalg.norm(A, axis=0)      # A の各列(=各軸)のスケール
    return ang, axis_scale, s


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="cj357_20221009_090000")
    ap.add_argument("--input-dir", default=str(ROOT / "outputs/3d"))
    ap.add_argument("--npz", default=str(ROOT / "outputs/reproj_eval/affine_log.npz"))
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/reproj_eval/affine_explained"))
    ap.add_argument("--n-frames", type=int, default=120)
    args = ap.parse_args()

    cfg_y = yaml.safe_load(open(CFG))
    cam_ids = [str(c) for c in cfg_y["camera_id"]]
    W, H = cfg_y["img_size"]
    bcams = load_cameras()
    proj_omni = _al._omnidir_projectors(cam_ids)

    # 1) 最適 A,t を npz から取得（無ければその場で fit）
    A0, t0 = aa._floorz_init()
    store = dict(np.load(args.npz, allow_pickle=True)) if Path(args.npz).exists() else {}
    items = aa.build_targets(args.session, cam_ids, bcams, proj_omni, W, H,
                             n_frames=args.n_frames, input_dir=args.input_dir)
    if not items:
        sys.exit(f"追跡不足: {args.session}")
    if args.session in store:
        p = np.asarray(store[args.session]); A, t = p[:9].reshape(3, 3), p[9:]
    else:
        A, t, *_ = aa.fit(items, cam_ids, bcams)

    # 2) 誤差分布 / 重畳 / A 分解
    e_before = per_kp_errors(items, cam_ids, bcams, A0, t0)
    e_after = per_kp_errors(items, cam_ids, bcams, A, t)
    cam, obs, bef, aft = overlay_one_view(items, cam_ids, bcams, proj_omni, A0, t0, A, t)
    ang, axis_scale, s = residual_rotation_and_scale(A0, A)

    # ── 描画 ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle(f"affine_align.py: 再投影誤差最小化による per-session 3Dアフィン当てはめ"
                 f"  ({args.session})", fontsize=13)

    # 左: 誤差分布
    hi = np.percentile(np.concatenate([e_before, e_after]), 99)
    bins = np.linspace(0, hi, 60)
    ax[0].hist(e_before, bins=bins, alpha=0.55, color="#d62728", label="floorz初期 A0,t0")
    ax[0].hist(e_after, bins=bins, alpha=0.55, color="#1f77b4", label="最適 A,t")
    for e, col in [(e_before, "#d62728"), (e_after, "#1f77b4")]:
        m = np.median(e)
        ax[0].axvline(m, color=col, ls="--", lw=1.8)
        ax[0].text(m, ax[0].get_ylim()[1] * 0.92, f" {m:.1f}px", color=col, fontsize=10)
    ax[0].set_xlabel("再投影誤差 (px)"); ax[0].set_ylabel("点数")
    ax[0].set_title("① 最適化で誤差が縮む(median)"); ax[0].legend()

    # 中央: 2D 重畳
    ax[1].scatter(obs[:, 0], obs[:, 1], s=60, facecolors="none",
                  edgecolors="#2ca02c", linewidths=1.8, label="実2D観測 O_c")
    ax[1].scatter(bef[:, 0], bef[:, 1], s=22, color="#d62728", label="初期投影 B_c(A0·X+t0)")
    ax[1].scatter(aft[:, 0], aft[:, 1], s=22, color="#1f77b4", label="最適投影 B_c(A·X+t)")
    for i in range(len(obs)):                   # 初期→最適 の移動量を矢印で
        ax[1].annotate("", xy=aft[i], xytext=bef[i],
                       arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax[1].set_aspect("equal"); ax[1].invert_yaxis()
    ax[1].set_xlabel("px"); ax[1].set_ylabel("px")
    ax[1].set_title(f"② cam{cam} 最良フレームの 2D 重畳"); ax[1].legend(fontsize=8)

    # 右: A の分解
    lbls = ["残差回転\n(deg)", "scale X", "scale Y", "scale Z"]
    vals = [ang, axis_scale[0], axis_scale[1], axis_scale[2]]
    cols = ["#9467bd", "#1f77b4", "#1f77b4", "#1f77b4"]
    bars = ax[2].bar(lbls, vals, color=cols, alpha=0.8)
    ax[2].axhline(1.0, color="0.5", ls=":", lw=1)   # scale=1 基準
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=10)
    ax[2].set_title("③ 学習した A の中身\n(floorz初期からの残差回転+軸スケール)")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"{args.session}_affine_explained.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fp, dpi=130)
    print(f"median {np.median(e_before):.2f}px -> {np.median(e_after):.2f}px, "
          f"残差回転={ang:.2f}deg, axis_scale={axis_scale.round(3).tolist()}")
    print(f"wrote {fp}")


if __name__ == "__main__":
    main()

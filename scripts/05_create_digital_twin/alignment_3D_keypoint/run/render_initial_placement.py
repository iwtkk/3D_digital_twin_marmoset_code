"""
Tool: render_initial_placement

Purpose:
    Render a session's digital twin with the animal posed at the INITIAL placement
    (P = A0 @ X_m + t0: fixed rotation + per-axis scale + translation, before the
    reprojection-fitted affine), from one real-camera viewpoint, for chosen kp3d frames.

    1. Initial-placement CSV: affine_align.build_aligned_df(session, A0, t0) - the same code path
       as the released aligned CSV (A, t), only the transform differs. Derived points
       (shoulder_mid, base, center_eye) are midpoints, so they follow the same transform.
    2. Copy the session .blend into --work-dir (the dataset .blend is never opened for writing).
    3. Re-keyframe the armature / face objects of the COPY with
       scripts/05_create_digital_twin/blender_animation.py (same script that built the dataset).
    4. Render the requested frames of the copy with scripts/05_create_digital_twin/render_range.py.
    blend frame = kp3d index + 1.

    --side-view (one --frames value): instead of a real camera, add a pinhole camera "SideCam"
    that looks at the animal from the side, just above the floor, so that a body sunk into
    (or floating above) the floor is visible. The camera is placed from the affine-fit pose
    (released kp3d) of that frame: target = mean xy of wrists/ankles at --floor-z + --side-height,
    direction = perpendicular to the nose->tail_1 axis, on the side facing the cage centre,
    --side-dist away, tilted down by --side-elev degrees. With --side-view the released
    (affine-fit) blend is also copied to {session}_fit.blend, given the same SideCam and
    rendered, for a before/after pair from an identical viewpoint.
    --with-fit (real camera): also render a copy of the released (affine-fit) blend from --camera,
    saved as {session}_fit_Camera.{camera_id}_{blend_frame:05d}.png.

Inputs:
    kp3d pickle: outputs/3d/{session}/kp3d_fxdJointLen.pickle (path_registry.data.kp3d_3d_root)
    session blend (read only): {blend-dir}/{session}.blend
        default: path_registry.data.dataset_work_npz / 3D_digital_twin_data/3D_digitaltwin_marmoset_blender_files

Outputs:
    {work-dir}/{session}/{session}_initial.csv        initial-placement CSV (same columns as aligned_csv)
    {work-dir}/{session}/{session}_initial.blend      copy of the session blend, re-keyframed
    {work-dir}/{session}/render/{session}_initial_Camera.{camera_id}_{blend_frame:05d}.png
                                                     2048x1536 RGB(A) PNG
    --side-view:
    {work-dir}/{session}/{session}_fit.blend          copy of the released blend + SideCam
    {work-dir}/{session}/render/{session}_{initial,fit}_SideCam_{blend_frame:05d}.png
                                                     --side-res square PNG
    default work-dir: outputs/reproj_eval/initial_placement (path_registry.outputs.reproj_initial_placement)

Registry:
    .agent_docs/docs/tool_registry.md#render_initial_placement

Usage:
    .venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/run/render_initial_placement.py \\
        --session cj357_20221009_090000 --frames 9585 19170 --camera 23511613 --gpu 2
    .venv/bin/python scripts/05_create_digital_twin/alignment_3D_keypoint/run/render_initial_placement.py \\
        --session cj611_20210909_180000 --frames 3818 --side-view --gpu 2
    Blender: snap 5.3.0 cannot enable CUDA with driver 565, so --blender defaults to the 5.2.2 binary.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/dev"))
sys.path.insert(0, str(ROOT / "scripts/05_create_digital_twin/alignment_3D_keypoint/run"))
sys.path.insert(0, str(ROOT / "scripts/dev/floorZ/run"))

ANIM = ROOT / "scripts/05_create_digital_twin/blender_animation.py"
RENDER = ROOT / "scripts/05_create_digital_twin/render_range.py"
FLOOR_Z = 0.734   # top of the cage floor grid (Cube.003), measured by ray_cast in the session blend
ADD_CAM = r"""
import bpy, json, sys
from mathutils import Vector
c = json.loads(sys.argv[sys.argv.index("--") + 1])
cam = bpy.data.objects.get("SideCam")
if cam is None:
    cam = bpy.data.objects.new("SideCam", bpy.data.cameras.new("SideCam"))
    bpy.context.scene.collection.objects.link(cam)
cam.data.type = "PERSP"; cam.data.lens = c["lens"]; cam.data.sensor_width = 36.0
cam.data.clip_start = 0.01
cam.location = Vector(c["loc"])
cam.rotation_mode = "QUATERNION"
cam.rotation_quaternion = (Vector(c["target"]) - Vector(c["loc"])).to_track_quat("-Z", "Y")
bpy.ops.wm.save_mainfile()
"""
DATASET_BLEND_DIR = Path("/data/iwata/3D_digital_twin_marmoset_data_work_npz/3D_digital_twin_data/"
                         "3D_digitaltwin_marmoset_blender_files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True, help="kp3d frame indices")
    ap.add_argument("--camera", help="real camera id, e.g. 23511613 (not needed with --side-view)")
    ap.add_argument("--side-view", action="store_true", help="render from an added floor-level side camera")
    ap.add_argument("--with-fit", action="store_true",
                    help="also render the released (affine-fit) blend, via a copy, from the same --camera")
    ap.add_argument("--floor-z", type=float, default=FLOOR_Z)
    ap.add_argument("--side-height", type=float, default=0.03, help="camera height above the floor [m]")
    ap.add_argument("--side-dist", type=float, default=0.35, help="horizontal camera distance [m]")
    ap.add_argument("--side-elev", type=float, default=5.0, help="downward tilt [deg]")
    ap.add_argument("--side-lens", type=float, default=50.0, help="focal length [mm], 36 mm sensor")
    ap.add_argument("--side-res", default="1200x1200")
    ap.add_argument("--blend-dir", default=str(DATASET_BLEND_DIR))
    ap.add_argument("--input-dir", default=str(ROOT / "outputs/3d"))
    ap.add_argument("--work-dir", default=str(ROOT / "outputs/reproj_eval/initial_placement"))
    ap.add_argument("--armature", default="NF_Armature")
    ap.add_argument("--blender", default="/snap/blender/7803/blender")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for the render")
    ap.add_argument("--resolution", default="2048x1536")
    ap.add_argument("--samples", type=int, default=0, help="<=0: use the .blend setting")
    args = ap.parse_args()
    assert args.side_view or args.camera, "--camera or --side-view is required"
    assert not args.side_view or len(args.frames) == 1, "--side-view takes one --frames value"

    from affine_align import _floorz_init, build_aligned_df

    work = Path(args.work_dir) / args.session
    work.mkdir(parents=True, exist_ok=True)
    src = Path(args.blend_dir) / f"{args.session}.blend"
    blend = work / f"{args.session}_initial.blend"
    csv = work / f"{args.session}_initial.csv"
    assert src.resolve() != blend.resolve()

    # 1. initial-placement CSV
    A0, t0 = _floorz_init()
    df = build_aligned_df(args.session, A0, t0, args.input_dir)
    df.to_csv(csv, index=False)
    print(f"CSV: {csv} ({len(df)} frames)")

    # 2. copy (never touch the dataset blend)
    shutil.copyfile(src, blend)
    print(f"copied {src} -> {blend}")

    # 3. re-keyframe the copy (blender_animation.py saves bpy.data.filepath = the copy)
    subprocess.run([args.blender, str(blend), "-b", "-P", str(ANIM), "--", str(csv),
                    "--armature", args.armature], check=True)

    # 4. render
    out = work / "render"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    if args.side_view:
        cam = side_camera(args)
        fit_blend = work / f"{args.session}_fit.blend"
        shutil.copyfile(src, fit_blend)
        targets = [("initial", blend), ("fit", fit_blend)]
        for _, b in targets:
            subprocess.run([args.blender, str(b), "-b", "--python-expr", ADD_CAM, "--", json.dumps(cam)],
                           check=True)
        cams, res = ["SideCam"], args.side_res
    else:
        targets, cams, res = [("initial", blend)], [f"Camera.{args.camera}"], args.resolution
        if args.with_fit:
            fit_blend = work / f"{args.session}_fit.blend"
            if not fit_blend.exists():
                shutil.copyfile(src, fit_blend)
            targets.append(("fit", fit_blend))
    for tag, b in targets:
        for camname in cams:
            for i in args.frames:
                f = i + 1
                subprocess.run([args.blender, str(b), "-b", "-P", str(RENDER), "--",
                                "--camera", camname, "--start", str(f), "--end", str(f),
                                "--out", str(out), "--prefix", f"{args.session}_{tag}_{camname}_",
                                "--resolution", res, "--samples", str(args.samples),
                                "--overwrite"], check=True, env=env)
    print(f"renders: {out}")


def side_camera(args):
    """SideCam placement from the affine-fit pose of the single requested frame (see docstring)."""
    import numpy as np
    z = np.load(ROOT / "outputs/reproj_eval/aligned_npz" / f"{args.session}.npz")
    names = [str(n) for n in z["kp_names"]]
    P = z["kp3d"][args.frames[0]]
    feet = P[[names.index(k) for k in ("l_wrist", "r_wrist", "l_ankle", "r_ankle")], :2]
    cxy = np.nanmean(feet, 0)
    axis = P[names.index("tail_1"), :2] - P[names.index("nose"), :2]
    perp = np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
    if np.dot(perp, -cxy) < 0:                  # face the cage centre (x=0 side, more room)
        perp = -perp
    zc = args.floor_z + args.side_height
    loc_xy = cxy + perp * args.side_dist
    loc_z = zc + args.side_dist * np.tan(np.radians(args.side_elev))
    return dict(loc=[*map(float, loc_xy), float(loc_z)], target=[*map(float, cxy), float(zc)],
                lens=args.side_lens)


if __name__ == "__main__":
    main()

"""
I/O ledger: 05_create_digital_twin (NPZ route: kp3d -> aligned NPZ -> Blender)

================================================================================
Tool: affine_align   (alignment_3D_keypoint/run/affine_align.py)
================================================================================
Purpose:
    Fit, per session, the affine transform from the rotated triangulation frame [m] to the Blender
    scene by minimising the reprojection error (real omnidir cameras vs Blender equisolid cameras).

Inputs:
    input_dir/{session}/kp3d_fxdJointLen.pickle (or kp3d npz)
        kp3d (1, T, 20, 3) float [mm], kp3d_score (1, T, 20) float
        path_source: Step 2 output (outputs/marmo3d/3d_v0p8_dark_fix_20_all) or outputs/3d (S3 pickle)
    config/marmo_cj425m/{config.yaml, cam_intrinsic.h5, cam_extrinsic_optim.h5}
    config/marmo_cj425m/blender_cam.json  (Blender cameras; tracked in git, made by dump_blender_cam.py)

Outputs:
    output_dir/affine_log.npz : {session: (12,) float64 = A (3x3, row-major) + t (3,)}
    output_dir/affine_fit.log : median px before -> after per session

================================================================================
Tool: export_aligned_npz   (alignment_3D_keypoint/run/export_aligned_npz.py)
================================================================================
Purpose:
    Write the aligned 3D keypoints as one Blender-ready NPZ per session (= Harvard Dataverse NPZ).

Inputs:
    input_dir (same as affine_align), affine_log.npz, vid/ (video metadata, only when frame_num.npy is missing)
    --frame-num-dir: root holding {session}/{camera_id}/frame_num.npy. Default: --input-dir.
        The released NPZ files were made with /data/iwata/iwtkmarmo/results/3d_v0p8_dark_fix_20_all_det
        (path_registry.data.marmo3dpose_step1_results); pass it explicitly to reproduce them.

Outputs:
    output_dir/{session}.npz
        kp3d (T, 23, 3) float64 [Blender m], kp_names (23,) str, head_euler_xyz (T, 3) float64 [rad],
        kp3d_mm / kp3d_score / kp3d_err (T, 20[, 3]), joint_len (30,), rot, affine_A, affine_t,
        i_frame (T,), camera_id (8,), frame_num (8, T), frame_min (8,), frame_num_source, frame_ref_camera, fps

Validation:
    cj357_20221009_090000 from the S3 pickle + video metadata: every array equals the released NPZ
    (max |diff| = 0); only frame_num_source differs ("reconstructed" vs "step1").

================================================================================
Tool: blender_animation   (blender_animation.py, run inside Blender)
================================================================================
Purpose:
    Keyframe the armature, face-part objects and head cameras of the opened .blend from 3D keypoints.

Inputs:
    -- <path> [--armature NF_Armature]
    <path> = {session}.npz (keys kp3d, kp_names, head_euler_xyz as above) or the legacy CSV
        (columns pts_<kp>_{x,y,z}, pts_{l_cam,r_cam,center_eye}_r{x,y,z}).
    NPZ head_euler_xyz is used for all of pts_{l_cam,r_cam,center_eye}_r*.

Outputs:
    The opened .blend is saved in place (blend frame i+1 = kp3d[i]).

Validation:
    Bone locations of Head_IK / Base / Tail_IK at frames 1, 5000, 20000 match the NPZ to < 4e-8 m
    (Blender 5.2.2).

================================================================================
Tool: run_animate_   (run_animate_.sh)
================================================================================
Purpose:
    Copy the base .blend per session and run blender_animation.py on it.

Inputs:
    --npz-dir (default outputs/npz) *.npz, or --csv-dir *.csv
    --blend (default outputs/digitaltwin/3D_digitaltwin_marmoset_blender.blend)
    --session (repeatable) / --count / --all, env BLENDER (default: blender)

Outputs:
    <output-dir>/work_{session}.blend, <output-dir>/logs/work_{session}.log
    Exit status 1 if any session failed (the end of its log is printed).

================================================================================
Tool: render_sessions   (render_sessions.py, via run_render.sh)
================================================================================
Change:
    --npz-dir: take the session list from {session}.npz (frame count = kp3d length) instead of --csv-dir.
    Blender executable from env BLENDER (also render_parallel.py / render_worker.py).
    Default cameras: Camera_Eqdst_70, Camera_Pinhole_orig (same in run_depth.sh); the old
    Camera_C/L/R do not exist in the released blend.

Inputs:
    <blend-dir>/work_{session}.blend for each input; --cam must name a camera in the .blend
    (Camera.{camera_id}, Camera_Eqdst_70/100, Camera_Pinhole_orig/x2 in the released blend).

Outputs:
    <output-dir>/{session}/{camera}/{session}_{camera}_{frame:05d}.png
"""

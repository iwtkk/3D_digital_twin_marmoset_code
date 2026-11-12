"""
Blender の実カメラ(Camera.235*) パラメータを config/marmo_cj425m/blender_cam.json に書き出す。

実行（python3 ではなく blender 経由）:
  blender --background <blend> --python scripts/05_create_digital_twin/alignment_3D_keypoint/run/dump_blender_cam.py

出力 json は blender_fisheye.load_cameras() が読む。matrix_world(camera→world),
fisheye_lens, fisheye_fov, sensor, resolution を含む。
"""
import bpy, math, json
from pathlib import Path

scn = bpy.context.scene
# 解像度は実カメラ img_size(2048x1536) に合わせる。blend シーンの既定(1000x1000)は
# レンダ用の暫定値で実カメラと違うため使わない。scale=res_x/sensor_width は解像度に線形。
RESOLUTION = [2048, 1536]
out = {"resolution": RESOLUTION, "cameras": {}}
for ob in bpy.data.objects:
    if ob.type != "CAMERA" or not ob.name.startswith("Camera.235"):
        continue
    c = ob.data
    out["cameras"][ob.name.replace("Camera.", "")] = {
        "object": ob.name,
        "matrix_world": [list(r) for r in ob.matrix_world],
        "fisheye_lens": c.fisheye_lens,
        "fisheye_fov_rad": c.fisheye_fov,
        "fisheye_fov_deg": math.degrees(c.fisheye_fov),
        "sensor_width": c.sensor_width,
        "sensor_height": c.sensor_height,
        "sensor_fit": c.sensor_fit,
        "panorama_type": c.panorama_type,
    }
# repo_root/config/marmo_cj425m/blender_cam.json
repo = Path(__file__).resolve().parents[4]
p = repo / "config/marmo_cj425m/blender_cam.json"
p.write_text(json.dumps(out, indent=1))
print("WROTE", p, "ncam", len(out["cameras"]))

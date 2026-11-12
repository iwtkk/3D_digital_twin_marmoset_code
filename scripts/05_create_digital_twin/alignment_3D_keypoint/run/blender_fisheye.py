"""
Blender FISHEYE_EQUISOLID カメラの OpenCV/numpy 再現。

Blender Cycles の fisheye-equisolid は等立体角射影（intern/cycles/kernel/camera/projection.h
の direction_to_fisheye_equisolid と一致）:
    theta = acos(-dir_z)   (光軸 = カメラ -Z, +Y up)
    r     = 2 * fisheye_lens * sin(theta/2)    [mm, センサー面]
    px = res_x/2 + (res_x/sensor_width) * r * cos(phi)
    py = res_y/2 - (res_x/sensor_width) * r * sin(phi)
画素化は中心 (res_x/2, res_y/2)、スケール res_x/sensor_width [px/mm]。
縦も同じスケール: Blender は正方ピクセルを保つため sensor_fit=HORIZONTAL では縦の実効
センサーが sensor_width*(res_y/res_x) に再計算され、sensor_height の生値は使われない。
解像度は実カメラ img_size = 2048x1536（blend シーン既定 1000x1000 ではなく）。
shift_x/shift_y はこの 8 台すべて 0（主点=画像中心）。
実 Blender レンダと照合済み: 残差 中央値 0.66px / 最大 0.86px @2048x1536
（残差は発光球が魚眼で歪んだ blob 重心の検出誤差由来。モデル自体はサブピクセル）。

注意:
  実カメラは omnidir(Mei) モデル (config/marmo_cj425m/cam_intrinsic.h5 の K,xi,D)。
  Blender は equisolid。radial 則が違うので、同一 3D 点でも 2D は一致しない
  （中心付近は近く、周辺ほど系統差が出る）。数値比較時はこの差を前提にすること。

カメラ定義は config/marmo_cj425m/blender_cam.json（dump_blender_cam.py で生成）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAM_JSON = Path(__file__).resolve().parents[4] / "config/marmo_cj425m/blender_cam.json"


class BlenderFisheyeCam:
    """1台の Blender equisolid カメラ。project() で Blender世界(m) 3D → 画素2D。"""

    def __init__(self, matrix_world, fisheye_lens, sensor_width,
                 resolution, fisheye_fov_rad=None):
        self.M = np.asarray(matrix_world, float)          # camera→world (4x4)
        self.Rt = np.linalg.inv(self.M)                   # world→camera
        self.lens = float(fisheye_lens)
        self.sw = float(sensor_width)
        self.res_x, self.res_y = int(resolution[0]), int(resolution[1])
        self.fov = float(fisheye_fov_rad) if fisheye_fov_rad is not None else None
        self.scale = self.res_x / self.sw                 # px/mm（両軸共通）

    def project(self, P_world, clip_fov=True):
        """P_world: (N,3) Blender世界座標(m) → (px, valid) px:(N,2), valid:(N,) bool。

        valid=False は背面 or FOV外。clip_fov=False で FOV クリップ無効。
        """
        P = np.atleast_2d(np.asarray(P_world, float))
        Ph = np.c_[P, np.ones(len(P))]
        d = (Ph @ self.Rt.T)[:, :3]                       # camera 座標
        n = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.maximum(n, 1e-12)
        cos_t = np.clip(-d[:, 2], -1.0, 1.0)              # 光軸 = -Z
        theta = np.arccos(cos_t)
        phi = np.arctan2(d[:, 1], d[:, 0])
        r = 2.0 * self.lens * np.sin(theta / 2.0)         # mm
        px = 0.5 * self.res_x + self.scale * r * np.cos(phi)
        py = 0.5 * self.res_y - self.scale * r * np.sin(phi)   # Blender v↑ → 画素行↓
        valid = cos_t > -0.999                            # 背面除外
        if clip_fov and self.fov is not None:
            valid &= theta <= self.fov / 2.0
        valid &= (px >= 0) & (px < self.res_x) & (py >= 0) & (py < self.res_y)
        return np.c_[px, py], valid


def load_cameras(json_path=CAM_JSON):
    """blender_cam.json → {camera_id(str): BlenderFisheyeCam}。"""
    d = json.loads(Path(json_path).read_text())
    res = d["resolution"]
    cams = {}
    for cid, c in d["cameras"].items():
        cams[cid] = BlenderFisheyeCam(
            c["matrix_world"], c["fisheye_lens"], c["sensor_width"],
            res, c.get("fisheye_fov_rad"))
    return cams


if __name__ == "__main__":
    cams = load_cameras()
    print(f"loaded {len(cams)} Blender fisheye cameras from {CAM_JSON.name}")
    for cid, cam in cams.items():
        print(f"  {cid}: lens={cam.lens:.3f} scale={cam.scale:.2f}px/mm "
              f"fov={np.degrees(cam.fov):.1f}deg")

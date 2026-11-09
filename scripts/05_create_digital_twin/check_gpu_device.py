"""Cycles が本当に GPU でレンダリングしているかを判定する診断スクリプト。

Cycles は Preferences 側の compute_device_type が未設定だと、シーンで Device=GPU を
選んでいても無言で CPU にフォールバックする（デスクトップで実際に発生）。
このスクリプトは3点を確認する。

  1. 実在デバイス（prefs.get_devices_for_type）と has_active_device()
     ※ prefs.devices は過去にそのマシンで見えたデバイスが userpref に残り続けるので
       当てにならない（GPU1枚の環境で13件並ぶことを実測）
  2. 同一フレームを GPU / CPU で、サンプル数を2水準振ってレンダリングした所要時間
  3. GPU が効いていればサンプル数を増やしても時間はほとんど増えない（判定の根拠）

使い方:
  blender outputs/blender/work_{session}.blend -b \\
    -P scripts/05_create_digital_twin/check_gpu_device.py -- \\
    --camera Camera_C --res 1000x1000 --frame 3000 --samples 32,256 --out /tmp/checkgpu

レンダリング中に別端末で `nvidia-smi` を見て、GPU 使用率と VRAM が動いているかも
併せて確認するとより確実（本番運用では render_worker.py が nvidia-smi の
compute-apps にワーカーPIDが載るかを自動チェックしている）。
"""
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


RES = opt("--res", "1000x1000")
FRAME = int(opt("--frame", "1"))
CAMERA = opt("--camera", "Camera_C")
OUT = opt("--out", "/tmp/check_gpu_device")
BACKEND = opt("--backend", "CUDA")
SAMPLE_LIST = [int(s) for s in opt("--samples", "32,256").split(",")]

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = BACKEND
prefs.get_devices()

print("\n===== 1. デバイス =====", flush=True)
try:
    present = list(prefs.get_devices_for_type(BACKEND))
except AttributeError:
    present = list(prefs.devices)
for d in prefs.devices:
    d.use = (d.type == BACKEND)
print(f"実在デバイス: {[(d.name, d.type) for d in present]}", flush=True)
print(f"prefs.devices（幽霊込み）: {len(prefs.devices)} 件", flush=True)
try:
    active = prefs.has_active_device()
except AttributeError:
    active = None
print(f"compute_device_type={prefs.compute_device_type!r} has_active_device={active}",
      flush=True)
if not active:
    print("=> GPU が有効になっていない。この状態では CPU レンダリングになる。")

w, h = RES.split("x")
scene.render.resolution_x, scene.render.resolution_y = int(w), int(h)
scene.render.resolution_percentage = 100
scene.render.use_persistent_data = True
if CAMERA not in bpy.data.objects:
    raise SystemExit(f"カメラ '{CAMERA}' が見つかりません: "
                     f"{[o.name for o in bpy.data.objects if o.type == 'CAMERA']}")
scene.camera = bpy.data.objects[CAMERA]
scene.frame_set(FRAME)
scene.render.image_settings.file_format = 'PNG'
# デノイザは CPU/GPU 比較のノイズになるので切り、パストレース時間だけを見る
scene.cycles.use_denoising = False

print(f"\n===== 2. GPU / CPU × samples={SAMPLE_LIST} でレンダリング "
      f"(res={RES} frame={FRAME} camera={CAMERA}) =====")
res = {}
for dev in ["GPU", "CPU"]:
    scene.cycles.device = dev
    for s in SAMPLE_LIST:
        scene.cycles.samples = s
        scene.render.filepath = f"{OUT}_{dev}_{s}.png"
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        res[(dev, s)] = time.time() - t0
        print(f"  device={dev:3} samples={s:5} -> {res[(dev, s)]:7.2f}s", flush=True)

print("\n===== 3. 判定 =====", flush=True)
lo, hi = SAMPLE_LIST[0], SAMPLE_LIST[-1]
for dev in ["GPU", "CPU"]:
    d = res[(dev, hi)] - res[(dev, lo)]
    print(f"{dev}: samples {lo}->{hi} で {d:+7.2f}s "
          f"({d / (hi - lo) * 1000:+7.1f} ms/サンプル)", flush=True)
gpu_d = res[("GPU", hi)] - res[("GPU", lo)]
cpu_d = res[("CPU", hi)] - res[("CPU", lo)]
if gpu_d <= 0 or cpu_d / max(gpu_d, 1e-6) >= 2.0:
    print("=> GPU でパストレースしている（サンプル増加に対して GPU はほぼ平坦）")
else:
    print("=> GPU と CPU の伸びが同程度。GPU が使われていない疑いが強い。"
          " Preferences の compute_device_type と CUDA/OptiX ドライバを確認すること。")
print("\n注: 低サンプル・低解像度では、1枚の時間が CPU 側処理（シーン同期・転送・"
      "film・PNG書き出し）で決まるため、GPU の方が遅く見えることがある。")

import bpy, pathlib

scene = bpy.context.scene

# 出力先
OUT_DIR = pathlib.Path(bpy.path.abspath("//renders_selected"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== どのカメラを対象にするか（いずれか1つを使う）=====

# A) 名前で明示指定（おすすめ：スクショの名前をコピペ）
TARGET_CAM_NAMES = [
    "Camera.23506214",
    "Camera.23506226",
    "Camera.23506236",
    'Camera.23506237',
    'Camera.23506239',
    'Camera.23511607',
    'Camera.23511613',
    'Camera.23511614'
    # 必要なカメラ名だけ残す
]
cams = [bpy.data.objects[name] for name in TARGET_CAM_NAMES if name in bpy.data.objects]

# B) 現在選択中のカメラだけ
# cams = [o for o in bpy.context.selected_objects if o.type == 'CAMERA']

# C) 特定コレクション配下のカメラだけ
# TARGET_COLLECTION_NAME = "Camera"  # 例：左上アウトライナーのコレクション名
# col = bpy.data.collections.get(TARGET_COLLECTION_NAME)
# cams = [o for o in col.all_objects if o.type == 'CAMERA'] if col else []

# ===== レンダーするフレーム指定 =====
# 範囲＋ステップ
FRAME_START, FRAME_END, FRAME_STEP =  162, 162, 5
frames = list(range(FRAME_START, FRAME_END + 1, FRAME_STEP))
# もしくはピンポイント指定
# frames = [1, 10, 25, 42]

# ===== 形式など必要なら調整 =====
scene.render.image_settings.file_format = 'PNG'  # 'PNG','JPEG','OPEN_EXR' など
# scene.render.engine = 'CYCLES'
# scene.render.resolution_x, scene.render.resolution_y = 1920, 1080

# ===== 実行 =====
if not cams:
    print("対象カメラが見つかりません。TARGET_CAM_NAMES/選択/コレクション名を確認してください。")

for cam in cams:
    (OUT_DIR / cam.name).mkdir(parents=True, exist_ok=True)
    scene.camera = cam
    for f in frames:
        scene.frame_set(f)
        scene.render.filepath = str(OUT_DIR / f'frame_{f}' / f"{cam.name}_{f}.png")
        bpy.ops.render.render(write_still=True)

print("Done.")

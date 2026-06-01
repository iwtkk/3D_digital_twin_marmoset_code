import bpy
import csv
import mathutils
import os

# CSVファイル名と保存先を指定
# "//" は現在開いている .blend ファイルがあるディレクトリを表す
#csv_filepath = os.path.join(bpy.path.abspath("//"), "object_dimensions.csv")
#csv_filepath = os.path.join(bpy.path.abspath("//"), "object_dimensions_world-coordinate.csv")
csv_filepath = os.path.join(bpy.path.abspath("//"), "height.csv")

# CSV書き込み用のファイルオープン
# newline='', encoding='utf-8' を指定すると文字化けや改行の不具合が起きにくいです
with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # ヘッダー行を書き込み (列のタイトル)
#    writer.writerow(["ObjectName", "DimX", "DimY", "DimZ"])
    writer.writerow(["ObjectName", "MaxZ", "MinZ"])

    # シーン内の全オブジェクトを走査
    for obj in bpy.context.selected_objects:
        
#        diml = obj.dimensions
#        
#        dim_x = diml.x
#        dim_y = diml.y
#        dim_z = diml.z
#        
        # ワールド座標系でのバウンディングボックス頂点を算出
        bb_world_corners = [
            obj.matrix_world @ mathutils.Vector(corner)
            for corner in obj.bound_box
        ]

        # 各軸方向の min/max を調べる
        min_x = min(v.x for v in bb_world_corners)
        max_x = max(v.x for v in bb_world_corners)
        min_y = min(v.y for v in bb_world_corners)
        max_y = max(v.y for v in bb_world_corners)
        min_z = min(v.z for v in bb_world_corners)
        max_z = max(v.z for v in bb_world_corners)

        # 寸法を計算
        dim_x = max_x - min_x
        dim_y = max_y - min_y
        dim_z = max_z - min_z
        
        # CSV に1行書き込み x1000することでmmに変換
        writer.writerow([obj.name, max_z*1000, min_z*1000])
#        writer.writerow([obj.name,dim_x*1000, dim_y*1000, dim_z*1000])
print(f"CSV 出力完了: {csv_filepath}")

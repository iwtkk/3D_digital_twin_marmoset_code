import bpy
import bmesh
import mathutils
import csv
import os

# アクティブオブジェクトを取得
obj = bpy.context.active_object

# オブジェクトが MESH かつ Edit モードであることを確認
if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
    # bmesh を作成 (Edit モードのメッシュを参照)
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # 保存先のCSVファイルパス
    # "//" は現在の.blendファイルと同じフォルダを指す
    csv_filepath = bpy.path.abspath("//selected_vertices.csv")

    # ファイルを書き込み用に開く
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # ヘッダー行を書き込み（必要に応じて列名を変えてください）
        writer.writerow(["VertexIndex", "LocalX", "LocalY", "LocalZ", "WorldX", "WorldY", "WorldZ"])

        # bmesh 上の頂点を走査
        for v in bm.verts:
            if v.select:  # 選択されている頂点だけ
                # ローカル座標 (オブジェクト内部のメッシュ座標)
                local_co = v.co
                # ワールド座標
                world_co = obj.matrix_world @ local_co

                # CSVに1行書き込み
                writer.writerow([
                    v.index,
                    local_co.x*1000, local_co.y*1000, local_co.z*1000,
                    world_co.x*1000, world_co.y*1000, world_co.z*1000
                ])

    # bmesh を更新して終了
    bmesh.update_edit_mesh(me)

    print(f"CSV出力完了: {csv_filepath}")

else:
    print("アクティブオブジェクトが MESH ではないか、Edit モードではありません。")

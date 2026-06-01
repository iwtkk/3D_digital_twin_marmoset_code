import bpy
import bmesh

# アクティブオブジェクトを取得
obj = bpy.context.active_object

# オブジェクトが MESH かつ Edit モードであることを確認
if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
    # bmesh を作成 (Edit モードのメッシュを参照)
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # bmesh 上の頂点を走査
    for v in bm.verts:
        if v.select:  # 選択されている頂点だけ
            # ローカル座標 (オブジェクト内部のメッシュ座標)
            local_co = v.co
            # ワールド座標
            world_co = obj.matrix_world @ local_co

            print(f"Vertex Index: {v.index}")
            print(f"  Local : {local_co}")
            print(f"  World : {world_co}")
            print("-------------")

    # bmesh を更新して終了
    bmesh.update_edit_mesh(me)

else:
    print("アクティブオブジェクトが MESH ではないか、Edit モードではありません。")

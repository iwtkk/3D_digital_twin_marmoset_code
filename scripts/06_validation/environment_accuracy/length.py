import bpy
import bmesh

# アクティブオブジェクトを取得
obj = bpy.context.active_object

# オブジェクトが MESH かつ Edit モードであることを確認
if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
    # メッシュデータの bmesh を取得
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # 選択されているエッジを走査
    for e in bm.edges:
        if e.select:
            # ローカル座標系（オブジェクト内部）の辺の長さ
            length_local = e.calc_length()

            # ワールド座標系での辺の長さ
            v1_world = obj.matrix_world @ e.verts[0].co
            v2_world = obj.matrix_world @ e.verts[1].co
            length_world = (v2_world - v1_world).length

            print(f"Edge index:{e.index}")
            print(f"  ローカル座標系の長さ: {length_local}")
            print(f"  ワールド座標系の長さ: {length_world}")
            print("----------")

    # bmesh を更新（編集モードのメッシュに反映）
    bmesh.update_edit_mesh(me)

else:
    print("アクティブオブジェクトが MESH ではないか、Edit モードではありません。")

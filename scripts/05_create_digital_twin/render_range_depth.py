"""Blender 側のレンダーワーカー（Cycles Depth(Z) パス / EXR）。

render_range.py の depth 版。GPU 設定・最適化・warm ループ（--serve）は
render_range.py の実装をそのまま再利用し、ここでは
  - Depth パスを EXR で書く File Output ノードの構成（ワーカー起動時に1回だけ）
  - ジョブごとの出力先・ファイル名の切り替え
だけを担当する。

  blender work.blend -b -P render_range_depth.py -- --camera Camera_C \\
      --start 1 --end 50 --out outputs/depth/s/Camera_C --prefix s_Camera_C_
  blender work.blend -b -P render_range_depth.py -- --serve   # warm モード

備考: Cycles の Depth パスは魚眼カメラでレイ距離、ピンホールで平面深度を返す。
"""
import os
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_range as rr  # noqa: E402

# File Output ノード（warm モードでジョブをまたいで使い回す）
_FILE_OUTPUT = None


def setup_depth_nodes(scene, out_dir):
    """Depth パスを EXR(32bit float) で書き出す File Output ノードを構成し返す。

    Blender 5.x: コンポジタは scene.compositing_node_group (CompositorNodeTree の
    node group) に置く。File Output は出力ファイル名を node.file_name が決める
    （フレーム番号は付かないので毎フレーム file_name を更新する）。
    """
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    tree = bpy.data.node_groups.new("DepthComp", "CompositorNodeTree")
    scene.compositing_node_group = tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.scene = scene
    rl.layer = vl.name
    fo = tree.nodes.new('CompositorNodeOutputFile')
    fo.directory = out_dir
    fo.file_output_items.clear()
    fo.file_output_items.new(socket_type='FLOAT', name='depth')
    item = fo.file_output_items[0]
    item.override_node_format = True
    item.format.file_format = 'OPEN_EXR'
    item.format.color_depth = '32'
    item.format.color_mode = 'RGB'
    depth_out = rl.outputs.get('Depth') or rl.outputs.get('Z')
    if depth_out is None:
        raise RuntimeError(f"No Depth/Z output on Render Layers: "
                           f"{[o.name for o in rl.outputs]}")
    tree.links.new(depth_out, fo.inputs['depth'])
    return fo


def depth_setup(scene, args):
    """ワーカー起動時に1回だけ呼ばれる depth 固有の初期化。"""
    global _FILE_OUTPUT
    # depth は決定的なのでサンプルは最小・デノイズ無し
    scene.cycles.samples = args.samples if args.samples is not None else 1
    scene.cycles.use_denoising = False
    # 出力先はジョブごとに差し替えるので、ここでは空で構成しておく
    _FILE_OUTPUT = setup_depth_nodes(scene, "")
    return [f"depth: samples={scene.cycles.samples} use_denoising=False"]


def render_depth_job(job, args):
    """1ジョブ（=1カメラ・連続フレーム範囲）の Depth を EXR 出力する。"""
    scene = bpy.context.scene
    rr.set_camera(scene, rr.job_value(job, args, "camera"))

    resolution = rr.job_value(job, args, "resolution")
    if resolution:
        rr.set_resolution(scene, resolution)
    samples = rr.job_value(job, args, "samples")
    if samples:
        scene.cycles.samples = int(samples)

    out_dir = bpy.path.abspath(rr.job_value(job, args, "out"))
    os.makedirs(out_dir, exist_ok=True)
    _FILE_OUTPUT.directory = out_dir

    prefix = rr.job_value(job, args, "prefix", "frame_")
    start, end = int(job["start"]), int(job["end"])
    overwrite = bool(rr.job_value(job, args, "overwrite", False))
    scene.frame_start, scene.frame_end = start, end

    rendered = skipped = 0
    for frame in range(start, end + 1):
        name = f"{prefix}{frame:05d}"
        exr_path = os.path.join(out_dir, name + ".exr")
        if not overwrite and os.path.exists(exr_path):
            skipped += 1
            continue
        t0 = time.time()
        scene.frame_set(frame)
        _FILE_OUTPUT.file_name = name  # -> out_dir/<name>.exr
        # write_still=False でも File Output ノードは render 時に EXR を書く
        bpy.ops.render.render(write_still=False)
        rendered += 1
        rr.emit({"event": "frame", "id": job.get("id"), "frame": frame,
                 "sec": round(time.time() - t0, 2)})
    return {"rendered": rendered, "skipped": skipped}


def main():
    # depth は決定的なのでサンプル数の既定は 1（カラーの既定 32 とは別）
    parser = rr.build_arg_parser(
        "Render Cycles Depth(Z) pass for a frame range (batch or warm/--serve)",
        default_samples=1)
    rr.run(rr.parse_args(parser), render_depth_job, extra_setup=depth_setup)


if __name__ == "__main__":
    main()

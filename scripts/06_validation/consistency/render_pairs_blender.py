"""[Run inside Blender] Render the Fig. 8a pairs of one Session from the published per-session scene.

I/O ledger: scripts/06_validation/consistency/io.md#render_pairs_blender

  blender -b -P render_pairs_blender.py -- --model detailed_fur \\
      --blend {root}/3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender_files/{session}.blend --jobs jobs.json
  simplified_fur also needs --simplified-blend {root}/3D_digital_twin_marmoset_data/3D_digitaltwin_marmoset_blender_simplified_marmoset/{session}.blend

Body models (the Detailed-model scene is always the base, so environment, lights and cameras are the same):
  detailed_fur   : the published scene as it is (posed with the Aligned keypoints).
  detailed_nofur : the same scene with the hair curves hidden from rendering.
  background     : every NF_Marmoset* collection excluded = the empty cage, one image per camera; compute_metrics.py
                   uses it as the background of the render silhouette.
  simplified_fur : the animal collection NF_Marmoset_lowpoly (body, fur shells and the animated
                   NF_Armature_simplified) is appended from the published Simplified-model scene of the same Session,
                   and the Detailed animal (NF_Marmoset.001) is excluded. The Simplified-model scenes have a different
                   environment build, so only the animal is taken from them.
Render settings: Cycles, 500 samples, GPU, opaque background, 2048x1536 PNG (RGB 8 bit).
Nothing is saved back to the .blend files.

--jobs: [{"camera": "Camera.23506214", "blend_frame": 9041, "out": ".../0009040.png"}, ...]
"""
import argparse
import json
import os
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../05_create_digital_twin"))
from render_range import setup_gpu  # noqa: E402

MODELS = ("detailed_fur", "detailed_nofur", "simplified_fur")
BACKGROUND = "background"   # the scene with every animal collection excluded (empty cage), for the render silhouette
ANIMAL_PREFIX = "NF_Marmoset"
ANIMAL_TYPES = ("MESH", "CURVES", "CURVE")
FUR_MATERIAL_PREFIX = "har_material"
SIMPLIFIED_COLLECTION = "NF_Marmoset_lowpoly"
DETAILED_COLLECTION = "NF_Marmoset.001"


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Render Fig. 8a pairs of one Session")
    p.add_argument("--model", required=True, choices=MODELS + (BACKGROUND,))
    p.add_argument("--blend", required=True, help="published Detailed-model scene of the Session")
    p.add_argument("--simplified-blend", help="published Simplified-model scene of the Session (simplified_fur)")
    p.add_argument("--jobs", required=True)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--backend", default="CUDA", choices=["CUDA", "OPTIX"])
    return p.parse_args(argv)


def layer_collections(lc, hidden=False):
    hidden = hidden or lc.exclude or lc.collection.hide_render
    yield lc, hidden
    for ch in lc.children:
        yield from layer_collections(ch, hidden)


def hide_fur(scene):
    """Hide hair curves (and fur shells) of the visible animal from rendering."""
    n = 0
    for lc, hidden in layer_collections(scene.view_layers[0].layer_collection):
        if hidden or not lc.collection.name.startswith(ANIMAL_PREFIX):
            continue
        for o in lc.collection.objects:
            mats = [m.name for m in (getattr(getattr(o, "data", None), "materials", None) or []) if m]
            fur = o.type in ("CURVES", "CURVE") or any(m.startswith(FUR_MATERIAL_PREFIX) for m in mats)
            if o.type in ANIMAL_TYPES and fur and not o.hide_render:
                o.hide_render = True; n += 1
    if n == 0:
        raise RuntimeError("no fur object found")
    print(f"[render] hid {n} fur objects", flush=True)


def exclude_animals(scene):
    n = 0
    for lc, _ in layer_collections(scene.view_layers[0].layer_collection):
        if lc.collection.name.startswith(ANIMAL_PREFIX) and not lc.exclude:
            lc.exclude = True; n += 1
    print(f"[render] excluded {n} animal collections (background)", flush=True)


def swap_in_simplified(scene, path):
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        if SIMPLIFIED_COLLECTION not in src.collections:
            raise RuntimeError(f"{path} has no collection {SIMPLIFIED_COLLECTION}")
        dst.collections = [SIMPLIFIED_COLLECTION]
    col = dst.collections[0]
    (bpy.data.collections.get(ANIMAL_PREFIX) or scene.collection).children.link(col)
    det = next((lc for lc, _ in layer_collections(scene.view_layers[0].layer_collection)
                if lc.collection.name == DETAILED_COLLECTION), None)
    if det is None:
        raise RuntimeError(f"collection {DETAILED_COLLECTION} not found")
    det.exclude = True
    print(f"[render] appended {col.name}, excluded {DETAILED_COLLECTION}", flush=True)


def main():
    a = parse_args()
    jobs = [j for j in json.load(open(a.jobs)) if not os.path.exists(j["out"])]
    print(f"[render] {a.model}: {len(jobs)} to render", flush=True)
    if not jobs:
        return
    bpy.ops.wm.open_mainfile(filepath=a.blend)
    scene = bpy.context.scene
    if a.model == "simplified_fur":
        if not a.simplified_blend:
            raise SystemExit("--simplified-blend is required for simplified_fur")
        swap_in_simplified(scene, a.simplified_blend)
    elif a.model == "detailed_nofur":
        hide_fur(scene)
    elif a.model == BACKGROUND:
        exclude_animals(scene)
    setup_gpu(a.backend)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = a.samples
    scene.render.film_transparent = False
    scene.render.resolution_x, scene.render.resolution_y = 2048, 1536
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    for j in sorted(jobs, key=lambda j: (j["camera"], j["blend_frame"])):
        scene.camera = bpy.data.objects[j["camera"]]
        scene.frame_set(int(j["blend_frame"]))
        os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
        scene.render.filepath = j["out"]
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[render] {j['camera']} blend_frame {j['blend_frame']} {time.time() - t0:.1f}s", flush=True)
    print("[render] done", flush=True)


if __name__ == "__main__":
    main()

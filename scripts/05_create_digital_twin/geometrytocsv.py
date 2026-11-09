import bpy, sys, csv, os

# ==== Settings ====
OBJ_NAME = "Body.001"
MOD_NAME = "GeometryNodes"
OUTPUTS  = ['Head','Body','tail begin mid','tail mid end','shoulder L','shoulder R','elbow L','elbow R','hand L',"hand R",'pelvis L','pelvis R','leg L','leg R','foot L','foot R',]  # Output names shown in Group Output (multiple allowed)
CSV_PATH = bpy.path.abspath("//gn_outputs_cli.csv")
FRAME_START = bpy.context.scene.frame_start
FRAME_END   = bpy.context.scene.frame_end
# =============

def map_name_to_identifier(mod):
    m = {}
    for it in mod.node_group.interface.items_tree:
        if getattr(it, "in_out", None) == "OUTPUT" and getattr(it, "item_type", None) == "SOCKET":
            m[it.name] = it.identifier  # e.g. 'minDist_Hand_R' -> 'Socket_3'
    return m

def read_output_value(eval_obj, eval_mod, base_mod, ui_name, ident):
    """
    1) Value socket: read from eval_mod['Socket_n']
    2) Attribute output: get the attribute name from base_mod['Socket_n_attribute_name'], then read from eval_obj.data.attributes[...]
    """
    # First, try as a value socket
    try:
        if ident in eval_mod.keys():
            return float(eval_mod[ident])
    except Exception:
        pass

    # Read as an attribute output
    attr_key = ident + "_attribute_name"
    attr_name = base_mod.get(attr_key)
    if not attr_name:  # Empty string / None -> attribute name not set
        # Returning None here makes the caller substitute NaN
        return None

    mesh = eval_obj.data
    attr = mesh.attributes.get(attr_name)
    if not attr or len(attr.data) == 0:
        return None

    # Assumes a scalar attribute. Adjust retrieval for vector/color attributes
    elem0 = attr.data[0]
    val = getattr(elem0, "value", None)
    if val is None and hasattr(elem0, "vector"):
        val = elem0.vector[0]
    if val is None and hasattr(elem0, "color"):
        val = elem0.color[0]
    return float(val) if val is not None else None

# Get target objects
obj = bpy.data.objects.get(OBJ_NAME)
if not obj:
    raise RuntimeError(f"Object '{OBJ_NAME}' not found.")
mod = obj.modifiers.get(MOD_NAME)
if not mod or mod.type != 'NODES':
    raise RuntimeError(f"Modifier '{MOD_NAME}' not found or not a Geometry Nodes modifier.")

name_to_id = map_name_to_identifier(mod)
missing = [n for n in OUTPUTS if n not in name_to_id]
if missing:
    raise RuntimeError(f"Group Output names not found: {missing}")

scene = bpy.context.scene
deps  = bpy.context.evaluated_depsgraph_get()

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["frame"] + OUTPUTS)

    for frame in range(FRAME_START, FRAME_END + 1):
        scene.frame_set(frame)
        deps.update()
        eval_obj = obj.evaluated_get(deps)
        eval_mod = eval_obj.modifiers[mod.name]

        row = [frame]
        for ui_name in OUTPUTS:
            ident = name_to_id[ui_name]
            v = read_output_value(eval_obj, eval_mod, mod, ui_name, ident)
            row.append(v if v is not None else float("nan"))
        w.writerow(row)

print(f"Wrote: {CSV_PATH}")

import bpy
import bmesh
import mathutils

# Get the active object
obj = bpy.context.active_object

# Check that the object is a MESH and in Edit mode
if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
    # Get the bmesh of the mesh data
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # Iterate over the selected edges
    for e in bm.edges:
        if e.select:
            # Edge length in local coordinates (object space)
            length_local = e.calc_length()

            # Edge length in world coordinates
            v1_world = obj.matrix_world @ e.verts[0].co
            v2_world = obj.matrix_world @ e.verts[1].co
            length_world = (v2_world - v1_world).length

            print(f"Edge index:{e.index}")
            print(f"  Length (local coordinates): {length_local}")
            print(f"  Length (world coordinates): {length_world}")
            print("----------")

    # Update the bmesh (apply to the Edit mode mesh)
    bmesh.update_edit_mesh(me)

else:
    print("The active object is not a MESH or is not in Edit mode.")

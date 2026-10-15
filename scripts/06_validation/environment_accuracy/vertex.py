import bpy
import bmesh

# Get the active object
obj = bpy.context.active_object

# Check that the object is a MESH and in Edit mode
if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
    # Create a bmesh (referencing the Edit mode mesh)
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # Iterate over the vertices in the bmesh
    for v in bm.verts:
        if v.select:  # selected vertices only
            # Local coordinates (mesh coordinates in object space)
            local_co = v.co
            # World coordinates
            world_co = obj.matrix_world @ local_co

            print(f"Vertex Index: {v.index}")
            print(f"  Local : {local_co}")
            print(f"  World : {world_co}")
            print("-------------")

    # Update the bmesh and finish
    bmesh.update_edit_mesh(me)

else:
    print("The active object is not a MESH or is not in Edit mode.")

import bpy
import bmesh
import mathutils
import csv
import os

# Get the active object
obj = bpy.context.active_object

# Check that the object is a MESH and in Edit mode
if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
    # Create a bmesh (referencing the Edit mode mesh)
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    # Output CSV file path
    # "//" refers to the same folder as the current .blend file
    csv_filepath = bpy.path.abspath("//selected_vertices.csv")

    # Open the file for writing
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write the header row (change the column names as needed)
        writer.writerow(["VertexIndex", "LocalX", "LocalY", "LocalZ", "WorldX", "WorldY", "WorldZ"])

        # Iterate over the vertices in the bmesh
        for v in bm.verts:
            if v.select:  # selected vertices only
                # Local coordinates (mesh coordinates in object space)
                local_co = v.co
                # World coordinates
                world_co = obj.matrix_world @ local_co

                # Write one row to CSV
                writer.writerow([
                    v.index,
                    local_co.x*1000, local_co.y*1000, local_co.z*1000,
                    world_co.x*1000, world_co.y*1000, world_co.z*1000
                ])

    # Update the bmesh and finish
    bmesh.update_edit_mesh(me)

    print(f"CSV export complete: {csv_filepath}")

else:
    print("The active object is not a MESH or is not in Edit mode.")

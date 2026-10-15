import bpy
import csv
import mathutils
import os

csv_filepath = os.path.join(bpy.path.abspath("//"), "height.csv")

with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    writer.writerow(["ObjectName", "MaxZ", "MinZ"])

    for obj in bpy.context.selected_objects:
        bb_world_corners = [
            obj.matrix_world @ mathutils.Vector(corner)
            for corner in obj.bound_box
        ]

        min_z = min(v.z for v in bb_world_corners)
        max_z = max(v.z for v in bb_world_corners)

        writer.writerow([obj.name, max_z*1000, min_z*1000])
print(f"CSV export complete: {csv_filepath}")

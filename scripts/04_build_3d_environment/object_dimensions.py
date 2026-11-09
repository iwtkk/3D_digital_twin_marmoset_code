import bpy
import csv
import mathutils
import os

# Specify the CSV file name and save location
# "//" refers to the directory of the currently open .blend file
#csv_filepath = os.path.join(bpy.path.abspath("//"), "object_dimensions.csv")
#csv_filepath = os.path.join(bpy.path.abspath("//"), "object_dimensions_world-coordinate.csv")
csv_filepath = os.path.join(bpy.path.abspath("//"), "height.csv")

# Open the file for CSV writing
# Specifying newline='', encoding='utf-8' helps avoid garbled characters and newline issues
with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Write the header row (column titles)
#    writer.writerow(["ObjectName", "DimX", "DimY", "DimZ"])
    writer.writerow(["ObjectName", "MaxZ", "MinZ"])

    # Iterate over all objects in the scene
    for obj in bpy.context.selected_objects:
        
#        diml = obj.dimensions
#        
#        dim_x = diml.x
#        dim_y = diml.y
#        dim_z = diml.z
#        
        # Compute bounding box vertices in world coordinates
        bb_world_corners = [
            obj.matrix_world @ mathutils.Vector(corner)
            for corner in obj.bound_box
        ]

        # Find min/max along each axis
        min_x = min(v.x for v in bb_world_corners)
        max_x = max(v.x for v in bb_world_corners)
        min_y = min(v.y for v in bb_world_corners)
        max_y = max(v.y for v in bb_world_corners)
        min_z = min(v.z for v in bb_world_corners)
        max_z = max(v.z for v in bb_world_corners)

        # Compute dimensions
        dim_x = max_x - min_x
        dim_y = max_y - min_y
        dim_z = max_z - min_z
        
        # Write one row to CSV; multiply by 1000 to convert to mm
        writer.writerow([obj.name, max_z*1000, min_z*1000])
#        writer.writerow([obj.name,dim_x*1000, dim_y*1000, dim_z*1000])
print(f"CSV export complete: {csv_filepath}")

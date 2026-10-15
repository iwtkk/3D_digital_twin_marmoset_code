import bpy
import sys
import os


def parse_args():
    # Blender arguments after "--" are for the user script
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    import argparse
    parser = argparse.ArgumentParser(description="Render specific frame range with specific camera")

    parser.add_argument("--camera", required=True, help="Camera object name")
    parser.add_argument("--start", type=int, required=True, help="Start frame")
    parser.add_argument("--end", type=int, required=True, help="End frame (inclusive)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--prefix", default="frame_", help="Output filename prefix (default: frame_)")
    parser.add_argument("--resolution", default=None, help="Resolution WxH (e.g. 1920x1080)")

    args = parser.parse_args(argv)
    return args


def main():
    args = parse_args()

    scene = bpy.context.scene

    # Enable GPU (CUDA) rendering
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True
    scene.cycles.device = 'GPU'

    # Set camera
    if args.camera not in bpy.data.objects:
        raise ValueError(f"Camera '{args.camera}' not found")
    cam = bpy.data.objects[args.camera]
    scene.camera = cam

    # Resolution
    if args.resolution:
        w, h = args.resolution.split("x")
        scene.render.resolution_x = int(w)
        scene.render.resolution_y = int(h)
        scene.render.resolution_percentage = 100

    # Render settings: PNG
    scene.render.image_settings.file_format = 'PNG'
    # Switch to RGBA if needed
    # scene.render.image_settings.color_mode = 'RGBA'

    # Output folder
    out_dir = bpy.path.abspath(args.out)  # Also handles //relative paths
    os.makedirs(out_dir, exist_ok=True)

    # Set frame range (optional; if you also want to change the Timeline)
    scene.frame_start = args.start
    scene.frame_end = args.end

    # Render frame by frame
    for frame in range(args.start, args.end + 1):
        filepath = os.path.join(out_dir, f"{args.prefix}{frame:05d}.png")
        if os.path.exists(filepath):
            print(f"Skip frame {frame} (already exists)")
            continue
        scene.frame_set(frame)
        scene.render.filepath = filepath
        print(f"Rendering frame {frame} -> {filepath}")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()

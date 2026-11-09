# images_to_video - Image sequence -> MP4

Converts the per-camera PNG sequences in a session directory to MP4 with ffmpeg.

| Category | Script | Input | Output |
|---|---|---|---|
| run | `images_to_video.sh` | `--input-dir` (session directory) | `output_{session}_{cam}.mp4` → `images_to_video_files/` |

Accepts `--input-dir / --output-dir / --session / --help`. `--framerate N` (default 24) is available.

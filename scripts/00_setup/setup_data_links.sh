#!/usr/bin/env bash
# =============================================================================
# setup_data_links.sh  —  link the dataset to the repository via symlinks
# =============================================================================
#
# [What this script does]
#   Creates symlinks to the actual data directly under the repository root (code/)
#   and inside outputs/.
#   Run once; all subsequent scripts then work with the same paths.
#
# [Symlinks created (relative to repository root)]
#
#   3D_digitaltwin_marmoset_code/                              ← repository root
#   ├── vid             -> <DATA_ROOT>/marmoset_keypoint_data/video_data/
#   └── outputs/
#       ├── 2d          -> <DATA_ROOT>/marmoset_keypoint_data/2D_temporal_keypoint_data/
#       ├── 3d          -> <DATA_ROOT>/marmoset_keypoint_data/3D_temporal_keypoint_data/
#       └── digitaltwin -> <DATA_ROOT>/3D_digital_twin_marmoset_data/
#
#   Note: vid/ and outputs/ are in .gitignore, so the links themselves are not tracked by git.
#
# [Usage]
#   # Specify via argument
#   bash scripts/00_setup/setup_data_links.sh /path/to/3D_digital_twin_marmoset_data
#
#   # Specify via environment variable
#   export DATA_DIR_ROOT=/data/iwata/3D_digital_twin_marmoset_data
#   bash scripts/00_setup/setup_data_links.sh
#
# [About DATA_ROOT]
#   DATA_ROOT must be a directory containing the following subdirectories:
#     <DATA_ROOT>/marmoset_keypoint_data/
#     <DATA_ROOT>/3D_digital_twin_marmoset_data/
#
# =============================================================================

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DATA_ROOT="${1:-${DATA_DIR_ROOT:?'Specify DATA_DIR_ROOT as the first argument or the DATA_DIR_ROOT environment variable'}}"
DATA_ROOT="${DATA_ROOT%/}"  # Strip trailing slash

MKP="${DATA_ROOT}/marmoset_keypoint_data"
DTW="${DATA_ROOT}/3D_digital_twin_data"

echo "DATA_ROOT    : ${DATA_ROOT}"
echo "PROJECT_ROOT : $(pwd)"
echo ""

# Check that the target directories exist
for d in "${MKP}/video_data" "${MKP}/2D_temporal_keypoint_data" "${MKP}/3D_temporal_keypoint_data" "${DTW}"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: directory not found: $d" >&2
        exit 1
    fi
done

# --- Directly under repository root ---
ln -sfn "${MKP}/video_data" vid
echo "Created: $(pwd)/vid"
echo "   -> ${MKP}/video_data"

# --- Inside outputs/ ---
mkdir -p outputs
ln -sfn "${MKP}/2D_temporal_keypoint_data" outputs/2d
echo "Created: $(pwd)/outputs/2d"
echo "   -> ${MKP}/2D_temporal_keypoint_data"

ln -sfn "${MKP}/3D_temporal_keypoint_data" outputs/3d
echo "Created: $(pwd)/outputs/3d"
echo "   -> ${MKP}/3D_temporal_keypoint_data"

ln -sfn "${DTW}" outputs/digitaltwin
echo "Created: $(pwd)/outputs/digitaltwin"
echo "   -> ${DTW}"

echo ""
echo "=== Verify ==="
for link in vid outputs/2d outputs/3d outputs/digitaltwin; do
    target=$(readlink "$link" 2>/dev/null || echo "(not a symlink)")
    echo "  $link -> $target"
done

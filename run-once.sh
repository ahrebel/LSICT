#!/bin/sh
# LSICT no-trace runner for macOS / Linux.
#
# Downloads everything (uv, Python, all dependencies, model weights) into ONE
# temporary folder, runs the app, and deletes that folder when the app stops
# or the terminal window is closed. Nothing is left behind.
#
# Usage:  curl -LsSf https://raw.githubusercontent.com/ahrebel/LSICT/main/run-once.sh | sh
set -eu

LSICT_SPEC="lsict[gui] @ https://github.com/ahrebel/LSICT/archive/refs/heads/main.tar.gz"

WORK=$(mktemp -d "${TMPDIR:-/tmp}/lsict-once.XXXXXX")
cleanup() {
    echo ""
    echo "==> Cleaning up — deleting $WORK ..."
    rm -rf "$WORK"
    echo "==> Done. Nothing was left on this machine."
}
trap cleanup EXIT INT TERM HUP

# Confine EVERYTHING to the temp folder: uv itself, its caches, the Python it
# downloads, the tool environment, and the model weights fetched at runtime.
export UV_INSTALL_DIR="$WORK/uv"
export UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/python"
export UV_TOOL_DIR="$WORK/tools"
export XDG_CACHE_HOME="$WORK/cache"
export HF_HOME="$WORK/cache/huggingface"
export TORCH_HOME="$WORK/cache/torch"
export YOLO_CONFIG_DIR="$WORK/cache/ultralytics"

if command -v uv >/dev/null 2>&1; then
    UV=uv
else
    echo "==> Downloading uv into the temp folder..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    UV="$WORK/uv/uv"
fi

echo "==> Downloading LSICT and its dependencies (~2 GB) into:"
echo "        $WORK"
echo "    Nothing outside this folder is touched, and it is deleted on exit."
echo "    (Because nothing is kept, every launch re-downloads everything —"
echo "     use install.sh instead if you'll run LSICT more than once.)"
cd "$WORK"
"$UV" tool run --python 3.12 --from "$LSICT_SPEC" lsict gui

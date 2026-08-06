#!/bin/sh
# LSICT one-command installer for macOS / Linux.
# Usage:  curl -LsSf https://raw.githubusercontent.com/ahrebel/LSICT/main/install.sh | sh
set -eu

LSICT_SPEC="lsict[gui] @ https://github.com/ahrebel/LSICT/archive/refs/heads/main.tar.gz"

echo "==> LSICT installer"

# 1) Make sure uv is available (it manages Python and the install for us).
if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv (Python tool manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin; make it visible for the rest of this script.
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2) Install LSICT (uv downloads a suitable Python if none is present).
echo "==> Installing LSICT and its dependencies (~2 GB, mostly PyTorch)."
echo "    This can take several minutes on the first install..."
uv tool install --force --python 3.12 "$LSICT_SPEC"

# 3) Make sure the tool directory is on PATH for future shells.
uv tool update-shell >/dev/null 2>&1 || true

echo ""
echo "==> Done! Open a NEW terminal window and run:"
echo ""
echo "        lsict gui"
echo ""
echo "    (Model weights download automatically the first time you run a job.)"

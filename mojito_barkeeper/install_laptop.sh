#!/usr/bin/env bash
set -euo pipefail

export PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_NAME="mojito_barkeeper"
PYTHON_VERSION="3.12"

if command -v mamba >/dev/null 2>&1; then
    PKG_MANAGER=mamba
elif command -v conda >/dev/null 2>&1; then
    PKG_MANAGER=conda
else
    echo "ERROR: install Miniconda/Anaconda or mamba first." >&2
    exit 1
fi

if $PKG_MANAGER env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> Environment '$ENV_NAME' already exists."
else
    echo "==> Creating conda environment '$ENV_NAME' (Python $PYTHON_VERSION)"
    $PKG_MANAGER create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

CONDA_BASE=$($PKG_MANAGER info --base)
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> Installing runtime dependencies"
pip install -r "${PROJ_ROOT}/requirements.txt"

echo "==> Installing mojito_barkeeper in editable mode"
pip install -e "${PROJ_ROOT}"

if python -c "import pytest" &>/dev/null; then
    echo "==> pytest already available"
else
    pip install pytest
fi

echo "==> Done. Activate with: conda activate $ENV_NAME"
echo "==> Run GUI with: mojito-barkeeper-gui"

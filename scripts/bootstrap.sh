#!/bin/bash
set -e

# Usage: bootstrap.sh [venv-destination]
#   venv-destination defaults to ./venv (dev mode)
#
# In production, Electron passes:
#   $1              = path to venv in userData
#   ONICHAT_UV      = path to bundled uv binary
#   ONICHAT_APP_ROOT = read-only app bundle root (cwd is set to this)

VENV_DIR="${1:-$(pwd)/venv}"

# ── Python / venv setup ───────────────────────────────────────────────────────
# Prefer bundled uv (production), fall back to system uv, then system python3
if [ -n "$ONICHAT_UV" ] && [ -x "$ONICHAT_UV" ]; then
  UV="$ONICHAT_UV"
  echo "Using bundled uv: $UV"
elif command -v uv &>/dev/null; then
  UV="$(which uv)"
  echo "Using system uv: $UV"
else
  UV=""
fi

if [ -n "$UV" ]; then
  "$UV" venv --python 3.12 "$VENV_DIR"
  "$UV" pip install --python "$VENV_DIR/bin/python" -r backend/requirements.txt
  "$UV" pip install --python "$VENV_DIR/bin/python" mlx-audio
else
  PYTHON=$(which python3.12 2>/dev/null || which python3.11 2>/dev/null || which python3.10 2>/dev/null || which python3 2>/dev/null || echo "")
  if [ -z "$PYTHON" ]; then
    echo "Error: python3 not found and uv is not available."
    exit 1
  fi
  echo "Using system python: $PYTHON"
  "$PYTHON" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install -r backend/requirements.txt
  "$VENV_DIR/bin/pip" install mlx-audio
fi

mkdir -p "$(dirname "$VENV_DIR")/data"
echo "Bootstrap complete."

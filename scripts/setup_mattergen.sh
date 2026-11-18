#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_dir="$(dirname "$project_dir")"
uv venv "$workspace_dir/.venvs/mattergen" --python 3.10 --allow-existing
uv pip install --python "$workspace_dir/.venvs/mattergen/bin/python" \
  --override "$project_dir/configs/blackwell-overrides.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  --find-links 'https://data.pyg.org/whl/torch-2.7.0+cu128.html' \
  -e "$workspace_dir/mattergen" -e "$project_dir" chgnet pyg-lib

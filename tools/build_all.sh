#!/usr/bin/env bash
# Rebuild every rig layer from the generated passes in gen/master.jpg + gen/edits/ (no API calls).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tools/stage_align.py
python3 tools/stage_colorfix.py
python3 tools/stage_hair.py
python3 tools/stage_parts.py
echo "layers written to web/assets/parts/"

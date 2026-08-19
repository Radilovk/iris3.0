#!/usr/bin/env bash
# Runs every automated check in the project.
#
# These exist so claims about localization accuracy can be falsified rather than
# asserted: the geometry suite checks roll recovery, sector round-trip, occlusion
# handling and the validity gate against synthetic eyes with known ground truth,
# and the zone suite covers the label-to-organ arithmetic that used to be asked
# of the AI model.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Python: syntax ==="
python3 -m py_compile app.py
echo "ok"

echo
echo "=== JavaScript: syntax ==="
node --check worker.js
node --check iris-geometry.js
echo "ok"

echo
echo "=== Geometry (synthetic ground truth) ==="
python3 tests/test_geometry.py

echo
echo "=== Zone matching and coordinate logic ==="
node tests/test_zones.mjs

echo
echo "All checks passed."

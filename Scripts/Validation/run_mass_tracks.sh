#!/usr/bin/env bash
# Drives Scripts/Validation/mass_tracks.py end to end for one target
# z=0 central stellar mass. py-as-is is skipped: Halogrowth can't run
# standalone there (G3) -- see mass_tracks.py's module docstring.
set -euo pipefail
cd "$(dirname "$0")/../.."

TARGET="${1:-11.5}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== py-corrected (py3.11, .) =="
env/py-legacy/bin/python Scripts/Validation/mass_tracks.py dump \
    --repo-root . --target "$TARGET" --out "$TMP/corrected.csv"

echo "== rs-steel =="
(cd rust && cargo run --release -p steel-postprocess --example dump_mass_tracks -- "$TARGET") > "$TMP/rust.csv"

echo "== combine + plot =="
env/py-legacy/bin/python Scripts/Validation/mass_tracks.py combine \
    --corrected "$TMP/corrected.csv" --rust "$TMP/rust.csv" \
    --target "$TARGET" --outdir Figures/PortValidation

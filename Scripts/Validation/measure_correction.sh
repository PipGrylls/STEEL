#!/usr/bin/env bash
# Measure one correction in isolation.
#
# Runs the reduced validation grid, snapshots the output, and diffs it
# against the previous snapshot. Only meaningful because
# `py-corrected 2/13` made runs bit-reproducible: before that, every run
# differed and no correction's effect could be separated from the noise.
#
# Usage:
#   Scripts/Validation/measure_correction.sh <label> [run-tuple]
#
# The first invocation just establishes a baseline. Each later one
# prints which output arrays the working-tree change moved, and by how
# much.
set -euo pipefail

LABEL="${1:?usage: measure_correction.sh <label> [run-tuple]}"
RUN="${2:-1.0,True,True,True,G19_DPL,G19_SE}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SNAPDIR="${STEEL_SNAPSHOT_DIR:-${TMPDIR:-/tmp}/steel-corrections}"
PY="${STEEL_PYTHON:-$ROOT/env/py-asis/bin/python}"

# STEEL.py builds the run directory name by joining the tuple fields.
DIRNAME="RunParam_$(echo "$RUN" | tr ',' '\n' | while read -r f; do printf '%s_' "$f"; done)"

mkdir -p "$SNAPDIR"
cd "$ROOT"

"$PY" Scripts/Validation/run_py_steel.py \
    --halo-min 11.0 --halo-max 12.6 --halo-bin 0.5 \
    --run "$RUN" >/dev/null 2>&1

rm -rf "$SNAPDIR/$LABEL"
mkdir -p "$SNAPDIR/$LABEL"
cp -r "Data/Model/Output/RunFiles/$DIRNAME" "$SNAPDIR/$LABEL/"

PREV_LINK="$SNAPDIR/.previous"
if [ -f "$PREV_LINK" ]; then
    PREV="$(cat "$PREV_LINK")"
    echo "=== $PREV -> $LABEL ==="
    # compare_runs.py exits 1 when shapes differ, which is an expected
    # outcome for several of these corrections -- don't let `set -e`
    # abort before the snapshot pointer is advanced.
    "$PY" Scripts/Validation/compare_runs.py \
        "$SNAPDIR/$PREV/$DIRNAME" "$SNAPDIR/$LABEL/$DIRNAME" \
        | grep -vE '^\S+ +0\.0000e\+00 +0\.0000e\+00 +0\.0000 +0\.0000 +0$' || true
else
    echo "=== baseline snapshot: $LABEL ==="
fi
echo "$LABEL" > "$PREV_LINK"

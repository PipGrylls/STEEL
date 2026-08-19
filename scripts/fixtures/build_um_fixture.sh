#!/usr/bin/env bash
# Builds upstream UniverseMachine (UM-SAGA branch) and dumps reference
# grids. Cloned out of tree; never committed. Spec section 6.
#
# Usage: build_um_fixture.sh <scratch-dir> <output-dir>
set -euo pipefail

SCRATCH="${1:?scratch dir required}"
OUTDIR="${2:?output dir required}"

REPO="https://bitbucket.org/RW-Stanford/universemachine-saga.git"
REF="saga"
EXPECTED_SHA="6aff8d792e81bf6049058e3e1bc6f2cfa616b525"

mkdir -p "$SCRATCH" "$OUTDIR"
cd "$SCRATCH"

if [ ! -d universemachine-saga ]; then
  git clone --branch "$REF" "$REPO" universemachine-saga
fi
cd universemachine-saga

ACTUAL_SHA="$(git rev-parse HEAD)"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "FATAL: upstream $REF HEAD $ACTUAL_SHA != pinned $EXPECTED_SHA" >&2
  echo "Upstream moved. Re-pin deliberately and re-verify." >&2
  exit 1
fi

echo "== locating the UM-SAGA best-fit parameter file =="
find . -name '*.param' -o -name '*fit*' -o -name '*param*' | head -40

echo "== building =="
make clean || true
make

echo "== upstream built at $ACTUAL_SHA =="
echo "Next: evaluate SFR(vMpeak, z) and f_Q(vMpeak, z) on the grid fixed"
echo "in the plan, using the UM-SAGA best-fit parameters."

#!/usr/bin/env bash
# Builds upstream EMERGE and dumps reference grids for regression tests.
#
# Upstream is cloned OUT OF TREE and never committed: it is an MPI
# whole-pipeline C code, not a library, so vendoring would add a C
# toolchain for no fidelity gain over pinned outputs. Spec section 6.
#
# Usage: build_emerge_fixture.sh <scratch-dir> <output-dir>
set -euo pipefail

SCRATCH="${1:?scratch dir required}"
OUTDIR="${2:?output dir required}"

REPO="https://github.com/bmoster/emerge.git"
REF="v1.0.2"
EXPECTED_SHA="2781b54c21a80acf237daf7f2e71ff6254da8c3b"

mkdir -p "$SCRATCH" "$OUTDIR"
cd "$SCRATCH"

if [ ! -d emerge ]; then
  git clone --branch "$REF" --depth 1 "$REPO" emerge
fi
cd emerge

ACTUAL_SHA="$(git rev-parse HEAD)"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "FATAL: upstream HEAD $ACTUAL_SHA != pinned $EXPECTED_SHA" >&2
  echo "Upstream moved. Do not proceed: re-pin deliberately and re-verify." >&2
  exit 1
fi

echo "== building =="
make clean || true
make

echo "== upstream built at $ACTUAL_SHA =="
echo "Next: run EMERGE with a Planck15-matched parameter file and dump"
echo "eps(M_h, z) and integrated M*(M_h, z) on the grid fixed in the plan."

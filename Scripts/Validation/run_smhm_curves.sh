#!/usr/bin/env bash
# Drives Scripts/Validation/smhm_curves.py end to end -- see
# run_paper2_figures.sh for the interpreter-version rationale.
set -euo pipefail
cd "$(dirname "$0")/../.."

PIPGRYLLS_ROOT="${PIPGRYLLS_ROOT:-../STEEL-pipgrylls}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== py-as-is (py3.10, $PIPGRYLLS_ROOT) =="
env/py-asis/bin/python Scripts/Validation/smhm_curves.py dump \
    --repo-root "$PIPGRYLLS_ROOT" --out "$TMP/asis.csv"

echo "== py-corrected (py3.11, .) =="
env/py-legacy/bin/python Scripts/Validation/smhm_curves.py dump \
    --repo-root . --out "$TMP/corrected.csv"

echo "== rs-steel =="
(cd rust && cargo run --release -p steel-plugins --example dump_smhm_curves) > "$TMP/rust.csv"

echo "== combine + plot =="
env/py-legacy/bin/python Scripts/Validation/smhm_curves.py combine \
    --asis "$TMP/asis.csv" --corrected "$TMP/corrected.csv" --rust "$TMP/rust.csv" \
    --outdir Figures/PortValidation

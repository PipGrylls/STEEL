#!/usr/bin/env bash
# Drives Scripts/Validation/paper2_figures.py end to end: dumps
# py-as-is (env/py-asis, 3.10, ../STEEL-pipgrylls worktree),
# py-corrected (env/py-legacy, 3.11, this checkout) and rs-steel
# (cargo examples), then combines all three into the Figure 6/7
# reproductions under Figures/PortValidation/.
set -euo pipefail
cd "$(dirname "$0")/../.."

PIPGRYLLS_ROOT="${PIPGRYLLS_ROOT:-../STEEL-pipgrylls}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== py-as-is (py3.10, $PIPGRYLLS_ROOT) =="
env/py-asis/bin/python Scripts/Validation/paper2_figures.py dump --figure 6 \
    --repo-root "$PIPGRYLLS_ROOT" --out "$TMP/fig6_asis.csv"
env/py-asis/bin/python Scripts/Validation/paper2_figures.py dump --figure 7 \
    --repo-root "$PIPGRYLLS_ROOT" --out "$TMP/fig7_asis.csv"

echo "== py-corrected (py3.11, .) =="
env/py-legacy/bin/python Scripts/Validation/paper2_figures.py dump --figure 6 \
    --repo-root . --out "$TMP/fig6_corrected.csv"
env/py-legacy/bin/python Scripts/Validation/paper2_figures.py dump --figure 7 \
    --repo-root . --out "$TMP/fig7_corrected.csv"

echo "== rs-steel =="
(cd rust && cargo run --release -p steel-plugins --example dump_quenching) > "$TMP/fig6_rust.csv"
(cd rust && cargo run --release -p steel-plugins --example dump_merger_time) > "$TMP/fig7_rust.csv"

echo "== combine + plot =="
env/py-legacy/bin/python Scripts/Validation/paper2_figures.py combine \
    --asis-fig6 "$TMP/fig6_asis.csv" --corrected-fig6 "$TMP/fig6_corrected.csv" --rust-fig6 "$TMP/fig6_rust.csv" \
    --asis-fig7 "$TMP/fig7_asis.csv" --corrected-fig7 "$TMP/fig7_corrected.csv" --rust-fig7 "$TMP/fig7_rust.csv" \
    --outdir Figures/PortValidation

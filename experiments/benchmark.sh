#!/usr/bin/env bash
# Benchmark power index scaling with increasing agent pool size.
# Requires: hyperfine, uv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results"
PNML_DIR="${RESULTS_DIR}/pnml"
mkdir -p "$PNML_DIR"

FORMATIONS="1-1-1,2-1-1,3-1-1,4-1-1,4-2-1,4-3-1,4-4-1,4-4-2"

echo "=== Power Index Scaling Benchmark ==="
echo "Results directory: ${RESULTS_DIR}"
echo ""

# --- Setup: pre-generate all PNML files ---
echo "--- Generating PNML files ---"
for formation in ${FORMATIONS//,/ }; do
    uv run "${SCRIPT_DIR}/bench_one.py" \
        --generate \
        --formation "$formation" \
        --pnml "${PNML_DIR}/${formation}.pnml"
done
echo ""

# --- Path-based indices (fast): all 10 formations ---
for index in usability gatekeeper; do
    echo "--- Benchmarking ${index} ---"
    hyperfine \
        --warmup 2 \
        --min-runs 10 \
        --parameter-list formation "$FORMATIONS" \
        --export-json "${RESULTS_DIR}/${index}.json" \
        "uv run ${SCRIPT_DIR}/bench_one.py --index ${index} --formation {formation} --pnml ${PNML_DIR}/{formation}.pnml"
    echo ""
done

# --- Coalition-based indices (slow): skip largest formations ---
for index in shapley-shubik banzhaf; do
    echo "--- Benchmarking ${index} ---"
    hyperfine \
        --warmup 2 \
        --min-runs 10 \
        --parameter-list formation "$FORMATIONS" \
        --export-json "${RESULTS_DIR}/${index}.json" \
        "uv run ${SCRIPT_DIR}/bench_one.py --index ${index} --formation {formation} --pnml ${PNML_DIR}/{formation}.pnml"
    echo ""
done

echo "=== Done. Results in ${RESULTS_DIR}/ ==="

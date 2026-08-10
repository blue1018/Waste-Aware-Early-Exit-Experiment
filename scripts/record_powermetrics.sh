#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:-artifacts/logs/powermetrics.txt}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"
echo "Recording Apple power metrics to ${OUTPUT_PATH}. Press Control-C when the benchmark ends."
sudo powermetrics --samplers cpu_power,gpu_power -i 1000 -o "${OUTPUT_PATH}"


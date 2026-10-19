#!/usr/bin/env bash
# Run the three retained KuaiRand baselines sequentially on one GPU.
set -euo pipefail
cd "$(dirname "$0")"
GPU_ID="${GPU_ID:-0}"
CONFIG_DIR="${CONFIG_DIR:-./config}"
for MODEL in LONGER RankMixer TokenMixer; do
    python run_expid.py --config "$CONFIG_DIR" --expid "${MODEL}_KuaiRand_Video_Action" --gpu "$GPU_ID"
done

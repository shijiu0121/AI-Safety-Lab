#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/launch-logs
for group in main long_tail; do
  aegis experiment --stage train --group "$group" 2>&1 | tee "data/launch-logs/train-${group}.log"
done
for stage in evaluate detect defend anomaly; do
  aegis experiment --stage "$stage" 2>&1 | tee "data/launch-logs/${stage}.log"
done
aegis export --include-weights 2>&1 | tee data/launch-logs/export.log

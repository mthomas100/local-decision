#!/bin/zsh
# run-all.sh — one GPU slot's worth of Clef testing (written 2026-10-03, before any model load).
#
# For each target: load it, run the planted controls (decide smoke), run the use-case suites,
# unload. Then write the report. The flash models go first, so a cut-short slot still
# yields a full pass on one model. MLX targets skip CLINC and the two robustness variants: they
# run the suites that measure agreement with llama.cpp, plus the image suites only they can run.
#
#   bench/run-all.sh                 all four targets
#   bench/run-all.sh clef clef-mlx   just these
#
# Refuses to start while anything else holds the GPU (decide's guard, via m3d gpu).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LLAMA_SUITES=banking77,banking77-reversed,clinc-oos,sms-spam,sms-spam-injected,halueval-qa,shell-success,latency
MLX_SUITES=banking77,sms-spam,halueval-qa,shell-success,latency,render-qa,subject-match,count-shapes,frame-checks
if (( $# )); then targets=("$@"); else targets=(clef-flash clef clef-flash-mlx clef-mlx); fi

bin/decide gpu || { echo "run-all: GPU busy, not starting"; exit 3; }
t0=$SECONDS
for t in $targets; do
  case $t in *mlx) suites=$MLX_SUITES ;; *) suites=$LLAMA_SUITES ;; esac
  echo "=== $t ($(( (SECONDS - t0) / 60 )) min in)"
  bin/decide serve $t || { echo "run-all: $t failed to start, skipping"; continue; }
  bin/decide smoke
  $PY bench/usecases.py run --suites $suites
  bin/decide stop || { echo "run-all: $t did not stop; halting so nothing loads on top of it"; exit 4; }
done
$PY bench/usecases.py report > /dev/null
echo "=== done in $(( (SECONDS - t0) / 60 )) min; report: bench/results/$(date +%F)/REPORT.md"

#!/usr/bin/env bash
set -euo pipefail

RUN="${1:?usage: scripts/analyze_run_full.sh runs/model/run_name}"
INTERVAL="${2:-16}"
LAYERS="${LAYERS:-1 10 21 32}"
MAX_POINTS="${MAX_POINTS:-12000}"
MAX_VECTORS="${MAX_VECTORS:-20000}"

python3 scripts/analyze.py "$RUN" --tool generation_summary
python3 scripts/analyze.py "$RUN" --tool activation_norms

for L in $LAYERS; do
  python3 scripts/analyze.py "$RUN" \
    --tool trajectory_projection \
    --layer "$L" \
    --interval "$INTERVAL" \
    --method pca \
    --max-points "$MAX_POINTS"

  python3 scripts/analyze.py "$RUN" \
    --tool pca_components \
    --layer "$L" \
    --n 24 \
    --max-vectors "$MAX_VECTORS"
done

echo
echo "Analysis written to: $RUN/analysis"
echo "Default: PCA only, interval=$INTERVAL, max_points=$MAX_POINTS, max_vectors=$MAX_VECTORS"
echo "Run t-SNE manually on one layer only, with a sparse interval, if needed."

#!/usr/bin/env bash
# record-demo.sh — Record the modality routing demo with asciinema
#
# Prerequisites:
#   - vLLM AR model on port 8100 (Qwen/Qwen2.5-14B-Instruct)
#   - vLLM-Omni diffusion on port 8091 (Qwen/Qwen-Image)
#   - Router running with scripts/demo-modality/router-config.yaml
#   - Envoy running on port 8801
#
# Output: scripts/demo-modality/demo.cast

set -euo pipefail

SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SR_ROOT}"

CAST_FILE="${SR_ROOT}/scripts/demo-modality/demo.cast"

echo "Recording modality routing demo to: ${CAST_FILE}"
echo ""
echo "Prerequisites:"
echo "  - vLLM AR model on port 8100"
echo "  - vLLM-Omni diffusion on port 8091"
echo "  - Router: bin/router -config=scripts/demo-modality/router-config.yaml"
echo "  - Envoy on port 8801 with scripts/demo-modality/envoy.yaml"
echo ""

asciinema rec \
    --title "vLLM Semantic Router — Modality Routing (AR / Diffusion / Both)" \
    --cols 90 \
    --rows 50 \
    --command "PAUSE=2 bash scripts/demo-modality/demo-modality.sh" \
    --overwrite \
    "${CAST_FILE}"

echo ""
echo "Recording saved to: ${CAST_FILE}"
echo "Play with:  asciinema play ${CAST_FILE}"
echo "Upload with: asciinema upload ${CAST_FILE}"

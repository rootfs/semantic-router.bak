#!/usr/bin/env bash
# record-demo.sh — Record the multi-provider demo with asciinema
#
# Prerequisites:
#   - Authorino port-forwarded to :50052
#   - Router running with config/testing/config.multi-provider.yaml
#   - Envoy running with scripts/demo-multi-provider/envoy.yaml on :8801
#   - K8s Secrets and AuthConfig applied
#
# Output: scripts/demo-multi-provider/demo.cast

set -euo pipefail

SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SR_ROOT}"

CAST_FILE="${SR_ROOT}/scripts/demo-multi-provider/demo.cast"

echo "Recording multi-provider demo to: ${CAST_FILE}"
echo "The demo auto-advances (WAIT=5s between tests for asciinema)."
echo "Set WAIT=N to change the pause duration."
echo ""

asciinema rec \
    --title "vLLM Semantic Router — Multi-Cloud Provider Routing (OpenAI, Azure, Anthropic)" \
    --cols 100 \
    --rows 50 \
    --command "PAUSE=2 WAIT=5 AUTOPLAY=1 bash scripts/demo-multi-provider/demo-multi-provider.sh" \
    --overwrite \
    "${CAST_FILE}"

echo ""
echo "Recording saved to: ${CAST_FILE}"
echo "Play with:   asciinema play ${CAST_FILE}"
echo "Upload with: asciinema upload ${CAST_FILE}"

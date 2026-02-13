#!/usr/bin/env bash
# record-demo.sh — Record the Anthropic vs Bedrock demo with asciinema
#
# Prerequisites:
#   - Authorino port-forwarded to :50052
#   - Router running with scripts/demo-multi-provider/anthropic-bedrock/config.yaml
#   - Envoy running with scripts/demo-multi-provider/anthropic-bedrock/envoy.yaml on :8801
#   - K8s Secrets and AuthConfig applied
#
# Output: scripts/demo-multi-provider/anthropic-bedrock/demo.cast

set -euo pipefail

SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${SR_ROOT}"

CAST_FILE="${SR_ROOT}/scripts/demo-multi-provider/anthropic-bedrock/demo.cast"

echo "Recording Anthropic vs Bedrock demo to: ${CAST_FILE}"
echo "The demo auto-advances (WAIT=5s between tests for asciinema)."
echo "Set WAIT=N to change the pause duration."
echo ""

asciinema rec \
    --title "Anthropic vs AWS Bedrock — Same Claude Model, Two Providers" \
    --cols 100 \
    --rows 50 \
    --command "PAUSE=2 WAIT=5 AUTOPLAY=1 bash scripts/demo-multi-provider/anthropic-bedrock/demo.sh" \
    --overwrite \
    "${CAST_FILE}"

echo ""
echo "Recording saved to: ${CAST_FILE}"
echo "Play with:   asciinema play ${CAST_FILE}"
echo "Upload with: asciinema upload ${CAST_FILE}"

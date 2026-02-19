#!/usr/bin/env bash
# Stop all demo containers.
set -euo pipefail

echo "Stopping demo containers..."
docker rm -f openclaw-demo 2>/dev/null && echo "  ✓ openclaw-demo" || echo "  - openclaw-demo (not running)"
docker rm -f envoy-test 2>/dev/null && echo "  ✓ envoy-test" || echo "  - envoy-test (not running)"
docker rm -f semantic-router-test 2>/dev/null && echo "  ✓ semantic-router-test" || echo "  - semantic-router-test (not running)"
docker rm -f vllm-qwen-14b-rack1-tools 2>/dev/null && echo "  ✓ vllm-qwen-14b-rack1-tools" || echo "  - vllm-qwen-14b-rack1-tools (not running)"
echo "Done."

#!/usr/bin/env bash
# Start all services for the OpenClaw + Semantic Router demo.
# All containers run on the host network so they can talk via localhost.
#
# Prerequisites:
#   - Docker
#   - ROCm GPU (for vLLM) or adapt --device flags for NVIDIA
#   - Built openclaw:local image
#   - Built semantic-router Go binary at demo/semantic-router/router
#   - Rust .so libs at demo/semantic-router/libs/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Starting OpenClaw + Semantic Router Demo ==="
echo "Demo directory: $DEMO_DIR"
echo ""

# ── 1. vLLM (Qwen2.5-14B with tool calling) ──────────────────────
echo "[1/4] Starting vLLM (Qwen/Qwen2.5-14B-Instruct, 32k context, tool calling)..."
docker rm -f vllm-qwen-14b-rack1-tools 2>/dev/null || true
docker run -d \
  --name vllm-qwen-14b-rack1-tools \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -p 8100:8000 \
  --entrypoint vllm \
  vllm/vllm-openai-rocm:v0.15.0 \
  serve \
  --model Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
echo "  → vLLM starting on port 8100 (may take ~90s to load model)"

# ── 2. Semantic Router (vector store + ext_proc) ──────────────────
echo "[2/4] Starting Semantic Router (vector store + LLM routing)..."
docker rm -f semantic-router-test 2>/dev/null || true

ROUTER_BIN="${DEMO_DIR}/semantic-router/router"
LIBS_DIR="${DEMO_DIR}/semantic-router/libs"
SR_CONFIG="${DEMO_DIR}/semantic-router/config.yaml"
SR_ROUTER_CONFIG="${DEMO_DIR}/semantic-router/router-config-generated.yaml"

if [[ ! -f "$ROUTER_BIN" ]]; then
  echo "  ⚠  Router binary not found at $ROUTER_BIN"
  echo "     Build it first: cd semantic-router.bak/src/semantic-router && go build -o $ROUTER_BIN ./cmd/main.go"
  exit 1
fi

docker run -d \
  --name semantic-router-test \
  --network host \
  -v "$SR_CONFIG":/app/config.yaml:ro \
  -v "$SR_ROUTER_CONFIG":/app/.vllm-sr/router-config-override.yaml:ro \
  -v "$ROUTER_BIN":/app/router-new:ro \
  -v "$LIBS_DIR":/app/libs-new:ro \
  --entrypoint sh \
  ghcr.io/vllm-project/semantic-router/vllm-sr:latest \
  -c '
    /app/start-router.sh /app/config.yaml /app/.vllm-sr &
    PID=$!
    while [ ! -f /app/.vllm-sr/router-config.yaml ]; do sleep 0.5; done
    sleep 2
    kill $PID 2>/dev/null; wait $PID 2>/dev/null
    cp /app/.vllm-sr/router-config-override.yaml /app/.vllm-sr/router-config.yaml
    export LD_LIBRARY_PATH=/app/libs-new:/usr/local/lib
    exec /app/router-new \
      -config=/app/.vllm-sr/router-config.yaml \
      -port=50051 -enable-api=true -api-port=8080
  '
echo "  → Semantic Router starting on ports 8080 (API) and 50051 (gRPC)"

# ── 3. Envoy (LLM proxy with ext_proc) ───────────────────────────
echo "[3/4] Starting Envoy proxy..."
docker rm -f envoy-test 2>/dev/null || true
docker run -d \
  --name envoy-test \
  --network host \
  -v "${DEMO_DIR}/envoy/envoy.yaml":/etc/envoy/envoy.yaml:ro \
  envoyproxy/envoy:v1.31-latest
echo "  → Envoy listening on port 8801"

# ── 4. OpenClaw Gateway ──────────────────────────────────────────
echo "[4/4] Starting OpenClaw gateway..."

# Prepare writable directories
mkdir -p "${DEMO_DIR}/openclaw/config-live" "${DEMO_DIR}/openclaw/state"
cp "${DEMO_DIR}/openclaw/openclaw.json" "${DEMO_DIR}/openclaw/config-live/openclaw.json"
chmod -R 777 "${DEMO_DIR}/openclaw/config-live" "${DEMO_DIR}/openclaw/state" "${DEMO_DIR}/workspace"

docker rm -f openclaw-demo 2>/dev/null || true
docker run -d \
  --name openclaw-demo \
  --network host \
  -e OPENCLAW_CONFIG_PATH=/config/openclaw.json \
  -e OPENCLAW_STATE_DIR=/state \
  -v "${DEMO_DIR}/openclaw/config-live":/config \
  -v "${DEMO_DIR}/workspace":/workspace \
  -v "${DEMO_DIR}/openclaw/state":/state \
  openclaw:local \
  node openclaw.mjs gateway --allow-unconfigured --bind lan
echo "  → OpenClaw gateway on port 18788"

echo ""
echo "=== All services started ==="
echo ""
echo "Wait ~90s for vLLM to load the model, then run:"
echo "  bash ${DEMO_DIR}/scripts/test-all.sh"
echo ""

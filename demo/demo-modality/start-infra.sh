#!/usr/bin/env bash
# start-infra.sh — Start all infrastructure for the modality routing demo
#
# Starts: Semantic Router + Envoy proxy
# (Assumes vLLM containers are already running on ports 8100 and 8091)
#
# Usage:
#   bash scripts/demo-modality/start-infra.sh          # start
#   bash scripts/demo-modality/start-infra.sh --stop   # stop

set -euo pipefail

SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SR_ROOT}"

ROUTER_CONFIG="scripts/demo-modality/router-config.yaml"
ENVOY_CONFIG="scripts/demo-modality/envoy.yaml"
ROUTER_PORT=50051
API_PORT=8080
ENVOY_PORT=8801
LOG_DIR="/tmp/modality-demo"
ENVOY_CONTAINER="envoy-modality-demo"
ENVOY_IMAGE="envoyproxy/envoy:v1.33-latest"

BOLD="\033[1m"
RESET="\033[0m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
DIM="\033[2m"

info() { echo -e "  ${CYAN}INFO${RESET} $1"; }
pass() { echo -e "  ${GREEN}  OK${RESET} $1"; }
fail() { echo -e "  ${RED}FAIL${RESET} $1"; }

# ── Stop ─────────────────────────────────────────────────────
stop_all() {
    echo -e "${BOLD}Stopping modality demo infrastructure...${RESET}"

    # Stop router
    if lsof -ti:"${ROUTER_PORT}" >/dev/null 2>&1; then
        lsof -ti:"${ROUTER_PORT}" | xargs kill -9 2>/dev/null || true
        info "Router stopped"
    fi
    if lsof -ti:"${API_PORT}" >/dev/null 2>&1; then
        lsof -ti:"${API_PORT}" | xargs kill -9 2>/dev/null || true
    fi

    # Stop envoy
    docker rm -f "${ENVOY_CONTAINER}" >/dev/null 2>&1 || true
    info "Envoy stopped"

    echo -e "${GREEN}Infrastructure stopped.${RESET}"
}

if [[ "${1:-}" == "--stop" ]]; then
    stop_all
    exit 0
fi

# ── Pre-checks ───────────────────────────────────────────────
echo ""
echo -e "${BOLD}Starting modality demo infrastructure${RESET}"
echo -e "${DIM}────────────────────────────────────────────────────────────────${RESET}"
echo ""

# Check vLLM endpoints
info "Checking vLLM endpoints..."
for label_port in "AR (Qwen2.5-14B)|8100" "Diffusion (Qwen-Image)|8091"; do
    label="${label_port%%|*}"
    port="${label_port##*|}"
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        pass "${label} @ :${port}"
    else
        fail "${label} @ :${port} — not running!"
        echo -e "    ${DIM}Start it before running this script.${RESET}"
    fi
done
echo ""

# ── Start Router ─────────────────────────────────────────────
info "Starting semantic router..."

    # Kill any existing router/envoy on the same ports
    lsof -ti:"${ROUTER_PORT}" | xargs kill -9 2>/dev/null || true
    lsof -ti:"${API_PORT}" | xargs kill -9 2>/dev/null || true
    lsof -ti:19000 | xargs kill -9 2>/dev/null || true
    docker rm -f "${ENVOY_CONTAINER}" >/dev/null 2>&1 || true
    sleep 1

mkdir -p "${LOG_DIR}"
export LD_LIBRARY_PATH="${SR_ROOT}/candle-binding/target/release:${SR_ROOT}/ml-binding/target/release"

"${SR_ROOT}/bin/router" \
    -config="${ROUTER_CONFIG}" \
    > "${LOG_DIR}/router.log" 2>&1 &
ROUTER_PID=$!

# Wait for router to be ready
elapsed=0
while ! curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [[ ${elapsed} -ge 60 ]]; then
        fail "Router did not start within 60s"
        echo "  Check: ${LOG_DIR}/router.log"
        tail -20 "${LOG_DIR}/router.log" 2>/dev/null
        exit 1
    fi
done
pass "Router ready (PID ${ROUTER_PID}, ${elapsed}s) — log: ${LOG_DIR}/router.log"

# ── Start Envoy ──────────────────────────────────────────────
info "Starting Envoy proxy..."

docker rm -f "${ENVOY_CONTAINER}" >/dev/null 2>&1 || true

docker run -d \
    --name "${ENVOY_CONTAINER}" \
    --network=host \
    -v "${SR_ROOT}/${ENVOY_CONFIG}:/etc/envoy/envoy.yaml:ro" \
    "${ENVOY_IMAGE}" \
    envoy -c /etc/envoy/envoy.yaml \
    > /dev/null 2>&1

sleep 2
if docker ps -q --filter "name=${ENVOY_CONTAINER}" | grep -q .; then
    pass "Envoy ready (container: ${ENVOY_CONTAINER}) — :${ENVOY_PORT}"
else
    fail "Envoy failed to start"
    docker logs "${ENVOY_CONTAINER}" 2>&1 | tail -10
    exit 1
fi

# ── Verify Pipeline ──────────────────────────────────────────
echo ""
info "Verifying end-to-end pipeline..."
sleep 1

VERIFY_RESP=$(curl -sf "${ENVOY_URL:-http://127.0.0.1:${ENVOY_PORT}}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "Qwen/Qwen2.5-14B-Instruct", "messages": [{"role": "user", "content": "Say hello in one word"}], "max_tokens": 5}' \
    2>/dev/null || echo "")

if [[ -n "${VERIFY_RESP}" ]] && echo "${VERIFY_RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('choices')" 2>/dev/null; then
    pass "Pipeline verified: Client → Envoy → Router → vLLM"
else
    fail "Pipeline verification failed"
    echo "  Response: ${VERIFY_RESP:0:200}"
fi

# ── Done ─────────────────────────────────────────────────────
echo ""
echo -e "${DIM}────────────────────────────────────────────────────────────────${RESET}"
echo -e "${GREEN}${BOLD}Infrastructure ready!${RESET}"
echo ""
echo -e "  Router PID:  ${ROUTER_PID}"
echo -e "  Router log:  ${LOG_DIR}/router.log"
echo -e "  Envoy:       ${ENVOY_CONTAINER} (docker)"
echo ""
echo -e "  Run demo:    ${BOLD}bash scripts/demo-modality/demo-modality.sh${RESET}"
echo -e "  Record:      ${BOLD}bash scripts/demo-modality/record-demo.sh${RESET}"
echo -e "  Stop:        ${BOLD}bash scripts/demo-modality/start-infra.sh --stop${RESET}"
echo ""

#!/usr/bin/env bash
# ─── Shared helpers for ext_authz demo scripts ───
# Source this file, then call the exported functions.

set -euo pipefail
cd /data/semantic-router.bak

LOG_DIR="/tmp/ext-authz-demo-logs"
mkdir -p "$LOG_DIR"

# ── Colours ──
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

banner() {
    echo ""
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

info()  { echo -e "${GREEN}▸ $1${NC}"; }
warn()  { echo -e "${YELLOW}▸ $1${NC}"; }
show()  { echo -e "${BOLD}$1${NC}"; }
dim()   { echo -e "${DIM}$1${NC}"; }

# ── Send a request through Envoy ──
send_request() {
    local TOKEN="$1" MODEL="$2" LABEL="$3"
    info "Token: ${TOKEN}"
    info "Model: ${MODEL}"
    echo ""
    HTTP_CODE=$(curl -s -o /tmp/resp.json -w "%{http_code}" \
        http://localhost:8801/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${TOKEN}" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}]}")
    echo -e "  HTTP Status: ${YELLOW}${HTTP_CODE}${NC}"
    if python3 -c "import json,sys; d=json.load(open('/tmp/resp.json')); print(json.dumps(d.get('error',d),indent=2))" 2>/dev/null; then
        true
    else
        cat /tmp/resp.json 2>/dev/null
    fi
    echo ""
    info "ext_authz log → $(tail -1 "$LOG_DIR/authz.log")"
    echo ""
}

# ── Denied-request helper (no token / bad token) ──
send_denied_request() {
    local TOKEN="$1" LABEL="$2"
    info "${LABEL}"
    if [ -z "$TOKEN" ]; then
        HTTP_CODE=$(curl -s -o /tmp/resp.json -w "%{http_code}" \
            http://localhost:8801/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}')
    else
        HTTP_CODE=$(curl -s -o /tmp/resp.json -w "%{http_code}" \
            http://localhost:8801/v1/chat/completions \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${TOKEN}" \
            -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}')
    fi
    echo -e "  HTTP Status: ${YELLOW}${HTTP_CODE}${NC}"
    cat /tmp/resp.json 2>/dev/null; echo ""
    echo ""
}

# ── Start ext_authz, router, envoy ──
# Usage: start_services <auth_tokens_yaml>
start_services() {
    local AUTH_CONFIG="$1"

    info "Starting ext_authz on :9001 (config: ${AUTH_CONFIG}) ..."
    ./bin/authz -config "$AUTH_CONFIG" -addr :9001 >"$LOG_DIR/authz.log" 2>&1 &
    AUTHZ_PID=$!
    sleep 1

    info "Starting semantic router on :50051 ..."
    export LD_LIBRARY_PATH=${PWD}/candle-binding/target/release:${PWD}/ml-binding/target/release
    ./bin/router -config=config/testing/config.ext-authz-demo.yaml >"$LOG_DIR/router.log" 2>&1 &
    ROUTER_PID=$!
    printf "  Waiting for router"
    for i in $(seq 1 120); do
        if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
            echo -e " ${GREEN}ready!${NC}"
            break
        fi
        printf "."
        sleep 2
    done

    info "Starting Envoy on :8801 ..."
    ./tools/bin/func-e run --config-path config/envoy.yaml \
        --component-log-level "ext_proc:warn,router:warn,http:warn,ext_authz:warn" \
        >"$LOG_DIR/envoy.log" 2>&1 &
    ENVOY_PID=$!
    sleep 3

    show "All services running  (ext_authz :9001 | router :50051 | envoy :8801)"
    sleep 1
}

# ── Cleanup ──
cleanup() {
    echo ""
    warn "Cleaning up..."
    kill "$AUTHZ_PID" "$ROUTER_PID" "$ENVOY_PID" 2>/dev/null || true
    wait "$AUTHZ_PID" "$ROUTER_PID" "$ENVOY_PID" 2>/dev/null || true
    info "Done."
}

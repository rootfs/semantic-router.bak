#!/usr/bin/env bash
# Demo: Envoy ext_authz — Three Token Mapping Patterns
#   1:1 BYOT      — user brings own provider key
#   1:N per-user   — admin maps one user → distinct provider keys
#   N:1 shared     — admin maps many users → same provider keys
set -euo pipefail
cd /data/semantic-router.bak

LOG_DIR="/tmp/ext-authz-demo-logs"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'
RED='\033[0;31m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

banner() { echo -e "\n${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}"; echo -e "${BOLD}${CYAN}  $1${NC}"; echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${NC}\n"; }
info()   { echo -e "${GREEN}▸ $1${NC}"; }
warn()   { echo -e "${YELLOW}▸ $1${NC}"; }
show()   { echo -e "${BOLD}$1${NC}"; }
dim()    { echo -e "${DIM}$1${NC}"; }

# Helper: send request, show HTTP status + response + authz log
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
    # Extract just the error message for OpenAI responses
    if python3 -c "import json,sys; d=json.load(open('/tmp/resp.json')); print(json.dumps(d.get('error',d),indent=2))" 2>/dev/null; then
        true
    else
        cat /tmp/resp.json 2>/dev/null
    fi
    echo ""
    info "ext_authz log → $(tail -1 "$LOG_DIR/authz.log")"
    echo ""
}

cleanup() {
    echo ""
    warn "Cleaning up..."
    kill "$AUTHZ_PID" "$ROUTER_PID" "$ENVOY_PID" 2>/dev/null || true
    wait "$AUTHZ_PID" "$ROUTER_PID" "$ENVOY_PID" 2>/dev/null || true
    info "Done."
}
trap cleanup EXIT

########################################################################
banner "1. Auth Tokens Config — Three Patterns"
########################################################################

cat config/auth_tokens.yaml
sleep 3

########################################################################
banner "2. Start Services (ext_authz + router + envoy)"
########################################################################

info "Starting ext_authz on :9001 ..."
./bin/authz -config config/auth_tokens.yaml -addr :9001 >"$LOG_DIR/authz.log" 2>&1 &
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
sleep 2

########################################################################
banner "3. Pattern 1 — BYOT (1:1): Bring Your Own Token"
########################################################################

show "User's access token IS their own OpenAI key."
show "ext_authz passes the same key through to the provider."
dim "access_token = sk-byot-user-own-openai-key-abc123"
dim "openai key   = sk-byot-user-own-openai-key-abc123  (same value)"
echo ""

info "Sending to OpenAI (gpt-4o):"
send_request "sk-byot-user-own-openai-key-abc123" "gpt-4o" "BYOT-OpenAI"
sleep 2

info "Sending to Anthropic (claude-3-5-sonnet-latest):"
send_request "sk-byot-user-own-openai-key-abc123" "claude-3-5-sonnet-latest" "BYOT-Anthropic"
sleep 2

########################################################################
banner "4. Pattern 2 — Admin per-user (1:N): Alice's local token"
########################################################################

show "Alice has a LOCAL token: sr-alice-local-token-7890"
show "Admin mapped it to SEPARATE provider keys:"
dim "openai    → sk-proj-alice-dedicated-openai-key"
dim "anthropic → sk-ant-alice-dedicated-anthropic-key"
echo ""

info "Alice → OpenAI (gpt-4o):"
send_request "sr-alice-local-token-7890" "gpt-4o" "Alice-OpenAI"
sleep 2

info "Alice → Anthropic (claude-3-5-sonnet-latest):"
send_request "sr-alice-local-token-7890" "claude-3-5-sonnet-latest" "Alice-Anthropic"
sleep 2

########################################################################
banner "5. Pattern 3 — Shared keys (N:1): Bob & Carol"
########################################################################

show "Bob  has token: sr-bob-team-token-1111"
show "Carol has token: sr-carol-team-token-2222"
show "Both map to the SAME provider keys:"
dim "openai    → sk-proj-shared-team-openai-key-TEAM"
dim "anthropic → sk-ant-shared-team-anthropic-key-TEAM"
echo ""

info "Bob → OpenAI (gpt-4o):"
send_request "sr-bob-team-token-1111" "gpt-4o" "Bob-OpenAI"
sleep 2

info "Carol → OpenAI (gpt-4o):"
send_request "sr-carol-team-token-2222" "gpt-4o" "Carol-OpenAI"
sleep 2

info "Bob → Anthropic (claude-3-5-sonnet-latest):"
send_request "sr-bob-team-token-1111" "claude-3-5-sonnet-latest" "Bob-Anthropic"
sleep 2

########################################################################
banner "6. Results Summary"
########################################################################

show "Pattern 1 — BYOT (1:1):"
echo "  User's own key used as both access token and provider key."
echo "  OpenAI error shows:  sk-byot-***-abc123  (user's own key)"
echo ""
show "Pattern 2 — Admin per-user (1:N):"
echo "  Local token sr-alice-local-token-7890 mapped to:"
echo "    OpenAI:    sk-proj-***-openai-key  (Alice-specific)"
echo "    Anthropic: sk-ant-***-anthropic-key (Alice-specific)"
echo ""
show "Pattern 3 — Shared keys (N:1):"
echo "  Bob  (sr-bob-team-token-1111)  ┐"
echo "  Carol (sr-carol-team-token-2222) ┘→ same sk-proj-***-key-TEAM"
echo ""
echo "  ┌───────────────────────────────────────────────────────┐"
echo "  │  Pattern   │  User Token → Provider Key               │"
echo "  │────────────┼──────────────────────────────────────────│"
echo "  │  1:1 BYOT  │  sk-byot... → sk-byot... (same key)     │"
echo "  │  1:N Admin  │  sr-alice.. → sk-proj-alice.. (unique)   │"
echo "  │  N:1 Shared │  sr-bob..  ┐                             │"
echo "  │             │  sr-carol..┘→ sk-proj-shared.. (shared)  │"
echo "  └───────────────────────────────────────────────────────┘"
echo ""
sleep 3
echo -e "${GREEN}Demo complete!${NC}"

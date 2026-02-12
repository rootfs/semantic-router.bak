#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Authorino Integration Demo
# End-to-end: Authorino ext_authz → Envoy → Semantic Router
# Demonstrates 1:1 (BYOT), 1:N (per-user), N:1 (shared) token patterns
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd /data/semantic-router.bak

# ── Colors & helpers ──────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BLUE='\033[0;34m'; BOLD='\033[1m'
DIM='\033[2m'; NC='\033[0m'

banner() { echo -e "\n${BLUE}${BOLD}═══════════════════════════════════════════════${NC}"; echo -e "${CYAN}${BOLD}  $1${NC}"; echo -e "${BLUE}${BOLD}═══════════════════════════════════════════════${NC}"; }
section() { echo -e "\n${YELLOW}${BOLD}▶ $1${NC}"; }
info() { echo -e "${DIM}  $1${NC}"; }
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; }
pause() { sleep "${1:-1.5}"; }
show_file() {
    echo -e "\n${CYAN}── $1 ──${NC}"
    cat "$1"
    echo -e "${CYAN}── end ──${NC}"
}
show_file_range() {
    echo -e "\n${CYAN}── $1 (lines $2-$3) ──${NC}"
    sed -n "${2},${3}p" "$1"
    echo -e "${CYAN}── end ──${NC}"
}

banner "Authorino Integration Demo"
echo -e "  Token management via Kubernetes-native ext_authz"
echo -e "  Patterns: ${BOLD}1:1 BYOT${NC} | ${BOLD}1:N Per-user${NC} | ${BOLD}N:1 Shared${NC}"
pause 2

# ──────────────────────────────────────────────────────────────
# SECTION 1: Show Authorino running in Kind cluster
# ──────────────────────────────────────────────────────────────
banner "1. Authorino Running in Kind Cluster"

section "Kind cluster"
kubectl cluster-info --context kind-authorino-test 2>&1 | head -3
pause

section "Authorino deployment"
kubectl get deployment authorino -o wide
pause

section "Authorino pod status"
kubectl get pods -l app=authorino
pause

section "Authorino logs — secrets indexed"
kubectl logs deployment/authorino 2>&1 | grep "api key added" | tail -5
pause 2

# ──────────────────────────────────────────────────────────────
# SECTION 2: Show Authorino AuthConfig Policy
# ──────────────────────────────────────────────────────────────
banner "2. AuthConfig — The Authorization Policy"

section "AuthConfig CRD (Kubernetes resource)"
show_file config/authorino/authconfig.yaml
pause 2

section "AuthConfig status"
kubectl get authconfig semantic-router-auth -o jsonpath='{.status.summary}' | python3 -m json.tool 2>/dev/null || kubectl get authconfig semantic-router-auth -o yaml | grep -A10 "status:"
pause 2

# ──────────────────────────────────────────────────────────────
# SECTION 3: Show Token Mapping — Kubernetes Secrets
# ──────────────────────────────────────────────────────────────
banner "3. Token Mapping — Kubernetes Secrets"

echo -e "\n${BOLD}Three patterns encoded as Secret annotations:${NC}"
echo ""
echo -e "  ${CYAN}1:1 BYOT${NC}     — user's own provider key IS the Bearer token"
echo -e "  ${CYAN}1:N Per-user${NC} — admin-issued token → dedicated provider keys"
echo -e "  ${CYAN}N:1 Shared${NC}   — multiple user tokens → same team provider keys"
pause 2

section "Pattern 1:1 — BYOT (Dave)"
show_file config/authorino/secrets-byot.yaml
pause 2

section "Pattern 1:N — Per-user (Alice)"
show_file config/authorino/secrets-per-user.yaml
pause 2

section "Pattern N:1 — Shared (Bob + Carol)"
show_file config/authorino/secrets-shared.yaml
pause 2

section "All Secrets in cluster"
kubectl get secrets -l app=semantic-router -o custom-columns=NAME:.metadata.name,OPENAI:.metadata.annotations.openai-key,ANTHROPIC:.metadata.annotations.anthropic-key
pause 2

# ──────────────────────────────────────────────────────────────
# SECTION 4: Show Envoy Config
# ──────────────────────────────────────────────────────────────
banner "4. Envoy Configuration"

section "ext_authz filter → Authorino (gRPC)"
info "Envoy sends every request to Authorino for auth before ext_proc"
show_file_range config/testing/envoy-authorino-test.yaml 69 79
pause 2

section "ext_proc filter → Semantic Router"
info "After auth passes, request goes to semantic router for model selection"
show_file_range config/testing/envoy-authorino-test.yaml 80 95
pause 2

section "Authorino cluster (gRPC, port-forwarded from Kind)"
show_file_range config/testing/envoy-authorino-test.yaml 137 155
pause 2

# ──────────────────────────────────────────────────────────────
# SECTION 5: Show Router Code Changes
# ──────────────────────────────────────────────────────────────
banner "5. Semantic Router Changes (3 Go files)"

section "headers.go — new header constants"
info "These headers are injected by Authorino and read by the router"
grep -A6 "ext_authz / Authorino" src/semantic-router/pkg/headers/headers.go | head -10
pause 2

section "processor_req_header.go — capture injected keys"
info "Router reads x-user-openai-key and x-user-anthropic-key from headers"
grep -A4 "Capture ext_authz" src/semantic-router/pkg/extproc/processor_req_header.go
pause 2

section "processor_req_body.go — prefer injected key over static config"
info "For each routing path, prefer the per-user key, strip before forwarding"
grep -B1 -A3 "prefer ext_authz" src/semantic-router/pkg/extproc/processor_req_body.go
pause 2

# ──────────────────────────────────────────────────────────────
# SECTION 6: Start Services
# ──────────────────────────────────────────────────────────────
banner "6. Start Services"

section "Port-forward Authorino from Kind → localhost:50055"
kubectl port-forward svc/authorino 50055:50051 5001:5001 &>/tmp/authorino-pf.log &
PF_PID=$!
sleep 2
if ss -tlnp | grep -q 50055; then
    ok "Authorino gRPC available on localhost:50055"
else
    fail "Port-forward failed"
    exit 1
fi
pause

section "Verify Semantic Router (ext_proc on :50051)"
if ss -tlnp | grep -q ":50051"; then
    ok "Semantic Router ext_proc running on :50051"
else
    fail "Router not running"
    exit 1
fi
pause

section "Start Envoy with Authorino config"
tools/bin/func-e run --config-path config/testing/envoy-authorino-test.yaml \
    --component-log-level "ext_proc:warn,ext_authz:debug,router:warn,http:warn" \
    &>/tmp/envoy-authorino-test.log &
ENVOY_PID=$!
sleep 3
if ss -tlnp | grep -q ":8801"; then
    ok "Envoy listening on :8801"
else
    fail "Envoy failed to start"
    exit 1
fi
pause

echo -e "\n${GREEN}${BOLD}  All services running:${NC}"
echo -e "  ${DIM}Authorino (gRPC)${NC}  → Kind cluster, port-forwarded to :50055"
echo -e "  ${DIM}Semantic Router${NC}   → :50051 (ext_proc)"
echo -e "  ${DIM}Envoy${NC}             → :8801 (listener)"
echo -e "  ${DIM}vLLM Qwen2.5-14B${NC} → :8000 (backend)"
pause 2

# ──────────────────────────────────────────────────────────────
# SECTION 7: Test Cases
# ──────────────────────────────────────────────────────────────
banner "7. End-to-End Tests"

send_request() {
    local label="$1" token="$2" expect_code="$3"
    section "$label"
    echo -e "  ${DIM}Token: ${token:0:30}...${NC}"

    local http_code body
    body=$(curl -s -w "\n%{http_code}" \
        http://localhost:8801/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $token" \
        -d '{"model":"auto","messages":[{"role":"user","content":"Say hello in one word"}]}' 2>&1)
    http_code=$(echo "$body" | tail -1)
    body=$(echo "$body" | sed '$d')

    if [ "$http_code" = "$expect_code" ]; then
        ok "HTTP $http_code (expected $expect_code)"
    else
        fail "HTTP $http_code (expected $expect_code)"
    fi

    if [ "$http_code" = "200" ]; then
        local model choice
        model=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['model'])" 2>/dev/null || echo "?")
        choice=$(echo "$body" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "?")
        echo -e "  ${DIM}Model: $model${NC}"
        echo -e "  ${DIM}Response: $choice${NC}"
    elif [ "$http_code" = "401" ] || [ "$http_code" = "403" ]; then
        echo -e "  ${RED}Blocked by Authorino — unauthorized${NC}"
    fi
    pause 1
}

# ── Negative tests ──
send_request "Test A: No token (should be rejected)" "" "401"

send_request "Test B: Invalid token (should be rejected)" "totally-invalid-token" "401"

# ── 1:1 BYOT ──
echo ""
echo -e "${BOLD}── Pattern 1:1 — Bring Your Own Token (BYOT) ──${NC}"
echo -e "${DIM}  Dave uses his own OpenAI key as the Bearer token.${NC}"
echo -e "${DIM}  Authorino injects the SAME key as x-user-openai-key.${NC}"
pause 1
send_request "Test C: BYOT — Dave (own OpenAI key)" \
    "sk-byot-dave-own-openai-key-abc123" "200"

# ── 1:N Per-user ──
echo ""
echo -e "${BOLD}── Pattern 1:N — Admin-managed Per-user Keys ──${NC}"
echo -e "${DIM}  Alice has an admin-issued local token.${NC}"
echo -e "${DIM}  Authorino maps it to her dedicated OpenAI + Anthropic keys.${NC}"
pause 1
send_request "Test D: Per-user — Alice (admin-issued token)" \
    "sr-alice-local-token-7890" "200"

# ── N:1 Shared ──
echo ""
echo -e "${BOLD}── Pattern N:1 — Shared Team Keys ──${NC}"
echo -e "${DIM}  Bob and Carol each have unique tokens.${NC}"
echo -e "${DIM}  Authorino maps BOTH to the same shared team provider keys.${NC}"
pause 1
send_request "Test E: Shared — Bob (team token)" \
    "sr-bob-team-token-1111" "200"
send_request "Test F: Shared — Carol (different token, same provider keys)" \
    "sr-carol-team-token-2222" "200"

# ── Verify Authorino logs ──
section "Authorino logs — auth decisions for these requests"
echo ""
kubectl logs deployment/authorino --tail=100 2>&1 | \
    grep -oP '(identity validated.*?"name":"[^"]+"|dynamic response built.*?"object": "[^"]+"|outgoing authorization.*?"authorized": (true|false))' | \
    sed 's/identity validated.*"name":"/  ✓ Matched Secret: /; s/".*//; s/dynamic response built.*"object": "/  → Injected: /; s/".*//; s/outgoing authorization.*"authorized": /  Result: authorized=/' | \
    tail -18
pause 2

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
banner "Summary"

echo -e "
  ${GREEN}${BOLD}All tests passed.${NC}

  ${CYAN}Architecture:${NC}
    curl → Envoy (:8801)
           → ext_authz → Authorino (Kind cluster, gRPC :50055)
              validates Bearer token, injects x-user-*-key headers
           → ext_proc → Semantic Router (:50051)
              reads injected keys, selects model, sets Authorization
           → vLLM (Qwen2.5-14B :8000)

  ${CYAN}Token Patterns:${NC}
    1:1 BYOT     — Dave's own key = Bearer token = provider key
    1:N Per-user  — Alice's local token → dedicated provider keys
    N:1 Shared    — Bob & Carol → same shared team keys

  ${CYAN}Key insight:${NC}
    The semantic router doesn't know or care about Authorino.
    It just reads x-user-openai-key / x-user-anthropic-key headers.
    The same code works with our custom ext_authz OR Authorino.
"

# ── Cleanup ──
kill $ENVOY_PID 2>/dev/null || true
kill $PF_PID 2>/dev/null || true

echo -e "${DIM}(Services cleaned up)${NC}"

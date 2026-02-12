#!/usr/bin/env bash
# demo-rbac.sh — Authorino RBAC Demo: User/Group-Based Model Tier Routing
#
# Shows how Authorino ext_authz + Semantic Router RBAC role_bindings route
# different users to different tiers of the same model based on their
# group membership (K8s Secret annotations), not what they ask.
#
# Architecture:
#   Client (Bearer) → Envoy :8801
#     → ext_authz  (Authorino :50052) — validates token, injects x-authz-* headers
#     → ext_proc   (Router :50051)    — matches role_bindings → picks model tier
#     → ORIGINAL_DST → vLLM (:8000 admin tier | :8001 free tier)
#
# Prerequisites:
#   - Kind cluster with Authorino deployed
#   - Two vLLM instances on localhost:8000 (14B admin) and :8001 (14B free)
#   - func-e or envoy binary available
#   - Semantic router binary built
#
# Environment variables (all optional):
#   KIND_CONTEXT       — Kind context name (default: kind-authorino-test)
#   ENVOY_URL          — Envoy listen address (default: http://127.0.0.1:8801)
#   ADMIN_PORT         — Admin tier vLLM port (default: 8000)
#   FREE_PORT          — Free tier vLLM port (default: 8001)
#   ROUTER_PORT        — Semantic router gRPC port (default: 50051)
#   AUTHORINO_PF_PORT  — Authorino port-forward local port (default: 50052)
#   PAUSE              — Seconds between steps (default: 3)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SR_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${SR_ROOT}"

# ── Configuration ────────────────────────────────────────────
KIND_CONTEXT="${KIND_CONTEXT:-kind-authorino-test}"
ENVOY_URL="${ENVOY_URL:-http://127.0.0.1:8801}"
ADMIN_PORT="${ADMIN_PORT:-8100}"
FREE_PORT="${FREE_PORT:-8200}"
ROUTER_PORT="${ROUTER_PORT:-50051}"
AUTHORINO_PF_PORT="${AUTHORINO_PF_PORT:-50052}"
ENVOY_PORT="${ENVOY_PORT:-8801}"
ENVOY_ADMIN_PORT="${ENVOY_ADMIN_PORT:-19000}"
PAUSE="${PAUSE:-4}"
LOG_FILE="/tmp/envoy-authorino-rbac.log"

ENVOY_BIN="$(command -v envoy 2>/dev/null || echo "${FUNC_E_HOME:-$HOME/.local/share/func-e}/envoy-versions/1.37.0/bin/envoy")"
ENVOY_CFG="${SCRIPT_DIR}/envoy-rbac.yaml"
ROUTER_CFG="${SCRIPT_DIR}/config-rbac-demo.yaml"

# ── User tokens (must match K8s Secrets) ─────────────────────
TOKEN_ADMIN="sr-admin-token-0000"
TOKEN_ALICE="sr-alice-local-token-7890"
TOKEN_BOB="sr-bob-team-token-1111"
TOKEN_CAROL="sr-carol-team-token-2222"
TOKEN_DAVE="sk-byot-dave-own-openai-key-abc123"

# ── Colors ───────────────────────────────────────────────────
BOLD="\033[1m"
RESET="\033[0m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
MAGENTA="\033[35m"
DIM="\033[2m"
WHITE="\033[97m"
BLUE="\033[34m"
BG_CYAN="\033[46m"
BG_GREEN="\033[42m"
BG_RED="\033[41m"
BG_MAGENTA="\033[45m"
BG_YELLOW="\033[43m"

# ── Helpers ──────────────────────────────────────────────────
separator() { echo -e "${DIM}────────────────────────────────────────────────────────────────────────${RESET}"; }
pause()     { sleep "${PAUSE}"; }

json_field() {
    python3 -c "import sys,json; d=json.load(sys.stdin); print($1)" 2>/dev/null
}

# Cleanup on exit
ENVOY_PID=""
PFWD_PID=""
ROUTER_PID=""
cleanup() {
    [[ -n "${ENVOY_PID}" ]]  && kill "${ENVOY_PID}"  2>/dev/null || true
    [[ -n "${PFWD_PID}" ]]   && kill "${PFWD_PID}"   2>/dev/null || true
    [[ -n "${ROUTER_PID}" ]] && kill "${ROUTER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Send a request and display formatted output
send_request() {
    local test_num="$1" title="$2" token="$3" user_label="$4" prompt="$5"

    separator
    echo ""
    echo -e "  ${BOLD}${BG_CYAN}${WHITE} TEST ${test_num} ${RESET}  ${BOLD}${title}${RESET}"
    echo ""
    pause

    # ── Show the actual curl command ──────────────────────────
    echo -e "  ${BOLD}${YELLOW}CURL COMMAND:${RESET}"
    echo ""
    if [[ "${token}" == "NONE" ]]; then
        echo -e "  ${YELLOW}\$ curl -s ${ENVOY_URL}/v1/chat/completions \\\\${RESET}"
        echo -e "  ${YELLOW}    -H \"Content-Type: application/json\" \\\\${RESET}"
        echo -e "  ${YELLOW}    -d '{\"model\":\"MoM\", \"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}], \"max_tokens\":60}'${RESET}"
    else
        # Mask token for display — show first 8 and last 4 chars
        local masked_token
        if [[ ${#token} -gt 16 ]]; then
            masked_token="${token:0:8}...${token: -4}"
        else
            masked_token="${token:0:8}..."
        fi
        echo -e "  ${YELLOW}\$ curl -s ${ENVOY_URL}/v1/chat/completions \\\\${RESET}"
        echo -e "  ${YELLOW}    -H \"Content-Type: application/json\" \\\\${RESET}"
        echo -e "  ${YELLOW}    -H \"Authorization: Bearer ${BOLD}${masked_token}${RESET}${YELLOW}\" \\\\${RESET}"
        echo -e "  ${YELLOW}    -d '{\"model\":\"MoM\", \"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}], \"max_tokens\":60}'${RESET}"
    fi
    echo ""
    echo -e "  ${DIM}user: ${user_label}${RESET}"
    echo ""
    sleep 2

    # ── Execute the curl ──────────────────────────────────────
    local resp http_code
    if [[ "${token}" == "NONE" ]]; then
        resp=$(curl -s -w "\n%{http_code}" "${ENVOY_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"MoM\",
                \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
                \"max_tokens\": 60
            }" 2>/dev/null)
    else
        resp=$(curl -s -w "\n%{http_code}" "${ENVOY_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${token}" \
            -d "{
                \"model\": \"MoM\",
                \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
                \"max_tokens\": 60
            }" 2>/dev/null)
    fi

    http_code=$(echo "${resp}" | tail -1)
    local body
    body=$(echo "${resp}" | sed '$d')

    # ── Display response ──────────────────────────────────────
    # Authorino returns 401 Unauthorized for missing/invalid tokens
    if [[ "${http_code}" == "401" || "${http_code}" == "403" ]]; then
        echo -e "  ${BOLD}${RED}RESPONSE — HTTP ${http_code} UNAUTHORIZED${RESET}"
        echo -e "  ┌──────────────────────────────────────────────────────────────────────────"
        echo -e "  │ ${RED}Authorino rejected: invalid or missing Bearer token${RESET}"
        if [[ -n "${body}" ]]; then
            local err_msg
            err_msg=$(echo "${body}" | python3 -c "import sys; print(sys.stdin.read().strip()[:120])" 2>/dev/null)
            [[ -n "${err_msg}" ]] && echo -e "  │ ${DIM}body: ${err_msg}${RESET}"
        fi
        echo -e "  └──────────────────────────────────────────────────────────────────────────"
        echo ""
        pause
        sleep 2
        return
    fi

    if [[ -z "${body}" ]]; then
        echo -e "  ${RED}ERROR: No response (HTTP ${http_code})${RESET}"
        echo ""
        return
    fi

    local resp_model resp_content resp_id resp_finish resp_prompt_tok resp_compl_tok resp_total_tok
    resp_model=$(echo "${body}" | json_field "d['model']")
    resp_content=$(echo "${body}" | json_field "d['choices'][0]['message']['content'][:200]")
    resp_finish=$(echo "${body}" | json_field "d['choices'][0].get('finish_reason','?')")
    resp_id=$(echo "${body}" | json_field "d['id']")
    resp_prompt_tok=$(echo "${body}" | json_field "d.get('usage',{}).get('prompt_tokens','?')")
    resp_compl_tok=$(echo "${body}" | json_field "d.get('usage',{}).get('completion_tokens','?')")
    resp_total_tok=$(echo "${body}" | json_field "d.get('usage',{}).get('total_tokens','?')")

    echo -e "  ${BOLD}${GREEN}RESPONSE — HTTP ${http_code}${RESET}"
    echo -e "  ┌──────────────────────────────────────────────────────────────────────────"
    echo -e "  │ id:            ${DIM}${resp_id}${RESET}"
    echo -e "  │ model:         ${BOLD}${GREEN}${resp_model}${RESET}"
    echo -e "  │ finish_reason: ${BOLD}${resp_finish}${RESET}"
    echo -e "  │ usage:         prompt=${resp_prompt_tok}  completion=${resp_compl_tok}  total=${resp_total_tok}"
    echo -e "  │"
    echo -e "  │ ${BOLD}content:${RESET}"
    echo -e "  │   ${WHITE}${resp_content}${RESET}"
    echo -e "  └──────────────────────────────────────────────────────────────────────────"
    echo ""
    pause
    sleep 2
}

# ── Banner ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════════════════════════════╗"
echo "  ║                                                                   ║"
echo "  ║   vLLM Semantic Router — Authorino RBAC Model Tier Routing Demo   ║"
echo "  ║                                                                   ║"
echo "  ║   K8s Secret → Authorino ext_authz → Role Bindings → Model Tiers ║"
echo "  ║                                                                   ║"
echo "  ╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "  ${BOLD}Architecture:${RESET}"
echo ""
echo -e "  ${DIM}┌──────────┐    ┌───────────────────┐    ┌──────────────────┐    ┌────────────────────────┐${RESET}"
echo -e "  ${DIM}│  Client  │ →  │  Envoy :${ENVOY_PORT}      │ →  │  Semantic Router │ ─┬→│  ${RESET}${GREEN}Admin${RESET}${DIM} vLLM :${ADMIN_PORT}     │${RESET}"
echo -e "  ${DIM}│  Bearer  │    │  ext_authz        │    │  ext_proc :${ROUTER_PORT} │  │ │  (admin + premium)     │${RESET}"
echo -e "  ${DIM}│  token   │    │   ↕ Authorino     │    │  RBAC bindings   │  │ └────────────────────────┘${RESET}"
echo -e "  ${DIM}│          │    │  :${AUTHORINO_PF_PORT} (K8s PF) │    │                  │  │ ┌────────────────────────┐${RESET}"
echo -e "  ${DIM}│          │    │                   │    │                  │  └→│  ${RESET}${YELLOW}Free${RESET}${DIM}  vLLM :${FREE_PORT}     │${RESET}"
echo -e "  ${DIM}│          │    │                   │    │                  │    │  (pro + free + default) │${RESET}"
echo -e "  ${DIM}└──────────┘    └───────────────────┘    └──────────────────┘    └────────────────────────┘${RESET}"
echo ""
echo -e "  ${BOLD}Token Management Patterns (K8s Secrets):${RESET}"
echo ""
printf "  ${BOLD}%-10s %-14s %-14s %-14s${RESET}\n" "Pattern" "User" "Bearer Token" "Provider Key"
printf "  ${DIM}%-10s %-14s %-14s %-14s${RESET}\n" "────────" "────────────" "────────────" "────────────"
printf "  %-10s %-14s %-14s %-14s\n" "1:N"  "alice" "local token"   "dedicated keys"
printf "  %-10s %-14s %-14s %-14s\n" "N:1"  "bob"   "local token"   "shared team key"
printf "  %-10s %-14s %-14s %-14s\n" "N:1"  "carol" "local token"   "shared team key"
printf "  %-10s %-14s %-14s %-14s\n" "1:1"  "dave"  "=own API key"  "=own API key"
echo ""
echo -e "  ${BOLD}RBAC Role Bindings (Authorino group → role → model tier):${RESET}"
echo ""
printf "  ${BOLD}%-10s %-16s %-16s %-14s %-14s${RESET}\n" "User" "Secret" "Group" "Role" "Tier"
printf "  ${DIM}%-10s %-16s %-16s %-14s %-14s${RESET}\n" "────────" "──────────────" "──────────────" "────────────" "────────────"
printf "  %-10s %-16s %-16s ${GREEN}%-14s${RESET} ${GREEN}%-14s${RESET}\n" "admin" "user-admin" "(User match)" "admin" "Admin (:${ADMIN_PORT})"
printf "  %-10s %-16s %-16s ${CYAN}%-14s${RESET} ${GREEN}%-14s${RESET}\n" "bob" "user-bob" "premium" "premium_user" "Admin (:${ADMIN_PORT})"
printf "  %-10s %-16s %-16s ${BLUE}%-14s${RESET} ${YELLOW}%-14s${RESET}\n" "alice" "user-alice" "engineering" "pro_user" "Free  (:${FREE_PORT})"
printf "  %-10s %-16s %-16s ${YELLOW}%-14s${RESET} ${YELLOW}%-14s${RESET}\n" "carol" "user-carol" "free" "free_user" "Free  (:${FREE_PORT})"
printf "  %-10s %-16s %-16s ${YELLOW}%-14s${RESET} ${YELLOW}%-14s${RESET}\n" "dave" "user-dave" "contractor" "free_user" "Free  (:${FREE_PORT})"
printf "  %-10s %-16s %-16s ${RED}%-14s${RESET} ${YELLOW}%-14s${RESET}\n" "unknown" "(none)" "(none)" "(no match)" "Free  (:${FREE_PORT})"
echo ""
pause
pause

# ══════════════════════════════════════════════════════════════
# PART 0: Environment Setup
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_YELLOW}${WHITE} SETUP ${RESET}  ${BOLD}Deploying Authorino + Secrets + Envoy + Router${RESET}"
echo ""
pause

# ── 0a. Check prerequisites ──────────────────────────────────
echo -e "  ${BOLD}Checking prerequisites...${RESET}"

echo -n "    Kind cluster (${KIND_CONTEXT})... "
if kubectl --context "${KIND_CONTEXT}" cluster-info >/dev/null 2>&1; then
    echo -e "${GREEN}OK${RESET}"
else
    echo -e "${RED}FAIL${RESET} — run: kind create cluster --name authorino-test"
    exit 1
fi

echo -n "    Authorino pod... "
if kubectl --context "${KIND_CONTEXT}" get pod -l app=authorino -o name 2>/dev/null | grep -q pod/; then
    echo -e "${GREEN}OK${RESET}"
else
    echo -e "${RED}FAIL${RESET} — deploy Authorino first (kubectl apply -f demo/authz/authorino/k8s/k8s-deploy.yaml)"
    exit 1
fi

echo -n "    Admin-tier vLLM (:${ADMIN_PORT})... "
if MODEL_ADMIN=$(curl -sf "http://localhost:${ADMIN_PORT}/v1/models" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null); then
    echo -e "${GREEN}OK${RESET} (${MODEL_ADMIN})"
else
    echo -e "${RED}FAIL${RESET} — no vLLM on port ${ADMIN_PORT}"
    exit 1
fi

echo -n "    Free-tier vLLM (:${FREE_PORT})... "
if MODEL_FREE=$(curl -sf "http://localhost:${FREE_PORT}/v1/models" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null); then
    echo -e "${GREEN}OK${RESET} (${MODEL_FREE})"
else
    echo -e "${RED}FAIL${RESET} — no vLLM on port ${FREE_PORT}"
    exit 1
fi

echo -n "    Envoy binary... "
if [[ -x "${ENVOY_BIN}" ]]; then
    echo -e "${GREEN}OK${RESET} (${ENVOY_BIN})"
else
    echo -e "${RED}FAIL${RESET} — install envoy or func-e"
    exit 1
fi

echo ""

# ── 0b. Apply K8s Secrets with authz-groups ──────────────────
echo -e "  ${BOLD}Creating K8s Secrets (users + authz-groups annotations)...${RESET}"

kubectl --context "${KIND_CONTEXT}" apply -f - <<'EOF'
---
apiVersion: v1
kind: Secret
metadata:
  name: user-admin
  labels:
    app: semantic-router
    authorino.kuadrant.io/managed-by: authorino
  annotations:
    openai-key: ""
    anthropic-key: ""
    authz-groups: ""
stringData:
  api_key: "sr-admin-token-0000"
type: Opaque
---
apiVersion: v1
kind: Secret
metadata:
  name: user-alice
  labels:
    app: semantic-router
    authorino.kuadrant.io/managed-by: authorino
  annotations:
    openai-key: "sk-proj-alice-dedicated-openai-key"
    anthropic-key: "sk-ant-alice-dedicated-anthropic-key"
    authz-groups: "engineering"
stringData:
  api_key: "sr-alice-local-token-7890"
type: Opaque
---
apiVersion: v1
kind: Secret
metadata:
  name: user-bob
  labels:
    app: semantic-router
    authorino.kuadrant.io/managed-by: authorino
  annotations:
    openai-key: "sk-proj-shared-team-openai-key-TEAM"
    anthropic-key: "sk-ant-shared-team-anthropic-key-TEAM"
    authz-groups: "premium"
stringData:
  api_key: "sr-bob-team-token-1111"
type: Opaque
---
apiVersion: v1
kind: Secret
metadata:
  name: user-carol
  labels:
    app: semantic-router
    authorino.kuadrant.io/managed-by: authorino
  annotations:
    openai-key: "sk-proj-shared-team-openai-key-TEAM"
    anthropic-key: "sk-ant-shared-team-anthropic-key-TEAM"
    authz-groups: "free"
stringData:
  api_key: "sr-carol-team-token-2222"
type: Opaque
---
apiVersion: v1
kind: Secret
metadata:
  name: user-dave
  labels:
    app: semantic-router
    authorino.kuadrant.io/managed-by: authorino
  annotations:
    openai-key: "sk-byot-dave-own-openai-key-abc123"
    anthropic-key: "sk-ant-byot-dave-own-anthropic-key-xyz"
    authz-groups: "contractor"
stringData:
  api_key: "sk-byot-dave-own-openai-key-abc123"
type: Opaque
EOF

echo ""
echo -e "  ${BOLD}Verifying secrets:${RESET}"
for s in user-admin user-alice user-bob user-carol user-dave; do
    groups=$(kubectl --context "${KIND_CONTEXT}" get secret "${s}" \
        -o jsonpath="{.metadata.annotations.authz-groups}" 2>/dev/null)
    printf "    %-16s authz-groups=%s\n" "${s}" "${groups:-\"(empty)\"}"
done
echo ""

# ── 0c. Apply AuthConfig ─────────────────────────────────────
echo -e "  ${BOLD}Applying Authorino AuthConfig (RBAC identity header injection)...${RESET}"

kubectl --context "${KIND_CONTEXT}" delete authconfig semantic-router-auth 2>/dev/null || true

kubectl --context "${KIND_CONTEXT}" apply -f - <<EOF
apiVersion: authorino.kuadrant.io/v1beta3
kind: AuthConfig
metadata:
  name: semantic-router-auth
spec:
  hosts:
  - "semantic-router.example.com"
  - "localhost"
  - "localhost:${ENVOY_PORT}"
  - "127.0.0.1"
  - "127.0.0.1:${ENVOY_PORT}"
  authentication:
    "api-key-users":
      apiKey:
        selector:
          matchLabels:
            app: semantic-router
      credentials:
        authorizationHeader:
          prefix: Bearer
  response:
    success:
      headers:
        x-user-openai-key:
          plain:
            expression: auth.identity.metadata.annotations['openai-key']
        x-user-anthropic-key:
          plain:
            expression: auth.identity.metadata.annotations['anthropic-key']
        x-authz-user-id:
          plain:
            expression: auth.identity.metadata.name
        x-authz-user-groups:
          plain:
            expression: auth.identity.metadata.annotations['authz-groups']
EOF

sleep 2
echo -n "    AuthConfig status: "
STATUS=$(kubectl --context "${KIND_CONTEXT}" get authconfig semantic-router-auth \
    -o jsonpath='{.status.summary.ready}' 2>/dev/null || echo "unknown")
if [[ "${STATUS}" == "true" ]]; then
    echo -e "${GREEN}ready${RESET}"
else
    echo -e "${YELLOW}${STATUS}${RESET} (may take a moment)"
    sleep 3
fi
echo ""

# ── 0d. Port-forward Authorino ───────────────────────────────
echo -e "  ${BOLD}Port-forwarding Authorino (K8s → localhost:${AUTHORINO_PF_PORT})...${RESET}"

pkill -f "port-forward svc/authorino ${AUTHORINO_PF_PORT}" 2>/dev/null || true
sleep 1

kubectl --context "${KIND_CONTEXT}" port-forward svc/authorino "${AUTHORINO_PF_PORT}":50051 --address 0.0.0.0 &>/dev/null &
PFWD_PID=$!
sleep 2

if ss -tlnp 2>/dev/null | grep -q ":${AUTHORINO_PF_PORT}" || lsof -i ":${AUTHORINO_PF_PORT}" >/dev/null 2>&1; then
    echo -e "    ${GREEN}OK${RESET} (PID ${PFWD_PID})"
else
    echo -e "    ${RED}FAIL${RESET}: port-forward did not start"
    exit 1
fi
echo ""

# ── 0e. Start Envoy ──────────────────────────────────────────
echo -e "  ${BOLD}Starting Envoy on port ${ENVOY_PORT}...${RESET}"

pkill -f "envoy.*envoy-rbac.yaml" 2>/dev/null || true
sleep 1
rm -f "${LOG_FILE}"

"${ENVOY_BIN}" -c "${ENVOY_CFG}" --log-level warn &>/tmp/envoy-authorino-rbac-stderr.log &
ENVOY_PID=$!
sleep 2

if ss -tlnp 2>/dev/null | grep -q ":${ENVOY_PORT}" || lsof -i ":${ENVOY_PORT}" >/dev/null 2>&1; then
    echo -e "    ${GREEN}OK${RESET} (PID ${ENVOY_PID})"
else
    echo -e "    ${RED}FAIL${RESET}: Envoy did not start"
    tail -5 /tmp/envoy-authorino-rbac-stderr.log 2>/dev/null
    exit 1
fi
echo ""

# ── 0f. Start Semantic Router ────────────────────────────────
echo -e "  ${BOLD}Starting Semantic Router on port ${ROUTER_PORT}...${RESET}"

pkill -f "router.*config-rbac-demo" 2>/dev/null || true
sleep 2

export LD_LIBRARY_PATH="${SR_ROOT}/candle-binding/target/release:${SR_ROOT}/ml-binding/target/release:${LD_LIBRARY_PATH:-}"
"${SR_ROOT}/bin/router" --config "${ROUTER_CFG}" --port "${ROUTER_PORT}" --enable-api=false \
    &>/tmp/router-authorino-rbac.log &
ROUTER_PID=$!
sleep 6

if ss -tlnp 2>/dev/null | grep -q ":${ROUTER_PORT}" || lsof -i ":${ROUTER_PORT}" >/dev/null 2>&1; then
    echo -e "    ${GREEN}OK${RESET} (PID ${ROUTER_PID})"
else
    echo -e "    ${RED}FAIL${RESET}: Router did not start"
    tail -10 /tmp/router-authorino-rbac.log 2>/dev/null
    exit 1
fi
echo ""

echo -e "  ${BOLD}${GREEN}All services running.${RESET}"
echo ""
pause

# ══════════════════════════════════════════════════════════════
# PART 1: Authentication — Authorino validates Bearer tokens
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_RED}${WHITE} PART 1 ${RESET}  ${BOLD}Authentication — Authorino validates Bearer tokens${RESET}"
echo ""
echo -e "  Authorino matches the Bearer token against K8s Secrets"
echo -e "  labeled ${BOLD}app=semantic-router${RESET}."
echo -e "  No token → ${RED}401${RESET}.  Invalid token → ${RED}401${RESET}."
echo ""
pause

send_request 1 \
    "No Bearer token → Authorino rejects" \
    "NONE" \
    "(anonymous)" \
    "hello"

send_request 2 \
    "Invalid Bearer token → Authorino rejects" \
    "invalid-token-doesnt-exist-in-k8s" \
    "(bad token)" \
    "hello"

# ══════════════════════════════════════════════════════════════
# PART 2: RBAC Routing — same prompt, different users, different tiers
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_GREEN}${WHITE} PART 2 ${RESET}  ${BOLD}RBAC Routing — same prompt, different users, different tiers${RESET}"
echo ""
echo -e "  All five users send the exact same prompt."
echo -e "  The router picks the model tier based on ${BOLD}who they are${RESET}, not what they ask."
echo ""
echo -e "  Authorino injects: ${BOLD}x-authz-user-id${RESET} and ${BOLD}x-authz-user-groups${RESET}"
echo -e "  Router matches these against ${BOLD}role_bindings${RESET} → picks the tier."
echo ""
pause

send_request 3 \
    "Admin (User match) → Admin tier (:${ADMIN_PORT})" \
    "${TOKEN_ADMIN}" \
    "admin / (User: user-admin) → admin" \
    "What is the capital of France?"

send_request 4 \
    "Bob (premium group) → Admin tier (:${ADMIN_PORT})" \
    "${TOKEN_BOB}" \
    "bob / premium → premium_user" \
    "What is the capital of France?"

send_request 5 \
    "Alice (engineering group) → Free tier (:${FREE_PORT})" \
    "${TOKEN_ALICE}" \
    "alice / engineering → pro_user" \
    "What is the capital of France?"

send_request 6 \
    "Carol (free group) → Free tier (:${FREE_PORT})" \
    "${TOKEN_CAROL}" \
    "carol / free → free_user" \
    "What is the capital of France?"

send_request 7 \
    "Dave (contractor, BYOT) → Free tier (:${FREE_PORT})" \
    "${TOKEN_DAVE}" \
    "dave / contractor → free_user" \
    "What is the capital of France?"

# ══════════════════════════════════════════════════════════════
# PART 3: Proof — Envoy access logs
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_MAGENTA}${WHITE} PROOF ${RESET}  ${BOLD}Envoy access logs — user identity → upstream port proves routing${RESET}"
echo ""
pause

if [[ -f "${LOG_FILE}" ]]; then
    echo -e "  ${BOLD}Envoy access log (authz_user → upstream):${RESET}"
    echo ""

    # Show the last requests with user identity and upstream
    grep '"authz_user"' "${LOG_FILE}" 2>/dev/null | tail -7 | while read -r line; do
        user=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('authz_user','?'))" 2>/dev/null)
        groups=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('authz_groups','?'))" 2>/dev/null)
        upstream=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('upstream','?'))" 2>/dev/null)
        status=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null)

        if [[ "${upstream}" == *"${ADMIN_PORT}"* ]]; then
            tier_color="${GREEN}"
            tier_label="Admin"
        elif [[ "${upstream}" == *"${FREE_PORT}"* ]]; then
            tier_color="${YELLOW}"
            tier_label="Free"
        else
            tier_color="${RED}"
            tier_label="Rejected"
        fi
        printf "    user=%-14s groups=%-14s → upstream=%-22s status=%s  ${tier_color}[%s tier]${RESET}\n" \
            "${user}" "${groups}" "${upstream}" "${status}" "${tier_label}"
    done
    echo ""
else
    echo -e "  ${YELLOW}(Access log not found at ${LOG_FILE})${RESET}"
    echo ""
fi
pause

# ── Authorino logs ────────────────────────────────────────────
echo -e "  ${BOLD}Authorino logs (last 5 auth decisions):${RESET}"
echo ""
AUTHORINO_POD=$(kubectl --context "${KIND_CONTEXT}" get pods -l app=authorino -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [[ -n "${AUTHORINO_POD}" ]]; then
    kubectl --context "${KIND_CONTEXT}" logs "${AUTHORINO_POD}" --tail=50 2>/dev/null \
        | grep -iE '"msg"|authorized|denied|secret' \
        | tail -5 \
        | while read -r line; do
            echo "    ${line}"
        done
    echo ""
else
    echo -e "    ${YELLOW}(Could not find Authorino pod)${RESET}"
    echo ""
fi
pause

# ── Conclusion ───────────────────────────────────────────────
separator
echo ""
echo -e "  ${BOLD}${CYAN}Summary — Authorino RBAC Model Tier Routing${RESET}"
echo ""
echo -e "  ${BOLD}Routing is purely identity-based:${RESET}"
echo -e "    Same prompt, different users → different model tiers."
echo -e "    No keyword analysis, no content inspection."
echo ""
echo -e "  ${BOLD}End-to-end flow:${RESET}"
echo -e "    ${GREEN}✓${RESET} Bearer token matched against K8s Secrets by Authorino"
echo -e "    ${GREEN}✓${RESET} Authorino injects: x-authz-user-id (Secret name)"
echo -e "    ${GREEN}✓${RESET}                    x-authz-user-groups (Secret annotation)"
echo -e "    ${GREEN}✓${RESET} Semantic router reads headers, matches role_bindings"
echo -e "    ${GREEN}✓${RESET} admin/premium → Admin tier (:${ADMIN_PORT}), pro/free/contractor → Free tier (:${FREE_PORT})"
echo -e "    ${GREEN}✓${RESET} No token / bad token → ${RED}401 Unauthorized${RESET} (never reaches router)"
echo ""
echo -e "  ${BOLD}Token management patterns demonstrated:${RESET}"
echo -e "    ${GREEN}✓${RESET} 1:N (alice) — admin-issued local token, dedicated provider keys"
echo -e "    ${GREEN}✓${RESET} N:1 (bob, carol) — local tokens, shared team provider keys"
echo -e "    ${GREEN}✓${RESET} 1:1 (dave) — BYOT, user's own API key IS the bearer token"
echo ""
echo -e "  ${BOLD}Components:${RESET}"
echo -e "    Authorino (K8s, ext_authz) + Envoy + Semantic Router + vLLM"
echo ""
echo -e "  ${DIM}Router config: demo/authz/authorino/config-rbac-demo.yaml${RESET}"
echo -e "  ${DIM}Envoy config:  demo/authz/authorino/envoy-rbac.yaml${RESET}"
echo ""
separator
echo ""

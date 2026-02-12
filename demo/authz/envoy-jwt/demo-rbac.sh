#!/usr/bin/env bash
# demo-authz-rbac.sh — Asciinema-friendly demo of RBAC user/group model routing
#
# Shows JWT-authenticated users routed to different model tiers based on
# their group membership (RBAC role bindings), not what they ask.

set -euo pipefail

SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SR_ROOT}"

# ── Configuration ────────────────────────────────────────────
ENVOY_URL="${ENVOY_URL:-http://127.0.0.1:8802}"
ADMIN_PORT=8100
FREE_PORT=8200
PAUSE="${PAUSE:-3}"

# Load JWT tokens
source scripts/authz/envoy-jwt/jwt-artifacts/tokens.env

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

# Send a request and display formatted output
send_request() {
    local test_num="$1" title="$2" jwt_var="$3" user_label="$4" prompt="$5"

    separator
    echo ""
    echo -e "  ${BOLD}${BG_CYAN}${WHITE} TEST ${test_num} ${RESET}  ${BOLD}${title}${RESET}"
    echo ""
    pause

    echo -e "  ${BOLD}${YELLOW}REQUEST${RESET}"
    echo -e "  ┌─────────────────────────────────────────────────────────────"
    echo -e "  │ user:   ${BOLD}${YELLOW}${user_label}${RESET}"
    echo -e "  │ model:  ${BOLD}MoM${RESET} (router picks based on identity)"
    echo -e "  │ prompt: \"${prompt}\""
    echo -e "  └─────────────────────────────────────────────────────────────"
    echo ""
    sleep 1

    local resp http_code
    if [[ "${jwt_var}" == "NONE" ]]; then
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
            -H "Authorization: Bearer ${jwt_var}" \
            -d "{
                \"model\": \"MoM\",
                \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
                \"max_tokens\": 60
            }" 2>/dev/null)
    fi

    http_code=$(echo "${resp}" | tail -1)
    local body
    body=$(echo "${resp}" | sed '$d')

    if [[ "${http_code}" == "401" ]]; then
        echo -e "  ${BOLD}${RED}RESPONSE — HTTP 401 UNAUTHORIZED${RESET}"
        echo -e "  ┌─────────────────────────────────────────────────────────────"
        echo -e "  │ Envoy rejected: invalid or missing JWT"
        echo -e "  └─────────────────────────────────────────────────────────────"
        echo ""
        pause
        return
    fi

    if [[ -z "${body}" ]]; then
        echo -e "  ${RED}ERROR: No response (HTTP ${http_code})${RESET}"
        echo ""
        return
    fi

    local resp_model resp_content resp_id
    resp_model=$(echo "${body}" | json_field "d['model']")
    resp_content=$(echo "${body}" | json_field "d['choices'][0]['message']['content'][:160]")
    resp_id=$(echo "${body}" | json_field "d['id']")

    echo -e "  ${BOLD}${GREEN}RESPONSE — HTTP ${http_code}${RESET}"
    echo -e "  ┌─────────────────────────────────────────────────────────────"
    echo -e "  │ id:    ${DIM}${resp_id}${RESET}"
    echo -e "  │ model: ${BOLD}${GREEN}${resp_model}${RESET}"
    echo -e "  │"
    echo -e "  │ ${WHITE}${resp_content}${RESET}"
    echo -e "  └─────────────────────────────────────────────────────────────"
    echo ""
    pause
}

# ── Banner ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════════════════════════════╗"
echo "  ║                                                                   ║"
echo "  ║   vLLM Semantic Router — RBAC User/Group Model Routing Demo       ║"
echo "  ║                                                                   ║"
echo "  ║   JWT Authentication → Role Bindings → Per-User Model Tiers       ║"
echo "  ║                                                                   ║"
echo "  ╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "  ${BOLD}Architecture:${RESET}"
echo ""
echo -e "  ${DIM}┌──────────┐    ┌───────────────────┐    ┌──────────────────┐    ┌──────────────────────┐${RESET}"
echo -e "  ${DIM}│  Client  │ →  │  Envoy :8802      │ →  │  Semantic Router │ ─┬→│  ${RESET}${GREEN}Admin${RESET}${DIM} vLLM :${ADMIN_PORT}  │${RESET}"
echo -e "  ${DIM}│  + JWT   │    │  jwt_authn        │    │  ext_proc :50053 │  │ │  (admin + premium)   │${RESET}"
echo -e "  ${DIM}│          │    │  claim→headers    │    │  RBAC bindings   │  │ └──────────────────────┘${RESET}"
echo -e "  ${DIM}│          │    │                   │    │                  │  │ ┌──────────────────────┐${RESET}"
echo -e "  ${DIM}│          │    │                   │    │                  │  └→│  ${RESET}${YELLOW}Free${RESET}${DIM}  vLLM :${FREE_PORT}  │${RESET}"
echo -e "  ${DIM}│          │    │                   │    │                  │    │  (free + default)    │${RESET}"
echo -e "  ${DIM}└──────────┘    └───────────────────┘    └──────────────────┘    └──────────────────────┘${RESET}"
echo ""
echo -e "  ${BOLD}RBAC Role Bindings (JWT group → role → model tier):${RESET}"
echo ""
printf "  ${BOLD}%-18s %-22s %-16s %-14s${RESET}\n" "User" "JWT Group" "Role" "Tier"
printf "  ${DIM}%-18s %-22s %-16s %-14s${RESET}\n" "────────────────" "────────────────────" "──────────────" "────────────"
printf "  %-18s %-22s ${GREEN}%-16s${RESET} ${GREEN}%-14s${RESET}\n" "alice"   "platform-admins"  "admin"        "Admin (:${ADMIN_PORT})"
printf "  %-18s %-22s ${CYAN}%-16s${RESET} ${GREEN}%-14s${RESET}\n" "bob"     "premium-tier"     "premium_user" "Admin (:${ADMIN_PORT})"
printf "  %-18s %-22s ${YELLOW}%-16s${RESET} ${YELLOW}%-14s${RESET}\n" "carol"   "free-tier"        "free_user"    "Free  (:${FREE_PORT})"
printf "  %-18s %-22s ${RED}%-16s${RESET} ${YELLOW}%-14s${RESET}\n" "unknown" "(none)"           "(no match)"   "Free  (:${FREE_PORT})"
echo ""
pause
pause

# ══════════════════════════════════════════════════════════════
# PART 1: JWT Authentication
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_RED}${WHITE} PART 1 ${RESET}  ${BOLD}JWT Authentication — Envoy validates tokens${RESET}"
echo ""
echo -e "  Envoy's jwt_authn filter checks RSA256 signature, issuer, audience."
echo -e "  No JWT → ${RED}401${RESET}.  Expired JWT → ${RED}401${RESET}."
echo ""
pause

send_request 1 \
    "No JWT → Envoy rejects (401)" \
    "NONE" \
    "(anonymous)" \
    "hello"

send_request 2 \
    "Expired JWT → Envoy rejects (401)" \
    "${JWT_EXPIRED}" \
    "expired (premium-tier, but token expired)" \
    "hello"

# ══════════════════════════════════════════════════════════════
# PART 2: RBAC Routing
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_GREEN}${WHITE} PART 2 ${RESET}  ${BOLD}RBAC Routing — same prompt, different users, different tiers${RESET}"
echo ""
echo -e "  All four users send the exact same prompt."
echo -e "  The router picks the model tier based on ${BOLD}who they are${RESET}, not what they ask."
echo ""
pause

send_request 3 \
    "Alice (admin) → Admin tier (:${ADMIN_PORT})" \
    "${JWT_ALICE}" \
    "alice / platform-admins → admin" \
    "What is the capital of France?"

send_request 4 \
    "Bob (premium) → Admin tier (:${ADMIN_PORT})" \
    "${JWT_BOB}" \
    "bob / premium-tier → premium_user" \
    "What is the capital of France?"

send_request 5 \
    "Carol (free) → Free tier (:${FREE_PORT})" \
    "${JWT_CAROL}" \
    "carol / free-tier → free_user" \
    "What is the capital of France?"

send_request 6 \
    "Unknown (no group) → Free tier default (:${FREE_PORT})" \
    "${JWT_UNKNOWN}" \
    "unknown / (no groups) → default" \
    "What is the capital of France?"

# ══════════════════════════════════════════════════════════════
# PART 3: Proof — Envoy access logs
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_MAGENTA}${WHITE} PROOF ${RESET}  ${BOLD}Envoy access logs — upstream port proves routing${RESET}"
echo ""
pause

echo -e "  ${BOLD}Envoy access log (jwt_sub → upstream):${RESET}"
echo ""
grep '"jwt_sub"' /tmp/envoy-authz-rbac.log | tail -4 | while read -r line; do
    sub=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('jwt_sub','?'))" 2>/dev/null)
    upstream=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('upstream','?'))" 2>/dev/null)
    status=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null)

    if [[ "${upstream}" == *"${ADMIN_PORT}"* ]]; then
        tier_color="${GREEN}"
        tier_label="Admin"
    else
        tier_color="${YELLOW}"
        tier_label="Free"
    fi
    printf "    jwt_sub=%-10s → upstream=%-22s status=%s  ${tier_color}[%s tier]${RESET}\n" \
        "${sub}" "${upstream}" "${status}" "${tier_label}"
done
echo ""
pause

# ── Conclusion ───────────────────────────────────────────────
separator
echo ""
echo -e "  ${BOLD}${CYAN}Summary — RBAC User/Group Model Routing${RESET}"
echo ""
echo -e "  ${BOLD}Routing is purely identity-based:${RESET}"
echo -e "    Same prompt, different users → different model tiers."
echo -e "    No keyword analysis, no content inspection."
echo ""
echo -e "  ${BOLD}End-to-end flow:${RESET}"
echo -e "    ${GREEN}✓${RESET} JWT signed with RSA256, validated by Envoy's jwt_authn filter"
echo -e "    ${GREEN}✓${RESET} Claims extracted to headers: sub → x-jwt-sub, groups → x-jwt-groups"
echo -e "    ${GREEN}✓${RESET} Semantic router reads headers, matches RBAC role_bindings"
echo -e "    ${GREEN}✓${RESET} admin/premium → Admin tier (:${ADMIN_PORT}), free/unknown → Free tier (:${FREE_PORT})"
echo -e "    ${GREEN}✓${RESET} No JWT or expired JWT → ${RED}401 Unauthorized${RESET} (never reaches router)"
echo ""
echo -e "  ${BOLD}No Kubernetes. No Authorino. Just:${RESET}"
echo -e "    Envoy (jwt_authn + ext_proc) + Semantic Router + vLLM"
echo ""
echo -e "  ${DIM}Config: config/testing/config.authz-rbac-demo.yaml${RESET}"
echo ""
separator
echo ""

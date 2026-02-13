#!/usr/bin/env bash
# demo-multi-provider.sh — Multi-Cloud Provider Routing Demo
#
# Demonstrates routing the same model to different cloud providers (OpenAI,
# Azure OpenAI, Anthropic) with per-user API keys injected by Authorino.
#
# Since these are FAKE tokens, we expect provider-specific error messages
# (e.g., "invalid api key" from OpenAI, "invalid x-api-key" from Anthropic).
# These errors PROVE the request reached the correct provider with the
# correct auth format — which is the whole point of the demo.
#
# Prerequisites:
#   1. Kind cluster with Authorino running
#      - K8s Secrets applied: kubectl apply -f k8s-secrets.yaml
#      - AuthConfig applied:  kubectl apply -f k8s-authconfig.yaml
#   2. Authorino port-forwarded:
#      kubectl port-forward -n default svc/authorino-authorino-authorization 50052:50051
#   3. Semantic Router running:
#      go run ./cmd/router -config config/testing/config.multi-provider.yaml
#   4. Envoy running:
#      envoy -c scripts/demo-multi-provider/envoy.yaml
#   5. (Optional) Local vLLM on port 8000 for code queries

set -euo pipefail

SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SR_ROOT}"

# ── Configuration ────────────────────────────────────────────
ENVOY_URL="${ENVOY_URL:-http://127.0.0.1:8801}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:9190/metrics}"
KIND_CONTEXT="${KIND_CONTEXT:-kind-authorino-test}"
PAUSE="${PAUSE:-3}"

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
BG_BLUE="\033[44m"

# ── Helpers ──────────────────────────────────────────────────
separator() { echo -e "${DIM}────────────────────────────────────────────────────────────────────────${RESET}"; }
pause()     { sleep "${PAUSE}"; }

# Wait for viewer to read the output before continuing.
# In interactive mode: "Press Enter to continue..."
# In asciinema/non-interactive mode: auto-continues after WAIT seconds.
WAIT="${WAIT:-5}"
AUTOPLAY="${AUTOPLAY:-0}"
wait_for_key() {
    if [[ "${AUTOPLAY}" == "1" ]]; then
        sleep "${WAIT}"
    elif [[ -t 0 ]]; then
        echo -e "  ${DIM}Press Enter to continue...${RESET}"
        read -r -s
    else
        sleep "${WAIT}"
    fi
}

json_field() {
    python3 -c "import sys,json; d=json.load(sys.stdin); print($1)" 2>/dev/null
}

# Pretty-print JSON body (compact, max 500 chars)
json_pretty() {
    python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(json.dumps(d, indent=2)[:500])
except:
    sys.stdin.seek(0)
    print(sys.stdin.read()[:500])
" 2>/dev/null
}

get_token() {
    local secret_name="$1"
    kubectl --context "$KIND_CONTEXT" get secret "$secret_name" \
        -o jsonpath='{.data.api_key}' 2>/dev/null | base64 -d 2>/dev/null
}

# Send a request, show full curl, show response with key info & body
send_request() {
    local test_num="$1" title="$2" token="$3" model="$4" prompt="$5"

    separator
    echo ""
    echo -e "  ${BOLD}${BG_CYAN}${WHITE} TEST ${test_num} ${RESET}  ${BOLD}${title}${RESET}"
    echo ""
    pause

    # Show the curl command being sent
    echo -e "  ${BOLD}${YELLOW}REQUEST${RESET}"
    echo -e "  ${DIM}curl -s ${ENVOY_URL}/v1/chat/completions \\${RESET}"
    echo -e "  ${DIM}  -H \"Authorization: Bearer ${token:0:20}...\" \\${RESET}"
    echo -e "  ${DIM}  -H \"Content-Type: application/json\" \\${RESET}"
    echo -e "  ${DIM}  -d '{\"model\": \"${model}\", \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}], \"max_tokens\": 60}'${RESET}"
    echo ""
    sleep 1

    local http_code tmpfile hdrfile
    tmpfile=$(mktemp)
    hdrfile=$(mktemp)

    # Capture both response headers (-D) and body (-o)
    http_code=$(curl -s -o "$tmpfile" -D "$hdrfile" -w '%{http_code}' \
        --max-time 120 \
        "${ENVOY_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${token}" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
            \"max_tokens\": 60
        }" 2>/dev/null) || http_code="000"

    # ── Show response header info ──
    local upstream_cluster upstream_host content_type
    upstream_cluster=$(grep -i "x-envoy-upstream-cluster:" "$hdrfile" 2>/dev/null | tr -d '\r' | awk '{print $2}') || upstream_cluster=""
    upstream_host=$(grep -i "x-envoy-upstream-service-time:" "$hdrfile" 2>/dev/null | tr -d '\r' | awk '{print $2}') || upstream_host=""
    content_type=$(grep -i "content-type:" "$hdrfile" 2>/dev/null | tr -d '\r' | sed 's/^[^:]*: //') || content_type=""

    echo -e "  ${BOLD}RESPONSE — HTTP ${http_code}${RESET}"
    echo -e "  ┌─────────────────────────────────────────────────────────────"
    echo -e "  │ ${BOLD}http_status:${RESET}     ${http_code}"
    [[ -n "${upstream_cluster}" ]] && echo -e "  │ ${BOLD}envoy_cluster:${RESET}   ${upstream_cluster}"
    [[ -n "${content_type}" ]] && echo -e "  │ ${BOLD}content-type:${RESET}    ${content_type}"

    # Check if body contains an error (router may return 200 with error body)
    local has_error
    has_error=$(python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if 'error' in d else 'no')" < "$tmpfile" 2>/dev/null) || has_error="no"

    if [[ "${http_code}" == "200" && "${has_error}" == "no" ]]; then
        # Success — show parsed fields
        local resp_model resp_content finish_reason resp_id
        resp_model=$(json_field "d['model']" < "$tmpfile") || resp_model="?"
        resp_content=$(json_field "d['choices'][0]['message']['content'][:200]" < "$tmpfile") || resp_content="?"
        finish_reason=$(json_field "d['choices'][0].get('finish_reason','?')" < "$tmpfile") || finish_reason="?"
        resp_id=$(json_field "d['id']" < "$tmpfile") || resp_id="?"

        echo -e "  │ ${BOLD}id:${RESET}              ${resp_id}"
        echo -e "  │ ${BOLD}model:${RESET}           ${GREEN}${resp_model}${RESET}"
        echo -e "  │ ${BOLD}finish_reason:${RESET}   ${finish_reason}"
        echo -e "  │"
        echo -e "  │ ${WHITE}${resp_content}${RESET}"
    else
        # Error — show the FULL response body so the error message is visible.
        # Fake tokens produce provider-specific errors that prove correct routing:
        #   OpenAI:    {"error":{"message":"Incorrect API key provided: sk-proj-..."}}
        #   Anthropic: {"error":{"type":"authentication_error","message":"invalid x-api-key"}}
        #   Azure:     {"error":{"code":"401","message":"Access denied due to invalid subscription key"}}
        #   Router:    {"error":{"code":401,"message":"no credential found..."}}
        echo -e "  │"
        echo -e "  │ ${BOLD}Response body (error details):${RESET}"

        # Pretty-print if JSON, otherwise raw
        local body_pretty
        body_pretty=$(json_pretty < "$tmpfile" 2>/dev/null) || body_pretty=$(head -c 500 "$tmpfile")

        # Indent each line of the body
        while IFS= read -r line; do
            echo -e "  │   ${line}"
        done <<< "${body_pretty}"
    fi

    echo -e "  └─────────────────────────────────────────────────────────────"
    echo ""
    rm -f "$tmpfile" "$hdrfile"
    wait_for_key
}

# Send a request with no Authorization header (for no-token test)
send_request_no_auth() {
    local test_num="$1" title="$2" model="$3" prompt="$4"

    separator
    echo ""
    echo -e "  ${BOLD}${BG_CYAN}${WHITE} TEST ${test_num} ${RESET}  ${BOLD}${title}${RESET}"
    echo ""
    pause

    echo -e "  ${BOLD}${YELLOW}REQUEST (no Authorization header)${RESET}"
    echo -e "  ${DIM}curl -s ${ENVOY_URL}/v1/chat/completions \\${RESET}"
    echo -e "  ${DIM}  -H \"Content-Type: application/json\" \\${RESET}"
    echo -e "  ${DIM}  -d '{\"model\": \"${model}\", ...}'${RESET}"
    echo ""
    sleep 1

    local http_code tmpfile
    tmpfile=$(mktemp)

    http_code=$(curl -s -o "$tmpfile" -w '%{http_code}' \
        --max-time 10 \
        "${ENVOY_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${model}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
            \"max_tokens\": 10
        }" 2>/dev/null) || http_code="000"

    echo -e "  ${BOLD}RESPONSE — HTTP ${http_code}${RESET}"
    echo -e "  ┌─────────────────────────────────────────────────────────────"
    echo -e "  │ ${BOLD}http_status:${RESET}     ${http_code}"
    echo -e "  │"

    local body_pretty
    body_pretty=$(json_pretty < "$tmpfile" 2>/dev/null) || body_pretty=$(head -c 500 "$tmpfile")
    while IFS= read -r line; do
        echo -e "  │   ${line}"
    done <<< "${body_pretty}"

    echo -e "  └─────────────────────────────────────────────────────────────"
    echo ""
    rm -f "$tmpfile"
    wait_for_key
}

# ── Banner ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════════════════════════════╗"
echo "  ║                                                                   ║"
echo "  ║   vLLM Semantic Router — Multi-Cloud Provider Routing Demo        ║"
echo "  ║                                                                   ║"
echo "  ║   Same model → different cloud providers with per-user auth       ║"
echo "  ║                                                                   ║"
echo "  ╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "  ${BOLD}Architecture:${RESET}"
echo ""
echo -e "  ${DIM}┌──────────┐    ┌───────────────┐    ┌──────────────────┐    ┌──────────────────────────┐${RESET}"
echo -e "  ${DIM}│          │    │ Envoy  :8801  │    │ Semantic Router  │ ─┬→│ ${RESET}${GREEN}OpenAI${RESET}${DIM}   api.openai.com │${RESET}"
echo -e "  ${DIM}│  Client  │ →  │               │ →  │ ext_proc :50051  │  │ └──────────────────────────┘${RESET}"
echo -e "  ${DIM}│          │    │ ext_authz ────│──→ │                  │  │ ┌──────────────────────────┐${RESET}"
echo -e "  ${DIM}│          │    │  Authorino    │    │                  │  ├→│ ${RESET}${BLUE}Azure${RESET}${DIM}  *.azure.com    │${RESET}"
echo -e "  ${DIM}│          │    │  :50052       │    │                  │  │ └──────────────────────────┘${RESET}"
echo -e "  ${DIM}│          │    │               │    │                  │  │ ┌──────────────────────────┐${RESET}"
echo -e "  ${DIM}│          │    │               │    │                  │  ├→│ ${RESET}${MAGENTA}Anthropic${RESET}${DIM}  api.anthr.. │${RESET}"
echo -e "  ${DIM}│          │    │               │    │                  │  │ └──────────────────────────┘${RESET}"
echo -e "  ${DIM}│          │    │               │    │                  │  │ ┌──────────────────────────┐${RESET}"
echo -e "  ${DIM}│          │    │               │    │                  │  └→│ ${RESET}${YELLOW}Local vLLM${RESET}${DIM}  :8000      │${RESET}"
echo -e "  ${DIM}└──────────┘    └───────────────┘    └──────────────────┘    └──────────────────────────┘${RESET}"
echo ""
echo -e "  ${BOLD}Provider Profiles:${RESET}"
echo ""
printf "  ${BOLD}%-22s %-16s %-28s %-20s${RESET}\n" "Profile" "Type" "Base URL" "Auth Header"
printf "  ${DIM}%-22s %-16s %-28s %-20s${RESET}\n" "────────────────────" "──────────────" "──────────────────────────" "──────────────────"
printf "  %-22s ${GREEN}%-16s${RESET} %-28s %-20s\n"   "openai-prod"     "openai"       "api.openai.com/v1"        "Authorization: Bearer"
printf "  %-22s ${BLUE}%-16s${RESET} %-28s %-20s\n"    "azure-east"      "azure-openai" "*.openai.azure.com"       "api-key"
printf "  %-22s ${MAGENTA}%-16s${RESET} %-28s %-20s\n" "anthropic-prod"  "anthropic"    "api.anthropic.com"        "x-api-key"
echo ""
echo -e "  ${BOLD}Users (from K8s Secrets):${RESET}"
echo ""
printf "  ${BOLD}%-14s %-24s %-24s %-20s${RESET}\n" "User" "OpenAI Key" "Azure Key" "Anthropic Key"
printf "  ${DIM}%-14s %-24s %-24s %-20s${RESET}\n" "────────────" "──────────────────────" "──────────────────────" "──────────────────"
printf "  %-14s ${GREEN}%-24s${RESET} ${BLUE}%-24s${RESET} ${MAGENTA}%-20s${RESET}\n" "alice" "yes" "yes" "yes"
printf "  %-14s ${GREEN}%-24s${RESET} ${BLUE}%-24s${RESET} ${DIM}%-20s${RESET}\n"     "bob"   "yes" "yes" "—"
printf "  %-14s ${DIM}%-24s${RESET} ${DIM}%-24s${RESET} ${MAGENTA}%-20s${RESET}\n"    "carol" "—"   "—"   "yes"
echo ""
echo -e "  ${DIM}NOTE: All tokens are FAKE. Provider error messages prove correct routing.${RESET}"
echo -e "  ${DIM}e.g., OpenAI returns \"Incorrect API key provided: sk-proj-alice-...\",${RESET}"
echo -e "  ${DIM}proving the request reached OpenAI with Alice's key in the right format.${RESET}"
echo ""
wait_for_key

# ── Discover tokens from K8s Secrets ─────────────────────────
separator
echo ""
echo -e "  ${BOLD}Discovering user tokens from K8s Secrets...${RESET}"
echo ""

TOKEN_ALICE=$(get_token "user-alice")
TOKEN_BOB=$(get_token "user-bob")
TOKEN_CAROL=$(get_token "user-carol")

echo -e "  user-alice: ${DIM}${TOKEN_ALICE:0:24}...${RESET}"
echo -e "  user-bob:   ${DIM}${TOKEN_BOB:0:24}...${RESET}"
echo -e "  user-carol: ${DIM}${TOKEN_CAROL:0:24}...${RESET}"
echo ""
wait_for_key

# ══════════════════════════════════════════════════════════════
# PART 1: Multi-Provider Routing (same model, different clouds)
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_CYAN}${WHITE} PART 1 ${RESET}  ${BOLD}Multi-Provider Routing — same model, different clouds${RESET}"
echo ""
echo -e "  Alice has keys for all providers. Requests route based on model name"
echo -e "  and the endpoint's provider_profile determines which key to use."
echo -e "  ${DIM}Expect: provider-specific auth errors (fake keys → 401 from provider)${RESET}"
echo ""
wait_for_key

send_request 1 \
    "Alice → gpt-4o (routes to OpenAI, uses openai key)" \
    "$TOKEN_ALICE" \
    "gpt-4o" \
    "What is 2 + 2? Reply in one word."

send_request 2 \
    "Alice → claude-sonnet-4 (routes to Anthropic, uses anthropic key)" \
    "$TOKEN_ALICE" \
    "claude-sonnet-4-20250514" \
    "What is the capital of France? Reply in one word."

# ══════════════════════════════════════════════════════════════
# PART 2: Per-User Key Isolation
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_MAGENTA}${WHITE} PART 2 ${RESET}  ${BOLD}Per-User Key Isolation — same model, different user keys${RESET}"
echo ""
echo -e "  Both Alice and Bob request gpt-4o. Each user's request uses their own"
echo -e "  OpenAI key (injected by Authorino from their K8s Secret)."
echo -e "  ${DIM}Expect: different key fingerprints in the error messages.${RESET}"
echo ""
wait_for_key

send_request 3 \
    "Alice → gpt-4o (Alice's OpenAI key: sk-proj-alice-...)" \
    "$TOKEN_ALICE" \
    "gpt-4o" \
    "Say hello. Reply in 5 words."

send_request 4 \
    "Bob → gpt-4o (Bob's OpenAI key: sk-proj-bob-...)" \
    "$TOKEN_BOB" \
    "gpt-4o" \
    "Say hello. Reply in 5 words."

# ══════════════════════════════════════════════════════════════
# PART 3: Missing Key Rejection
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_RED}${WHITE} PART 3 ${RESET}  ${BOLD}Missing Key Rejection — fail-closed per provider${RESET}"
echo ""
echo -e "  Carol only has an Anthropic key. Requesting gpt-4o requires an OpenAI key."
echo -e "  The router rejects with a credential error (no fallback, no silent failure)."
echo -e "  ${DIM}Expect: router 500 with \"no credential found\" for openai provider.${RESET}"
echo ""
wait_for_key

send_request 5 \
    "Carol → gpt-4o (NO OpenAI key — expect router rejection)" \
    "$TOKEN_CAROL" \
    "gpt-4o" \
    "What is 2 + 2?"

send_request 6 \
    "Carol → claude-sonnet-4 (HAS Anthropic key — expect Anthropic auth error)" \
    "$TOKEN_CAROL" \
    "claude-sonnet-4-20250514" \
    "What is the capital of Japan? Reply in one word."

# ══════════════════════════════════════════════════════════════
# PART 4: Invalid / No Token
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_RED}${WHITE} PART 4 ${RESET}  ${BOLD}Authentication — Authorino rejects invalid tokens${RESET}"
echo ""
echo -e "  ${DIM}Expect: HTTP 401/403 from Authorino (request never reaches router/provider).${RESET}"
echo ""
wait_for_key

send_request 7 \
    "Invalid token → expect 401 from Authorino" \
    "invalid-token-does-not-exist" \
    "gpt-4o" \
    "This should fail"

send_request_no_auth 8 \
    "No token → expect 401 from Authorino" \
    "gpt-4o" \
    "This should also fail"

# ── Conclusion ───────────────────────────────────────────────
separator
echo ""
echo -e "  ${BOLD}${CYAN}Summary — Multi-Cloud Provider Routing${RESET}"
echo ""
echo -e "  The same router instance handles ${BOLD}multiple cloud LLM providers${RESET}."
echo -e "  Provider-specific auth (header name, prefix, path) is driven by ${BOLD}provider_profiles${RESET}."
echo -e "  API keys come from ${BOLD}Authorino${RESET} (K8s Secrets → header injection)."
echo ""
echo -e "  ${BOLD}What the error messages prove:${RESET}"
echo -e "    ${GREEN}✓${RESET} OpenAI errors mention \"sk-proj-alice-...\" → correct key, correct provider"
echo -e "    ${GREEN}✓${RESET} Anthropic errors say \"invalid x-api-key\" → correct auth header format"
echo -e "    ${GREEN}✓${RESET} Alice vs Bob get different key fingerprints → per-user isolation"
echo -e "    ${GREEN}✓${RESET} Carol → gpt-4o gets router rejection → fail-closed, no fallback"
echo -e "    ${GREEN}✓${RESET} Invalid/no token → Authorino 401 → never reaches provider"
echo ""
echo -e "  ${BOLD}Key files:${RESET}"
echo -e "    ${DIM}Router config:  config/testing/config.multi-provider.yaml${RESET}"
echo -e "    ${DIM}Envoy config:   scripts/demo-multi-provider/envoy.yaml${RESET}"
echo -e "    ${DIM}AuthConfig:     scripts/demo-multi-provider/k8s-authconfig.yaml${RESET}"
echo -e "    ${DIM}Secrets:        scripts/demo-multi-provider/k8s-secrets.yaml${RESET}"
echo ""
separator
echo ""

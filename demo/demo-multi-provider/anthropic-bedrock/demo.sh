#!/usr/bin/env bash
# demo.sh — Anthropic vs AWS Bedrock: Same Claude Model, Two Providers
#
# Demonstrates routing the same Claude Sonnet 4 model to either
# Anthropic's API or AWS Bedrock, with per-user keys from Authorino.
#
# Since these are FAKE tokens, we expect provider-specific error messages:
#   Anthropic: {"error":{"type":"authentication_error","message":"invalid x-api-key"}}
#   Bedrock:   {"message":"The security token included in the request is invalid"}
# These errors PROVE the request reached the correct provider with the
# correct auth format — which is the whole point of the demo.
#
# Prerequisites:
#   1. Kind cluster with Authorino running
#      - kubectl apply -f k8s-secrets.yaml
#      - kubectl apply -f k8s-authconfig.yaml
#   2. Authorino port-forwarded:
#      kubectl port-forward -n default svc/authorino-authorino-authorization 50052:50051
#   3. Semantic Router:
#      go run ./cmd/router -config scripts/demo-multi-provider/anthropic-bedrock/config.yaml
#   4. Envoy:
#      envoy -c scripts/demo-multi-provider/anthropic-bedrock/envoy.yaml

set -euo pipefail

SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${SR_ROOT}"

# ── Configuration ────────────────────────────────────────────
ENVOY_URL="${ENVOY_URL:-http://127.0.0.1:8801}"
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

PASS=0
FAIL=0
TOTAL=0

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

# Send a request, show the curl, show key info, show full response body
send_request() {
    local test_num="$1" title="$2" token="$3" model="$4" prompt="$5" expect_http="$6"

    TOTAL=$((TOTAL + 1))
    separator
    echo ""
    echo -e "  ${BOLD}${BG_CYAN}${WHITE} TEST ${test_num} ${RESET}  ${BOLD}${title}${RESET}"
    echo ""
    pause

    # Show the curl command
    echo -e "  ${BOLD}${YELLOW}REQUEST${RESET}"
    echo -e "  ${DIM}curl -s ${ENVOY_URL}/v1/chat/completions \\${RESET}"
    echo -e "  ${DIM}  -H \"Authorization: Bearer ${token:0:20}...\" \\${RESET}"
    echo -e "  ${DIM}  -H \"Content-Type: application/json\" \\${RESET}"
    echo -e "  ${DIM}  -d '{\"model\": \"${model}\", \"messages\": [{...}], \"max_tokens\": 60}'${RESET}"
    echo -e "  ${DIM}  # expected: HTTP ${expect_http}${RESET}"
    echo ""
    sleep 1

    local http_code tmpfile hdrfile
    tmpfile=$(mktemp)
    hdrfile=$(mktemp)

    # Capture response headers (-D) and body (-o)
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

    # ── Extract key info from response headers ──
    local upstream_cluster content_type server_header
    upstream_cluster=$(grep -i "x-envoy-upstream-cluster:" "$hdrfile" 2>/dev/null | tr -d '\r' | awk '{print $2}') || upstream_cluster=""
    content_type=$(grep -i "content-type:" "$hdrfile" 2>/dev/null | tr -d '\r' | sed 's/^[^:]*: //') || content_type=""
    server_header=$(grep -i "^server:" "$hdrfile" 2>/dev/null | tr -d '\r' | sed 's/^[^:]*: //') || server_header=""

    # ── Determine pass/fail ──
    local result_color result_label
    if [[ "${http_code}" == "${expect_http}" ]]; then
        result_color="${GREEN}"
        result_label="PASS"
        PASS=$((PASS + 1))
    else
        result_color="${RED}"
        result_label="FAIL"
        FAIL=$((FAIL + 1))
    fi

    echo -e "  ${BOLD}${result_color}${result_label} — HTTP ${http_code} (expected ${expect_http})${RESET}"
    echo -e "  ┌─────────────────────────────────────────────────────────────"
    echo -e "  │ ${BOLD}http_status:${RESET}     ${http_code}"
    [[ -n "${upstream_cluster}" ]] && echo -e "  │ ${BOLD}envoy_cluster:${RESET}   ${upstream_cluster}"
    [[ -n "${server_header}" ]] && echo -e "  │ ${BOLD}server:${RESET}          ${server_header}"
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
        # Error — show the FULL response body.
        # The error message is the proof:
        #   Anthropic 401: {"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}
        #   Bedrock  403:  {"message":"The security token included in the request is invalid"}
        #   Router   500:  {"error":"no credential found for provider ..."}
        #   Authorino 401: {"code":"UNAUTHENTICATED","message":"..."}
        echo -e "  │"
        echo -e "  │ ${BOLD}Response body (error details):${RESET}"

        local body_pretty
        body_pretty=$(json_pretty < "$tmpfile" 2>/dev/null) || body_pretty=$(head -c 500 "$tmpfile")

        while IFS= read -r line; do
            echo -e "  │   ${line}"
        done <<< "${body_pretty}"
    fi

    echo -e "  └─────────────────────────────────────────────────────────────"
    echo ""
    rm -f "$tmpfile" "$hdrfile"
    wait_for_key
}

# Send with no Authorization header
send_request_no_auth() {
    local test_num="$1" title="$2" model="$3" prompt="$4" expect_http="$5"

    TOTAL=$((TOTAL + 1))
    separator
    echo ""
    echo -e "  ${BOLD}${BG_CYAN}${WHITE} TEST ${test_num} ${RESET}  ${BOLD}${title}${RESET}"
    echo ""
    pause

    echo -e "  ${BOLD}${YELLOW}REQUEST (no Authorization header)${RESET}"
    echo -e "  ${DIM}curl -s ${ENVOY_URL}/v1/chat/completions \\${RESET}"
    echo -e "  ${DIM}  -H \"Content-Type: application/json\" \\${RESET}"
    echo -e "  ${DIM}  -d '{\"model\": \"${model}\", ...}'${RESET}"
    echo -e "  ${DIM}  # expected: HTTP ${expect_http}${RESET}"
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

    local result_color result_label
    if [[ "${http_code}" == "${expect_http}" || ("${expect_http}" == "401" && "${http_code}" == "403") ]]; then
        result_color="${GREEN}"
        result_label="PASS"
        PASS=$((PASS + 1))
    else
        result_color="${RED}"
        result_label="FAIL"
        FAIL=$((FAIL + 1))
    fi

    echo -e "  ${BOLD}${result_color}${result_label} — HTTP ${http_code} (expected ${expect_http})${RESET}"
    echo -e "  ┌─────────────────────────────────────────────────────────────"
    echo -e "  │ ${BOLD}http_status:${RESET}     ${http_code}"
    echo -e "  │"
    echo -e "  │ ${BOLD}Response body:${RESET}"

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
echo "  ║   Anthropic vs AWS Bedrock — Same Claude Model, Two Providers     ║"
echo "  ║                                                                   ║"
echo "  ║   Per-user keys from Authorino · Provider-specific auth formats   ║"
echo "  ║                                                                   ║"
echo "  ╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "  ${BOLD}Architecture:${RESET}"
echo ""
echo -e "  ${DIM}┌──────────┐   ┌──────────────┐   ┌────────────────┐   ┌───────────────────────────────┐${RESET}"
echo -e "  ${DIM}│          │   │ Envoy :8801  │   │ Semantic Router│ ┬→│ ${RESET}${MAGENTA}Anthropic${RESET}${DIM}  api.anthropic.com  │${RESET}"
echo -e "  ${DIM}│  Client  │ → │              │ → │ ext_proc :50051│ │ │  x-api-key: {key}             │${RESET}"
echo -e "  ${DIM}│          │   │ ext_authz ───│─→ │                │ │ └───────────────────────────────┘${RESET}"
echo -e "  ${DIM}│          │   │ Authorino    │   │                │ │ ┌───────────────────────────────┐${RESET}"
echo -e "  ${DIM}│          │   │ :50052       │   │                │ └→│ ${RESET}${YELLOW}Bedrock${RESET}${DIM}  bedrock-runtime..  │${RESET}"
echo -e "  ${DIM}│          │   │              │   │                │   │  Authorization: Bearer {sigv4} │${RESET}"
echo -e "  ${DIM}└──────────┘   └──────────────┘   └────────────────┘   └───────────────────────────────┘${RESET}"
echo ""
echo -e "  ${BOLD}Same model, different auth:${RESET}"
echo ""
printf "  ${BOLD}%-18s %-18s %-24s %-22s${RESET}\n" "Provider" "Profile Type" "Auth Header" "Path"
printf "  ${DIM}%-18s %-18s %-24s %-22s${RESET}\n" "────────────────" "────────────────" "──────────────────────" "────────────────────"
printf "  ${MAGENTA}%-18s${RESET} %-18s %-24s %-22s\n" "Anthropic"  "anthropic" "x-api-key: {key}"        "/v1/messages"
printf "  ${YELLOW}%-18s${RESET} %-18s %-24s %-22s\n" "AWS Bedrock" "bedrock"  "Authorization: Bearer"   "/model/.../converse"
echo ""
echo -e "  ${BOLD}Users:${RESET}"
echo ""
printf "  ${BOLD}%-12s %-20s %-20s${RESET}\n" "User" "Anthropic Key" "Bedrock Key"
printf "  ${DIM}%-12s %-20s %-20s${RESET}\n" "──────────" "──────────────────" "──────────────────"
printf "  %-12s ${MAGENTA}%-20s${RESET} ${YELLOW}%-20s${RESET}\n" "alice" "sk-ant-alice-..." "alice-bedrock-sigv4-..."
printf "  %-12s ${MAGENTA}%-20s${RESET} ${DIM}%-20s${RESET}\n"    "bob"   "sk-ant-bob-..."   "—"
printf "  %-12s ${DIM}%-20s${RESET} ${YELLOW}%-20s${RESET}\n"     "carol" "—"                "carol-bedrock-sigv4-..."
echo ""
echo -e "  ${DIM}NOTE: All tokens are FAKE. Provider error messages prove correct routing:${RESET}"
echo -e "  ${DIM}  Anthropic: \"invalid x-api-key\" → reached Anthropic, x-api-key header used${RESET}"
echo -e "  ${DIM}  Bedrock:   \"security token...is invalid\" → reached Bedrock, Bearer used${RESET}"
echo ""
wait_for_key

# ── Discover tokens ──────────────────────────────────────────
separator
echo ""
echo -e "  ${BOLD}Discovering user tokens from K8s Secrets...${RESET}"
echo ""

TOKEN_ALICE=$(get_token "user-alice")
TOKEN_BOB=$(get_token "user-bob")
TOKEN_CAROL=$(get_token "user-carol")

echo -e "  user-alice: ${DIM}${TOKEN_ALICE}${RESET}"
echo -e "  user-bob:   ${DIM}${TOKEN_BOB}${RESET}"
echo -e "  user-carol: ${DIM}${TOKEN_CAROL}${RESET}"
echo ""
wait_for_key

# ══════════════════════════════════════════════════════════════
# PART 1: Same model → different providers (Alice has both keys)
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_CYAN}${WHITE} PART 1 ${RESET}  ${BOLD}Same Claude model → Anthropic vs Bedrock${RESET}"
echo ""
echo -e "  Alice has keys for both providers."
echo -e "  Explicit model aliases force routing to a specific provider."
echo -e "  ${DIM}Expect: Anthropic returns \"invalid x-api-key\", Bedrock returns \"security token...invalid\"${RESET}"
echo ""
wait_for_key

send_request 1 \
    "Alice → claude-sonnet-4-anthropic (Anthropic direct, x-api-key header)" \
    "$TOKEN_ALICE" \
    "claude-sonnet-4-anthropic" \
    "What is 2 + 2? Reply in one word." \
    "401"

send_request 2 \
    "Alice → claude-sonnet-4-bedrock (AWS Bedrock, Authorization: Bearer header)" \
    "$TOKEN_ALICE" \
    "claude-sonnet-4-bedrock" \
    "What is the capital of France? Reply in one word." \
    "403"

# ══════════════════════════════════════════════════════════════
# PART 2: Failover — Anthropic preferred, Bedrock fallback
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_MAGENTA}${WHITE} PART 2 ${RESET}  ${BOLD}Failover — claude-sonnet-4-20250514 (Anthropic → Bedrock)${RESET}"
echo ""
echo -e "  The default model prefers Anthropic but can fail over to Bedrock."
echo -e "  Alice (both keys) routes to preferred Anthropic."
echo -e "  ${DIM}Expect: Anthropic auth error (proves it went to preferred provider)${RESET}"
echo ""
wait_for_key

send_request 3 \
    "Alice → claude-sonnet-4-20250514 (prefers Anthropic)" \
    "$TOKEN_ALICE" \
    "claude-sonnet-4-20250514" \
    "Say hello in 5 words." \
    "401"

# ══════════════════════════════════════════════════════════════
# PART 3: Per-user key isolation
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_GREEN}${WHITE} PART 3 ${RESET}  ${BOLD}Per-User Key Isolation${RESET}"
echo ""
echo -e "  Bob has only an Anthropic key → succeeds reaching Anthropic."
echo -e "  Carol has only a Bedrock key → succeeds reaching Bedrock."
echo -e "  ${DIM}Expect: different error messages proving different providers + different keys${RESET}"
echo ""
wait_for_key

send_request 4 \
    "Bob → claude-sonnet-4-anthropic (Bob's Anthropic key: sk-ant-bob-...)" \
    "$TOKEN_BOB" \
    "claude-sonnet-4-anthropic" \
    "What is 3 + 3? Reply in one word." \
    "401"

send_request 5 \
    "Carol → claude-sonnet-4-bedrock (Carol's Bedrock key: carol-bedrock-sigv4-...)" \
    "$TOKEN_CAROL" \
    "claude-sonnet-4-bedrock" \
    "What is the capital of Japan? Reply in one word." \
    "403"

# ══════════════════════════════════════════════════════════════
# PART 4: Missing key rejection (fail-closed)
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_RED}${WHITE} PART 4 ${RESET}  ${BOLD}Missing Key Rejection — fail-closed per provider${RESET}"
echo ""
echo -e "  Bob has NO Bedrock key → requesting bedrock endpoint fails at router level."
echo -e "  Carol has NO Anthropic key → requesting anthropic endpoint fails at router level."
echo -e "  ${DIM}Expect: router 200 with error body \"no credential found\" (NOT a provider error)${RESET}"
echo ""
wait_for_key

send_request 6 \
    "Bob → claude-sonnet-4-bedrock (NO Bedrock key — router rejects)" \
    "$TOKEN_BOB" \
    "claude-sonnet-4-bedrock" \
    "This should fail" \
    "200"

send_request 7 \
    "Carol → claude-sonnet-4-anthropic (NO Anthropic key — router rejects)" \
    "$TOKEN_CAROL" \
    "claude-sonnet-4-anthropic" \
    "This should fail" \
    "200"

# ══════════════════════════════════════════════════════════════
# PART 5: Auth rejection (invalid / no token)
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_RED}${WHITE} PART 5 ${RESET}  ${BOLD}Authentication — Authorino rejects bad tokens${RESET}"
echo ""
echo -e "  ${DIM}Expect: HTTP 401/403 from Authorino (request never reaches router or provider)${RESET}"
echo ""
wait_for_key

send_request 8 \
    "Invalid token → Authorino rejects" \
    "invalid-token-does-not-exist" \
    "claude-sonnet-4-20250514" \
    "This should fail" \
    "401"

send_request_no_auth 9 \
    "No token → Authorino rejects" \
    "claude-sonnet-4-20250514" \
    "This should fail" \
    "401"

# ── Summary ──────────────────────────────────────────────────
separator
echo ""
echo -e "  ${BOLD}${CYAN}Results${RESET}"
echo ""
echo -e "  ${GREEN}${PASS} passed${RESET}, ${RED}${FAIL} failed${RESET}, ${TOTAL} total"
echo ""

echo -e "  ${BOLD}${CYAN}Summary — Anthropic vs AWS Bedrock${RESET}"
echo ""
echo -e "  The same Claude Sonnet 4 model is served by ${BOLD}two cloud providers${RESET}."
echo -e "  Provider-specific auth (header, prefix, path) is driven by ${BOLD}provider_profiles${RESET}."
echo -e "  API keys come from ${BOLD}Authorino${RESET} (K8s Secrets → header injection)."
echo ""
echo -e "  ${BOLD}What the error messages prove:${RESET}"
echo ""
echo -e "    ${BOLD}Test${RESET}  ${BOLD}Error from${RESET}   ${BOLD}Error message proves${RESET}"
echo -e "    ${DIM}────  ──────────  ─────────────────────────────────────────${RESET}"
echo -e "    1     Anthropic    \"invalid x-api-key\" → correct auth header"
echo -e "    2     Bedrock      \"security token invalid\" → correct Bearer format"
echo -e "    3     Anthropic    reached preferred provider (not Bedrock fallback)"
echo -e "    4     Anthropic    Bob's key (different from Alice's in test 1)"
echo -e "    5     Bedrock      Carol's key reached Bedrock"
echo -e "    6     Router       200 + \"no credential\" → fail-closed, no fallback"
echo -e "    7     Router       200 + \"no credential\" → fail-closed, no fallback"
echo -e "    8-9   Authorino    401 → request never reached router/provider"
echo ""
echo -e "  ${BOLD}Key files:${RESET}"
echo -e "    ${DIM}Router config:  scripts/demo-multi-provider/anthropic-bedrock/config.yaml${RESET}"
echo -e "    ${DIM}Envoy config:   scripts/demo-multi-provider/anthropic-bedrock/envoy.yaml${RESET}"
echo -e "    ${DIM}AuthConfig:     scripts/demo-multi-provider/anthropic-bedrock/k8s-authconfig.yaml${RESET}"
echo -e "    ${DIM}Secrets:        scripts/demo-multi-provider/anthropic-bedrock/k8s-secrets.yaml${RESET}"
echo ""
separator
echo ""

if [[ "${FAIL}" -gt 0 ]]; then
    exit 1
fi

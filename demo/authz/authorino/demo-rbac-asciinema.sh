#!/usr/bin/env bash
# demo-authz-rbac-asciinema.sh — Asciinema demo: RBAC Authz Signal with Semantic Router
#
# This script is designed to be recorded with asciinema.
# It shows the full end-to-end flow with real running systems.
# All values are discovered dynamically — no hardcoded output.
#
# Prerequisites:
#   All infrastructure must be running (use setup-authz-rbac-demo.sh)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ─── Configurable ────────────────────────────────────────────────────
ENVOY_PORT="${ENVOY_PORT:-8801}"
VLLM_14B_PORT="${VLLM_14B_PORT:-8000}"
VLLM_7B_PORT="${VLLM_7B_PORT:-8001}"
KIND_CONTEXT="${KIND_CONTEXT:-kind-authorino-test}"
PAUSE="${PAUSE:-5}"

# ─── Presentation helpers ────────────────────────────────────────────
BOLD="\033[1m"
DIM="\033[2m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

title()   { echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${RESET}"; echo -e "${BOLD}${CYAN}  $1${RESET}"; echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${RESET}\n"; sleep "$PAUSE"; }
section() { echo -e "\n${BOLD}${YELLOW}── $1 ──${RESET}\n"; sleep 2; }
step()    { echo -e "${GREEN}▶${RESET} ${BOLD}$1${RESET}"; }
info()    { echo -e "  ${DIM}$1${RESET}"; }
pause()   { sleep "$PAUSE"; }
run_cmd() { echo -e "\n${DIM}\$ $1${RESET}"; eval "$1"; echo ""; }

# ─── Dynamic discovery (read from live systems, not hardcoded) ───────
read_token() {
  kubectl --context "${KIND_CONTEXT}" get secret "$1" \
    -o jsonpath='{.data.api_key}' 2>/dev/null | base64 -d
}
read_groups() {
  kubectl --context "${KIND_CONTEXT}" get secret "$1" \
    -o jsonpath="{.metadata.annotations.authz-groups}" 2>/dev/null
}
extract_model() {
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d['model'])" 2>/dev/null
}

# ═══════════════════════════════════════════════════════════════════════
#  DEMO START
# ═══════════════════════════════════════════════════════════════════════

clear

title "RBAC Authz Signal — End-to-End Demo"

echo -e "${BOLD}Architecture:${RESET}"
echo ""
echo "  Client (curl) → Envoy (:${ENVOY_PORT}) → Authorino (ext_authz) → Semantic Router (ext_proc) → vLLM"
echo ""
echo "  1. Client sends Bearer token to Envoy"
echo "  2. Envoy calls Authorino — validates token against K8s Secrets"
echo "  3. Authorino injects x-authz-user-id + x-authz-user-groups headers"
echo "  4. Envoy calls Semantic Router (ext_proc) with enriched headers"
echo "  5. Router matches role_bindings → fires authz signal → decision engine selects model"
echo "  6. Envoy routes request to the selected vLLM backend"
pause

# ═══════════════════════════════════════════════════════════════════════
title "Step 1: Verify Running Infrastructure"

step "Docker containers"
run_cmd "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | head -10"
pause

step "Discover models from live vLLM endpoints"
MODEL_14B=$(curl -sf "http://localhost:${VLLM_14B_PORT}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
MODEL_7B=$(curl -sf "http://localhost:${VLLM_7B_PORT}/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo -e "  14B model (port ${VLLM_14B_PORT}): ${GREEN}${MODEL_14B}${RESET}"
echo -e "  7B model  (port ${VLLM_7B_PORT}): ${GREEN}${MODEL_7B}${RESET}"
pause

step "Authorino port-forward"
run_cmd "ss -tlnp | grep ':50052' | head -1"
pause

step "Semantic Router (ext_proc)"
run_cmd "ss -tlnp | grep ':50051' | head -1"
pause

# ═══════════════════════════════════════════════════════════════════════
title "Step 2: Kubernetes Secrets (User Identity Store)"

step "Each user has a K8s Secret with an API key + authz-groups annotation"
info "Authorino reads these to authenticate users and inject RBAC headers"
echo ""

for secret in user-admin user-alice-per-user user-bob-shared user-carol-shared user-dave-byot; do
  groups=$(read_groups "${secret}")
  echo -e "  ${BOLD}${secret}${RESET}  →  authz-groups: ${CYAN}${groups:-\"(empty)\"}${RESET}"
done
pause

step "Example: show user-bob-shared secret annotations"
run_cmd "kubectl --context ${KIND_CONTEXT} get secret user-bob-shared -o jsonpath='{.metadata.annotations}' | python3 -m json.tool"
pause

# ═══════════════════════════════════════════════════════════════════════
title "Step 3: Authorino AuthConfig (Token → Header Injection)"

step "Authorino AuthConfig controls what headers get injected after auth"
info "x-authz-user-id ← Secret metadata.name"
info "x-authz-user-groups ← Secret annotation 'authz-groups'"
echo ""

run_cmd "kubectl --context ${KIND_CONTEXT} get authconfig semantic-router-auth -o yaml | grep -A 20 'response:'"
pause

# ═══════════════════════════════════════════════════════════════════════
title "Step 4: Router Config — Role Bindings"

step "role_bindings map users/groups → roles (pure RBAC, K8s style)"
echo ""

# Show just the role_bindings section from the live config
run_cmd "sed -n '/^role_bindings:/,/^decisions:/p' ${REPO_ROOT}/config/testing/config.authz-rbac-live.yaml | head -35"
pause

# ═══════════════════════════════════════════════════════════════════════
title "Step 5: Router Config — Decisions"

step "Decisions combine authz roles + other signals → model selection"
info "Higher priority = checked first. authz signal feeds into AND/OR conditions."
echo ""

run_cmd "sed -n '/^decisions:/,\$p' ${REPO_ROOT}/config/testing/config.authz-rbac-live.yaml | head -50"
pause

# ═══════════════════════════════════════════════════════════════════════
title "Step 6: Live Tests — Send Real Requests"

info "Every request goes through: curl → Envoy → Authorino → Router → vLLM"
info "You will see: the exact curl command, the raw HTTP status code,"
info "the model name from the JSON response, and a snippet of the LLM output."
echo ""
pause

# Read tokens dynamically
TOKEN_ADMIN=$(read_token "user-admin")
TOKEN_ALICE=$(read_token "user-alice-per-user")
TOKEN_BOB=$(read_token "user-bob-shared")
TOKEN_CAROL=$(read_token "user-carol-shared")
TOKEN_DAVE=$(read_token "user-dave-byot")

ENVOY_URL="http://localhost:${ENVOY_PORT}"
PASS=0
FAIL=0

# ─── Test helper: show curl, run it, display raw results ─────────────
# Writes response body to $TMPDIR/resp_body.json and HTTP code to $TMPDIR/resp_code
TMPDIR=$(mktemp -d)

run_test() {
  local test_num="$1"
  local test_label="$2"
  local why="$3"
  local token="$4"
  local prompt="$5"
  local max_tokens="${6:-30}"
  local expected_model="${7:-}"      # empty for status-only tests
  local expected_status="${8:-200}"  # default 200

  section "Test ${test_num}/8: ${test_label}"
  step "${why}"
  echo ""

  # Show the exact curl command the viewer can copy-paste
  echo -e "${DIM}Running:${RESET}"
  if [[ -n "${token}" ]]; then
    echo -e "${DIM}  curl -s -w '\\nHTTP_STATUS:%{http_code}' ${ENVOY_URL}/v1/chat/completions \\\\${RESET}"
    echo -e "${DIM}    -H 'Authorization: Bearer ${token}' \\\\${RESET}"
    echo -e "${DIM}    -H 'Content-Type: application/json' \\\\${RESET}"
    echo -e "${DIM}    -d '{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens}}'${RESET}"
  else
    echo -e "${DIM}  curl -s -w '\\nHTTP_STATUS:%{http_code}' ${ENVOY_URL}/v1/chat/completions \\\\${RESET}"
    echo -e "${DIM}    -H 'Content-Type: application/json' \\\\${RESET}"
    echo -e "${DIM}    -d '{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens}}'${RESET}"
  fi
  echo ""
  sleep 2

  # Actually run it — capture both body and status code
  local raw_output
  if [[ -n "${token}" ]]; then
    raw_output=$(curl -s -w '\nHTTP_STATUS:%{http_code}' "${ENVOY_URL}/v1/chat/completions" \
      -H "Authorization: Bearer ${token}" \
      -H "Content-Type: application/json" \
      -H "Host: localhost:${ENVOY_PORT}" \
      -d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens}}" 2>&1)
  else
    raw_output=$(curl -s -w '\nHTTP_STATUS:%{http_code}' "${ENVOY_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -H "Host: localhost:${ENVOY_PORT}" \
      -d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"${prompt}\"}],\"max_tokens\":${max_tokens}}" 2>&1)
  fi

  # Split body and status
  local http_status
  http_status=$(echo "${raw_output}" | grep -oP 'HTTP_STATUS:\K\d+' | tail -1)
  local body
  body=$(echo "${raw_output}" | sed '/^HTTP_STATUS:/d')

  # Display HTTP status code
  echo -e "  ${BOLD}HTTP Status:${RESET} ${http_status}"

  if [[ -n "${expected_model}" ]]; then
    # Model routing test — parse and display response fields
    local actual_model actual_content
    actual_model=$(echo "${body}" | python3 -c "import sys,json; print(json.load(sys.stdin)['model'])" 2>/dev/null || echo "PARSE_ERROR")
    actual_content=$(echo "${body}" | python3 -c "
import sys,json
d=json.load(sys.stdin)
txt=d['choices'][0]['message']['content']
# Show first 120 chars, one line
txt=txt.replace('\n',' ').strip()
print(txt[:120] + ('...' if len(txt)>120 else ''))
" 2>/dev/null || echo "(could not parse content)")

    echo -e "  ${BOLD}Model:${RESET}       ${actual_model}"
    echo -e "  ${BOLD}Response:${RESET}    ${DIM}${actual_content}${RESET}"
    echo ""

    # Verdict
    if [[ "${actual_model}" == "${expected_model}" && "${http_status}" == "${expected_status}" ]]; then
      echo -e "  ${GREEN}✓ PASS${RESET}  model=${actual_model}, HTTP ${http_status}"
      PASS=$((PASS + 1))
    else
      echo -e "  ${RED}✗ FAIL${RESET}  expected model=${expected_model} HTTP ${expected_status}, got model=${actual_model} HTTP ${http_status}"
      FAIL=$((FAIL + 1))
    fi
  else
    # Status-only test (401, 403 etc) — show the error body too
    local error_body
    error_body=$(echo "${body}" | head -c 200)
    if [[ -n "${error_body}" ]]; then
      echo -e "  ${BOLD}Body:${RESET}        ${DIM}${error_body}${RESET}"
    fi
    echo ""

    if [[ "${http_status}" == "${expected_status}" ]]; then
      echo -e "  ${GREEN}✓ PASS${RESET}  HTTP ${http_status}"
      PASS=$((PASS + 1))
    else
      echo -e "  ${RED}✗ FAIL${RESET}  expected HTTP ${expected_status}, got HTTP ${http_status}"
      FAIL=$((FAIL + 1))
    fi
  fi
  pause
}

# ── Test 1: Admin ──
run_test "1" "Admin → expects ${MODEL_14B}" \
  "admin has role 'admin' → decision 'admin_unrestricted' → 14B" \
  "${TOKEN_ADMIN}" "What is 2+2?" 30 "${MODEL_14B}" "200"

# ── Test 2: Alice ──
run_test "2" "Alice (engineering) → expects ${MODEL_7B}" \
  "alice group=engineering → role 'pro_tier' → decision 'pro_default' → 7B" \
  "${TOKEN_ALICE}" "What is 2+2?" 30 "${MODEL_7B}" "200"

# ── Test 3: Bob complex ──
run_test "3" "Bob (premium) + complex query → expects ${MODEL_14B}" \
  "bob group=premium → role 'premium_tier' + keyword 'analyze' → decision 'premium_complex' → 14B" \
  "${TOKEN_BOB}" "Please analyze and explain why the sky is blue, think step by step" 50 "${MODEL_14B}" "200"

# ── Test 4: Bob simple ──
run_test "4" "Bob (premium) + simple query → expects ${MODEL_7B}" \
  "bob 'premium_tier' but no keyword match → decision 'premium_default' → 7B (cost savings)" \
  "${TOKEN_BOB}" "Hi there" 30 "${MODEL_7B}" "200"

# ── Test 5: Carol ──
run_test "5" "Carol (free) → expects ${MODEL_7B}" \
  "carol group=free → role 'free_tier' → decision 'free_short' or 'free_default' → 7B" \
  "${TOKEN_CAROL}" "What is the capital of France?" 30 "${MODEL_7B}" "200"

# ── Test 6: Dave ──
run_test "6" "Dave (contractor) → expects ${MODEL_7B}" \
  "dave group=contractor → role 'free_tier' → 7B" \
  "${TOKEN_DAVE}" "Hello world" 30 "${MODEL_7B}" "200"

# ── Test 7: Invalid token ──
run_test "7" "Invalid token → expects HTTP 401" \
  "Authorino rejects unknown tokens before they reach the router" \
  "fake-invalid-token-xyz" "Hello" 10 "" "401"

# ── Test 8: No token ──
run_test "8" "No token → expects HTTP 401" \
  "Missing Authorization header is rejected by Authorino" \
  "" "Hello" 10 "" "401"

rm -rf "${TMPDIR}"

# ═══════════════════════════════════════════════════════════════════════
title "Results"

echo -e "  ${BOLD}${PASS} passed, ${FAIL} failed${RESET} (total $((PASS + FAIL)))"
echo ""

if [[ ${FAIL} -gt 0 ]]; then
  echo -e "  ${RED}Some tests failed — check infrastructure.${RESET}"
  exit 1
fi

echo -e "  ${GREEN}All tests passed against live infrastructure.${RESET}"
echo ""
echo -e "  ${BOLD}Key points:${RESET}"
echo "  - Every result above came from a real HTTP request through Envoy → Authorino → Router → vLLM"
echo "  - Model names were discovered from vLLM /v1/models endpoints, not hardcoded"
echo "  - Tokens were read from K8s Secrets via kubectl, not hardcoded"
echo "  - The 'model' field in each response is what vLLM actually served, not what was expected"
echo ""

#!/usr/bin/env bash
# test-limitor.sh — Comprehensive local-limiter rate limiting test suite
#
# Starts the router + Envoy, runs all test cases against live vLLM, stops infra.
# No silent fallback: every test asserts exact HTTP status codes and headers.
#
# Architecture:
#   Client → Envoy :8804 (ext_proc) → Router :50055 (local-limiter) → vLLM :8100
#
# Test cases:
#   1. Free-tier user: requests within limit → 200
#   2. Free-tier user: burst exceeding 3 RPM → 429
#   3. 429 response has correct rate limit headers
#   4. 429 response body is valid JSON error
#   5. Premium-tier user: 5 requests within 10 RPM limit → all 200
#   6. User with no group (no matching rule) → unlimited → 200
#   7. Separate users have independent rate limit buckets
#   8. Admin tier: high limit, 5 requests → all 200

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR"

ROUTER_PORT=50055
ENVOY_PORT=8804
ENVOY_ADMIN=19004
VLLM_PORT=8100
ROUTER_PID=""
ENVOY_PID=""

PASS=0
FAIL=0
TOTAL=0

# ── Cleanup on exit ──
cleanup() {
    echo ""
    echo "── Cleanup ──"
    [ -n "$ROUTER_PID" ] && kill "$ROUTER_PID" 2>/dev/null && echo "  Stopped router (PID $ROUTER_PID)"
    [ -n "$ENVOY_PID" ] && kill "$ENVOY_PID" 2>/dev/null && echo "  Stopped envoy (PID $ENVOY_PID)"
    rm -f "$LAST_BODY_FILE" "$LAST_HDRS_FILE" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

assert_eq() {
    local test_name="$1" expected="$2" actual="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$expected" = "$actual" ]; then
        echo -e "  \033[1;32mPASS\033[0m  $test_name (got $actual)"
        PASS=$((PASS + 1))
    else
        echo -e "  \033[1;31mFAIL\033[0m  $test_name (expected $expected, got $actual)"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    local test_name="$1" haystack="$2" needle="$3"
    TOTAL=$((TOTAL + 1))
    if echo "$haystack" | grep -qi "$needle"; then
        echo -e "  \033[1;32mPASS\033[0m  $test_name"
        PASS=$((PASS + 1))
    else
        echo -e "  \033[1;31mFAIL\033[0m  $test_name (\"$needle\" not found)"
        FAIL=$((FAIL + 1))
    fi
}

LAST_BODY_FILE=$(mktemp)
LAST_HDRS_FILE=$(mktemp)

send() {
    local user="$1" groups="$2" prompt="$3" max_tok="${4:-10}"

    local curl_args=(
        -s -o "$LAST_BODY_FILE" -w '%{http_code}' -D "$LAST_HDRS_FILE"
        --max-time 120
        "http://127.0.0.1:${ENVOY_PORT}/v1/chat/completions"
        -H "Content-Type: application/json"
    )
    [ -n "$user" ]   && curl_args+=(-H "x-jwt-sub: $user")
    [ -n "$groups" ] && curl_args+=(-H "x-jwt-groups: $groups")
    curl_args+=(-d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tok}")

    local code
    code=$(curl "${curl_args[@]}" 2>/dev/null) || code="000"
    echo "$code"
}

# ══════════════════════════════════════════════════════════════
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Local Limiter — Comprehensive Test Suite               ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Preflight: check vLLM ──
echo ""
echo "── Preflight ──"
echo -n "  vLLM :$VLLM_PORT … "
vllm_code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" --max-time 5 2>/dev/null) || vllm_code="unreachable"
if [ "$vllm_code" != "200" ]; then
    echo "FAIL (HTTP $vllm_code)"
    echo "  ERROR: vLLM must be running on port $VLLM_PORT"
    exit 1
fi
echo "OK"

# ── Kill stale processes ──
for p in $ROUTER_PORT $ENVOY_PORT $ENVOY_ADMIN; do
    lsof -ti :"$p" 2>/dev/null | xargs kill -9 2>/dev/null || true
done
sleep 1

# ── Start router ──
echo ""
echo "── Starting router ──"
echo "  Config: $CONFIG_DIR/config.yaml"
echo "  Port:   $ROUTER_PORT"

export LD_LIBRARY_PATH="${REPO_ROOT}/candle-binding/target/release:${REPO_ROOT}/ml-binding/target/release:${REPO_ROOT}/nlp-binding/target/release"
/tmp/router-ratelimit --config "$CONFIG_DIR/config.yaml" --port "$ROUTER_PORT" > /tmp/test-limitor-router.log 2>&1 &
ROUTER_PID=$!
echo "  PID:    $ROUTER_PID"

echo -n "  Waiting for router"
for i in $(seq 1 15); do
    sleep 1
    echo -n "."
    if grep -q "Starting insecure LLM Router ExtProc server" /tmp/test-limitor-router.log 2>/dev/null; then
        echo " ready"
        break
    fi
    if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
        echo " FAILED (process exited)"
        tail -10 /tmp/test-limitor-router.log
        exit 1
    fi
done

# Verify rate limiter was initialized
if ! grep -q "local-limiter" /tmp/test-limitor-router.log; then
    echo "  ERROR: Rate limiter not initialized. Router log:"
    grep -i "ratelimit\|limiter\|error" /tmp/test-limitor-router.log | tail -10
    exit 1
fi
echo "  Rate limiter: $(grep 'RateLimit:' /tmp/test-limitor-router.log | head -2 | sed 's/.*"msg":"/  /' | sed 's/".*//')"

# ── Start Envoy ──
echo ""
echo "── Starting Envoy ──"
func-e run -c "$CONFIG_DIR/envoy.yaml" --base-id 5 --log-level warn > /tmp/test-limitor-envoy.log 2>&1 &
ENVOY_PID=$!
echo "  PID:    $ENVOY_PID"
sleep 3

if ! kill -0 "$ENVOY_PID" 2>/dev/null; then
    echo "  FAILED (process exited)"
    tail -10 /tmp/test-limitor-envoy.log
    exit 1
fi
echo "  Envoy listening on :$ENVOY_PORT"

# ── Verify connectivity ──
echo ""
echo "── Verifying end-to-end connectivity ──"
warmup_code=$(send "warmup-user" "premium-tier" "ping" 5)
if [ "$warmup_code" != "200" ]; then
    echo "  ERROR: Warmup request returned HTTP $warmup_code"
    echo "  Body: $LAST_BODY"
    exit 1
fi
echo "  Warmup: HTTP $warmup_code — pipeline is healthy"

# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running test cases"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Test 1: Free-tier within limit ──
echo ""
echo "── Test 1: Free-tier user — requests within 3 RPM limit → 200 ──"
FREE_USER="free-test-$$"
for i in 1 2 3; do
    code=$(send "$FREE_USER" "free-tier" "say hello $i" 5)
    assert_eq "Free-tier request $i → 200" "200" "$code"
done

# ── Test 2: Free-tier burst exceeds limit ──
echo ""
echo "── Test 2: Free-tier user — 4th request exceeds 3 RPM → 429 ──"
code=$(send "$FREE_USER" "free-tier" "this should fail" 5)
assert_eq "Free-tier request 4 → 429" "429" "$code"

# ── Test 3: 429 response has correct headers ──
echo ""
echo "── Test 3: 429 response includes rate limit headers ──"
HDRS=$(cat "$LAST_HDRS_FILE")
BODY=$(cat "$LAST_BODY_FILE")

assert_contains "Retry-After header present" "$HDRS" "retry-after"
assert_contains "X-RateLimit-Limit header present" "$HDRS" "x-ratelimit-limit"
assert_contains "X-RateLimit-Remaining header present" "$HDRS" "x-ratelimit-remaining"
assert_contains "X-RateLimit-Reset header present" "$HDRS" "x-ratelimit-reset"

rl_limit=$(echo "$HDRS" | grep -i "x-ratelimit-limit" | awk '{print $2}' | tr -d '\r')
rl_remaining=$(echo "$HDRS" | grep -i "x-ratelimit-remaining" | awk '{print $2}' | tr -d '\r')
assert_eq "X-RateLimit-Limit = 3" "3" "$rl_limit"
assert_eq "X-RateLimit-Remaining = 0" "0" "$rl_remaining"

# ── Test 4: 429 body is valid JSON ──
echo ""
echo "── Test 4: 429 response body is valid JSON error ──"
json_valid=$(echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); print('valid')" 2>/dev/null || echo "invalid")
assert_eq "JSON body is parseable" "valid" "$json_valid"

error_type=$(echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['error']['type'])" 2>/dev/null || echo "")
assert_eq "Error type = rate_limit_error" "rate_limit_error" "$error_type"

error_code=$(echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['error']['code'])" 2>/dev/null || echo "")
assert_eq "Error code = 429" "429" "$error_code"

# ── Test 5: Premium-tier has higher limit ──
echo ""
echo "── Test 5: Premium-tier user — 5 requests within 10 RPM → all 200 ──"
PREM_USER="premium-test-$$"
for i in 1 2 3 4 5; do
    code=$(send "$PREM_USER" "premium-tier" "premium request $i" 5)
    assert_eq "Premium request $i → 200" "200" "$code"
done

# ── Test 6: User with no group → no matching rule → unlimited ──
echo ""
echo "── Test 6: User with no group (no rule match) → unlimited → 200 ──"
NOGRP_USER="nogroup-test-$$"
for i in 1 2 3 4 5; do
    code=$(send "$NOGRP_USER" "" "unlimited $i" 5)
    assert_eq "No-group request $i → 200" "200" "$code"
done

# ── Test 7: Independent rate limit buckets per user ──
echo ""
echo "── Test 7: Separate free-tier users have independent buckets ──"
USER_A="free-a-$$"
USER_B="free-b-$$"

# User A: exhaust limit
for i in 1 2 3; do
    send "$USER_A" "free-tier" "a $i" 5 > /dev/null
done
code_a=$(send "$USER_A" "free-tier" "a overflow" 5)
assert_eq "User A (4th request) → 429" "429" "$code_a"

# User B: should still have full budget
code_b=$(send "$USER_B" "free-tier" "b first" 5)
assert_eq "User B (1st request) → 200" "200" "$code_b"

# ── Test 8: Admin tier has high limit ──
echo ""
echo "── Test 8: Admin tier — 5 requests within 500 RPM → all 200 ──"
ADMIN_USER="admin-test-$$"
for i in 1 2 3 4 5; do
    code=$(send "$ADMIN_USER" "platform-admins" "admin $i" 5)
    assert_eq "Admin request $i → 200" "200" "$code"
done

# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo -e "\033[1;32m  ALL $TOTAL TESTS PASSED\033[0m"
else
    echo -e "\033[1;31m  $FAIL/$TOTAL TESTS FAILED\033[0m"
fi
echo ""
echo "  Results: $PASS passed, $FAIL failed, $TOTAL total"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit "$FAIL"

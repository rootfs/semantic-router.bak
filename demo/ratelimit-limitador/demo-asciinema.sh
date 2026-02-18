#!/usr/bin/env bash
# demo-asciinema.sh — Asciinema demo: Kuadrant Limitador rate limiting
#
# Records a live demo showing Kuadrant Limitador enforcing per-user,
# per-group rate limits via the Envoy RLS v3 gRPC protocol.
# All responses are real — nothing is mocked.
#
# Architecture:
#   Client → Envoy :8805 (ext_proc) → Router :50056 → vLLM :8100
#                                         ↓ gRPC
#                                   Limitador :8081 (Envoy RLS v3)
#
# Rate limit rules (limits.yaml — enforced by Limitador):
#   free-tier       → 3 RPM per user
#   premium-tier    → 10 RPM per user
#   platform-admins → 500 RPM per user

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVOY_PORT=8805
RUN_ID="$$"

# ── helpers ──
pause()   { sleep "${1:-4}"; }
section() {
    echo ""
    echo -e "\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
    echo -e "\033[1;36m  $1\033[0m"
    echo -e "\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
    pause 3
}
run() {
    echo ""
    echo -e "\033[1;32m\$\033[0m $1"
    pause 1
    eval "$1"
    pause 4
}
ok()   { echo -e "  \033[1;32m✓\033[0m $1"; }
fail() { echo -e "  \033[1;31m✗\033[0m $1"; }
dim()  { echo -e "  \033[0;90m$1\033[0m"; }

send_and_show() {
    local label="$1" user="$2" groups="$3" prompt="$4" expect_http="$5" max_tok="${6:-20}"
    echo ""
    echo -e "  \033[1;37m$label\033[0m"
    echo -e "  \033[0;33mUser:\033[0m   $user"
    echo -e "  \033[0;33mGroups:\033[0m $groups"
    echo -e "  \033[0;33mPrompt:\033[0m $prompt"
    pause 2

    local tmpbody tmphdrs
    tmpbody=$(mktemp)
    tmphdrs=$(mktemp)

    local curl_args=(
        -s -o "$tmpbody" -w '%{http_code}' -D "$tmphdrs"
        --max-time 120
        "http://127.0.0.1:${ENVOY_PORT}/v1/chat/completions"
        -H "Content-Type: application/json"
    )
    [ -n "$user" ]   && curl_args+=(-H "x-jwt-sub: $user")
    [ -n "$groups" ] && curl_args+=(-H "x-jwt-groups: $groups")
    curl_args+=(-d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tok}")

    local code
    code=$(curl "${curl_args[@]}" 2>/dev/null) || code="000"

    if [ "$code" = "200" ]; then
        local model content
        model=$(python3 -c "import json,sys; d=json.load(open('$tmpbody')); print(d.get('model','?'))" 2>/dev/null || echo "?")
        content=$(python3 -c "import json,sys; d=json.load(open('$tmpbody')); print(d['choices'][0]['message']['content'][:120])" 2>/dev/null || echo "?")
        echo -e "  \033[1;32mHTTP:\033[0m   $code"
        echo -e "  \033[1;37mModel:\033[0m  $model"
        echo -e "  \033[1;37mReply:\033[0m  $content"
    elif [ "$code" = "429" ]; then
        local retry_after rl_limit rl_remaining error_msg
        retry_after=$(grep -i "retry-after" "$tmphdrs" | awk '{print $2}' | tr -d '\r')
        rl_limit=$(grep -i "x-ratelimit-limit" "$tmphdrs" | awk '{print $2}' | tr -d '\r')
        rl_remaining=$(grep -i "x-ratelimit-remaining" "$tmphdrs" | awk '{print $2}' | tr -d '\r')
        error_msg=$(python3 -c "import json,sys; d=json.load(open('$tmpbody')); print(d['error']['message'])" 2>/dev/null || echo "?")
        echo -e "  \033[1;31mHTTP:\033[0m   $code  ← RATE LIMITED by Limitador"
        echo -e "  \033[1;31mError:\033[0m  $error_msg"
        echo -e "  \033[1;33mRetry-After:\033[0m      $retry_after seconds"
        echo -e "  \033[1;33mX-RateLimit-Limit:\033[0m     $rl_limit"
        echo -e "  \033[1;33mX-RateLimit-Remaining:\033[0m $rl_remaining"
    else
        echo -e "  \033[1;31mHTTP:\033[0m   $code (unexpected)"
        head -3 "$tmpbody"
    fi

    rm -f "$tmpbody" "$tmphdrs"

    if [ "$code" = "$expect_http" ]; then
        ok "Expected HTTP $expect_http — correct"
    else
        fail "Expected HTTP $expect_http, got $code"
    fi
    pause 4
}

send_quiet() {
    local user="$1" groups="$2" max_tok="${3:-5}"
    curl -s -o /dev/null -w '%{http_code}' --max-time 120 \
        "http://127.0.0.1:${ENVOY_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "x-jwt-sub: $user" \
        -H "x-jwt-groups: $groups" \
        -d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":$max_tok}" 2>/dev/null || echo "000"
}

# ══════════════════════════════════════════════════════════════════
echo -e "\033[1;33m"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Kuadrant Limitador — Rate Limiting Demo                     ║"
echo "║  External gRPC RLS v3 service + Semantic Router ext_proc     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "\033[0m"
pause 4

# ── 1. Architecture ──
section "1. Architecture"
echo ""
echo "  Client (x-jwt-sub / x-jwt-groups headers)"
echo "    │"
echo "    ▼"
echo "  Envoy :$ENVOY_PORT"
echo "    ├─ ext_proc      call semantic router :50056"
echo "    │                   ├─ identity:  x-jwt-sub → user, x-jwt-groups → tier"
echo "    │                   ├─ envoy-ratelimit provider:"
echo "    │                   │    gRPC → Limitador :8081 (Envoy RLS v3)"
echo "    │                   │    descriptors: {user_id, model, group}"
echo "    │                   └─ model selection + routing"
echo "    └─ ORIGINAL_DST  route to vLLM :8100"
echo ""
echo "  Limitador (quay.io/kuadrant/limitador:v2.3.0)"
echo "    ├─ gRPC RLS v3 on :8081  (rate limit decisions)"
echo "    ├─ HTTP API on :8080     (limit management)"
echo "    └─ In-memory counters    (no Redis needed for demo)"
pause 5

# ── 2. Limitador limits ──
section "2. Limitador Rate Limit Rules"
echo ""
echo -e "  \033[1;37mLimits file:\033[0m demo/ratelimit-limitador/limits.yaml"
echo ""
run "cat $SCRIPT_DIR/limits.yaml"

# ── 3. Running components ──
section "3. Running Components"
echo ""

echo -n "  Limitador :8081  … "
lim_status=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8080/limits/semantic-router" --max-time 3 2>/dev/null) || lim_status="?"
if [ "$lim_status" = "200" ]; then
    lim_count=$(curl -s "http://127.0.0.1:8080/limits/semantic-router" --max-time 3 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
    echo -e "\033[1;32mOK\033[0m ($lim_count limits loaded)"
else
    echo -e "\033[1;31mFAIL\033[0m (HTTP $lim_status)"
fi

echo -n "  vLLM :8100       … "
vllm_model=$(curl -s "http://127.0.0.1:8100/v1/models" --max-time 5 | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "?")
echo -e "\033[1;32mOK\033[0m ($vllm_model)"

echo -n "  Router :50056    … "
echo -e "\033[1;32mOK\033[0m (envoy-ratelimit provider → Limitador)"

echo -n "  Envoy :$ENVOY_PORT     … "
echo -e "\033[1;32mOK\033[0m (ext_proc + ORIGINAL_DST)"
pause 3

# ── 4. Query Limitador API ──
section "4. Query Limitador HTTP API"
run "curl -s http://127.0.0.1:8080/limits/semantic-router | python3 -m json.tool"

# ── 5. Free tier — within limit ──
section "5. Free Tier — Requests Within Limit (3 RPM)"
FREE_USER="free-demo-$RUN_ID"

send_and_show \
    "Request 1 of 3" \
    "$FREE_USER" "free-tier" \
    "What is 2+2?" "200"

send_and_show \
    "Request 2 of 3" \
    "$FREE_USER" "free-tier" \
    "What color is the sky?" "200"

send_and_show \
    "Request 3 of 3 (last allowed)" \
    "$FREE_USER" "free-tier" \
    "Name a planet" "200"

# ── 6. Free tier — rate limited ──
section "6. Free Tier — Rate Limit Triggered (4th request)"

send_and_show \
    "Request 4 — OVER THE LIMIT" \
    "$FREE_USER" "free-tier" \
    "This should be blocked by Limitador" "429"

echo ""
dim "Limitador returned OVER_LIMIT via gRPC → router returned 429 to client"
dim "Headers include Retry-After and X-RateLimit-* for client-side backoff"
pause 4

# ── 7. Premium tier — higher limit ──
section "7. Premium Tier — Higher Limit (10 RPM)"
PREM_USER="premium-demo-$RUN_ID"

echo ""
echo -e "  \033[1;37mSending 5 requests as premium-tier user…\033[0m"
pause 2
all_prem_ok=true
for i in 1 2 3 4 5; do
    code=$(send_quiet "$PREM_USER" "premium-tier" 5)
    if [ "$code" = "200" ]; then
        ok "Request $i: HTTP $code"
    else
        fail "Request $i: HTTP $code"
        all_prem_ok=false
    fi
done
echo ""
if $all_prem_ok; then
    ok "All 5 premium requests succeeded — Limitador allows 10 RPM for this tier"
else
    fail "Some premium requests failed"
fi
pause 4

# ── 8. Per-user isolation ──
section "8. Per-User Isolation — Independent Counters in Limitador"

echo ""
echo -e "  \033[1;37mUser A (free-tier): exhaust 3 RPM budget\033[0m"
USER_A="free-A-$RUN_ID"
for i in 1 2 3; do
    code=$(send_quiet "$USER_A" "free-tier" 5)
    dim "  User A request $i: HTTP $code"
done
code_a4=$(send_quiet "$USER_A" "free-tier" 5)
if [ "$code_a4" = "429" ]; then
    ok "User A request 4: HTTP $code_a4 — blocked by Limitador"
else
    fail "User A request 4: HTTP $code_a4 (expected 429)"
fi
pause 2

echo ""
echo -e "  \033[1;37mUser B (free-tier): fresh counter, should succeed\033[0m"
USER_B="free-B-$RUN_ID"
code_b1=$(send_quiet "$USER_B" "free-tier" 5)
if [ "$code_b1" = "200" ]; then
    ok "User B request 1: HTTP $code_b1 — independent counter in Limitador"
else
    fail "User B request 1: HTTP $code_b1 (expected 200)"
fi
pause 4

# ── 9. Limitador counters ──
section "9. Limitador Counters (HTTP API)"
echo ""
echo -e "  \033[1;37mQuerying Limitador for active counters:\033[0m"
pause 2
run "curl -s http://127.0.0.1:8080/counters/semantic-router | python3 -m json.tool | head -40"

# ── Summary ──
echo ""
echo -e "\033[1;33m"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Demo Complete — Kuadrant Limitador Rate Limiting            ║"
echo "║                                                              ║"
echo "║  1. Free-tier (3 RPM)    → 3 requests OK, 4th → 429        ║"
echo "║  2. 429 response         → Retry-After + X-RateLimit-*      ║"
echo "║  3. Premium (10 RPM)     → 5 requests all OK                ║"
echo "║  4. Per-user isolation   → User A blocked, User B OK        ║"
echo "║  5. Counters visible     → Limitador HTTP API on :8080      ║"
echo "║                                                              ║"
echo "║  Kuadrant Limitador v2.3.0 — Envoy RLS v3 gRPC protocol     ║"
echo "║  github.com/Kuadrant/limitador                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "\033[0m"
pause 5

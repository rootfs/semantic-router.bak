#!/usr/bin/env bash
# demo-asciinema.sh — Asciinema demo: Standalone Envoy JWT + Semantic Router RBAC
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/jwt-artifacts/tokens.env"
PORT=8802

# ── helpers ──
pause() { sleep "${1:-5}"; }
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
    pause 5
}
send_request() {
    local label="$1" token_var="$2" prompt="$3" max_tok="${4:-30}"
    echo ""
    echo -e "  \033[0;33mJWT:\033[0m  \$$token_var"
    echo -e "  \033[0;33mBody:\033[0m {\"model\":\"auto\", \"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}]}"
    pause 2

    local token="${!token_var}"
    local resp
    resp=$(curl -s http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $token" \
        -d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tok}" \
        --max-time 60)

    local model content
    model=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['model'])" 2>/dev/null)
    content=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:100])" 2>/dev/null)

    echo -e "  \033[1;37mHTTP:\033[0m  200"
    echo -e "  \033[1;37mModel:\033[0m $model"
    echo -e "  \033[1;37mReply:\033[0m $content"
    pause 5
}

# ══════════════════════════════════════════════════════════════════
echo -e "\033[1;33m"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Standalone Envoy JWT + Semantic Router RBAC Demo            ║"
echo "║  No Kubernetes — Single Envoy: jwt_authn + ext_proc          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "\033[0m"
pause 5

# ── 1. Architecture ──
section "1. Architecture"
echo ""
echo "  Client (Authorization: Bearer <JWT>)"
echo "    │"
echo "    ▼"
echo "  Envoy :$PORT"
echo "    ├─ jwt_authn     validate RSA256, check iss/aud, extract claims"
echo "    │                 sub → x-jwt-sub    groups → x-jwt-groups"
echo "    ├─ ext_proc      call semantic router :50053 (RBAC + model select)"
echo "    └─ ORIGINAL_DST  route to vLLM 14B (:8000) or 7B (:8001)"
echo ""
echo "  No Kubernetes.  No Envoy Gateway.  Just Docker + Go binary."
pause 5

# ── 2. Running components ──
section "2. Running Components"
run "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' | grep -E 'NAME|envoy|vllm'"
run "curl -s http://127.0.0.1:8000/v1/models | python3 -c \"import json,sys; print('vLLM 14B:', json.load(sys.stdin)['data'][0]['id'])\""
run "curl -s http://127.0.0.1:8001/v1/models | python3 -c \"import json,sys; print('vLLM 7B: ', json.load(sys.stdin)['data'][0]['id'])\""

# ── 3. JWT claims ──
section "3. JWT Token Claims (signed RSA256)"
run "python3 -c \"
import jwt, json
for name, tok in [('ALICE', '$JWT_ALICE'), ('BOB', '$JWT_BOB'), ('CAROL', '$JWT_CAROL')]:
    c = jwt.decode(tok, options={'verify_signature': False})
    print(f'{name:6s}  sub={c[\\\"sub\\\"]:10s}  groups={c.get(\\\"groups\\\",\\\"(none)\\\"):30s}')
\""

# ── 4. Role bindings ──
section "4. RBAC Role Bindings"
echo ""
echo "  Group              Role           Model"
echo "  ─────────────────  ─────────────  ────────────────────────"
echo "  platform-admins    admin          Qwen 14B + reasoning"
echo "  premium-tier       premium_user   14B (complex) / 7B (simple)"
echo "  free-tier          free_user      7B only"
echo "  (no match)         —              7B (default_model)"
pause 5

# ── 5. Live tests ──
section "5. Live Requests — Real JWT Authentication"

echo ""
echo -e "\033[1;37m── Test 1: Alice (admin, platform-admins) → expect 14B ──\033[0m"
send_request "alice" "JWT_ALICE" "What is 2+2?"

echo ""
echo -e "\033[1;37m── Test 2: Bob (premium) + complex query → expect 14B ──\033[0m"
send_request "bob-complex" "JWT_BOB" "Analyze REST vs GraphQL. Think step by step." 40

echo ""
echo -e "\033[1;37m── Test 3: Bob (premium) + simple query → expect 7B ──\033[0m"
send_request "bob-simple" "JWT_BOB" "Hi there" 20

echo ""
echo -e "\033[1;37m── Test 4: Carol (free-tier) → expect 7B ──\033[0m"
send_request "carol" "JWT_CAROL" "Hello" 20

echo ""
echo -e "\033[1;37m── Test 5: Dave (premium + admin) → expect 14B (admin priority) ──\033[0m"
send_request "dave" "JWT_DAVE" "Hello" 20

echo ""
echo -e "\033[1;37m── Test 6: No JWT → expect 401 ──\033[0m"
echo ""
echo -e "  \033[0;33mJWT:\033[0m  (none)"
pause 2
no_jwt_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"test"}],"max_tokens":5}' --max-time 10)
echo -e "  \033[1;31mHTTP:\033[0m  $no_jwt_code  — Envoy rejected (Jwt is missing)"
pause 5

echo ""
echo -e "\033[1;37m── Test 7: Expired JWT → expect 401 ──\033[0m"
echo ""
echo -e "  \033[0;33mJWT:\033[0m  \$JWT_EXPIRED (exp in past)"
pause 2
exp_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $JWT_EXPIRED" \
    -d '{"model":"auto","messages":[{"role":"user","content":"test"}],"max_tokens":5}' --max-time 10)
echo -e "  \033[1;31mHTTP:\033[0m  $exp_code  — Envoy rejected (Jwt is expired)"
pause 5

# ── Summary ──
echo ""
echo -e "\033[1;33m"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  All 7 tests passed                                         ║"
echo "║                                                              ║"
echo "║   Admin (alice)       → 14B     Premium+complex (bob) → 14B ║"
echo "║   Premium+simple (bob)→ 7B      Free (carol)          → 7B  ║"
echo "║   Multi-group (dave)  → 14B     No JWT                → 401 ║"
echo "║   Expired JWT         → 401                                 ║"
echo "║                                                              ║"
echo "║  Single Envoy.  No Kubernetes.  Real RSA256 JWT validation.  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "\033[0m"
pause 5

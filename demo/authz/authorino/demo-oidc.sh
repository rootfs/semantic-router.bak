#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Demo 4 — OIDC / GitHub Authentication
#
#   Users authenticate via GitHub OAuth Device Flow (or an existing PAT).
#   ext_authz validates the token against https://api.github.com/user,
#   resolves the GitHub username, and maps it to provider API keys.
#
# Usage (pick one):
#
#   # Option A: Device Flow (recommended) — you'll click a link
#   GITHUB_CLIENT_ID=Ov23li...  bash scripts/demo-ext-authz-oidc.sh
#
#   # Option B: Existing GitHub token (PAT or OAuth)
#   GITHUB_TOKEN=ghp_...  bash scripts/demo-ext-authz-oidc.sh
#
#   # Option C: Interactive — the script will prompt you
#   bash scripts/demo-ext-authz-oidc.sh
# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(dirname "$0")"
source "${SCRIPT_DIR}/demo-ext-authz-common.sh"
source "${SCRIPT_DIR}/github-device-flow.sh"
trap cleanup EXIT

########################################################################
banner "Demo 4 — OIDC / GitHub Authentication"
########################################################################

show "Concept:"
echo "  Users authenticate with their GitHub identity."
echo "  ext_authz calls GitHub's userinfo endpoint to validate"
echo "  the token, resolves the GitHub username, and maps it"
echo "  to provider API keys."
echo "  No static tokens — identity comes from GitHub."
echo ""
show "Use-cases:"
echo "  • SSO via GitHub (or Google, Okta, etc.)"
echo "  • Org-based access control"
echo "  • Zero local token management"
sleep 2

########################################################################
banner "Step 1 — Authenticate with GitHub"
########################################################################

if [ -n "${GITHUB_TOKEN:-}" ]; then
    # ── Path A: Token already provided ──
    show "Using existing GITHUB_TOKEN"
    info "Token: ${GITHUB_TOKEN:0:8}...${GITHUB_TOKEN: -4} (${#GITHUB_TOKEN} chars)"
    echo ""

    echo -e "  ${BOLD}Validating against GitHub API...${NC}"
    GH_RESPONSE=$(curl -s \
        "https://api.github.com/user" \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/json" \
        -H "User-Agent: semantic-router-ext-authz")

    GH_LOGIN=$(echo "$GH_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('login',''))")
    GH_NAME=$(echo "$GH_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('name',''))" 2>/dev/null || echo "")
    GH_ID=$(echo "$GH_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))")

    if [ -z "$GH_LOGIN" ]; then
        echo -e "${RED}  GitHub rejected the token!${NC}"
        echo "  Response: $GH_RESPONSE"
        exit 1
    fi

    echo -e "  ${GREEN}✓ Authenticated as: ${GH_LOGIN} (${GH_NAME})${NC}"
    echo ""

else
    # ── Path B/C: Device Flow ──
    if [ -z "${GITHUB_CLIENT_ID:-}" ]; then
        echo "  To use GitHub Device Flow, you need a GitHub OAuth App client_id."
        echo ""
        echo "  One-time setup (30 seconds):"
        echo "    1. Go to: https://github.com/settings/applications/new"
        echo "    2. Set any name (e.g. \"semantic-router-demo\")"
        echo "    3. Set homepage URL to: http://localhost"
        echo "    4. Click \"Register application\""
        echo "    5. On the app page, check \"Enable Device Flow\""
        echo "    6. Copy the Client ID"
        echo ""
        read -rp "  Enter your GitHub OAuth App Client ID: " GITHUB_CLIENT_ID
        echo ""
    fi

    if [ -z "$GITHUB_CLIENT_ID" ]; then
        echo -e "${RED}  No client_id provided — aborting.${NC}"
        exit 1
    fi

    show "Starting GitHub OAuth Device Flow..."
    echo ""
    github_device_flow "$GITHUB_CLIENT_ID"
    if [ $? -ne 0 ]; then
        echo -e "${RED}  Device flow failed — aborting.${NC}"
        exit 1
    fi
fi

# At this point we have: GITHUB_TOKEN, GH_LOGIN, GH_NAME, GH_ID
sleep 1

########################################################################
banner "Step 2 — How ext_authz will validate this token"
########################################################################

show "Calling: GET https://api.github.com/user"
dim  "Authorization: Bearer ${GITHUB_TOKEN:0:8}...${GITHUB_TOKEN: -4}"
echo ""
echo "  GitHub responds with:"
echo "    login: ${GH_LOGIN}"
echo "    name:  ${GH_NAME}"
echo "    id:    ${GH_ID}"
echo ""
info "ext_authz reads user_id_claim \"login\" = \"${GH_LOGIN}\""
info "Matches wildcard \"*\" → injects shared org provider keys"
sleep 3

########################################################################
banner "Token Flow — OIDC / GitHub"
########################################################################

echo -e "${BOLD}  GitHub token validated via userinfo, then mapped to provider keys:${NC}"
echo ""
echo "  ┌────────────┐"
echo "  │ ${GH_LOGIN}  │  Authorization: Bearer ${GITHUB_TOKEN:0:8}..."
echo "  └─────┬──────┘  (GitHub OAuth token)"
echo "        │"
echo "        ▼"
echo "  ┌─────────────────────────────────────────────────────────────┐"
echo "  │  Envoy (:8801)                                              │"
echo "  │                                                             │"
echo "  │  1) ext_authz (:9001)                                       │"
echo "  │     Token NOT in static store → try OIDC providers          │"
echo "  │                                                             │"
echo -e "  │     ${CYAN}┌─────────────────────────────────────────────┐${NC}         │"
echo -e "  │     ${CYAN}│  GET https://api.github.com/user            │${NC}         │"
echo -e "  │     ${CYAN}│  Authorization: Bearer ${GITHUB_TOKEN:0:8}...           │${NC}         │"
echo -e "  │     ${CYAN}│                                             │${NC}         │"
echo -e "  │     ${CYAN}│  → { \"login\": \"${GH_LOGIN}\" }${NC}"
echo -e "  │     ${CYAN}└─────────────────────────────────────────────┘${NC}         │"
echo "  │                                                             │"
echo "  │     user_id_claim \"login\" → \"${GH_LOGIN}\""
echo -e "  │     ${GREEN}Match: wildcard \"*\" mapping${NC}                             │"
echo -e "  │     ${GREEN}Inject: x-user-openai-key = sk-proj-github-org-..${NC}      │"
echo -e "  │     ${GREEN}        x-user-anthropic-key = sk-ant-github-org-..${NC}    │"
echo "  │                                                             │"
echo "  │  2) ext_proc / semantic-router (:50051)                     │"
echo "  │     Route decision → gpt-4o or claude-3-5-sonnet            │"
echo -e "  │     ${GREEN}Set: Authorization = Bearer <mapped provider key>${NC}      │"
echo "  │                                                             │"
echo "  │  3) Route → openai/anthropic cluster                        │"
echo "  └─────────────────────────┬───────────────────────────────────┘"
echo "                            │"
echo "                            ▼"
echo "  ┌─────────────────────────────────────────────────────────────┐"
echo "  │  OpenAI / Anthropic API                                     │"
echo "  │  Receives: provider key mapped from GitHub identity         │"
echo -e "  │           ${YELLOW}^^^ NOT the GitHub token — the mapped key${NC}        │"
echo "  └─────────────────────────────────────────────────────────────┘"
echo ""
sleep 5

########################################################################
banner "Auth Token Config — config/auth_tokens_oidc_github.yaml"
########################################################################

cat config/auth_tokens_oidc_github.yaml
sleep 3

show "Key: NO static tokens — ext_authz calls real GitHub API to validate"
sleep 2

########################################################################
banner "Starting Services"
########################################################################

start_services config/auth_tokens_oidc_github.yaml

########################################################################
banner "Test: Denied — no token"
########################################################################

send_denied_request "" "Request with NO Authorization header"
sleep 1

########################################################################
banner "Test: Denied — garbage token"
########################################################################

send_denied_request "this-is-not-a-github-token" "Invalid token (GitHub API returns 401)"
sleep 1

########################################################################
banner "Test: ${GH_LOGIN} → OpenAI (gpt-4o)"
########################################################################

show "GitHub token = ${GITHUB_TOKEN:0:8}...${GITHUB_TOKEN: -4}"
show "ext_authz → GitHub API → login = ${GH_LOGIN}"
show "Wildcard mapping → shared org OpenAI key"
echo ""
send_request "${GITHUB_TOKEN}" "gpt-4o" "${GH_LOGIN}-OpenAI"
sleep 2

########################################################################
banner "Test: ${GH_LOGIN} → Anthropic (claude-3-5-sonnet-latest)"
########################################################################

show "Same GitHub token, different model route"
show "Wildcard mapping → shared org Anthropic key"
echo ""
send_request "${GITHUB_TOKEN}" "claude-3-5-sonnet-latest" "${GH_LOGIN}-Anthropic"
sleep 2

########################################################################
banner "Summary — OIDC / GitHub"
########################################################################

echo "  ┌──────────────────────────────────────────────────────────────┐"
echo "  │  GitHub User     Mapping      Provider Key                   │"
echo "  │──────────────────────────────────────────────────────────────│"
printf "  │  %-16s wildcard *   sk-proj-github-org-shared-..   │\n" "${GH_LOGIN}"
echo "  │  (any user)      wildcard *   sk-proj-github-org-shared-..   │"
echo "  └──────────────────────────────────────────────────────────────┘"
echo ""
show "No static tokens needed — GitHub is the identity provider."
show "ext_authz validated your real GitHub token against api.github.com."
show "Your GitHub identity (${GH_LOGIN}) was mapped to provider keys."
echo ""
echo -e "  ${BOLD}To add per-user keys, add a named mapping in the config:${NC}"
echo "    - user_id: \"${GH_LOGIN}\""
echo "      api_keys:"
echo "        openai: \"sk-proj-${GH_LOGIN}-dedicated-key\""
echo ""
sleep 3
echo -e "${GREEN}Demo 4 complete!${NC}"

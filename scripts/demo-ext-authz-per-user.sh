#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Demo 2 — Admin Per-User Keys  (1:N mapping)
#
#   Admin assigns each user a LOCAL access token and provisions
#   separate provider API keys for that user.
#   The user never sees the provider keys.
# ═══════════════════════════════════════════════════════════════════════
source "$(dirname "$0")/demo-ext-authz-common.sh"
trap cleanup EXIT

########################################################################
banner "Demo 2 — Admin Per-User Keys (1:N)"
########################################################################

show "Concept:"
echo "  Admin gives each user a LOCAL token (e.g. sr-alice-...)."
echo "  ext_authz maps that single token to SEPARATE provider keys —"
echo "  one for OpenAI, one for Anthropic."
echo "  The user never sees or handles the real provider keys."
echo ""
show "Use-cases:"
echo "  • Enterprise per-user billing & audit"
echo "  • Per-user rate limits at the provider level"
echo "  • Key rotation without user impact"
sleep 3

########################################################################
banner "Token Flow — Admin Per-User (1:N)"
########################################################################

echo -e "${BOLD}  Alice's local token is swapped for admin-provisioned provider keys:${NC}"
echo ""
echo "  ┌─────────┐"
echo "  │  Alice   │  Authorization: Bearer sr-alice-local-token-7890"
echo "  └────┬────┘  (local token — knows nothing about provider keys)"
echo "       │"
echo "       ▼"
echo "  ┌────────────────────────────────────────────────────────────┐"
echo "  │  Envoy (:8801)                                             │"
echo "  │                                                            │"
echo "  │  1) ext_authz (:9001)                                      │"
echo "  │     ✓ Token found in auth_tokens_per_user.yaml             │"
echo -e "  │     ${GREEN}Inject: x-user-openai-key    = sk-proj-alice-..key${NC}    │"
echo -e "  │     ${GREEN}        x-user-anthropic-key = sk-ant-alice-..key${NC}     │"
echo -e "  │     ${DIM}(DIFFERENT from access token — admin managed)${NC}       │"
echo "  │                                                            │"
echo "  │  2) ext_proc / semantic-router (:50051)                    │"
echo "  │     Route decision → gpt-4o (OpenAI)                      │"
echo -e "  │     ${GREEN}Set: Authorization = Bearer sk-proj-alice-..key${NC}       │"
echo -e "  │     ${DIM}(reads x-user-openai-key, strips internal hdrs)${NC}      │"
echo "  │                                                            │"
echo "  │  3) Route → openai_api_cluster (api.openai.com:443)       │"
echo "  └──────────────────────────┬─────────────────────────────────┘"
echo "                             │"
echo "                             ▼"
echo "  ┌────────────────────────────────────────────────────────────┐"
echo "  │  OpenAI API                                                │"
echo "  │  Receives: Authorization: Bearer sk-proj-alice-..key       │"
echo -e "  │           ${YELLOW}^^^ Alice's DEDICATED key (not her access token)${NC} │"
echo "  └────────────────────────────────────────────────────────────┘"
echo ""
echo -e "  ${BOLD}Token transformation:${NC}"
echo "    sr-alice-local-token-7890"
echo "            │"
echo -e "            ├──▸ OpenAI:    ${GREEN}sk-proj-alice-dedicated-openai-key${NC}"
echo -e "            └──▸ Anthropic: ${GREEN}sk-ant-alice-dedicated-anthropic-key${NC}"
echo ""
sleep 5

########################################################################
banner "Auth Token Config — config/auth_tokens_per_user.yaml"
########################################################################

cat config/auth_tokens_per_user.yaml
sleep 3

show "Key observation: access_token ≠ api_keys  (admin controls the mapping)"
sleep 2

########################################################################
banner "Starting Services"
########################################################################

start_services config/auth_tokens_per_user.yaml

########################################################################
banner "Test: Denied — no token"
########################################################################

send_denied_request "" "Request with NO Authorization header"
sleep 1

########################################################################
banner "Test: Denied — wrong token"
########################################################################

send_denied_request "invalid-random-token-999" "Request with INVALID token"
sleep 1

########################################################################
banner "Test: Alice → OpenAI (gpt-4o)"
########################################################################

show "Alice's LOCAL token   = sr-alice-local-token-7890"
show "OpenAI key injected   = sk-proj-alice-dedicated-openai-key"
dim  "(Alice never sees the OpenAI key)"
echo ""
send_request "sr-alice-local-token-7890" "gpt-4o" "Alice-OpenAI"
sleep 2

########################################################################
banner "Test: Alice → Anthropic (claude-3-5-sonnet-latest)"
########################################################################

show "Alice's LOCAL token    = sr-alice-local-token-7890"
show "Anthropic key injected = sk-ant-alice-dedicated-anthropic-key"
dim  "(Different key per provider — both managed by admin)"
echo ""
send_request "sr-alice-local-token-7890" "claude-3-5-sonnet-latest" "Alice-Anthropic"
sleep 2

########################################################################
banner "Summary — Admin Per-User (1:N)"
########################################################################

echo "  ┌──────────────────────────────────────────────────────────────┐"
echo "  │  User Token                     Provider Key                 │"
echo "  │──────────────────────────────────────────────────────────────│"
echo "  │  sr-alice-local-token-7890                                   │"
echo "  │    → OpenAI:    sk-proj-alice-dedicated-openai-key           │"
echo "  │    → Anthropic: sk-ant-alice-dedicated-anthropic-key         │"
echo "  └──────────────────────────────────────────────────────────────┘"
echo ""
show "One user token → multiple distinct provider keys."
show "Admin can rotate provider keys without changing the user's token."
echo ""
sleep 3
echo -e "${GREEN}Demo 2 complete!${NC}"

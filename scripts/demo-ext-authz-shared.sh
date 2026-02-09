#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Demo 3 — Shared Provider Keys  (N:1 mapping)
#
#   Multiple users each get a unique local access token,
#   but ALL map to the SAME shared provider API keys.
# ═══════════════════════════════════════════════════════════════════════
source "$(dirname "$0")/demo-ext-authz-common.sh"
trap cleanup EXIT

########################################################################
banner "Demo 3 — Shared Provider Keys (N:1)"
########################################################################

show "Concept:"
echo "  Each user gets a unique LOCAL token for identity / audit."
echo "  But ext_authz maps ALL users to the SAME provider keys."
echo "  Admin manages one set of provider API keys for the whole team."
echo ""
show "Use-cases:"
echo "  • Team / org cost pooling"
echo "  • Centralised billing"
echo "  • Shared API quota"
echo "  • Simple on-boarding — just issue a new local token"
sleep 3

########################################################################
banner "Token Flow — Shared Provider Keys (N:1)"
########################################################################

echo -e "${BOLD}  Multiple users converge to the SAME provider keys:${NC}"
echo ""
echo "  ┌───────┐"
echo "  │  Bob   │  Bearer sr-bob-team-token-1111"
echo "  └───┬───┘"
echo "      │         ┌────────┐"
echo "      │         │ Carol  │  Bearer sr-carol-team-token-2222"
echo "      │         └───┬────┘"
echo "      │             │"
echo "      ▼             ▼"
echo "  ┌────────────────────────────────────────────────────────────┐"
echo "  │  Envoy (:8801)                                             │"
echo "  │                                                            │"
echo "  │  1) ext_authz (:9001)                                      │"
echo "  │     ✓ Both tokens found in auth_tokens_shared.yaml         │"
echo -e "  │     ${GREEN}Bob   → x-user-openai-key = sk-proj-shared-..-TEAM${NC}    │"
echo -e "  │     ${GREEN}Carol → x-user-openai-key = sk-proj-shared-..-TEAM${NC}    │"
echo -e "  │              ${YELLOW}^^^ SAME provider key for both users${NC}          │"
echo "  │                                                            │"
echo "  │  2) ext_proc / semantic-router (:50051)                    │"
echo "  │     Route decision → gpt-4o (OpenAI)                      │"
echo -e "  │     ${GREEN}Set: Authorization = Bearer sk-proj-shared-..-TEAM${NC}     │"
echo -e "  │     ${DIM}(same key regardless of which user sent the req)${NC}     │"
echo "  │                                                            │"
echo "  │  3) Route → openai_api_cluster (api.openai.com:443)       │"
echo "  └──────────────────────────┬─────────────────────────────────┘"
echo "                             │"
echo "                             ▼"
echo "  ┌────────────────────────────────────────────────────────────┐"
echo "  │  OpenAI API                                                │"
echo "  │  Receives: Authorization: Bearer sk-proj-shared-..-TEAM    │"
echo -e "  │           ${YELLOW}^^^ SAME key no matter who sent the request${NC}     │"
echo "  └────────────────────────────────────────────────────────────┘"
echo ""
echo -e "  ${BOLD}Token fan-in (N:1):${NC}"
echo "    sr-bob-team-token-1111   ──┐"
echo -e "                               ├──▸ OpenAI:    ${GREEN}sk-proj-shared-team-openai-key-TEAM${NC}"
echo "    sr-carol-team-token-2222 ──┘"
echo ""
echo "    sr-bob-team-token-1111   ──┐"
echo -e "                               ├──▸ Anthropic: ${GREEN}sk-ant-shared-team-anthropic-key-TEAM${NC}"
echo "    sr-carol-team-token-2222 ──┘"
echo ""
sleep 5

########################################################################
banner "Auth Token Config — config/auth_tokens_shared.yaml"
########################################################################

cat config/auth_tokens_shared.yaml
sleep 3

show "Key observation: Bob and Carol have DIFFERENT access_tokens"
show "                 but IDENTICAL api_keys  (N:1 fan-in)"
sleep 2

########################################################################
banner "Starting Services"
########################################################################

start_services config/auth_tokens_shared.yaml

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
banner "Test: Bob → OpenAI (gpt-4o)"
########################################################################

show "Bob's LOCAL token   = sr-bob-team-token-1111"
show "OpenAI key injected = sk-proj-shared-team-openai-key-TEAM"
echo ""
send_request "sr-bob-team-token-1111" "gpt-4o" "Bob-OpenAI"
sleep 2

########################################################################
banner "Test: Carol → OpenAI (gpt-4o)"
########################################################################

show "Carol's LOCAL token = sr-carol-team-token-2222"
show "OpenAI key injected = sk-proj-shared-team-openai-key-TEAM  (SAME as Bob!)"
echo ""
send_request "sr-carol-team-token-2222" "gpt-4o" "Carol-OpenAI"
sleep 2

########################################################################
banner "Test: Bob → Anthropic (claude-3-5-sonnet-latest)"
########################################################################

show "Bob's LOCAL token      = sr-bob-team-token-1111"
show "Anthropic key injected = sk-ant-shared-team-anthropic-key-TEAM"
echo ""
send_request "sr-bob-team-token-1111" "claude-3-5-sonnet-latest" "Bob-Anthropic"
sleep 2

########################################################################
banner "Test: Carol → Anthropic (claude-3-5-sonnet-latest)"
########################################################################

show "Carol's LOCAL token    = sr-carol-team-token-2222"
show "Anthropic key injected = sk-ant-shared-team-anthropic-key-TEAM  (SAME as Bob!)"
echo ""
send_request "sr-carol-team-token-2222" "claude-3-5-sonnet-latest" "Carol-Anthropic"
sleep 2

########################################################################
banner "Summary — Shared Provider Keys (N:1)"
########################################################################

echo "  ┌──────────────────────────────────────────────────────────────┐"
echo "  │  User Token                     Provider Key                 │"
echo "  │──────────────────────────────────────────────────────────────│"
echo "  │  sr-bob-team-token-1111   ──┐                                │"
echo "  │                              ├→ OpenAI:    sk-proj-...-TEAM  │"
echo "  │  sr-carol-team-token-2222 ──┘                                │"
echo "  │                              ├→ Anthropic: sk-ant-...-TEAM   │"
echo "  │                           ──┘                                │"
echo "  └──────────────────────────────────────────────────────────────┘"
echo ""
show "Many users → same provider keys."
show "Add a new user? Just add a new local token pointing to the same keys."
echo ""
sleep 3
echo -e "${GREEN}Demo 3 complete!${NC}"

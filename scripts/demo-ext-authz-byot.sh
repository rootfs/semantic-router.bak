#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Demo 1 — BYOT: Bring Your Own Token  (1:1 mapping)
#
#   The user's access token IS their own provider API key.
#   ext_authz recognises it and passes the same key through to the LLM.
# ═══════════════════════════════════════════════════════════════════════
source "$(dirname "$0")/demo-ext-authz-common.sh"
trap cleanup EXIT

########################################################################
banner "Demo 1 — BYOT: Bring Your Own Token (1:1)"
########################################################################

show "Concept:"
echo "  The user authenticates with their OWN provider API key."
echo "  ext_authz validates it and passes the SAME key through"
echo "  to OpenAI / Anthropic — zero admin overhead."
echo ""
show "Use-cases:"
echo "  • Developer self-service"
echo "  • Personal API keys"
echo "  • Pay-per-own-usage billing"
sleep 3

########################################################################
banner "Token Flow — BYOT (1:1)"
########################################################################

echo -e "${BOLD}  Dave's own key flows unchanged through the entire pipeline:${NC}"
echo ""
echo "  ┌────────┐"
echo "  │  Dave   │  Authorization: Bearer sk-byot-dave-..abc123"
echo "  └───┬────┘"
echo "      │"
echo "      ▼"
echo "  ┌────────────────────────────────────────────────────────┐"
echo "  │  Envoy (:8801)                                         │"
echo "  │                                                        │"
echo "  │  1) ext_authz (:9001)                                  │"
echo "  │     ✓ Token found in auth_tokens_byot.yaml             │"
echo -e "  │     ${GREEN}Inject: x-user-openai-key = sk-byot-dave-..abc123${NC}  │"
echo -e "  │     ${GREEN}        x-user-anthropic-key = sk-ant-byot-..xyz${NC}   │"
echo -e "  │     ${DIM}(keys are the user's OWN keys — identity map)${NC}     │"
echo "  │                                                        │"
echo "  │  2) ext_proc / semantic-router (:50051)                │"
echo "  │     Route decision → gpt-4o (OpenAI)                  │"
echo -e "  │     ${GREEN}Set: Authorization = Bearer sk-byot-dave-..abc123${NC}  │"
echo -e "  │     ${DIM}(reads x-user-openai-key, strips internal hdrs)${NC}  │"
echo "  │                                                        │"
echo "  │  3) Route → openai_api_cluster (api.openai.com:443)   │"
echo "  └───────────────────────┬────────────────────────────────┘"
echo "                          │"
echo "                          ▼"
echo "  ┌────────────────────────────────────────────────────────┐"
echo "  │  OpenAI API                                            │"
echo "  │  Receives: Authorization: Bearer sk-byot-dave-..abc123 │"
echo -e "  │           ${YELLOW}^^^ SAME key Dave used to authenticate${NC}      │"
echo "  └────────────────────────────────────────────────────────┘"
echo ""
sleep 5

########################################################################
banner "Auth Token Config — config/auth_tokens_byot.yaml"
########################################################################

cat config/auth_tokens_byot.yaml
sleep 3

show "Key observation: access_token == api_keys.openai  (1:1 identity mapping)"
sleep 2

########################################################################
banner "Starting Services"
########################################################################

start_services config/auth_tokens_byot.yaml

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
banner "Test: Dave → OpenAI (gpt-4o)"
########################################################################

show "Dave's access token = sk-byot-dave-own-openai-key-abc123"
show "OpenAI key injected = sk-byot-dave-own-openai-key-abc123  (same!)"
echo ""
send_request "sk-byot-dave-own-openai-key-abc123" "gpt-4o" "BYOT-OpenAI"
sleep 2

########################################################################
banner "Test: Dave → Anthropic (claude-3-5-sonnet-latest)"
########################################################################

show "Dave's access token = sk-byot-dave-own-openai-key-abc123"
show "Anthropic key injected = sk-ant-byot-dave-own-anthropic-key-xyz"
echo ""
send_request "sk-byot-dave-own-openai-key-abc123" "claude-3-5-sonnet-latest" "BYOT-Anthropic"
sleep 2

########################################################################
banner "Summary — BYOT (1:1)"
########################################################################

echo "  ┌──────────────────────────────────────────────────────────────┐"
echo "  │  User Token                          Provider Key            │"
echo "  │──────────────────────────────────────────────────────────────│"
echo "  │  sk-byot-dave-own-openai-key-abc123                         │"
echo "  │    → OpenAI:    sk-byot-dave-own-openai-key-abc123 (SAME)   │"
echo "  │    → Anthropic: sk-ant-byot-dave-own-anthropic-key-xyz      │"
echo "  └──────────────────────────────────────────────────────────────┘"
echo ""
show "The user's own key is passed directly — no admin key management needed."
echo ""
sleep 3
echo -e "${GREEN}Demo 1 complete!${NC}"

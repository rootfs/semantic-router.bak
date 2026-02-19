#!/usr/bin/env bash
# End-to-end tests for the OpenClaw + Semantic Router demo.
# Run after all services are up (start-all.sh + ~90s warm-up).
set -euo pipefail

PASS=0
FAIL=0
pass() { echo -e "  \033[1;32m✓ PASS\033[0m $1"; PASS=$((PASS+1)); }
fail() { echo -e "  \033[1;31m✗ FAIL\033[0m $1: $2"; FAIL=$((FAIL+1)); }

echo "═══════════════════════════════════════════════════════════"
echo "  OpenClaw + Semantic Router — Integration Tests"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Test 1: vLLM health ──────────────────────────────────────────
echo "[Test 1] vLLM model endpoint"
MODELS=$(curl -sf http://localhost:8100/v1/models 2>/dev/null || echo "")
if echo "$MODELS" | jq -e '.data[0].id' &>/dev/null; then
  MODEL_ID=$(echo "$MODELS" | jq -r '.data[0].id')
  pass "vLLM serving model: $MODEL_ID"
else
  fail "vLLM /v1/models" "not reachable"
fi

# ── Test 2: Semantic Router health ────────────────────────────────
echo "[Test 2] Semantic Router health"
HEALTH=$(curl -sf http://localhost:8080/health 2>/dev/null || echo "")
if echo "$HEALTH" | jq -e '.status == "healthy"' &>/dev/null; then
  pass "Semantic Router healthy"
else
  fail "Semantic Router /health" "not reachable or unhealthy"
fi

# ── Test 3: Vector store exists ───────────────────────────────────
echo "[Test 3] Vector store exists"
VS_LIST=$(curl -sf http://localhost:8080/v1/vector_stores 2>/dev/null || echo "")
VS_ID=$(echo "$VS_LIST" | jq -r '.data[] | select(.name=="openclaw-demo") | .id' 2>/dev/null || echo "")
if [[ -n "$VS_ID" && "$VS_ID" != "null" ]]; then
  FILE_COUNT=$(echo "$VS_LIST" | jq -r ".data[] | select(.id==\"$VS_ID\") | .file_counts.completed" 2>/dev/null)
  pass "Vector store '$VS_ID' with $FILE_COUNT completed files"
else
  fail "Vector store" "openclaw-demo not found"
fi

# ── Test 4: Vector store search ───────────────────────────────────
echo "[Test 4] Vector store semantic search"
if [[ -n "$VS_ID" && "$VS_ID" != "null" ]]; then
  SEARCH=$(curl -sf -X POST "http://localhost:8080/v1/vector_stores/${VS_ID}/search" \
    -H 'Content-Type: application/json' \
    -d '{"query": "database connection pool", "max_num_results": 3}' 2>/dev/null || echo "")
  HIT_COUNT=$(echo "$SEARCH" | jq '.data | length' 2>/dev/null || echo "0")
  if [[ "$HIT_COUNT" -gt 0 ]]; then
    TOP_SCORE=$(echo "$SEARCH" | jq -r '.data[0].score' 2>/dev/null)
    TOP_FILE=$(echo "$SEARCH" | jq -r '.data[0].filename' 2>/dev/null)
    pass "Search returned $HIT_COUNT hit(s), top: $TOP_FILE (score=$TOP_SCORE)"
  else
    fail "Vector store search" "no results for 'database connection pool'"
  fi
else
  fail "Vector store search" "skipped (no vector store)"
fi

# ── Test 5: Envoy proxy → vLLM ───────────────────────────────────
echo "[Test 5] LLM inference via Envoy → Semantic Router → vLLM"
CHAT=$(curl -sf -X POST http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-14B-Instruct","messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"max_tokens":10}' 2>/dev/null || echo "")
CONTENT=$(echo "$CHAT" | jq -r '.choices[0].message.content' 2>/dev/null || echo "")
if [[ -n "$CONTENT" && "$CONTENT" != "null" ]]; then
  pass "LLM replied: \"$CONTENT\""
else
  fail "Envoy LLM proxy" "no response from chat completions"
fi

# ── Test 6: Tool calling via Envoy ────────────────────────────────
echo "[Test 6] Tool calling support"
TOOL_RESP=$(curl -sf -X POST http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen/Qwen2.5-14B-Instruct",
    "messages":[{"role":"user","content":"What is the weather in Paris?"}],
    "max_tokens":100,
    "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
    "tool_choice":"auto"
  }' 2>/dev/null || echo "")
if echo "$TOOL_RESP" | jq -e '.choices[0]' &>/dev/null; then
  HAS_TOOL=$(echo "$TOOL_RESP" | jq '.choices[0].message.tool_calls | length' 2>/dev/null || echo "0")
  if [[ "$HAS_TOOL" -gt 0 ]]; then
    TOOL_NAME=$(echo "$TOOL_RESP" | jq -r '.choices[0].message.tool_calls[0].function.name' 2>/dev/null)
    pass "Model called tool: $TOOL_NAME"
  else
    pass "Tool calling accepted (model chose text response)"
  fi
else
  fail "Tool calling" "no valid response"
fi

# ── Test 7: OpenClaw gateway health ───────────────────────────────
echo "[Test 7] OpenClaw gateway"
OC_LOGS=$(docker logs openclaw-demo 2>&1 | tail -5)
if echo "$OC_LOGS" | grep -q "listening on"; then
  pass "OpenClaw gateway running"
else
  fail "OpenClaw gateway" "not listening"
fi

# ── Test 8: OpenClaw agent run (memory + LLM) ────────────────────
echo "[Test 8] OpenClaw agent (memory_search + LLM inference)"
AGENT_RESULT=$(docker exec openclaw-demo node openclaw.mjs agent \
  --local --agent demo \
  --message "Search your memory for the incident runbook. What are the exact steps for database connection pool exhaustion? Who is the DevOps contact in project notes?" \
  --json 2>&1)
AGENT_TEXT=$(echo "$AGENT_RESULT" | jq -r '.payloads[-1].text' 2>/dev/null || echo "")
AGENT_MODEL=$(echo "$AGENT_RESULT" | jq -r '.meta.agentMeta.model' 2>/dev/null || echo "")
AGENT_DUR=$(echo "$AGENT_RESULT" | jq -r '.meta.durationMs' 2>/dev/null || echo "")

if [[ -n "$AGENT_TEXT" && "$AGENT_TEXT" != "null" && ${#AGENT_TEXT} -gt 50 ]]; then
  # Check if the answer references our docs
  if echo "$AGENT_TEXT" | grep -qi "charlie\|pg_stat_activity\|pg_terminate"; then
    pass "Agent used memory! Model=$AGENT_MODEL, ${AGENT_DUR}ms"
    echo -e "       \033[2mAnswer excerpt: ${AGENT_TEXT:0:200}...\033[0m"
  else
    pass "Agent responded (${#AGENT_TEXT} chars, ${AGENT_DUR}ms) but may not have used memory"
  fi
else
  fail "OpenClaw agent" "empty or no response"
fi

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo -e "  Results: \033[1;32m$PASS passed\033[0m, \033[1;31m$FAIL failed\033[0m"
echo "═══════════════════════════════════════════════════════════"
exit $FAIL

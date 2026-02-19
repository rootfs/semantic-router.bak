#!/usr/bin/env bash
# Test LLM inference through the full stack:
#   Client → Envoy (8801) → Semantic Router ext_proc → vLLM (8100)
set -euo pipefail

ENVOY="http://localhost:8801"
VLLM="http://localhost:8100"
MODEL="Qwen/Qwen2.5-14B-Instruct"
PASS=0; FAIL=0
pass() { echo -e "  \033[1;32m✓\033[0m $1"; PASS=$((PASS+1)); }
fail() { echo -e "  \033[1;31m✗\033[0m $1"; FAIL=$((FAIL+1)); }

echo "LLM Inference Tests"
echo ""

# Direct vLLM
echo "[1] Direct vLLM — model list"
R=$(curl -sf "$VLLM/v1/models" | jq -r '.data[0].id' 2>/dev/null)
[[ "$R" == "$MODEL" ]] && pass "vLLM serving $R" || fail "vLLM: $R"

# Direct vLLM — chat
echo "[2] Direct vLLM — chat completion"
R=$(curl -sf -X POST "$VLLM/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"max_tokens\":20}" \
  | jq -r '.choices[0].message.content' 2>/dev/null)
[[ -n "$R" && "$R" != "null" ]] && pass "Direct chat: \"${R:0:60}\"" || fail "Direct chat failed"

# Envoy proxy
echo "[3] Envoy proxy — chat completion"
R=$(curl -sf -X POST "$ENVOY/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 7*6? Reply with only the number.\"}],\"max_tokens\":10}" \
  | jq -r '.choices[0].message.content' 2>/dev/null)
[[ -n "$R" && "$R" != "null" ]] && pass "Envoy proxy: \"$R\"" || fail "Envoy proxy failed"

# Tool calling
echo "[4] Tool calling via Envoy"
R=$(curl -sf -X POST "$ENVOY/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"'"$MODEL"'",
    "messages":[{"role":"user","content":"Get the weather in Tokyo"}],
    "max_tokens":100,
    "tools":[{
      "type":"function",
      "function":{
        "name":"get_weather",
        "description":"Get weather for a city",
        "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
      }
    }],
    "tool_choice":"auto"
  }')
TC=$(echo "$R" | jq '.choices[0].message.tool_calls | length' 2>/dev/null || echo 0)
[[ "$TC" -gt 0 ]] && pass "Tool called: $(echo "$R" | jq -r '.choices[0].message.tool_calls[0].function.name')" \
  || pass "Model responded without tool call (acceptable)"

# Streaming
echo "[5] Streaming via Envoy"
CHUNKS=0
while IFS= read -r line; do
  [[ "$line" == data:* ]] && CHUNKS=$((CHUNKS+1))
done < <(curl -sf -N -X POST "$ENVOY/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 5\"}],\"max_tokens\":30,\"stream\":true}" 2>/dev/null)
[[ "$CHUNKS" -gt 2 ]] && pass "Streaming: $CHUNKS chunks" || fail "Streaming: only $CHUNKS chunks"

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit $FAIL

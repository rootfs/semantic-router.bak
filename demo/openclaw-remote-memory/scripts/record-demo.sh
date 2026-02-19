#!/usr/bin/env bash
set -euo pipefail

# Colors & helpers
BOLD='\033[1m'
CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[1;35m'
DIM='\033[2m'
RESET='\033[0m'

banner() { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; echo -e "${BOLD}  $1${RESET}"; echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"; }
info()   { echo -e "${GREEN}▸${RESET} $1"; }
cmd()    { echo -e "${DIM}\$ $1${RESET}"; eval "$1"; }
pause()  { sleep "${1:-1.5}"; }

clear
echo -e "${MAGENTA}"
cat << 'EOF'
   ___                    ____ _                
  / _ \ _ __   ___ _ __  / ___| | __ ___      __
 | | | | '_ \ / _ \ '_ \| |   | |/ _` \ \ /\ / /
 | |_| | |_) |  __/ | | | |___| | (_| |\ V  V / 
  \___/| .__/ \___|_| |_|\____|_|\__,_| \_/\_/  
       |_|                                       
   +  Semantic Router  +  Remote Vector Memory
EOF
echo -e "${RESET}"
echo -e "  End-to-end demo: LLM inference & memory both served by semantic-router"
echo ""
pause 3

# ─────────────────────────────────────────────────────────────────
banner "1 ─ Running Containers"
# ─────────────────────────────────────────────────────────────────
info "Four containers power this demo:"
echo ""
cmd "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'vllm-qwen|semantic-router|envoy|openclaw'"
pause 3

# ─────────────────────────────────────────────────────────────────
banner "2 ─ OpenClaw Configuration"
# ─────────────────────────────────────────────────────────────────
info "The openclaw.json configures LLM provider and remote memory:"
echo ""
echo -e "${YELLOW}── models (LLM via Envoy → Semantic Router → vLLM) ──${RESET}"
cmd "jq '.models' /data/openclaw-demo/config/openclaw.json"
pause 2

echo ""
echo -e "${YELLOW}── memory (remote vector store via Semantic Router) ──${RESET}"
cmd "jq '.memory' /data/openclaw-demo/config/openclaw.json"
pause 2

echo ""
echo -e "${YELLOW}── agents ──${RESET}"
cmd "jq '.agents' /data/openclaw-demo/config/openclaw.json"
pause 3

# ─────────────────────────────────────────────────────────────────
banner "3 ─ Workspace Memory Files"
# ─────────────────────────────────────────────────────────────────
info "These markdown files live in the agent workspace and are synced"
info "to the remote vector store for semantic search:"
echo ""
cmd "ls -la /data/openclaw-demo/workspace/memory/"
pause 2

echo ""
echo -e "${YELLOW}── memory/project-notes.md (excerpt) ──${RESET}"
head -20 /data/openclaw-demo/workspace/memory/project-notes.md
pause 2

echo ""
echo -e "${YELLOW}── memory/runbook.md (excerpt) ──${RESET}"
head -15 /data/openclaw-demo/workspace/memory/runbook.md
pause 3

# ─────────────────────────────────────────────────────────────────
banner "4 ─ Semantic Router Health & Vector Store"
# ─────────────────────────────────────────────────────────────────
info "Semantic Router API (port 8080) — health check:"
cmd "curl -s http://localhost:8080/health | jq ."
pause 1

echo ""
info "Vector store created by OpenClaw (files synced from workspace):"
cmd "curl -s http://localhost:8080/v1/vector_stores | jq '.data[] | {id, name, status, file_counts}'"
pause 2

echo ""
info "Files in the vector store:"
VS_ID=$(curl -s http://localhost:8080/v1/vector_stores | jq -r '.data[] | select(.name=="openclaw-demo") | .id')
cmd "curl -s http://localhost:8080/v1/vector_stores/${VS_ID}/files | jq '.data[] | {id, file_id, status}'"
pause 3

# ─────────────────────────────────────────────────────────────────
banner "5 ─ Test: Vector Store Search (Semantic Router)"
# ─────────────────────────────────────────────────────────────────
info "Searching the vector store directly — query: 'database connection pool'"
echo ""
cmd "curl -s -X POST http://localhost:8080/v1/vector_stores/${VS_ID}/search -H 'Content-Type: application/json' -d '{\"query\": \"database connection pool\", \"max_num_results\": 2}' | jq '.data[] | {filename, score, content: (.content[:120] + \"...\") }'"
pause 3

# ─────────────────────────────────────────────────────────────────
banner "6 ─ Test: LLM Inference (Envoy → Semantic Router → vLLM)"
# ─────────────────────────────────────────────────────────────────
info "Chat completion through the full routing chain (port 8801):"
echo ""
cmd "curl -s -X POST http://localhost:8801/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"Qwen/Qwen2.5-14B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer in one sentence.\"}], \"max_tokens\": 50}' | jq '{model: .model, content: .choices[0].message.content, tokens: .usage}'"
pause 3

# ─────────────────────────────────────────────────────────────────
banner "7 ─ OpenClaw Agent Demo (Memory + LLM)"
# ─────────────────────────────────────────────────────────────────
info "Running OpenClaw agent with a question that requires memory search."
info "The agent will:"
info "  1. Call memory_search → Semantic Router vector store API"
info "  2. Get relevant docs from runbook.md & project-notes.md"
info "  3. Call LLM → Envoy → Semantic Router → vLLM"
info "  4. Return an answer grounded in workspace knowledge"
echo ""
echo -e "${YELLOW}Question: \"Search your memory for the incident runbook. What are"
echo -e "the exact steps for database connection pool exhaustion? Who is"
echo -e "the DevOps contact in project notes?\"${RESET}"
echo ""
info "Running agent..."
echo ""

RESULT=$(docker exec openclaw-demo node openclaw.mjs agent \
  --local --agent demo \
  --message "Search your memory for the incident runbook. What are the exact steps for database connection pool exhaustion? Also, who is the DevOps contact listed in the project notes?" \
  --json 2>&1)

echo "$RESULT" | jq -r '.payloads[-1].text' 2>/dev/null || echo "$RESULT"
pause 2

echo ""
echo -e "${YELLOW}── Agent Metadata ──${RESET}"
echo "$RESULT" | jq '{
  provider: .meta.agentMeta.provider,
  model: .meta.agentMeta.model,
  duration_ms: .meta.durationMs,
  input_tokens: .meta.agentMeta.usage.input,
  output_tokens: .meta.agentMeta.usage.output
}' 2>/dev/null
pause 3

# ─────────────────────────────────────────────────────────────────
banner "8 ─ Architecture Summary"
# ─────────────────────────────────────────────────────────────────
echo -e "${BOLD}Data Flow:${RESET}"
echo ""
echo "  ┌──────────────────────────────────────────────────────────┐"
echo "  │                   OpenClaw Agent                         │"
echo "  │              (container, port 18788)                     │"
echo "  └──────┬───────────────────────────────┬───────────────────┘"
echo "         │                               │"
echo "         │ memory_search tool            │ LLM inference"
echo "         ▼                               ▼"
echo "  ┌──────────────────┐          ┌──────────────────┐"
echo "  │ Semantic Router  │          │      Envoy       │"
echo "  │   API :8080      │          │     :8801        │"
echo "  │                  │          └────────┬─────────┘"
echo "  │  /v1/vector_     │                   │ ext_proc gRPC"
echo "  │   stores/search  │                   ▼"
echo "  │                  │          ┌──────────────────┐"
echo "  │  mmbert embed    │          │ Semantic Router  │"
echo "  │  in-memory store │          │  ExtProc :50051  │"
echo "  └──────────────────┘          └────────┬─────────┘"
echo "                                         │ model routing"
echo "                                         ▼"
echo "                                ┌──────────────────┐"
echo "                                │      vLLM        │"
echo "                                │     :8100        │"
echo "                                │ Qwen2.5-14B-Inst │"
echo "                                │ 32k ctx + tools  │"
echo "                                └──────────────────┘"
echo ""
pause 3

echo -e "${GREEN}${BOLD}✓ Demo complete!${RESET} Both LLM inference and vector memory are served"
echo -e "  by the semantic-router infrastructure."
echo ""

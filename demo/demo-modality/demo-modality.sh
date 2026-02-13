#!/usr/bin/env bash
# demo-modality.sh — Asciinema-friendly demo of modality routing
#
# Shows intelligent routing between text (AR) and image (Diffusion) models
# through the full Envoy → Semantic Router → vLLM pipeline.
#
# Architecture:
#   Client → Envoy :8801 → Semantic Router (ext_proc) → AR :8100 / Diffusion :8091
#
# Prerequisites:
#   - vLLM AR model on port 8100 (Qwen/Qwen2.5-14B-Instruct)
#   - vLLM-Omni diffusion on port 8091 (Qwen/Qwen-Image)
#   - Semantic Router with scripts/demo-modality/router-config.yaml
#   - Envoy with scripts/demo-modality/envoy.yaml on :8801
#
# Usage:
#   bash scripts/demo-modality/demo-modality.sh

set -euo pipefail

SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SR_ROOT}"

# ── Configuration ────────────────────────────────────────────
ENVOY_URL="${ENVOY_URL:-http://127.0.0.1:8801}"
AR_URL="${AR_URL:-http://127.0.0.1:8100}"
DIFFUSION_URL="${DIFFUSION_URL:-http://127.0.0.1:8091}"
VLLM_MODEL="Qwen/Qwen2.5-14B-Instruct"
DIFFUSION_MODEL="Qwen/Qwen-Image"
OUTPUT_DIR="/tmp/modality-demo"
PAUSE="${PAUSE:-3}"

mkdir -p "${OUTPUT_DIR}"

# ── Colors ───────────────────────────────────────────────────
BOLD="\033[1m"
RESET="\033[0m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
MAGENTA="\033[35m"
BLUE="\033[34m"
DIM="\033[2m"
WHITE="\033[97m"
BG_CYAN="\033[46m"
BG_GREEN="\033[42m"
BG_MAGENTA="\033[45m"
BG_YELLOW="\033[43m"
BG_BLACK="\033[40m"

# ── Helpers ──────────────────────────────────────────────────
separator() { echo -e "${DIM}────────────────────────────────────────────────────────────────────────${RESET}"; }
pause()     { sleep "${PAUSE}"; }

json_field() {
    python3 -c "import sys,json; d=json.load(sys.stdin); print($1)" 2>/dev/null
}

# Extract base64 image from chat completions response and save to file
# Usage: extract_image "$response_json" "/path/to/output.png"
extract_image() {
    local output_path="$2"
    IMG_OUT="${output_path}" python3 -c "
import json, sys, base64, os
data = json.load(sys.stdin)
content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
if isinstance(content, list):
    for part in content:
        if part.get('type') == 'image_url':
            url = part['image_url']['url']
            if url.startswith('data:image/'):
                b64 = url.split(',', 1)[1]
                with open(os.environ['IMG_OUT'], 'wb') as f:
                    f.write(base64.b64decode(b64))
                print('ok')
                sys.exit(0)
print('no_image')
" <<< "$1" 2>/dev/null
}

# Extract text content from response (handles both string and array content)
extract_text() {
    local response="$1"
    python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
if isinstance(content, list):
    texts = [p.get('text', '') for p in content if p.get('type') == 'text']
    print(' '.join(texts)[:500])
elif isinstance(content, str):
    print(content[:500])
" <<< "${response}" 2>/dev/null
}

# Show image in terminal via chafa
show_image() {
    local path="$1" width="${2:-60}" height="${3:-25}"
    if [[ ! -f "${path}" ]]; then
        echo -e "  ${RED}Image not found: ${path}${RESET}"
        return
    fi
    local size_kb
    size_kb=$(( $(stat -c%s "${path}" 2>/dev/null || echo 0) / 1024 ))
    echo -e "  ${DIM}[${size_kb} KB, 512×512]${RESET}"
    echo ""
    chafa --size "${width}x${height}" --colors direct \
          --symbols "block+border+space+extra" --color-space din99d \
          "${path}" 2>/dev/null || echo -e "  ${RED}Could not display image (chafa error)${RESET}"
    echo ""
}

# ── Banner ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════════════════════════════╗"
echo "  ║                                                                   ║"
echo "  ║   vLLM Semantic Router — Modality Routing Demo                    ║"
echo "  ║                                                                   ║"
echo "  ║   Automatic AR / Diffusion / Both routing via signal evaluation   ║"
echo "  ║                                                                   ║"
echo "  ╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "  ${BOLD}Architecture:${RESET}"
echo ""
echo -e "  ${DIM}┌──────────┐    ┌───────────┐    ┌──────────────────┐    ┌──────────────────────────┐${RESET}"
echo -e "  ${DIM}│${RESET}  Client  ${DIM}│ →  │${RESET}   Envoy   ${DIM}│ →  │${RESET}  Semantic Router ${DIM}│ ─┬→│${RESET}  ${GREEN}AR${RESET}   Qwen2.5-14B :8100  ${DIM}│${RESET}"
echo -e "  ${DIM}│${RESET}          ${DIM}│    │${RESET}  :8801    ${DIM}│    │${RESET}  ext_proc :50051 ${DIM}│  │ │${RESET}  (text generation)      ${DIM}│${RESET}"
echo -e "  ${DIM}│${RESET}          ${DIM}│    │${RESET}           ${DIM}│    │${RESET}                  ${DIM}│  │ └──────────────────────────┘${RESET}"
echo -e "  ${DIM}│${RESET}          ${DIM}│    │${RESET}           ${DIM}│    │${RESET}  mmBERT-32K      ${DIM}│  │ ┌──────────────────────────┐${RESET}"
echo -e "  ${DIM}│${RESET}          ${DIM}│    │${RESET}           ${DIM}│    │${RESET}  modality signal ${DIM}│  └→│${RESET}  ${MAGENTA}DIF${RESET}  Qwen-Image   :8091  ${DIM}│${RESET}"
echo -e "  ${DIM}│${RESET}          ${DIM}│    │${RESET}           ${DIM}│    │${RESET}                  ${DIM}│    │${RESET}  (image generation)     ${DIM}│${RESET}"
echo -e "  ${DIM}└──────────┘    └───────────┘    └──────────────────┘    └──────────────────────────┘${RESET}"
echo ""
echo -e "  ${BOLD}Modality Signals:${RESET}"
echo -e "    ${GREEN}AR${RESET}         Text-only prompts       → Qwen/Qwen2.5-14B-Instruct (vLLM)"
echo -e "    ${MAGENTA}DIFFUSION${RESET}  Image generation prompts → Qwen/Qwen-Image (vLLM-Omni)"
echo -e "    ${YELLOW}BOTH${RESET}       Text + image prompts     → AR text + Diffusion image (parallel)"
echo ""
echo -e "  ${DIM}Config: scripts/demo-modality/router-config.yaml${RESET}"
echo ""
pause
pause

# ── Check Endpoints ──────────────────────────────────────────
separator
echo ""
echo -e "  ${BOLD}Checking infrastructure...${RESET}"
echo ""
sleep 1

for label_url in "AR Model (Qwen2.5-14B)|${AR_URL}/health" "Diffusion (Qwen-Image)|${DIFFUSION_URL}/health" "Envoy Proxy|http://127.0.0.1:19000/ready" "Semantic Router|http://127.0.0.1:8080/health"; do
    label="${label_url%%|*}"
    url="${label_url##*|}"
    if curl -sf "${url}" >/dev/null 2>&1; then
        echo -e "    ${GREEN}✓${RESET} ${label} @ ${DIM}${url}${RESET}"
    else
        echo -e "    ${RED}✗${RESET} ${label} @ ${DIM}${url}${RESET}"
    fi
done
echo ""
pause

# ══════════════════════════════════════════════════════════════
# DEMO 1: Text Generation (AR)
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_GREEN}${WHITE} DEMO 1 ${RESET}  ${BOLD}Text Generation (AR Modality)${RESET}"
echo ""
pause

PROMPT1="Explain the theory of relativity in three sentences"
echo -e "  ${BOLD}${YELLOW}PROMPT${RESET}"
echo -e "  ┌─────────────────────────────────────────────────────────────"
echo -e "  │ ${BLUE}${PROMPT1}${RESET}"
echo -e "  └─────────────────────────────────────────────────────────────"
echo ""
sleep 1

echo -e "  ${DIM}Sending to Envoy → Router (modality classification) → ...${RESET}"
START_TIME=$(date +%s%N)

RESP1=$(curl -sf "${ENVOY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${VLLM_MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT1}\"}],
        \"max_tokens\": 200
    }" 2>/dev/null)

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

if [[ -z "${RESP1}" ]]; then
    echo -e "  ${RED}ERROR: No response received${RESET}"
else
    RESP1_MODEL=$(echo "${RESP1}" | json_field "d['model']")
    RESP1_TEXT=$(extract_text "${RESP1}")
    RESP1_TOKENS=$(echo "${RESP1}" | json_field "d['usage']['completion_tokens']")

    echo ""
    echo -e "  ${BOLD}${GREEN}RESPONSE — Routed to AR${RESET}  ${DIM}(${ELAPSED_MS}ms)${RESET}"
    echo -e "  ┌─────────────────────────────────────────────────────────────"
    echo -e "  │ model:   ${BOLD}${GREEN}${RESP1_MODEL}${RESET}"
    echo -e "  │ tokens:  ${RESP1_TOKENS}"
    echo -e "  │"
    # Word-wrap the response text
    echo "${RESP1_TEXT}" | fold -s -w 65 | while IFS= read -r line; do
        echo -e "  │ ${WHITE}${line}${RESET}"
    done
    echo -e "  └─────────────────────────────────────────────────────────────"
fi
echo ""
pause
pause

# ══════════════════════════════════════════════════════════════
# DEMO 2: Image Generation (DIFFUSION)
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_MAGENTA}${WHITE} DEMO 2 ${RESET}  ${BOLD}Image Generation (DIFFUSION Modality)${RESET}"
echo ""
pause

PROMPT2="Generate a photorealistic image of a golden retriever puppy playing in autumn leaves"
echo -e "  ${BOLD}${YELLOW}PROMPT${RESET}"
echo -e "  ┌─────────────────────────────────────────────────────────────"
echo -e "  │ ${BLUE}${PROMPT2}${RESET}"
echo -e "  └─────────────────────────────────────────────────────────────"
echo ""
sleep 1

echo -e "  ${DIM}Sending to Envoy → Router (modality: DIFFUSION) → Qwen-Image...${RESET}"
echo -e "  ${DIM}(Image generation may take 15-60 seconds)${RESET}"
START_TIME=$(date +%s%N)

RESP2=$(curl -sf "${ENVOY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${VLLM_MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT2}\"}],
        \"max_tokens\": 100
    }" 2>/dev/null)

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
ELAPSED_S=$(( ELAPSED_MS / 1000 ))

IMG_PATH="${OUTPUT_DIR}/demo_diffusion.png"
IMG_RESULT=$(extract_image "${RESP2}" "${IMG_PATH}")

if [[ "${IMG_RESULT}" == "ok" ]]; then
    echo ""
    echo -e "  ${BOLD}${MAGENTA}RESPONSE — Routed to DIFFUSION${RESET}  ${DIM}(${ELAPSED_S}s)${RESET}"
    echo -e "  ┌─────────────────────────────────────────────────────────────"
    echo -e "  │ model:   ${BOLD}${MAGENTA}${DIFFUSION_MODEL}${RESET} (via vLLM-Omni)"
    echo -e "  │ status:  ${GREEN}✓ Image generated${RESET}"
    echo -e "  └─────────────────────────────────────────────────────────────"
    echo ""
    show_image "${IMG_PATH}" 60 25
else
    echo ""
    echo -e "  ${RED}Image extraction failed. Raw response:${RESET}"
    echo "${RESP2}" | python3 -m json.tool 2>/dev/null | head -20 || echo "${RESP2}" | head -200
fi
echo ""
pause
pause

# ══════════════════════════════════════════════════════════════
# DEMO 3: Multimodal (BOTH)
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${BG_YELLOW}${BG_BLACK}${YELLOW} DEMO 3 ${RESET}  ${BOLD}Multimodal Response (BOTH Modality)${RESET}"
echo ""
pause

PROMPT3="Explain what a fractal is in two sentences and generate an image of the Mandelbrot set"
echo -e "  ${BOLD}${YELLOW}PROMPT${RESET}"
echo -e "  ┌─────────────────────────────────────────────────────────────"
echo -e "  │ ${BLUE}${PROMPT3}${RESET}"
echo -e "  └─────────────────────────────────────────────────────────────"
echo ""
sleep 1

echo -e "  ${DIM}Sending to Envoy → Router (modality: BOTH) → AR + Diffusion in parallel...${RESET}"
echo -e "  ${DIM}(Parallel generation: text + image)${RESET}"
START_TIME=$(date +%s%N)

RESP3=$(curl -sf "${ENVOY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${VLLM_MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT3}\"}],
        \"max_tokens\": 200
    }" 2>/dev/null)

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
ELAPSED_S=$(( ELAPSED_MS / 1000 ))

echo ""
echo -e "  ${BOLD}${YELLOW}RESPONSE — Routed to BOTH${RESET}  ${DIM}(${ELAPSED_S}s)${RESET}"
echo ""

# Extract text part
RESP3_TEXT=$(extract_text "${RESP3}")
if [[ -n "${RESP3_TEXT}" ]]; then
    echo -e "  ${BOLD}[1/2] Text (from ${GREEN}Qwen2.5-14B${RESET}${BOLD})${RESET}"
    echo -e "  ┌─────────────────────────────────────────────────────────────"
    echo "${RESP3_TEXT}" | fold -s -w 65 | while IFS= read -r line; do
        echo -e "  │ ${WHITE}${line}${RESET}"
    done
    echo -e "  └─────────────────────────────────────────────────────────────"
    echo ""
fi

# Extract image part
IMG_PATH3="${OUTPUT_DIR}/demo_both.png"
IMG_RESULT3=$(extract_image "${RESP3}" "${IMG_PATH3}")
if [[ "${IMG_RESULT3}" == "ok" ]]; then
    echo -e "  ${BOLD}[2/2] Image (from ${MAGENTA}Qwen-Image${RESET}${BOLD})${RESET}"
    echo -e "  ┌─────────────────────────────────────────────────────────────"
    echo -e "  │ ${GREEN}✓ Image generated${RESET}"
    echo -e "  └─────────────────────────────────────────────────────────────"
    echo ""
    show_image "${IMG_PATH3}" 60 25
else
    echo -e "  ${DIM}[2/2] No image in response (modality may have been classified differently)${RESET}"
fi
echo ""
pause
pause

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
separator
echo ""
echo -e "  ${BOLD}${CYAN}Summary — Modality Routing Demo${RESET}"
echo ""
echo -e "  The semantic router automatically classifies prompts and routes them"
echo -e "  to the appropriate backend — no client-side logic needed."
echo ""
echo -e "  ${BOLD}Routing Results:${RESET}"
echo ""
printf "  ${BOLD}%-5s %-55s %-12s${RESET}\n" "#" "Prompt" "Modality"
printf "  ${DIM}%-5s %-55s %-12s${RESET}\n" "─────" "───────────────────────────────────────────────────────" "────────────"
printf "  %-5s %-55s ${GREEN}%-12s${RESET}\n" "1" "${PROMPT1:0:52}..." "AR"
printf "  %-5s %-55s ${MAGENTA}%-12s${RESET}\n" "2" "${PROMPT2:0:52}..." "DIFFUSION"
printf "  %-5s %-55s ${YELLOW}%-12s${RESET}\n" "3" "${PROMPT3:0:52}..." "BOTH"
echo ""
echo -e "  ${BOLD}Infrastructure:${RESET}"
echo -e "    AR Model:       ${GREEN}Qwen/Qwen2.5-14B-Instruct${RESET}     @ localhost:8100 (vLLM)"
echo -e "    Diffusion:      ${MAGENTA}Qwen/Qwen-Image${RESET}               @ localhost:8091 (vLLM-Omni)"
echo -e "    Classifier:     ${CYAN}mmBERT-32K modality router${RESET}     (hybrid: classifier + keyword)"
echo -e "    Router:         ${CYAN}Semantic Router${RESET}                @ :50051 (ext_proc)"
echo -e "    Proxy:          ${CYAN}Envoy${RESET}                          @ :8801"
echo -e "    Config:         scripts/demo-modality/router-config.yaml"
echo ""
echo -e "${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════════════════════════════╗"
echo "  ║                       Demo Complete!                              ║"
echo "  ╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

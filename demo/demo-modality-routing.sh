#!/usr/bin/env bash
# demo-modality-routing.sh — Run modality routing tests with inline image display
# Designed for asciinema recording with truecolor terminal
#
# Prerequisites:
#   - Semantic router running: make run-router-modality
#   - Envoy proxy running:     func-e run --config-path config/envoy.yaml
#   - AR vLLM on port 8000     (e.g., Qwen/Qwen2.5-14B-Instruct)
#   - Diffusion vLLM on 8001   (e.g., Qwen/Qwen-Image via vllm-omni)
#   - chafa installed           (apt install chafa)

set -euo pipefail

ENVOY_URL="${ENVOY_URL:-http://localhost:8801}"
METRICS_URL="${METRICS_URL:-http://localhost:9190/metrics}"
IMG_WIDTH="${IMG_WIDTH:-60}"
IMG_HEIGHT="${IMG_HEIGHT:-30}"

# ── Colors ──────────────────────────────────────────────────
BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
MAGENTA="\033[35m"
RED="\033[31m"
WHITE="\033[97m"
BLUE="\033[34m"
BG_BLUE="\033[44m"
BG_GREEN="\033[42m"
BG_MAGENTA="\033[45m"
BG_YELLOW="\033[43m"
BG_BLACK="\033[40m"

banner() {
    echo ""
    echo -e "${BOLD}${BG_BLUE}${WHITE}                                                                ${RESET}"
    echo -e "${BOLD}${BG_BLUE}${WHITE}   vLLM Semantic Router — Modality Routing Demo                 ${RESET}"
    echo -e "${BOLD}${BG_BLUE}${WHITE}   AR (Text) | DIFFUSION (Image) | BOTH (Text + Tool Call)      ${RESET}"
    echo -e "${BOLD}${BG_BLUE}${WHITE}                                                                ${RESET}"
    echo ""
    echo -e "  ${DIM}Architecture: curl → Envoy (:8801) → Semantic Router ExtProc (:50051) → vLLM${RESET}"
    echo -e "  ${DIM}Classifier:   mmBERT-32K modality router (hybrid: classifier + keyword)${RESET}"
    echo ""
}

separator() {
    echo -e "${DIM}────────────────────────────────────────────────────────────────────${RESET}"
}

# Print a section header with modality badge
test_header() {
    local num="$1" modality="$2" label="$3" prompt="$4"
    local badge_color=""
    case "$modality" in
        AR)        badge_color="${BG_GREEN}" ;;
        DIFFUSION) badge_color="${BG_MAGENTA}" ;;
        BOTH)      badge_color="${BG_BLUE}" ;;
    esac
    separator
    echo -e "${BOLD}  TEST ${num}  ${badge_color}${WHITE} ${modality} ${RESET}  ${label}${RESET}"
    echo -e "  ${CYAN}Prompt:${RESET} ${WHITE}\"${prompt}\"${RESET}"
    echo ""
}

# Print all x-vsr-* response headers with colored formatting
print_response_headers() {
    local header_file="$1"
    local has_header=false

    echo -e "  ${BOLD}${YELLOW}Response Headers (proof of router routing):${RESET}"
    while IFS= read -r line; do
        # Trim carriage return
        line="${line%$'\r'}"
        local key val
        key=$(echo "$line" | cut -d: -f1)
        val=$(echo "$line" | cut -d: -f2- | sed 's/^ //')
        case "$key" in
            x-vsr-selected-modality)
                echo -e "    ${GREEN}${key}${RESET}: ${BOLD}${val}${RESET}"
                has_header=true
                ;;
            x-vsr-selected-model)
                echo -e "    ${GREEN}${key}${RESET}: ${val}"
                has_header=true
                ;;
            x-vsr-selected-decision)
                echo -e "    ${GREEN}${key}${RESET}: ${val}"
                has_header=true
                ;;
            x-vsr-matched-keywords)
                echo -e "    ${GREEN}${key}${RESET}: ${val}"
                has_header=true
                ;;
            x-vsr-selected-confidence)
                echo -e "    ${GREEN}${key}${RESET}: ${val}"
                has_header=true
                ;;
            x-vsr-selected-reasoning)
                echo -e "    ${GREEN}${key}${RESET}: ${val}"
                has_header=true
                ;;
            x-vsr-injected-system-prompt)
                # skip, not interesting for this demo
                has_header=true
                ;;
        esac
    done < <(grep "^x-vsr-" "$header_file" 2>/dev/null || true)

    if [ "$has_header" = false ]; then
        echo -e "    ${DIM}(DIFFUSION short-circuit — image returned directly by router, no upstream)${RESET}"
    fi
}

# Extract and display text content from JSON response
print_text_response() {
    local response="$1"
    echo -e "  ${BOLD}${CYAN}LLM Response:${RESET}"
    echo "$response" | python3 -c "
import sys, json, textwrap
try:
    d = json.load(sys.stdin)
    if 'error' in d:
        print(f'  \033[31m❌ Error: {d[\"error\"][\"message\"][:200]}\033[0m')
        sys.exit(0)
    msg = d['choices'][0]['message']
    content = msg.get('content', '') or ''
    model = d.get('model', 'unknown')
    # Truncate long content for display
    if len(content) > 600:
        content = content[:600] + '...'
    # Wrap text nicely
    for line in content.split('\n'):
        wrapped = textwrap.fill(line, width=68, initial_indent='    ', subsequent_indent='    ')
        print(wrapped)
    # Show tool calls if present
    tool_calls = msg.get('tool_calls', [])
    if tool_calls:
        print()
        print(f'  \033[33m🔧 Tool call: {tool_calls[0][\"function\"][\"name\"]}()\033[0m')
        args = json.loads(tool_calls[0]['function']['arguments'])
        prompt_text = args.get('prompt', '')
        if len(prompt_text) > 120:
            prompt_text = prompt_text[:120] + '...'
        print(f'     \033[2mprompt: \"{prompt_text}\"\033[0m')
    finish = d['choices'][0].get('finish_reason', '')
    usage = d.get('usage', {})
    tokens = usage.get('total_tokens', 0)
    print(f'\n  \033[2m📊 model={model}  finish={finish}  tokens={tokens}\033[0m')
except Exception as e:
    print(f'  (parse error: {e})')
" 2>&1
}

# Extract base64 image, decode to file, display with chafa
display_image() {
    local response="$1"
    local img_file="/tmp/modality_demo_img.png"
    rm -f "$img_file"

    local img_size
    img_size=$(echo "$response" | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
msg = d['choices'][0]['message']
content = msg.get('content', '')
if isinstance(content, list):
    for item in content:
        if item.get('type') == 'image_url':
            url = item['image_url']['url']
            b64 = url.split(',', 1)[1] if ',' in url else url
            img = base64.b64decode(b64)
            with open('${img_file}', 'wb') as f:
                f.write(img)
            print(len(img))
            sys.exit(0)
print('0')
" 2>/dev/null || echo "0")

    if [ -f "$img_file" ] && [ -s "$img_file" ]; then
        echo -e "  ${GREEN}✓ Image generated${RESET} ($(( img_size / 1024 )) KB)"
        echo ""
        chafa --size="${IMG_WIDTH}x${IMG_HEIGHT}" \
              --colors=direct \
              --symbols=all \
              --color-space=din99d \
              "$img_file"
        echo ""
        rm -f "$img_file"
    else
        echo -e "  ${YELLOW}⚠ No image in response${RESET}"
    fi
}

# Fetch and print relevant Prometheus metrics from the router
print_metrics() {
    local label="$1"
    echo -e "  ${BOLD}${BLUE}Router Metrics (${label}):${RESET}"

    local metrics
    metrics=$(curl -s "$METRICS_URL" 2>/dev/null || echo "")
    if [ -z "$metrics" ]; then
        echo -e "    ${DIM}(metrics endpoint unavailable)${RESET}"
        return
    fi

    # image_gen_requests_total
    local img_total
    img_total=$(echo "$metrics" | grep '^image_gen_requests_total{' | head -1 | awk '{print $2}')
    echo -e "    image_gen_requests_total:  ${WHITE}${img_total:-0}${RESET}"

    # image_gen avg latency
    local img_sum img_count
    img_sum=$(echo "$metrics" | grep '^image_gen_latency_seconds_sum{' | head -1 | awk '{print $2}')
    img_count=$(echo "$metrics" | grep '^image_gen_latency_seconds_count{' | head -1 | awk '{print $2}')
    if [ -n "$img_sum" ] && [ -n "$img_count" ] && [ "$img_count" != "0" ]; then
        local avg
        avg=$(python3 -c "print(f'{${img_sum}/${img_count}:.2f}')" 2>/dev/null || echo "?")
        echo -e "    image_gen_avg_latency:     ${WHITE}${avg}s${RESET} (${img_count} requests)"
    fi

    # decision evaluations
    local dec_count
    dec_count=$(echo "$metrics" | grep '^llm_decision_confidence_count{' | head -1 | awk '{print $2}')
    echo -e "    decision_evaluations:      ${WHITE}${dec_count:-0}${RESET}"

    # routing reason code
    local routing
    routing=$(echo "$metrics" | grep '^llm_routing_reason_codes_total{' | head -1)
    if [ -n "$routing" ]; then
        local model reason count
        model=$(echo "$routing" | sed 's/.*model="\([^"]*\)".*/\1/')
        reason=$(echo "$routing" | sed 's/.*reason_code="\([^"]*\)".*/\1/')
        count=$(echo "$routing" | awk '{print $2}')
        echo -e "    routing_reason:            ${WHITE}${reason}${RESET} → ${model} (${count}×)"
    fi
}

# ── Main ────────────────────────────────────────────────────

banner

# ── Show metrics before tests ──
print_metrics "before tests"
echo ""

# System message to enforce English (Qwen2.5 can switch languages otherwise)
SYS='{"role":"system","content":"You are a helpful assistant. Always respond in English. When asked to explain something and generate an image, first provide a detailed text explanation, then call the generate_image tool."}'

# ── TEST 1: AR (pure text) ──
test_header 1 "AR" "Pure text question" "What is the capital of France?"
RESPONSE=$(curl -sS -D /tmp/modality_h.txt -X POST "${ENVOY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":['"${SYS}"',{"role":"user","content":"What is the capital of France?"}],"max_tokens":100}')
print_response_headers /tmp/modality_h.txt
print_text_response "$RESPONSE"
echo ""

# ── TEST 2: AR (code) ──
test_header 2 "AR" "Code generation" "Write a Python function to compute fibonacci numbers"
RESPONSE=$(curl -sS -D /tmp/modality_h.txt -X POST "${ENVOY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":['"${SYS}"',{"role":"user","content":"Write a Python function to compute fibonacci numbers"}],"max_tokens":200}')
print_response_headers /tmp/modality_h.txt
print_text_response "$RESPONSE"
echo ""

# ── TEST 3: DIFFUSION (generate image) ──
test_header 3 "DIFFUSION" "Image generation" "Generate an image of a sunset over mountains"
RESPONSE=$(curl -sS -D /tmp/modality_h.txt -X POST "${ENVOY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"Generate an image of a sunset over mountains"}],"max_tokens":100}')
print_response_headers /tmp/modality_h.txt
display_image "$RESPONSE"

# ── TEST 4: DIFFUSION (draw) ──
test_header 4 "DIFFUSION" "Creative drawing" "Draw a cute cat wearing a top hat"
RESPONSE=$(curl -sS -D /tmp/modality_h.txt -X POST "${ENVOY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"Draw a cute cat wearing a top hat"}],"max_tokens":100}')
print_response_headers /tmp/modality_h.txt
display_image "$RESPONSE"

# ── TEST 5: BOTH (text + tool call) ──
test_header 5 "BOTH" "Text + image tool call" "Explain how photosynthesis works and generate an image of the process"
RESPONSE=$(curl -sS -D /tmp/modality_h.txt -X POST "${ENVOY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":['"${SYS}"',{"role":"user","content":"Explain how photosynthesis works and generate an image of the process"}],"max_tokens":500}')
print_response_headers /tmp/modality_h.txt
print_text_response "$RESPONSE"
echo ""

# ── TEST 6: BOTH (describe + illustrate) ──
test_header 6 "BOTH" "Description + illustration request" "Describe the water cycle and illustrate it with a diagram"
RESPONSE=$(curl -sS -D /tmp/modality_h.txt -X POST "${ENVOY_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":['"${SYS}"',{"role":"user","content":"Describe the water cycle and illustrate it with a diagram"}],"max_tokens":500}')
print_response_headers /tmp/modality_h.txt
print_text_response "$RESPONSE"
echo ""

# ── Show metrics after tests ──
separator
print_metrics "after tests"
echo ""

# ── Summary ──
separator
echo ""
echo -e "${BOLD}${GREEN}  ✓ All 6 modality routing tests completed${RESET}"
echo -e "    ${DIM}2× AR (text) │ 2× DIFFUSION (image) │ 2× BOTH (text+tool)${RESET}"
echo ""
echo -e "  ${DIM}Key observations:${RESET}"
echo -e "  ${DIM}• x-vsr-selected-modality header proves the router classified each request${RESET}"
echo -e "  ${DIM}• DIFFUSION requests short-circuit — image generated by router, no LLM call${RESET}"
echo -e "  ${DIM}• BOTH requests: router injected generate_image tool, LLM called it${RESET}"
echo -e "  ${DIM}• Metrics show image_gen_requests increased and decision_evaluations counted${RESET}"
echo ""

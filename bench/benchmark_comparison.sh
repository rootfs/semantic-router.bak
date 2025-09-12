#!/bin/bash

# Comprehensive benchmark script to compare Router vs Direct vLLM
# Usage: ./benchmark_comparison.sh [dataset] [samples_per_category]

set -e

# Configuration
DATASET=${1:-"arc"}                    # Default to ARC dataset
SAMPLES_PER_CATEGORY=${2:-5}           # Default to 5 samples per category
CONCURRENT_REQUESTS=${3:-2}            # Default to 2 concurrent requests

# Router configuration (via Envoy proxy)
ROUTER_ENDPOINT="http://127.0.0.1:8801/v1"
ROUTER_API_KEY="1234"
ROUTER_MODEL="auto"                    # Router decides the model

# Direct vLLM configuration  
VLLM_ENDPOINT="http://127.0.0.1:8000/v1"
VLLM_API_KEY="1234"
VLLM_MODEL="openai/gpt-oss-20b"        # Direct model access

# Benchmark parameters
MAX_TOKENS=1024
TEMPERATURE=0.0
OUTPUT_DIR="results/comparison_$(date +%Y%m%d_%H%M%S)"

echo "🎯 SEMANTIC ROUTER vs DIRECT vLLM BENCHMARK"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Samples per category: $SAMPLES_PER_CATEGORY"
echo "Concurrent requests: $CONCURRENT_REQUESTS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Ensure we're in the bench directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🔄 PHASE 1: ROUTER EVALUATION (via Envoy)"
echo "------------------------------------------"
echo "Endpoint: $ROUTER_ENDPOINT"
echo "Model: $ROUTER_MODEL (router decides)"
echo ""

# Run router benchmark
python3 router_reason_bench_v2.py \
    --dataset "$DATASET" \
    --samples-per-category "$SAMPLES_PER_CATEGORY" \
    --concurrent-requests "$CONCURRENT_REQUESTS" \
    --router-endpoint "$ROUTER_ENDPOINT" \
    --router-api-key "$ROUTER_API_KEY" \
    --router-models "$ROUTER_MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output-dir "$OUTPUT_DIR" \
    --run-router

echo ""
echo "🔄 PHASE 2: DIRECT vLLM EVALUATION"
echo "-----------------------------------"
echo "Endpoint: $VLLM_ENDPOINT"
echo "Model: $VLLM_MODEL (direct access)"
echo ""

# Run direct vLLM benchmark
python3 router_reason_bench_v2.py \
    --dataset "$DATASET" \
    --samples-per-category "$SAMPLES_PER_CATEGORY" \
    --concurrent-requests "$CONCURRENT_REQUESTS" \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --vllm-api-key "$VLLM_API_KEY" \
    --vllm-models "$VLLM_MODEL" \
    --vllm-exec-modes "NR" "XC" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output-dir "$OUTPUT_DIR" \
    --run-vllm

echo ""
echo "📊 BENCHMARK COMPLETED!"
echo "======================="
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Display quick summary if results exist
echo "📈 QUICK SUMMARY:"
echo "-----------------"

# Find and display router results
ROUTER_RESULT=$(find "$OUTPUT_DIR" -name "*router*auto*" -type d | head -1)
if [ -n "$ROUTER_RESULT" ] && [ -f "$ROUTER_RESULT/summary.json" ]; then
    echo "🔀 Router (via Envoy):"
    python3 -c "
import json, sys
try:
    with open('$ROUTER_RESULT/summary.json') as f:
        data = json.load(f)
    print(f\"  Accuracy: {data.get('overall_accuracy', 0):.3f}\")
    print(f\"  Avg Latency: {data.get('avg_response_time', 0):.2f}s\")
    print(f\"  Avg Tokens: {data.get('avg_total_tokens', 0):.0f}\")
    print(f\"  Questions: {data.get('successful_queries', 0)}/{data.get('total_questions', 0)}\")
except Exception as e:
    print(f\"  Error reading router results: {e}\")
"
fi

# Find and display vLLM results
VLLM_RESULT=$(find "$OUTPUT_DIR" -name "*vllm*gpt-oss*" -type d | head -1)
if [ -n "$VLLM_RESULT" ] && [ -f "$VLLM_RESULT/summary.json" ]; then
    echo "🎯 Direct vLLM:"
    python3 -c "
import json, sys
try:
    with open('$VLLM_RESULT/summary.json') as f:
        data = json.load(f)
    print(f\"  Accuracy: {data.get('overall_accuracy', 0):.3f}\")
    print(f\"  Avg Latency: {data.get('avg_response_time', 0):.2f}s\")
    print(f\"  Avg Tokens: {data.get('avg_total_tokens', 0):.0f}\")
    print(f\"  Questions: {data.get('successful_queries', 0)}/{data.get('total_questions', 0)}\")
    
    # Show breakdown by mode if available
    by_mode = data.get('by_mode', {})
    if by_mode:
        print(\"  Mode Breakdown:\")
        for mode, metrics in by_mode.items():
            if 'accuracy' in metrics:
                print(f\"    {mode}: {metrics['accuracy']:.3f} acc, {metrics.get('avg_response_time', 0):.2f}s\")
except Exception as e:
    print(f\"  Error reading vLLM results: {e}\")
"
fi

echo ""
echo "🔍 DETAILED ANALYSIS:"
echo "--------------------"
echo "- Router results: $ROUTER_RESULT"
echo "- vLLM results: $VLLM_RESULT"
echo "- Compare CSV files for detailed question-by-question analysis"
echo "- Check summary.json files for comprehensive metrics"
echo ""

echo "✅ Benchmark comparison complete!"
echo "Run with different datasets: $0 mmlu 10"
echo "Run with different datasets: $0 arc-challenge 3"

# Cluster Router Tutorial

## 1. Start vLLM Containers

```bash
# Math model (port 8081)
docker run --gpus '"device=0"' -p 8081:8000 vllm/vllm-openai:latest \
  vllm serve --model Qwen/Qwen2.5-Math-7B-Instruct --trust-remote-code --max-model-len 4096

# Coder model (port 8082)
docker run --gpus '"device=1"' -p 8082:8000 vllm/vllm-openai:latest \
  vllm serve --model Qwen/Qwen2.5-Coder-7B-Instruct --trust-remote-code --max-model-len 4096

# General model (port 8083)
docker run --gpus '"device=2"' -p 8083:8000 vllm/vllm-openai:latest \
  vllm serve --model Qwen/Qwen2.5-14B-Instruct-AWQ --trust-remote-code --quantization awq --max-model-len 4096
```

## 2. Configuration

Use `config/config.cluster_router.yaml`:

```yaml
intelligent_routing:
  cluster_router:
    enabled: true
    n_clusters: 10
    alpha: 0.5  # Cost-performance balance (see below)
    experience_db_path: "examples/cluster_router/mmlu_pro_experience_db_qwen3.json"
    embedding_model: "qwen3"
    embedding_dim: 1024
    model_costs:
      math: 7.0      # 7B model
      coder: 7.0     # 7B model
      general: 14.0  # 14B model (2x cost)
```

## 3. Alpha Parameter: Cost vs Performance

The `alpha` parameter controls the trade-off between accuracy and cost:

| Alpha | Behavior | Use Case |
|-------|----------|----------|
| 1.0 | Always pick best performing model | Maximum accuracy, ignore cost |
| 0.8 | Strongly favor performance | High accuracy priority |
| **0.5** | **Balanced (recommended)** | **Best accuracy + cost savings** |
| 0.2 | Strongly favor cost | Budget-constrained |
| 0.0 | Always pick cheapest model | Minimum cost, ignore accuracy |

**Impact on OOD (Out-of-Distribution) data:**

| Alpha | Router Distribution | Accuracy |
|-------|---------------------|----------|
| 0.8 | general: 100% | 73.7% |
| **0.5** | **math: 33%, coder: 40%, general: 27%** | **78.6%** |

With alpha=0.5, the router:
- Selects diverse models based on query semantics
- Achieves higher accuracy by matching query to specialist
- Reduces cost by preferring cheaper 7B models when appropriate

## 4. Generate Experience Database

```bash
# Requires: datasets, sentence_transformers, requests
python generate_mmlu_pro_experience.py
```

## 5. Run Router

```bash
cd /path/to/semantic-router
make build
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=candle-binding/target/release:$LD_LIBRARY_PATH \
  ./bin/router -config config/config.cluster_router.yaml
```

## 6. Test

```bash
# Single request through Envoy
curl http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is 2+2?"}]}'

# MMLU-Pro E2E accuracy test
python test_router_e2e.py

# OOD (Out-of-Distribution) E2E test
python test_ood_e2e.py 20  # Test 20 samples
```

## 7. Hot Reload

The router supports hot-reload of the experience database:

```bash
# Update experience DB - router automatically reloads
cp new_experience_db.json examples/cluster_router/mmlu_pro_experience_db_qwen3.json
# Router logs: "Hot-reloading experience database..."
```

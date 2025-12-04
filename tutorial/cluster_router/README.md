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
# Key settings
intelligent_routing:
  cluster_router:
    enabled: true
    n_clusters: 10
    alpha: 0.8  # 1.0=performance only, 0.0=cost only
    experience_db_path: "tutorial/cluster_router/mmlu_pro_experience_db_qwen3.json"
    embedding_model: "qwen3"
    embedding_dim: 1024
    model_costs:
      math: 7.0
      coder: 7.0
      general: 14.0
```

## 3. Generate Experience Database

```bash
# Requires: datasets, sentence_transformers, requests
python generate_mmlu_pro_experience.py
```

This queries MMLU-Pro questions against all models, evaluates correctness, and generates embeddings.

## 4. Run Router

```bash
cd /path/to/semantic-router
make build
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=candle-binding/target/release:$LD_LIBRARY_PATH \
  ./bin/router -config config/config.cluster_router.yaml
```

## 5. Test

```bash
# Through Envoy (port 8801)
curl http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is 2+2?"}]}'

# E2E accuracy test
python test_router_e2e.py
```

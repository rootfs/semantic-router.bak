# Cluster Router Tutorial

This tutorial demonstrates how to use the cluster-based model routing system.

## Files

| File | Description |
|------|-------------|
| `ood_evaluation_cache.json` | Cached inference results from multiple LLMs on OOD datasets |
| `convert_cache_to_experience_db.py` | Converts cache to experience database format |
| `evaluate_ood_generalization.py` | Original Python evaluation script (reference) |
| `analyze_*.py` | Various analysis scripts for routing performance |

## Quick Start

### 1. Convert Cached Inference to Experience Database

```bash
cd tutorial/cluster_router
python convert_cache_to_experience_db.py
```

This creates:
- `experience_db.json` - Full experience database
- `experience_db_sample.json` - 100-entry sample for testing

### 2. Use with Go Cluster Router

```go
package main

import (
    "log"
    
    candle "github.com/vllm-project/semantic-router/candle-binding"
    "github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
    "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func main() {
    // Load experience database
    cfg := &config.ClusterRouterConfig{
        NClusters:     10,
        Alpha:         0.8,  // 80% performance, 20% cost
        TopK:          3,
        Beta:          9.0,
        EmbeddingModel: "qwen3",
        EmbeddingDim:   768,
    }
    
    expDB := classification.NewExperienceDatabase(cfg)
    if err := expDB.LoadFromFile("tutorial/cluster_router/experience_db.json"); err != nil {
        log.Fatalf("Failed to load experience: %v", err)
    }
    
    // Create and train router
    modelCosts := map[string]float32{
        "math":    7.0,   // 7B model
        "coder":   7.0,   // 7B model
        "general": 14.0,  // 14B model
    }
    
    router := classification.NewClusterRouter(cfg, "general", []string{"math", "coder", "general"})
    records := expDB.ToExperienceRecords()
    
    if err := router.Train(records); err != nil {
        log.Fatalf("Failed to train: %v", err)
    }
    
    // Export for later use
    if err := candle.ExportClusterRouter("models/cluster_router"); err != nil {
        log.Fatalf("Failed to export: %v", err)
    }
    
    // Route a query
    result, err := router.RouteWithText("What is the derivative of x^2?")
    if err != nil {
        log.Fatalf("Failed to route: %v", err)
    }
    
    log.Printf("Routed to: %s (confidence: %.2f)", result.ModelName, result.Confidence)
}
```

## Experience Database Format

```json
[
    {
        "query_text": "What is 2+2?",
        "model_scores": {
            "math": 1.0,
            "coder": 0.0,
            "general": 1.0
        },
        "metadata": {
            "dataset": "arc-challenge",
            "question_id": "q123",
            "correct_answer": "A"
        }
    }
]
```

- `query_text`: The question/prompt (embedding will be generated)
- `model_scores`: Performance scores (1.0 = correct, 0.0 = incorrect)
- `metadata`: Optional debugging info

## Datasets in Cache

The `ood_evaluation_cache.json` contains results from:

| Dataset | Description | Type |
|---------|-------------|------|
| arc-challenge | Science reasoning | Multiple choice |
| openbookqa | Common sense | Multiple choice |
| sciq | Science QA | Multiple choice |
| commonsenseqa | Common sense reasoning | Multiple choice |
| truthfulqa | Factual accuracy | Multiple choice |
| hellaswag | Sentence completion | Multiple choice |
| gpqa | Graduate-level science | Multiple choice |

## Analysis Scripts

### Cost-Aware Routing Analysis
```bash
python analyze_cost_aware_routing.py
```
Analyzes the cost-performance tradeoff with different alpha values.

### Pareto Frontier Analysis
```bash
python analyze_pareto_frontier.py
```
Generates Pareto frontier plots comparing accuracy vs cost.

### MMLU-Pro Specific Analysis
```bash
python analyze_mmlu_pro_pareto.py
```
Analyzes routing on MMLU-Pro benchmark specifically.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clusters` | 10 | Number of K-means clusters |
| `alpha` | 1.0 | Cost-performance balance (1.0 = performance only) |
| `top_k` | 3 | Number of clusters for aggregation |
| `beta` | 9.0 | Softmax temperature (higher = more focused) |

## Model Costs

Example cost configuration based on model size:

```yaml
model_costs:
  math: 7.0      # 7B parameter model
  coder: 7.0     # 7B parameter model  
  general: 14.0  # 14B parameter model
```

With `alpha=0.8`, the router will prefer performance but consider cost when models are similar in accuracy.


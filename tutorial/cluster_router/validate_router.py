#!/usr/bin/env python3
"""
Validate Cluster Router Algorithm

This script validates the cluster router algorithm using the cached inference data.
It simulates what the Go cluster router does using pure Python for verification.

Usage:
    python validate_router.py
"""

import json
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from typing import List, Dict, Any, Tuple

# Model names
MODEL_NAMES = ["math", "coder", "general"]

# Model costs (based on model size)
MODEL_COSTS = {
    "math": 7.0,    # 7B
    "coder": 7.0,   # 7B
    "general": 14.0 # 14B
}

def load_cache(path: str = "ood_evaluation_cache.json") -> List[Dict]:
    """Load the OOD evaluation cache."""
    with open(path, "r") as f:
        return json.load(f)

def extract_model_scores(entry: Dict) -> Dict[str, float]:
    """Extract model scores from a cache entry."""
    model_responses = entry.get("model_responses", {})
    scores = {}
    for model in MODEL_NAMES:
        if model in model_responses:
            resp = model_responses[model]
            if isinstance(resp, dict):
                scores[model] = 1.0 if resp.get("is_correct", False) else 0.0
    return scores

def generate_simple_embeddings(entries: List[Dict], dim: int = 64) -> np.ndarray:
    """
    Generate simple embeddings based on text characteristics.
    This is a placeholder - real embeddings would come from a model.
    """
    np.random.seed(42)
    embeddings = np.random.randn(len(entries), dim).astype(np.float32)
    
    # Add some structure based on dataset
    dataset_vectors = {
        "arc-challenge": np.array([1, 0, 0, 0, 0, 0]),
        "openbookqa": np.array([0, 1, 0, 0, 0, 0]),
        "sciq": np.array([0, 0, 1, 0, 0, 0]),
        "commonsenseqa": np.array([0, 0, 0, 1, 0, 0]),
        "truthfulqa": np.array([0, 0, 0, 0, 1, 0]),
        "hellaswag": np.array([0, 0, 0, 0, 0, 1]),
    }
    
    for i, entry in enumerate(entries):
        dataset = entry.get("dataset", "unknown")
        if dataset in dataset_vectors:
            embeddings[i, :6] = dataset_vectors[dataset] * 2  # Bias embedding by dataset
    
    return embeddings

class SimpleClusterRouter:
    """Simple implementation of AvengersPro-style cluster routing."""
    
    def __init__(self, n_clusters: int = 10, top_k: int = 3, beta: float = 9.0, alpha: float = 1.0):
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.beta = beta
        self.alpha = alpha
        self.normalizer = Normalizer(norm='l2')
        self.kmeans = None
        self.cluster_rankings = {}
        self.cluster_scores = {}
    
    def train(self, embeddings: np.ndarray, model_scores: List[Dict[str, float]]):
        """Train the cluster router."""
        # Normalize embeddings
        embeddings_norm = self.normalizer.fit_transform(embeddings)
        
        # Fit K-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(embeddings_norm)
        
        # Compute cluster-wise rankings
        cluster_data = defaultdict(list)
        for i, cluster_id in enumerate(cluster_labels):
            cluster_data[cluster_id].append(model_scores[i])
        
        for cluster_id, scores_list in cluster_data.items():
            # Aggregate scores
            model_avg = defaultdict(list)
            for scores in scores_list:
                for model, score in scores.items():
                    model_avg[model].append(score)
            
            avg_scores = {model: np.mean(s) for model, s in model_avg.items()}
            
            # Sort by score descending
            sorted_models = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            
            self.cluster_rankings[cluster_id] = [m for m, _ in sorted_models]
            self.cluster_scores[cluster_id] = dict(sorted_models)
        
        print(f"Trained with {self.n_clusters} clusters")
        for cid in sorted(self.cluster_rankings.keys()):
            best = self.cluster_rankings[cid][0] if self.cluster_rankings[cid] else "N/A"
            score = self.cluster_scores[cid].get(best, 0)
            print(f"  Cluster {cid}: Best={best} ({score:.2%})")
    
    def route(self, embedding: np.ndarray) -> Tuple[str, float, int]:
        """Route a query to the best model."""
        # Normalize
        embedding_norm = self.normalizer.transform(embedding.reshape(1, -1))
        
        # Compute distances to all clusters
        distances = np.linalg.norm(self.kmeans.cluster_centers_ - embedding_norm, axis=1)
        
        # Get top-k clusters
        topk_indices = np.argsort(distances)[:self.top_k]
        topk_distances = distances[topk_indices]
        
        # Convert to probabilities (softmax with temperature beta)
        logits = -self.beta * topk_distances
        logits = logits - logits.max()  # For numerical stability
        probs = np.exp(logits) / np.exp(logits).sum()
        
        # Aggregate scores across clusters
        model_scores = defaultdict(float)
        for cluster_idx, prob in zip(topk_indices, probs):
            if cluster_idx not in self.cluster_rankings:
                continue
            
            rankings = self.cluster_rankings[cluster_idx]
            for rank, model in enumerate(rankings):
                rank_score = 1.0 / (rank + 1)
                model_scores[model] += prob * rank_score
        
        # Apply cost adjustment if alpha < 1
        if self.alpha < 1.0:
            max_cost = max(MODEL_COSTS.values())
            for model in model_scores:
                cost = MODEL_COSTS.get(model, 1.0)
                normalized_cost = cost / max_cost
                model_scores[model] = self.alpha * model_scores[model] - (1 - self.alpha) * normalized_cost
        
        # Find best model
        best_model = max(model_scores.items(), key=lambda x: x[1])
        primary_cluster = topk_indices[0]
        
        return best_model[0], best_model[1], primary_cluster

def evaluate_router(router: SimpleClusterRouter, test_embeddings: np.ndarray, 
                   test_scores: List[Dict[str, float]]) -> Dict[str, Any]:
    """Evaluate router accuracy."""
    correct = 0
    total = 0
    routing_dist = defaultdict(int)
    
    for i, (embedding, scores) in enumerate(zip(test_embeddings, test_scores)):
        model, confidence, cluster = router.route(embedding)
        routing_dist[model] += 1
        
        # Check if routed model got the answer correct
        if scores.get(model, 0) > 0.5:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    # Find oracle (best single model) accuracy
    model_correct = defaultdict(int)
    for scores in test_scores:
        for model, score in scores.items():
            if score > 0.5:
                model_correct[model] += 1
    
    best_single_model = max(model_correct.items(), key=lambda x: x[1])
    oracle_acc = best_single_model[1] / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "oracle_accuracy": oracle_acc,
        "best_single_model": best_single_model[0],
        "routing_distribution": dict(routing_dist),
        "total_queries": total
    }

def main():
    print("=== Cluster Router Validation ===\n")
    
    # Load cache
    print("Loading cached inference results...")
    cache = load_cache()
    print(f"Loaded {len(cache)} entries\n")
    
    # Extract model scores
    model_scores = [extract_model_scores(e) for e in cache]
    
    # Generate simple embeddings (placeholder)
    print("Generating embeddings...")
    embeddings = generate_simple_embeddings(cache)
    print(f"Embedding shape: {embeddings.shape}\n")
    
    # Split train/test (80/20)
    split_idx = int(len(cache) * 0.8)
    train_embeddings = embeddings[:split_idx]
    train_scores = model_scores[:split_idx]
    test_embeddings = embeddings[split_idx:]
    test_scores = model_scores[split_idx:]
    
    print(f"Train: {len(train_embeddings)}, Test: {len(test_embeddings)}\n")
    
    # Test different configurations
    configs = [
        {"n_clusters": 5, "top_k": 1, "alpha": 1.0, "name": "5 clusters, top-1, perf only"},
        {"n_clusters": 10, "top_k": 3, "alpha": 1.0, "name": "10 clusters, top-3, perf only"},
        {"n_clusters": 10, "top_k": 3, "alpha": 0.8, "name": "10 clusters, top-3, cost-aware"},
        {"n_clusters": 20, "top_k": 5, "alpha": 1.0, "name": "20 clusters, top-5, perf only"},
    ]
    
    print("=" * 70)
    print(f"{'Configuration':<35} {'Accuracy':>10} {'vs Oracle':>12} {'Routing':<20}")
    print("=" * 70)
    
    for cfg in configs:
        router = SimpleClusterRouter(
            n_clusters=cfg["n_clusters"],
            top_k=cfg["top_k"],
            alpha=cfg["alpha"]
        )
        router.train(train_embeddings, train_scores)
        
        results = evaluate_router(router, test_embeddings, test_scores)
        
        routing_str = ", ".join(f"{k}:{v}" for k, v in sorted(results["routing_distribution"].items()))
        delta = results["accuracy"] - results["oracle_accuracy"]
        
        print(f"{cfg['name']:<35} {results['accuracy']:>10.1%} {delta:>+10.1%} {routing_str}")
    
    print("=" * 70)
    print(f"\nOracle (best single model): {results['best_single_model']} ({results['oracle_accuracy']:.1%})")
    print(f"Model costs: {MODEL_COSTS}")
    
    # Per-dataset breakdown for best config
    print("\n=== Per-Dataset Performance (10 clusters, top-3) ===")
    router = SimpleClusterRouter(n_clusters=10, top_k=3, alpha=1.0)
    router.train(train_embeddings, train_scores)
    
    # Group test data by dataset
    dataset_results = defaultdict(lambda: {"correct": 0, "total": 0, "routing": defaultdict(int)})
    
    for i, (embedding, scores, entry) in enumerate(zip(test_embeddings, test_scores, cache[split_idx:])):
        dataset = entry.get("dataset", "unknown")
        model, _, _ = router.route(embedding)
        
        dataset_results[dataset]["total"] += 1
        dataset_results[dataset]["routing"][model] += 1
        if scores.get(model, 0) > 0.5:
            dataset_results[dataset]["correct"] += 1
    
    print(f"\n{'Dataset':<20} {'Router Acc':>12} {'Best Single':>12} {'Routing':<30}")
    print("-" * 80)
    
    for dataset in sorted(dataset_results.keys()):
        data = dataset_results[dataset]
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        
        # Find best single model for this dataset
        dataset_entries = [e for e in cache[split_idx:] if e.get("dataset") == dataset]
        model_correct = defaultdict(int)
        for entry in dataset_entries:
            scores = extract_model_scores(entry)
            for model, score in scores.items():
                if score > 0.5:
                    model_correct[model] += 1
        
        best = max(model_correct.items(), key=lambda x: x[1]) if model_correct else ("N/A", 0)
        best_acc = best[1] / len(dataset_entries) if dataset_entries else 0
        
        routing_str = ", ".join(f"{k}:{v}" for k, v in sorted(data["routing"].items()))
        print(f"{dataset:<20} {acc:>12.1%} {best_acc:>12.1%} {routing_str}")

if __name__ == "__main__":
    main()


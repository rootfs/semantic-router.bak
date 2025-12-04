#!/usr/bin/env python3
"""
Cost-Aware Routing Analysis

Analyzes the tradeoff between model accuracy and inference cost using
AvengersPro-style cluster routing with cost sensitivity.

Model costs based on parameter count:
- math: 7B (1.0x baseline)
- coder: 7B (1.0x)
- general: 14B (2.0x)
"""

import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import random
import os

# Set GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

random.seed(42)
np.random.seed(42)

# Model costs (relative, based on parameter count)
MODEL_COSTS = {
    'math': 1.0,      # 7B params - baseline cost
    'coder': 1.0,     # 7B params - same as math
    'general': 2.0,   # 14B params - 2x more expensive
}

MODELS = ['math', 'coder', 'general']


def extract_answer_key(response):
    """Extract answer key from model response."""
    if not response:
        return "NONE"
    response = response.strip()
    
    # Try \boxed{X} format (math model)
    match = re.search(r'\\boxed\{([A-Z])\}', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Try Answer: [X] or Answer: X format
    match = re.search(r'Answer:\s*\[?([A-Z])\]?', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Try "the answer is X" format
    match = re.search(r'(?:the\s+)?answer(?:\s+is)?[:\s]+\[?([A-Z])\]?', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Try leading letter
    match = re.match(r'^\[?([A-Z])\]?[.\s)]', response.upper())
    if match:
        return match.group(1)
    
    # Try trailing letter
    match = re.search(r'\b\[?([A-Z])\]?\s*$', response.upper())
    if match:
        return match.group(1)
    
    return "NONE"


class CostAwareClusterRouter:
    """
    Cluster-based router with cost sensitivity.
    
    Uses K-means clustering on embeddings to group similar queries,
    then learns which model performs best per cluster, adjusted for cost.
    
    Args:
        n_clusters: Number of clusters for K-means
        alpha: Weight for performance (0=cost only, 1=performance only)
        cost_sensitivity: How much to penalize cost (higher = more cost-sensitive)
    """
    
    def __init__(self, n_clusters=10, alpha=0.5, cost_sensitivity=1.0):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cost_sensitivity = cost_sensitivity
        self.cluster_centers = None
        self.cluster_best = None
        self.cluster_scores = None
    
    def fit(self, embeddings, records, model_costs):
        """
        Fit the router on training data.
        
        Args:
            embeddings: Query embeddings (N x D)
            records: List of dicts with model correctness {model: 0.0 or 1.0}
            model_costs: Dict of model costs {model: cost}
        """
        actual_clusters = min(self.n_clusters, len(embeddings) // 3)
        
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        self.cluster_centers = kmeans.cluster_centers_
        
        cluster_data = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            cluster_data[label].append(records[i])
        
        self.cluster_best = {}
        self.cluster_scores = {}
        
        for cluster_id, cluster_records in cluster_data.items():
            # Calculate performance scores
            perf_scores = {m: np.mean([r[m] for r in cluster_records]) for m in MODELS}
            
            # Calculate cost-adjusted scores
            max_cost = max(model_costs.values())
            cost_adjusted = {}
            for m in MODELS:
                perf = perf_scores[m]
                normalized_cost = model_costs[m] / max_cost
                # balance_score = alpha * performance - (1-alpha) * cost_penalty
                cost_adjusted[m] = self.alpha * perf - (1 - self.alpha) * self.cost_sensitivity * normalized_cost
            
            self.cluster_best[cluster_id] = max(cost_adjusted, key=cost_adjusted.get)
            self.cluster_scores[cluster_id] = {
                'perf': perf_scores,
                'cost_adjusted': cost_adjusted,
                'n': len(cluster_records)
            }
    
    def route(self, query_embedding):
        """Route a query to the best model based on nearest cluster."""
        distances = 1 - query_embedding @ self.cluster_centers.T
        nearest = np.argmin(distances)
        return self.cluster_best.get(nearest, 'general')


def load_and_prepare_data(cache_path):
    """Load OOD cache and prepare train/test split."""
    with open(cache_path) as f:
        ood_data = json.load(f)
    
    # Re-extract answers
    for r in ood_data:
        gt = r['ground_truth']
        for m in MODELS:
            raw = r['model_responses'][m]['raw_response']
            extracted = extract_answer_key(raw)
            r['model_responses'][m]['is_correct'] = (extracted == gt)
    
    # 80/20 split per dataset
    train_indices, test_indices = [], []
    for ds in set(r['dataset'] for r in ood_data):
        ds_indices = [i for i, r in enumerate(ood_data) if r['dataset'] == ds]
        random.shuffle(ds_indices)
        split = int(len(ds_indices) * 0.8)
        train_indices.extend(ds_indices[:split])
        test_indices.extend(ds_indices[split:])
    
    return ood_data, train_indices, test_indices


def prepare_texts_and_records(ood_data, indices):
    """Prepare text and correctness records for given indices."""
    texts, records, datasets = [], [], []
    for i in indices:
        r = ood_data[i]
        text = r['question']
        if r.get('options'):
            text += "\n" + "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(r['options'])])
        texts.append(text)
        records.append({m: 1.0 if r['model_responses'][m]['is_correct'] else 0.0 for m in MODELS})
        datasets.append(r['dataset'])
    return texts, records, datasets


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, 'ood_evaluation_cache.json')
    
    print("Loading data...")
    ood_data, train_indices, test_indices = load_and_prepare_data(cache_path)
    
    train_texts, train_records, _ = prepare_texts_and_records(ood_data, train_indices)
    test_texts, test_records, test_datasets = prepare_texts_and_records(ood_data, test_indices)
    
    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    # Load embedder
    print("Loading embedding model...")
    embedder = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda:0')
    normalizer = Normalizer(norm='l2')
    
    # Embed
    print("Embedding...")
    train_emb = normalizer.fit_transform(
        embedder.encode(train_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    )
    test_emb = normalizer.transform(
        embedder.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    )
    
    # Baselines
    single_correct = {m: sum(1 for r in test_records if r[m] == 1.0) for m in MODELS}
    n_test = len(test_records)
    
    print(f"\n{'='*100}")
    print("COST-AWARE ROUTING ANALYSIS")
    print(f"Model Costs: math=7B (1.0x), coder=7B (1.0x), general=14B (2.0x)")
    print("="*100)
    
    print(f"\nBaselines ({n_test} questions):")
    for m in MODELS:
        acc = 100 * single_correct[m] / n_test
        cost = MODEL_COSTS[m]
        efficiency = acc / cost
        print(f"  {m:8s}: {acc:5.1f}% accuracy, {cost:4.1f}x cost, efficiency={efficiency:5.1f}%/cost")
    
    # Alpha sweep
    print(f"\n{'Alpha':<8} | {'Accuracy':>10} | {'Avg Cost':>10} | {'Efficiency':>12} | Routing Distribution")
    print("-" * 90)
    
    results = []
    for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        router = CostAwareClusterRouter(n_clusters=10, alpha=alpha, cost_sensitivity=1.0)
        router.fit(train_emb, train_records, MODEL_COSTS)
        
        correct = 0
        total_cost = 0
        routing_dist = defaultdict(int)
        
        for i in range(len(test_emb)):
            selected = router.route(test_emb[i])
            routing_dist[selected] += 1
            total_cost += MODEL_COSTS[selected]
            if test_records[i][selected] == 1.0:
                correct += 1
        
        accuracy = 100 * correct / n_test
        avg_cost = total_cost / n_test
        efficiency = accuracy / avg_cost
        
        results.append({
            'alpha': alpha,
            'accuracy': accuracy,
            'avg_cost': avg_cost,
            'efficiency': efficiency,
            'routing': dict(routing_dist)
        })
        
        dist_str = f"m:{routing_dist['math']:3d}, c:{routing_dist['coder']:3d}, g:{routing_dist['general']:3d}"
        print(f"{alpha:<8.1f} | {accuracy:>9.1f}% | {avg_cost:>9.2f}x | {efficiency:>10.1f}%/cost | {dist_str}")
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print("="*100)
    
    best_perf = max(results, key=lambda x: x['accuracy'])
    best_eff = max(results, key=lambda x: x['efficiency'])
    
    print(f"\nBest Performance: α={best_perf['alpha']:.1f} -> {best_perf['accuracy']:.1f}% at {best_perf['avg_cost']:.2f}x cost")
    print(f"Best Efficiency:  α={best_eff['alpha']:.1f} -> {best_eff['accuracy']:.1f}% at {best_eff['avg_cost']:.2f}x cost = {best_eff['efficiency']:.1f}%/cost")


if __name__ == "__main__":
    main()


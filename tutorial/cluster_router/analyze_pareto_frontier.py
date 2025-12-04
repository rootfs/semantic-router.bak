#!/usr/bin/env python3
"""
Pareto Frontier Analysis for LLM Routing

Compares routing performance and cost tradeoffs between:
1. MMLU-Pro only training data
2. MMLU-Pro + OOD combined training data

Generates Pareto frontier plots showing optimal accuracy-cost tradeoffs
for both single models and AvengersPro-style cluster routing.

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set GPU before importing torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

random.seed(42)
np.random.seed(42)

MODEL_COSTS = {'math': 1.0, 'coder': 1.0, 'general': 2.0}
MODELS = ['math', 'coder', 'general']


def extract_answer_key(response):
    """Extract answer key from model response."""
    if not response:
        return "NONE"
    response = response.strip()
    
    match = re.search(r'\\boxed\{([A-Z])\}', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'Answer:\s*\[?([A-Z])\]?', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.search(r'(?:the\s+)?answer(?:\s+is)?[:\s]+\[?([A-Z])\]?', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    match = re.match(r'^\[?([A-Z])\]?[.\s)]', response.upper())
    if match:
        return match.group(1)
    
    match = re.search(r'\b\[?([A-Z])\]?\s*$', response.upper())
    if match:
        return match.group(1)
    
    return "NONE"


class CostAwareClusterRouter:
    """
    AvengersPro-style cluster router with cost awareness.
    
    Args:
        n_clusters: Number of K-means clusters
        alpha: Performance weight (0=cost only, 1=performance only)
    """
    
    def __init__(self, n_clusters=25, alpha=0.5):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = None
        self.cluster_best = None
    
    def fit(self, embeddings, records):
        actual_clusters = min(self.n_clusters, len(embeddings) // 3)
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        self.cluster_centers = kmeans.cluster_centers_
        
        cluster_data = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            cluster_data[label].append(records[i])
        
        self.cluster_best = {}
        for cluster_id, cluster_records in cluster_data.items():
            perf_scores = {m: np.mean([r[m] for r in cluster_records]) for m in MODELS}
            max_cost = max(MODEL_COSTS.values())
            cost_adjusted = {}
            for m in MODELS:
                normalized_cost = MODEL_COSTS[m] / max_cost
                cost_adjusted[m] = self.alpha * perf_scores[m] - (1 - self.alpha) * normalized_cost
            self.cluster_best[cluster_id] = max(cost_adjusted, key=cost_adjusted.get)
    
    def route(self, query_embedding):
        distances = 1 - query_embedding @ self.cluster_centers.T
        nearest = np.argmin(distances)
        return self.cluster_best.get(nearest, 'general')


def load_mmlu_data(cache_path):
    """Load MMLU-Pro cached data."""
    with open(cache_path) as f:
        mmlu_cache = json.load(f)['questions']
    
    texts, records = [], []
    seen = set()
    for item in mmlu_cache:
        q = item['question']
        if q in seen:
            continue
        seen.add(q)
        
        text = q
        opts = item.get('options', [])
        if opts:
            text += "\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(opts)])
        
        rec = {m: 1.0 if item['model_responses'].get(m, {}).get('is_correct', False) else 0.0 
               for m in MODELS}
        texts.append(text)
        records.append(rec)
    
    return texts, records


def load_ood_data(cache_path):
    """Load OOD cached data."""
    with open(cache_path) as f:
        ood_data = json.load(f)
    
    for r in ood_data:
        gt = r['ground_truth']
        for m in MODELS:
            raw = r['model_responses'][m]['raw_response']
            extracted = extract_answer_key(raw)
            r['model_responses'][m]['is_correct'] = (extracted == gt)
    
    texts, records = [], []
    for r in ood_data:
        text = r['question']
        if r.get('options'):
            text += "\n" + "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(r['options'])])
        texts.append(text)
        records.append({m: 1.0 if r['model_responses'][m]['is_correct'] else 0.0 for m in MODELS})
    
    return texts, records


def split_data(texts, records, train_ratio=0.8):
    """Split data into train/test sets."""
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(len(indices) * train_ratio)
    train_idx, test_idx = indices[:split], indices[split:]
    return (
        [texts[i] for i in train_idx],
        [records[i] for i in train_idx],
        [texts[i] for i in test_idx],
        [records[i] for i in test_idx]
    )


def evaluate_config(train_emb, train_records, test_emb, test_records, n_clusters, alpha):
    """Evaluate a router configuration."""
    router = CostAwareClusterRouter(n_clusters=n_clusters, alpha=alpha)
    router.fit(train_emb, train_records)
    
    correct, total_cost = 0, 0
    routing_dist = defaultdict(int)
    for i in range(len(test_emb)):
        selected = router.route(test_emb[i])
        routing_dist[selected] += 1
        total_cost += MODEL_COSTS[selected]
        if test_records[i][selected] == 1.0:
            correct += 1
    
    n = len(test_records)
    return {
        'accuracy': 100 * correct / n,
        'cost': total_cost / n,
        'routing': dict(routing_dist)
    }


def get_single_model_points(test_records):
    """Get accuracy/cost points for single models."""
    n = len(test_records)
    points = []
    for m in MODELS:
        acc = 100 * sum(1 for r in test_records if r[m] == 1.0) / n
        points.append({'name': m, 'accuracy': acc, 'cost': MODEL_COSTS[m]})
    return points


def get_router_points(train_emb, train_records, test_emb, test_records):
    """Get accuracy/cost points for various router configurations."""
    points = []
    for n_clusters in [10, 25, 50]:
        for alpha in np.arange(0.0, 1.05, 0.1):
            result = evaluate_config(train_emb, train_records, test_emb, test_records, n_clusters, alpha)
            points.append({
                'n_clusters': n_clusters,
                'alpha': alpha,
                'accuracy': result['accuracy'],
                'cost': result['cost']
            })
    return points


def find_pareto_frontier(points):
    """Find Pareto optimal points (not dominated by any other point)."""
    pareto = []
    for p in points:
        dominated = False
        for other in points:
            if (other['accuracy'] > p['accuracy'] and other['cost'] <= p['cost']) or \
               (other['accuracy'] >= p['accuracy'] and other['cost'] < p['cost']):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return sorted(pareto, key=lambda x: x['cost'])


def plot_pareto_frontiers(mmlu_data, combined_data, output_path):
    """Generate Pareto frontier comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    model_colors = {'math': '#e74c3c', 'coder': '#3498db', 'general': '#2ecc71'}
    
    for ax, (title, single_points, router_points, pareto_points) in zip(axes, [mmlu_data, combined_data]):
        # Plot all router points (gray)
        router_costs = [p['cost'] for p in router_points]
        router_accs = [p['accuracy'] for p in router_points]
        ax.scatter(router_costs, router_accs, s=40, c='lightgray', alpha=0.6, 
                   label='Router configs', zorder=1)
        
        # Plot single models (large squares)
        for p in single_points:
            ax.scatter(p['cost'], p['accuracy'], s=250, c=model_colors[p['name']], 
                       marker='s', label=f"Always {p['name']}", zorder=4, 
                       edgecolors='black', linewidths=2)
            ax.annotate(f"{p['name']}\n{p['accuracy']:.1f}%", 
                        (p['cost'], p['accuracy']), 
                        textcoords="offset points", xytext=(15, 0), 
                        fontsize=10, fontweight='bold')
        
        # Plot Pareto frontier routers (orange stars)
        pareto_routers = [p for p in pareto_points if 'n_clusters' in p]
        if pareto_routers:
            pareto_costs = [p['cost'] for p in pareto_routers]
            pareto_accs = [p['accuracy'] for p in pareto_routers]
            ax.scatter(pareto_costs, pareto_accs, s=200, c='orange', marker='*', 
                       label='Pareto optimal routers', zorder=3, edgecolors='black', linewidths=1)
        
        # Draw Pareto frontier line
        all_pareto_sorted = sorted(pareto_points, key=lambda x: x['cost'])
        pareto_line_costs = [p['cost'] for p in all_pareto_sorted]
        pareto_line_accs = [p['accuracy'] for p in all_pareto_sorted]
        ax.plot(pareto_line_costs, pareto_line_accs, 'k--', alpha=0.7, linewidth=2, 
                label='Pareto frontier', zorder=2)
        
        # Labels and styling
        ax.set_xlabel('Average Cost (relative to 7B model)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set axis limits with padding
        all_costs = router_costs + [p['cost'] for p in single_points]
        all_accs = router_accs + [p['accuracy'] for p in single_points]
        ax.set_xlim(min(all_costs) - 0.1, max(all_costs) + 0.2)
        ax.set_ylim(min(all_accs) - 3, max(all_accs) + 3)
    
    plt.suptitle('Pareto Frontiers: Accuracy vs Cost\nAvengersPro-style Cluster Routing with Cost Awareness', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mmlu_cache_path = os.path.join(script_dir, '..', 'mmlu_specialized_cache.json')
    ood_cache_path = os.path.join(script_dir, 'ood_evaluation_cache.json')
    
    # Load data
    print("Loading data...")
    mmlu_texts, mmlu_records = load_mmlu_data(mmlu_cache_path)
    ood_texts, ood_records = load_ood_data(ood_cache_path)
    
    combined_texts = mmlu_texts + ood_texts
    combined_records = mmlu_records + ood_records
    
    print(f"MMLU-Pro: {len(mmlu_texts)}, OOD: {len(ood_texts)}, Combined: {len(combined_texts)}")
    
    # Load embedder
    print("Loading embedding model...")
    embedder = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda:0')
    normalizer = Normalizer(norm='l2')
    
    # Split data
    mmlu_train_t, mmlu_train_r, mmlu_test_t, mmlu_test_r = split_data(mmlu_texts, mmlu_records)
    combined_train_t, combined_train_r, combined_test_t, combined_test_r = split_data(combined_texts, combined_records)
    
    print(f"MMLU-Pro: Train={len(mmlu_train_t)}, Test={len(mmlu_test_t)}")
    print(f"Combined: Train={len(combined_train_t)}, Test={len(combined_test_t)}")
    
    # Embed
    print("Embedding...")
    mmlu_train_emb = normalizer.fit_transform(
        embedder.encode(mmlu_train_t, convert_to_numpy=True, normalize_embeddings=True, batch_size=32))
    mmlu_test_emb = normalizer.transform(
        embedder.encode(mmlu_test_t, convert_to_numpy=True, normalize_embeddings=True, batch_size=32))
    combined_train_emb = normalizer.transform(
        embedder.encode(combined_train_t, convert_to_numpy=True, normalize_embeddings=True, batch_size=32))
    combined_test_emb = normalizer.transform(
        embedder.encode(combined_test_t, convert_to_numpy=True, normalize_embeddings=True, batch_size=32))
    
    # Evaluate
    print("\nEvaluating configurations...")
    
    mmlu_single = get_single_model_points(mmlu_test_r)
    mmlu_router = get_router_points(mmlu_train_emb, mmlu_train_r, mmlu_test_emb, mmlu_test_r)
    mmlu_pareto = find_pareto_frontier(mmlu_single + mmlu_router)
    
    combined_single = get_single_model_points(combined_test_r)
    combined_router = get_router_points(combined_train_emb, combined_train_r, combined_test_emb, combined_test_r)
    combined_pareto = find_pareto_frontier(combined_single + combined_router)
    
    # Print results
    print(f"\n{'='*80}")
    print("MMLU-PRO ONLY")
    print("="*80)
    print("\nSingle Models:")
    for p in mmlu_single:
        print(f"  {p['name']:8s}: {p['accuracy']:.1f}% @ {p['cost']:.1f}x")
    print("\nPareto Frontier:")
    for p in mmlu_pareto:
        name = p.get('name', f"Router(c={p.get('n_clusters')},α={p.get('alpha', 0):.1f})")
        print(f"  {name:<30s}: {p['accuracy']:.1f}% @ {p['cost']:.2f}x")
    
    print(f"\n{'='*80}")
    print("MMLU-PRO + OOD (Combined)")
    print("="*80)
    print("\nSingle Models:")
    for p in combined_single:
        print(f"  {p['name']:8s}: {p['accuracy']:.1f}% @ {p['cost']:.1f}x")
    print("\nPareto Frontier:")
    for p in combined_pareto:
        name = p.get('name', f"Router(c={p.get('n_clusters')},α={p.get('alpha', 0):.1f})")
        print(f"  {name:<30s}: {p['accuracy']:.1f}% @ {p['cost']:.2f}x")
    
    # Generate plot
    print("\nGenerating plot...")
    output_path = os.path.join(script_dir, 'pareto_comparison.png')
    plot_pareto_frontiers(
        ("MMLU-Pro Only", mmlu_single, mmlu_router, mmlu_pareto),
        ("MMLU-Pro + OOD (Combined)", combined_single, combined_router, combined_pareto),
        output_path
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    
    mmlu_best_single = max(mmlu_single, key=lambda x: x['accuracy'])
    mmlu_best_router = max(mmlu_router, key=lambda x: x['accuracy'])
    combined_best_single = max(combined_single, key=lambda x: x['accuracy'])
    combined_best_router = max(combined_router, key=lambda x: x['accuracy'])
    
    print(f"\n{'Metric':<35} | {'MMLU-Pro Only':>15} | {'Combined':>15}")
    print("-" * 75)
    print(f"{'Best Single Model':<35} | {mmlu_best_single['name']:>8s} {mmlu_best_single['accuracy']:>5.1f}% | "
          f"{combined_best_single['name']:>8s} {combined_best_single['accuracy']:>5.1f}%")
    print(f"{'Best Router Accuracy':<35} | {mmlu_best_router['accuracy']:>14.1f}% | "
          f"{combined_best_router['accuracy']:>14.1f}%")
    print(f"{'Router vs Best Single':<35} | {mmlu_best_router['accuracy'] - mmlu_best_single['accuracy']:>+14.1f}% | "
          f"{combined_best_router['accuracy'] - combined_best_single['accuracy']:>+14.1f}%")


if __name__ == "__main__":
    main()


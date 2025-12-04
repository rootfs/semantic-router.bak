#!/usr/bin/env python3
"""
MMLU-Pro Pareto Frontier Analysis

Detailed analysis of accuracy-cost tradeoffs using AvengersPro-style
cluster routing on MMLU-Pro dataset only.

Generates:
1. Pareto frontier of router configurations
2. Routing distribution by alpha (cost sensitivity)
3. Comparison with single model baselines
4. Pareto frontier visualization plot

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


class CostAwareClusterRouter:
    """AvengersPro-style cluster router with cost awareness."""
    
    def __init__(self, n_clusters=25, alpha=0.5, cost_sensitivity=1.0):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cost_sensitivity = cost_sensitivity
        self.cluster_centers = None
        self.cluster_best = None
    
    def fit(self, embeddings, records, model_costs):
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
            perf_scores = {m: np.mean([r[m] for r in cluster_records]) for m in MODELS}
            
            max_cost = max(model_costs.values())
            cost_adjusted = {}
            for m in MODELS:
                perf = perf_scores[m]
                normalized_cost = model_costs[m] / max_cost
                cost_adjusted[m] = self.alpha * perf - (1 - self.alpha) * self.cost_sensitivity * normalized_cost
            
            self.cluster_best[cluster_id] = max(cost_adjusted, key=cost_adjusted.get)
            self.cluster_scores[cluster_id] = {'perf': perf_scores, 'n': len(cluster_records)}
    
    def route(self, query_embedding):
        distances = 1 - query_embedding @ self.cluster_centers.T
        nearest = np.argmin(distances)
        return self.cluster_best.get(nearest, 'general')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mmlu_cache_path = os.path.join(script_dir, '..', 'mmlu_specialized_cache.json')
    
    # Load MMLU-Pro data
    print("Loading MMLU-Pro data...")
    with open(mmlu_cache_path) as f:
        mmlu_cache = json.load(f)['questions']
    print(f"Loaded {len(mmlu_cache)} MMLU-Pro entries")
    
    # Load embedder
    print("Loading bge-base-en-v1.5...")
    embedder = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda:0')
    normalizer = Normalizer(norm='l2')
    
    # Prepare data
    mmlu_texts, mmlu_records = [], []
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
        mmlu_texts.append(text)
        mmlu_records.append(rec)
    
    # 80/20 split
    indices = list(range(len(mmlu_texts)))
    random.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_indices = indices[:split]
    test_indices = indices[split:]
    
    train_texts = [mmlu_texts[i] for i in train_indices]
    train_records = [mmlu_records[i] for i in train_indices]
    test_texts = [mmlu_texts[i] for i in test_indices]
    test_records = [mmlu_records[i] for i in test_indices]
    
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")
    
    # Embed
    print("Embedding...")
    train_emb = normalizer.fit_transform(
        embedder.encode(train_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32))
    test_emb = normalizer.transform(
        embedder.encode(test_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32))
    
    # Calculate baselines
    n_test = len(test_records)
    single_correct = {m: sum(1 for r in test_records if r[m] == 1.0) for m in MODELS}
    oracle_correct = sum(1 for r in test_records if any(r[m] == 1.0 for m in MODELS))
    
    print(f"\n{'='*100}")
    print("MMLU-PRO PARETO FRONTIER ANALYSIS")
    print("="*100)
    
    print(f"\n--- Single Model Baselines ({n_test} questions) ---")
    single_model_points = []
    for m in MODELS:
        acc = 100 * single_correct[m] / n_test
        cost = MODEL_COSTS[m]
        eff = acc / cost
        single_model_points.append({'name': f'Always {m}', 'accuracy': acc, 'cost': cost, 'efficiency': eff})
        print(f"  {m:8s}: {acc:5.1f}% accuracy, {cost:.1f}x cost, {eff:.1f}%/cost")
    
    print(f"\n  Oracle:   {100*oracle_correct/n_test:.1f}% accuracy")
    
    # Grid search over router configurations
    print(f"\n--- Router Configurations ---")
    router_points = []
    
    for n_clusters in [10, 25, 50]:
        for alpha in np.arange(0.0, 1.05, 0.1):
            router = CostAwareClusterRouter(n_clusters=n_clusters, alpha=alpha, cost_sensitivity=1.0)
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
            
            router_points.append({
                'name': f'Router(c={n_clusters},α={alpha:.1f})',
                'n_clusters': n_clusters,
                'alpha': alpha,
                'accuracy': accuracy,
                'cost': avg_cost,
                'efficiency': efficiency,
                'routing': dict(routing_dist)
            })
    
    # Find Pareto frontier
    def is_pareto_optimal(point, all_points):
        for other in all_points:
            if other['accuracy'] > point['accuracy'] and other['cost'] < point['cost']:
                return False
            if other['accuracy'] >= point['accuracy'] and other['cost'] < point['cost']:
                return False
            if other['accuracy'] > point['accuracy'] and other['cost'] <= point['cost']:
                return False
        return True
    
    all_points = single_model_points + router_points
    pareto_points = [p for p in all_points if is_pareto_optimal(p, all_points)]
    pareto_points.sort(key=lambda x: x['cost'])
    
    print(f"\n{'='*100}")
    print("PARETO FRONTIER (Optimal Accuracy-Cost Tradeoffs)")
    print("="*100)
    
    print(f"\n{'Configuration':<35} | {'Accuracy':>10} | {'Cost':>8} | {'Efficiency':>12}")
    print("-" * 75)
    
    for p in pareto_points:
        print(f"{p['name']:<35} | {p['accuracy']:>9.1f}% | {p['cost']:>7.2f}x | {p['efficiency']:>10.1f}%/cost")
    
    # Top configurations
    print(f"\n{'='*100}")
    print("TOP ROUTER CONFIGURATIONS BY ACCURACY")
    print("="*100)
    
    router_points.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n{'Config':<35} | {'Accuracy':>10} | {'Cost':>8} | {'Efficiency':>12} | Routing")
    print("-" * 110)
    
    for p in router_points[:10]:
        routing_str = f"m:{p['routing'].get('math',0):3d}, c:{p['routing'].get('coder',0):3d}, g:{p['routing'].get('general',0):3d}"
        print(f"{p['name']:<35} | {p['accuracy']:>9.1f}% | {p['cost']:>7.2f}x | {p['efficiency']:>10.1f}%/cost | {routing_str}")
    
    # Routing distribution by alpha
    print(f"\n{'='*100}")
    print("ROUTING DISTRIBUTION BY ALPHA (n_clusters=25)")
    print("="*100)
    
    print(f"\n{'Alpha':<8} | {'math':>8} | {'coder':>8} | {'general':>8} | {'Accuracy':>10} | {'Cost':>8}")
    print("-" * 70)
    
    for p in sorted([x for x in router_points if x['n_clusters'] == 25], key=lambda x: x['alpha'], reverse=True):
        r = p['routing']
        total = sum(r.values())
        m_pct = 100 * r.get('math', 0) / total
        c_pct = 100 * r.get('coder', 0) / total
        g_pct = 100 * r.get('general', 0) / total
        print(f"{p['alpha']:<8.1f} | {m_pct:>7.1f}% | {c_pct:>7.1f}% | {g_pct:>7.1f}% | {p['accuracy']:>9.1f}% | {p['cost']:>7.2f}x")
    
    # Generate plot
    print(f"\n{'='*100}")
    print("GENERATING PARETO FRONTIER PLOT")
    print("="*100)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot single models
    model_colors = {'Always math': '#e74c3c', 'Always coder': '#3498db', 'Always general': '#2ecc71'}
    for p in single_model_points:
        ax.scatter(p['cost'], p['accuracy'], s=200, c=model_colors[p['name']], 
                   marker='s', label=p['name'], zorder=5, edgecolors='black', linewidths=2)
    
    # Plot router points
    router_accs = [p['accuracy'] for p in router_points]
    router_costs = [p['cost'] for p in router_points]
    ax.scatter(router_costs, router_accs, s=50, c='gray', alpha=0.5, label='Router configs')
    
    # Highlight Pareto frontier
    pareto_router = [p for p in pareto_points if 'Router' in p['name']]
    if pareto_router:
        pareto_costs = [p['cost'] for p in pareto_router]
        pareto_accs = [p['accuracy'] for p in pareto_router]
        ax.scatter(pareto_costs, pareto_accs, s=150, c='orange', marker='*', 
                   label='Pareto optimal routers', zorder=4, edgecolors='black')
        
        # Connect Pareto points
        all_pareto = sorted(pareto_points, key=lambda x: x['cost'])
        pareto_costs_all = [p['cost'] for p in all_pareto]
        pareto_accs_all = [p['accuracy'] for p in all_pareto]
        ax.plot(pareto_costs_all, pareto_accs_all, 'k--', alpha=0.5, label='Pareto frontier')
    
    ax.set_xlabel('Average Cost (relative to 7B model)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('MMLU-Pro: Accuracy vs Cost Pareto Frontier\n(AvengersPro-style Cluster Routing)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(script_dir, 'mmlu_pro_pareto_frontier.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print("="*100)
    
    best_single = max(single_model_points, key=lambda x: x['accuracy'])
    best_router_acc = max(router_points, key=lambda x: x['accuracy'])
    best_router_eff = max(router_points, key=lambda x: x['efficiency'])
    
    print(f"\n{'Strategy':<35} | {'Accuracy':>10} | {'Cost':>8} | {'Efficiency':>12}")
    print("-" * 75)
    print(f"{'Best Single (general)':<35} | {best_single['accuracy']:>9.1f}% | {best_single['cost']:>7.2f}x | {best_single['efficiency']:>10.1f}%/cost")
    print(f"{'Best Router (accuracy)':<35} | {best_router_acc['accuracy']:>9.1f}% | {best_router_acc['cost']:>7.2f}x | {best_router_acc['efficiency']:>10.1f}%/cost")
    print(f"{'Best Router (efficiency)':<35} | {best_router_eff['accuracy']:>9.1f}% | {best_router_eff['cost']:>7.2f}x | {best_router_eff['efficiency']:>10.1f}%/cost")
    
    print(f"\n*** Router beats best single model by {best_router_acc['accuracy'] - best_single['accuracy']:+.1f}% accuracy")
    print(f"*** Best efficiency router saves {(1 - best_router_eff['cost']/best_single['cost'])*100:.0f}% cost vs always-general")


if __name__ == "__main__":
    main()


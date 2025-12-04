#!/usr/bin/env python3
"""
Training Domain Analysis

Compares routing performance when training on:
1. MMLU-Pro (in-domain for the original router)
2. OOD datasets (arc-challenge, openbookqa, sciq, commonsenseqa, hellaswag)

Key finding: Training domain matters significantly for generalization.
"""

import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import random
import os

# Set GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

random.seed(42)
np.random.seed(42)

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


class HardClusterRouter:
    """Simple cluster router that routes to nearest cluster's best model."""
    
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.cluster_centers = None
        self.cluster_best = None
        self.cluster_scores = None
    
    def fit(self, embeddings, records):
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
            model_scores = {m: np.mean([r[m] for r in cluster_records]) for m in MODELS}
            self.cluster_best[cluster_id] = max(model_scores, key=model_scores.get)
            self.cluster_scores[cluster_id] = {'scores': model_scores, 'n': len(cluster_records)}
    
    def route(self, query_embedding):
        distances = 1 - query_embedding @ self.cluster_centers.T
        nearest = np.argmin(distances)
        return self.cluster_best.get(nearest, 'general')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ood_cache_path = os.path.join(script_dir, 'ood_evaluation_cache.json')
    mmlu_cache_path = os.path.join(script_dir, '..', 'mmlu_specialized_cache.json')
    
    # Load OOD cache
    print("Loading OOD data...")
    with open(ood_cache_path) as f:
        ood_data = json.load(f)
    
    for r in ood_data:
        gt = r['ground_truth']
        for m in MODELS:
            raw = r['model_responses'][m]['raw_response']
            extracted = extract_answer_key(raw)
            r['model_responses'][m]['is_correct'] = (extracted == gt)
    
    print(f"Loaded {len(ood_data)} OOD entries")
    print(f"By dataset: {dict(Counter(r['dataset'] for r in ood_data))}")
    
    # Load MMLU-Pro cache
    print("\nLoading MMLU-Pro data...")
    with open(mmlu_cache_path) as f:
        mmlu_cache = json.load(f)['questions']
    print(f"Loaded {len(mmlu_cache)} MMLU-Pro entries")
    
    # Load embedder
    print("\nLoading embedding model...")
    embedder = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda:0')
    normalizer = Normalizer(norm='l2')
    
    # Prepare MMLU-Pro training data
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
        
        rec = {}
        for m in MODELS:
            rec[m] = 1.0 if item['model_responses'].get(m, {}).get('is_correct', False) else 0.0
        
        mmlu_texts.append(text)
        mmlu_records.append(rec)
    
    # Prepare OOD test data (all of it)
    ood_texts, ood_records, ood_datasets = [], [], []
    for r in ood_data:
        text = r['question']
        if r.get('options'):
            text += "\n" + "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(r['options'])])
        ood_texts.append(text)
        ood_records.append({m: 1.0 if r['model_responses'][m]['is_correct'] else 0.0 for m in MODELS})
        ood_datasets.append(r['dataset'])
    
    # OOD 80/20 split for OOD-trained router
    ood_train_indices, ood_test_indices = [], []
    for ds in set(r['dataset'] for r in ood_data):
        ds_indices = [i for i, r in enumerate(ood_data) if r['dataset'] == ds]
        random.shuffle(ds_indices)
        split = int(len(ds_indices) * 0.8)
        ood_train_indices.extend(ds_indices[:split])
        ood_test_indices.extend(ds_indices[split:])
    
    ood_train_texts = [ood_texts[i] for i in ood_train_indices]
    ood_train_records = [ood_records[i] for i in ood_train_indices]
    ood_test_texts = [ood_texts[i] for i in ood_test_indices]
    ood_test_records = [ood_records[i] for i in ood_test_indices]
    ood_test_datasets = [ood_datasets[i] for i in ood_test_indices]
    
    print(f"\nMMLP-Pro Train: {len(mmlu_texts)}")
    print(f"OOD Train: {len(ood_train_texts)}, OOD Test: {len(ood_test_texts)}")
    
    # Embed all data
    print("\nEmbedding...")
    mmlu_emb = normalizer.fit_transform(
        embedder.encode(mmlu_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    )
    ood_train_emb = normalizer.transform(
        embedder.encode(ood_train_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    )
    ood_test_emb = normalizer.transform(
        embedder.encode(ood_test_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    )
    ood_all_emb = normalizer.transform(
        embedder.encode(ood_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    )
    
    # Baselines on OOD test set
    single_correct = {m: sum(1 for r in ood_test_records if r[m] == 1.0) for m in MODELS}
    n_test = len(ood_test_records)
    best_model = max(single_correct, key=single_correct.get)
    best_single_acc = 100 * single_correct[best_model] / n_test
    
    print(f"\n{'='*100}")
    print("TRAINING DOMAIN COMPARISON")
    print("="*100)
    
    print(f"\nBaselines on OOD Test ({n_test} questions):")
    for m in MODELS:
        marker = " <-- BEST" if m == best_model else ""
        print(f"  {m:8s}: {100*single_correct[m]/n_test:.1f}%{marker}")
    
    # Test 1: MMLU-Pro trained -> OOD test (all)
    print(f"\n{'='*100}")
    print("1. MMLU-PRO TRAINED -> OOD TEST (ALL)")
    print("="*100)
    
    router_mmlu = HardClusterRouter(n_clusters=50)
    router_mmlu.fit(mmlu_emb, mmlu_records)
    
    correct = 0
    routing_dist = defaultdict(int)
    for i in range(len(ood_all_emb)):
        selected = router_mmlu.route(ood_all_emb[i])
        routing_dist[selected] += 1
        if ood_records[i][selected] == 1.0:
            correct += 1
    
    mmlu_acc = 100 * correct / len(ood_records)
    mmlu_best = max(single_correct, key=single_correct.get)
    mmlu_best_acc = 100 * sum(1 for r in ood_records if r[mmlu_best] == 1.0) / len(ood_records)
    
    print(f"Router Accuracy: {mmlu_acc:.1f}%")
    print(f"Best Single ({mmlu_best}): {mmlu_best_acc:.1f}%")
    print(f"Delta: {mmlu_acc - mmlu_best_acc:+.1f}%")
    print(f"Routing: {dict(routing_dist)}")
    
    # Test 2: OOD trained -> OOD test (80/20)
    print(f"\n{'='*100}")
    print("2. OOD TRAINED -> OOD TEST (80/20 split)")
    print("="*100)
    
    router_ood = HardClusterRouter(n_clusters=10)
    router_ood.fit(ood_train_emb, ood_train_records)
    
    correct = 0
    routing_dist = defaultdict(int)
    for i in range(len(ood_test_emb)):
        selected = router_ood.route(ood_test_emb[i])
        routing_dist[selected] += 1
        if ood_test_records[i][selected] == 1.0:
            correct += 1
    
    ood_acc = 100 * correct / n_test
    
    print(f"Router Accuracy: {ood_acc:.1f}%")
    print(f"Best Single ({best_model}): {best_single_acc:.1f}%")
    print(f"Delta: {ood_acc - best_single_acc:+.1f}%")
    print(f"Routing: {dict(routing_dist)}")
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY: Training Domain Impact")
    print("="*100)
    
    print(f"\n{'Training Data':<25} | {'Accuracy':>10} | {'vs Best Single':>15}")
    print("-" * 55)
    print(f"{'MMLU-Pro':<25} | {mmlu_acc:>9.1f}% | {mmlu_acc - mmlu_best_acc:>+14.1f}%")
    print(f"{'OOD (80/20)':<25} | {ood_acc:>9.1f}% | {ood_acc - best_single_acc:>+14.1f}%")
    print(f"\nGap: {ood_acc - mmlu_acc:+.1f}% (OOD training is better)")


if __name__ == "__main__":
    main()


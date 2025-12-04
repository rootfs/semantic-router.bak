#!/usr/bin/env python3
"""
Embedding Model Comparison for Routing

Compares different embedding models for cluster-based routing performance.
Tests BGE and GTE models of various sizes.
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
import torch

# Set GPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

random.seed(42)
np.random.seed(42)

MODELS = ['math', 'coder', 'general']

# Embedding models to compare
EMBEDDING_MODELS = [
    ("BAAI/bge-small-en-v1.5", "bge-small (33M)"),
    ("BAAI/bge-base-en-v1.5", "bge-base (110M)"),
    ("BAAI/bge-large-en-v1.5", "bge-large (335M)"),
    ("thenlper/gte-large", "gte-large (335M)"),
]


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
    
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
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
            model_scores = {m: np.mean([r[m] for r in cluster_records]) for m in MODELS}
            self.cluster_best[cluster_id] = max(model_scores, key=model_scores.get)
    
    def route(self, query_embedding):
        distances = 1 - query_embedding @ self.cluster_centers.T
        nearest = np.argmin(distances)
        return self.cluster_best.get(nearest, 'general')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, 'ood_evaluation_cache.json')
    
    # Load data
    print("Loading data...")
    with open(cache_path) as f:
        ood_data = json.load(f)
    
    for r in ood_data:
        gt = r['ground_truth']
        for m in MODELS:
            raw = r['model_responses'][m]['raw_response']
            extracted = extract_answer_key(raw)
            r['model_responses'][m]['is_correct'] = (extracted == gt)
    
    # 80/20 split
    train_indices, test_indices = [], []
    for ds in set(r['dataset'] for r in ood_data):
        ds_indices = [i for i, r in enumerate(ood_data) if r['dataset'] == ds]
        random.shuffle(ds_indices)
        split = int(len(ds_indices) * 0.8)
        train_indices.extend(ds_indices[:split])
        test_indices.extend(ds_indices[split:])
    
    # Prepare data
    train_texts, train_records = [], []
    for i in train_indices:
        r = ood_data[i]
        text = r['question']
        if r.get('options'):
            text += "\n" + "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(r['options'])])
        train_texts.append(text)
        train_records.append({m: 1.0 if r['model_responses'][m]['is_correct'] else 0.0 for m in MODELS})
    
    test_texts, test_records = [], []
    for i in test_indices:
        r = ood_data[i]
        text = r['question']
        if r.get('options'):
            text += "\n" + "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(r['options'])])
        test_texts.append(text)
        test_records.append({m: 1.0 if r['model_responses'][m]['is_correct'] else 0.0 for m in MODELS})
    
    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    # Baselines
    single_correct = {m: sum(1 for r in test_records if r[m] == 1.0) for m in MODELS}
    n_test = len(test_records)
    best_model = max(single_correct, key=single_correct.get)
    best_single_acc = 100 * single_correct[best_model] / n_test
    
    print(f"\nBaselines: math={100*single_correct['math']/n_test:.1f}%, "
          f"coder={100*single_correct['coder']/n_test:.1f}%, "
          f"general={100*single_correct['general']/n_test:.1f}%")
    print(f"Best Single: {best_model} ({best_single_acc:.1f}%)")
    
    normalizer = Normalizer(norm='l2')
    
    # Compare embedding models
    print(f"\n{'='*85}")
    print("EMBEDDING MODEL COMPARISON (Hard Cluster Router, n_clusters=8)")
    print("="*85)
    
    print(f"\n{'Model':<25} | {'Dim':>6} | {'Accuracy':>10} | {'vs Best':>10} | Routing")
    print("-" * 85)
    
    results = []
    
    for model_path, model_name in EMBEDDING_MODELS:
        try:
            print(f"Loading {model_name}...", end=" ", flush=True)
            embedder = SentenceTransformer(model_path, device='cuda:0')
            
            # Embed
            train_emb = embedder.encode(train_texts, convert_to_numpy=True, 
                                        normalize_embeddings=True, batch_size=32)
            test_emb = embedder.encode(test_texts, convert_to_numpy=True, 
                                       normalize_embeddings=True, batch_size=32)
            
            train_emb = normalizer.fit_transform(train_emb)
            test_emb = normalizer.transform(test_emb)
            
            dim = train_emb.shape[1]
            
            # Train and evaluate
            router = HardClusterRouter(n_clusters=8)
            router.fit(train_emb, train_records)
            
            correct = 0
            routing_dist = defaultdict(int)
            
            for i in range(len(test_emb)):
                selected = router.route(test_emb[i])
                routing_dist[selected] += 1
                if test_records[i][selected] == 1.0:
                    correct += 1
            
            accuracy = 100 * correct / n_test
            vs_best = accuracy - best_single_acc
            
            results.append((model_name, dim, accuracy, vs_best, dict(routing_dist)))
            
            dist_str = f"m:{routing_dist['math']:3d}, c:{routing_dist['coder']:3d}, g:{routing_dist['general']:3d}"
            print(f"\r{model_name:<25} | {dim:>6} | {accuracy:>9.1f}% | {vs_best:>+9.1f}% | {dist_str}")
            
            # Clear GPU memory
            del embedder
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\r{model_name:<25} | ERROR: {str(e)[:40]}")
    
    # Summary
    print(f"\n{'='*85}")
    print("SUMMARY")
    print("="*85)
    
    results.sort(key=lambda x: x[2], reverse=True)
    print(f"\nBest Embedding Model: {results[0][0]} with {results[0][2]:.1f}% accuracy")
    print(f"vs Best Single Model ({best_model}): {results[0][3]:+.1f}%")


if __name__ == "__main__":
    main()


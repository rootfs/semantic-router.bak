#!/usr/bin/env python3
"""
OOD (Out-of-Distribution) E2E Test for Cluster Router

Tests router accuracy on datasets NOT in the training experience database.
The router was trained on MMLU-Pro; this tests on ARC, SciQ, TruthfulQA, etc.
"""

import json
import requests
import random
from collections import Counter

ROUTER_URL = "http://localhost:8801/v1/chat/completions"
OOD_CACHE = "examples/cluster_router/ood_evaluation_cache.json"

MODEL_MAP = {
    'Qwen/Qwen2.5-Math-7B-Instruct': 'math',
    'Qwen/Qwen2.5-Coder-7B-Instruct': 'coder',
    'Qwen/Qwen2.5-14B-Instruct-AWQ': 'general'
}

def run_ood_e2e(num_samples=20, seed=42):
    print("=" * 70, flush=True)
    print(f"OOD E2E: Router vs Oracle ({num_samples} samples)", flush=True)
    print("=" * 70, flush=True)

    with open(OOD_CACHE) as f:
        data = json.load(f)

    random.seed(seed)
    test_qs = random.sample(data, num_samples)

    correct = 0
    total = 0
    router_dist = Counter()
    oracle_dist = Counter()

    print(f"\n{'#':>3} | {'Router':^8} | {'Oracle':^8} | {'OK?':^5} | Dataset", flush=True)
    print("-" * 60, flush=True)

    for i, q in enumerate(test_qs):
        # Get oracle (first correct model from cached inference)
        oracle = None
        for m in ['math', 'coder', 'general']:
            if q['model_responses'].get(m, {}).get('is_correct', False):
                oracle = m
                break
        
        if oracle:
            oracle_dist[oracle] += 1
        
        try:
            resp = requests.post(ROUTER_URL, json={
                "model": "auto",
                "messages": [{"role": "user", "content": q['question']}],
                "max_tokens": 16
            }, timeout=60)
            
            if resp.status_code == 200:
                vllm_model = resp.json().get('model', '')
                router = MODEL_MAP.get(vllm_model, '???')
                router_dist[router] += 1
                
                router_correct = q['model_responses'].get(router, {}).get('is_correct', False)
                
                if oracle:
                    total += 1
                    if router_correct:
                        correct += 1
                        ok = "✓"
                    else:
                        ok = "✗"
                else:
                    ok = "-"
                
                ds = q['dataset'][:12]
                print(f"{i+1:3} | {router:^8} | {oracle or 'N/A':^8} | {ok:^5} | {ds}", flush=True)
            else:
                print(f"{i+1:3} | {'ERR':^8} | {oracle or 'N/A':^8} | {'?':^5} | HTTP{resp.status_code}", flush=True)
        except Exception as e:
            print(f"{i+1:3} | {'TIMEOUT':^8} | {oracle or 'N/A':^8} | {'?':^5} | timeout", flush=True)

    print("-" * 60, flush=True)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if total > 0:
        print(f"\nRouter Accuracy: {correct/total*100:.1f}% ({correct}/{total})")
    
    print(f"\nRouter selections: {dict(router_dist)}")
    print(f"Oracle distribution: {dict(oracle_dist)}")
    
    # Single model baselines
    print(f"\nSingle Model Baselines:")
    for m in ['math', 'coder', 'general']:
        single_correct = sum(1 for q in test_qs 
            if q['model_responses'].get(m, {}).get('is_correct', False)
            and any(q['model_responses'].get(x, {}).get('is_correct', False) 
                   for x in ['math', 'coder', 'general']))
        if total > 0:
            print(f"  {m:10s}: {single_correct}/{total} = {single_correct/total*100:.1f}%")

if __name__ == "__main__":
    import sys
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_ood_e2e(num_samples=num)


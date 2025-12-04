#!/usr/bin/env python3
"""
Evaluate OOD Generalization of the Consensus-ProxRouter.

1. Train Router on MMLU-Pro (Ground Truth)
2. Evaluate on OOD Datasets (ARC-Challenge, OpenBookQA, Hellaswag)
3. Cache results for replay
"""

import os
# Set CUDA device BEFORE importing torch/sentence_transformers
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import json
import asyncio
import re
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import httpx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add project root to path to find bench
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from bench.vllm_semantic_router_bench.dataset_factory import DatasetFactory
    from bench.vllm_semantic_router_bench.dataset_interface import Question
except ImportError:
    print("Error: Could not import benchmark utils. Make sure you are running from correct location.")
    sys.path.append("/home/ubuntu/rootfs/back/semantic-router.bak")
    from bench.vllm_semantic_router_bench.dataset_factory import DatasetFactory
    from bench.vllm_semantic_router_bench.dataset_interface import Question

# Configuration
CACHE_FILE_MMLU = Path(__file__).parent.parent.parent / "scripts/mmlu_specialized_cache.json"
CACHE_FILE_OOD = Path(__file__).parent / "ood_evaluation_cache.json"

MODELS = {
    "math": "Qwen/Qwen2.5-Math-7B-Instruct",
    "coder": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "general": "Qwen/Qwen2.5-14B-Instruct-AWQ"
}

ENDPOINTS = {
    "math": "http://localhost:8081/v1/chat/completions",
    "coder": "http://localhost:8082/v1/chat/completions",
    "general": "http://localhost:8083/v1/chat/completions"
}

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# OOD Datasets to evaluate (all multiple-choice for easy evaluation)
DATASETS_TO_EVAL = [
    "arc-challenge",      # Science reasoning (hard)
    "openbookqa",         # Science facts + reasoning
    "sciq",               # Science questions
    "commonsenseqa",      # Commonsense reasoning
    "hellaswag",          # Commonsense completion
    "truthfulqa",         # Truthfulness / factual accuracy
]
SAMPLES_PER_DATASET = 200  # Increased for larger experience database


def extract_answer_key(response: str) -> str:
    if not response:
        return "NONE"
    response = response.strip()
    
    # Boxed answer (common in math models)
    # r'\\boxed' = 2 backslashes in raw string = regex matches 1 literal backslash
    match = re.search(r'\\boxed\{([A-Z])\}', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for "Answer: X" pattern first (standardized in our prompts)
    match = re.search(r'Answer:\s*([A-Z])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallbacks
    match = re.search(r'(?:the\s+)?answer(?:\s+is)?[:\s]+([A-Z])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.match(r'^([A-Z])[.\s)]', response.upper())
    if match:
        return match.group(1)
    match = re.search(r'\b([A-Z])\s*$', response.upper())
    if match:
        return match.group(1)
    
    return "NONE"


class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device='cuda:0')

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)


class ProxRouter:
    def __init__(self, embedder: EmbeddingService, k: int = 20, tau: float = 0.1):
        self.embedder = embedder
        self.k = k
        self.tau = tau
        self.experience_embeddings: Optional[np.ndarray] = None
        self.experience_models: Optional[List[str]] = None
        
    def train_on_ground_truth(self, questions: list[dict]):
        """Build index using Ground Truth (is_correct) from MMLU-Pro."""
        embeddings = []
        models = []
        
        print("Building Router Index from MMLU-Pro Ground Truth...")
        
        # Pre-compute text for batch embedding
        texts = []
        valid_indices = []
        
        for i, q in enumerate(questions):
            # Check if ANY model was correct
            correct_models = [
                m for m in ["math", "coder", "general"] 
                if q['model_responses'].get(m, {}).get('is_correct', False)
            ]
            
            if correct_models:
                text = q['question']
                if q.get('options'):
                    text += "\n" + "\n".join([f"{chr(65+idx)}. {opt}" for idx, opt in enumerate(q['options'])])
                texts.append(text)
                
                # Strategy: Add ONE entry per correct model? Or one entry with a randomly selected correct model?
                # The paper says: "create multiple training examples for each consensus query"
                # For Ground Truth, if multiple are correct, we can add all or pick one.
                # Let's add ALL correct models as valid neighbors (Multi-label approach)
                
                for m in correct_models:
                    valid_indices.append((len(texts)-1, m))
        
        if not texts:
            print("Warning: No ground truth data found!")
            return

        print(f"Embedding {len(texts)} unique questions...")
        batch_embeddings = self.embedder.embed_batch(texts)
        
        final_embeddings = []
        final_models = []
        
        for text_idx, model in valid_indices:
            final_embeddings.append(batch_embeddings[text_idx])
            final_models.append(model)
            
        self.experience_embeddings = np.array(final_embeddings)
        self.experience_models = final_models
        
        print(f"Index built: {len(self.experience_embeddings)} vectors")
        from collections import Counter
        print(f"Distribution: {dict(Counter(self.experience_models))}")

    def route(self, query_embedding: np.ndarray) -> str:
        if self.experience_embeddings is None:
            return "general"
            
        similarities = np.dot(self.experience_embeddings, query_embedding)
        
        # Top-K
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]
        top_k_sims = similarities[top_k_indices]
        
        # Softmax weights
        weights = np.exp(self.tau * top_k_sims)
        weights = weights / weights.sum()
        
        votes = {m: 0.0 for m in ["math", "coder", "general"]}
        for i, idx in enumerate(top_k_indices):
            votes[self.experience_models[idx]] += weights[i]
            
        return max(votes, key=votes.get)


async def query_model(client: httpx.AsyncClient, model_name: str, prompt: str) -> str:
    endpoint = ENDPOINTS[model_name]
    model_id = MODELS[model_name]
    
    try:
        response = await client.post(
            endpoint,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.0
            },
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying {model_name}: {e}")
        return ""


async def process_ood_dataset(
    dataset_name: str, 
    limit: int = 100,
    cache_data: Dict = None,
    save_callback = None  # Callback to save incrementally
) -> List[Dict]:
    
    print(f"\nProcessing {dataset_name}...")
    
    # Load dataset
    try:
        dataset = DatasetFactory.create_dataset(dataset_name)
        questions, info = dataset.load_dataset(samples_per_category=limit) # This samples per category, might result in more than limit total
        
        # Simple truncation to limit
        if len(questions) > limit:
            import random
            random.seed(42)
            questions = random.sample(questions, limit)
            
        print(f"Loaded {len(questions)} questions from {dataset_name}")
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return []

    results = []
    
    async with httpx.AsyncClient() as client:
        for q in tqdm(questions, desc=f"Inference {dataset_name}"):
            q_id = f"{dataset_name}_{q.question_id}"
            
            # Check cache
            if cache_data and q_id in cache_data:
                results.append(cache_data[q_id])
                continue
                
            # Format prompt (handle different dataset APIs)
            try:
                prompt = dataset.format_prompt(q, prompt_style="plain")
            except TypeError:
                prompt = dataset.format_prompt(q, style="plain")
            
            # Query models
            model_responses = {}
            tasks = []
            for m in ["math", "coder", "general"]:
                tasks.append(query_model(client, m, prompt))
            
            responses = await asyncio.gather(*tasks)
            
            # Process results
            is_any_correct = False
            
            # Correct answer is usually an index (0, 1, 2) or letter (A, B, C)
            # We need to standardize to Letter for comparison
            ground_truth = str(q.correct_answer).upper()
            
            # Attempt to map numeric ground truth to letter if needed
            if ground_truth.isdigit():
                idx = int(ground_truth)
                if 0 <= idx < 26:
                    ground_truth = chr(65 + idx)
            
            for i, m in enumerate(["math", "coder", "general"]):
                resp_text = responses[i]
                extracted = extract_answer_key(resp_text)
                is_correct = (extracted == ground_truth)
                if is_correct:
                    is_any_correct = True
                    
                model_responses[m] = {
                    "raw_response": resp_text,
                    "extracted_answer": extracted,
                    "is_correct": is_correct
                }
            
            entry = {
                "dataset": dataset_name,
                "question_id": q_id,
                "question": q.question,
                "options": q.options,
                "ground_truth": ground_truth,
                "full_prompt": prompt,
                "model_responses": model_responses,
                "any_correct": is_any_correct
            }
            
            results.append(entry)
            
            # Incremental save every 10 questions
            if save_callback and len(results) % 10 == 0:
                save_callback(results)
    
    # Final save
    if save_callback:
        save_callback(results)
            
    return results


async def main():
    # 1. Setup
    if not CACHE_FILE_MMLU.exists():
        print(f"Error: MMLU cache not found at {CACHE_FILE_MMLU}")
        return

    embedder = EmbeddingService()
    router = ProxRouter(embedder)
    
    # 2. Train Router on MMLU-Pro
    with open(CACHE_FILE_MMLU) as f:
        mmlu_data = json.load(f)
    
    # Filter for train split (use 80% or same split as before)
    all_questions = mmlu_data['questions']
    random.seed(42)
    random.shuffle(all_questions)
    train_size = int(len(all_questions) * 0.8)
    train_questions = all_questions[:train_size]
    
    router.train_on_ground_truth(train_questions)
    
    # 3. Load/Run OOD Evaluation
    existing_cache = {}
    if CACHE_FILE_OOD.exists():
        with open(CACHE_FILE_OOD) as f:
            raw_cache = json.load(f)
            # Index by question_id
            for item in raw_cache:
                existing_cache[item['question_id']] = item
        print(f"Loaded {len(existing_cache)} cached OOD results")
    
    all_ood_results = list(existing_cache.values())  # Start with existing cache
    
    def save_incremental(new_results):
        """Save results incrementally to prevent data loss."""
        # Merge new results with existing
        merged = {r['question_id']: r for r in all_ood_results}
        for r in new_results:
            merged[r['question_id']] = r
        with open(CACHE_FILE_OOD, 'w') as f:
            json.dump(list(merged.values()), f, indent=2)
        print(f"  [Saved {len(merged)} total results]")
    
    for ds_name in DATASETS_TO_EVAL:
        ds_results = await process_ood_dataset(ds_name, SAMPLES_PER_DATASET, existing_cache, save_incremental)
        # Add only new results (not already in cache)
        for r in ds_results:
            if r['question_id'] not in existing_cache:
                all_ood_results.append(r)
    
    # Re-extract answers with fixed regex (in case cache has old extractions)
    print("Re-extracting answers with fixed regex...")
    for r in all_ood_results:
        gt = r['ground_truth']
        any_correct = False
        for m in ["math", "coder", "general"]:
            raw = r['model_responses'][m]['raw_response']
            extracted = extract_answer_key(raw)
            is_correct = (extracted == gt)
            r['model_responses'][m]['extracted_answer'] = extracted
            r['model_responses'][m]['is_correct'] = is_correct
            if is_correct:
                any_correct = True
        r['any_correct'] = any_correct
    
    # Save cache with updated extractions
    with open(CACHE_FILE_OOD, 'w') as f:
        json.dump(all_ood_results, f, indent=2)
    print(f"Saved {len(all_ood_results)} OOD results to cache")
    
    # 4. Evaluate Router
    print("\n" + "="*50)
    print("OOD EVALUATION RESULTS")
    print("="*50)
    
    # Prepare batches for embedding
    eval_texts = []
    for r in all_ood_results:
        text = r['question']
        if r.get('options'):
            text += "\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(r['options'])])
        eval_texts.append(text)
        
    print(f"Embedding {len(eval_texts)} OOD queries...")
    eval_embeddings = embedder.embed_batch(eval_texts)
    
    # Metrics
    total = 0
    router_correct = 0
    oracle_correct = 0
    single_correct = {"math": 0, "coder": 0, "general": 0}
    
    dataset_metrics = {}
    
    for i, r in enumerate(all_ood_results):
        ds = r['dataset']
        if ds not in dataset_metrics:
            dataset_metrics[ds] = {"total": 0, "router": 0, "oracle": 0, "single": {"math": 0, "coder": 0, "general": 0}}
            
        # Router prediction
        selected_model = router.route(eval_embeddings[i])
        
        # Check correctness
        is_router_correct = r['model_responses'][selected_model]['is_correct']
        
        # Update counters
        total += 1
        dataset_metrics[ds]["total"] += 1
        
        if is_router_correct:
            router_correct += 1
            dataset_metrics[ds]["router"] += 1
            
        if r['any_correct']:
            oracle_correct += 1
            dataset_metrics[ds]["oracle"] += 1
            
        for m in ["math", "coder", "general"]:
            if r['model_responses'][m]['is_correct']:
                single_correct[m] += 1
                dataset_metrics[ds]["single"][m] += 1
                
    # Print Global Results
    print(f"\nOverall OOD Performance ({total} questions):")
    print(f"Oracle (Upper Bound):   {oracle_correct/total*100:.1f}%")
    
    best_single_acc = 0
    best_single_name = ""
    for m, count in single_correct.items():
        acc = count/total*100
        print(f"Single Model ({m}):   {acc:.1f}%")
        if acc > best_single_acc:
            best_single_acc = acc
            best_single_name = m
            
    print(f"Consensus-ProxRouter:   {router_correct/total*100:.1f}%")
    print(f"vs Best Single:         {router_correct/total*100 - best_single_acc:+.1f}%")

    # Analyze Routing Distribution
    from collections import Counter
    router_choices = []
    for i, r in enumerate(all_ood_results):
        selected_model = router.route(eval_embeddings[i])
        router_choices.append(selected_model)
    
    print(f"\nRouting Distribution: {dict(Counter(router_choices))}")
    
    # Print Per-Dataset Results
    print("\nPer-Dataset Breakdown:")
    print(f"{'Dataset':<20} | {'Router':<8} | {'Best Single':<12} | {'Oracle':<8} | {'Delta':<8}")
    print("-" * 70)
    
    for ds, m in dataset_metrics.items():
        t = m["total"]
        if t == 0: continue
        
        r_acc = m["router"] / t * 100
        o_acc = m["oracle"] / t * 100
        
        bs_acc = 0
        for model_count in m["single"].values():
            acc = model_count / t * 100
            if acc > bs_acc:
                bs_acc = acc
                
        delta = r_acc - bs_acc
        print(f"{ds:<20} | {r_acc:>6.1f}% | {bs_acc:>10.1f}% | {o_acc:>6.1f}% | {delta:>+6.1f}%")


if __name__ == "__main__":
    asyncio.run(main())


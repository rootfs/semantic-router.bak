#!/usr/bin/env python3
"""
Generate MMLU-Pro Experience Database for Cluster Router

This script:
1. Loads MMLU-Pro dataset
2. Queries all three models (math, coder, general)
3. Evaluates correctness
4. Saves to experience_db format with embeddings
"""

import json
import os
import re
import requests
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Model endpoints
MODELS = {
    "math": {"url": "http://localhost:8081/v1/chat/completions", "model_id": "Qwen/Qwen2.5-Math-7B-Instruct"},
    "coder": {"url": "http://localhost:8082/v1/chat/completions", "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct"},
    "general": {"url": "http://localhost:8083/v1/chat/completions", "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ"},
}

# Answer extraction pattern
ANSWER_PATTERN = re.compile(r'(?:answer\s*(?:is|:)?\s*)?[\[\(]?([A-J])[\]\)]?', re.IGNORECASE)
BOXED_PATTERN = re.compile(r'\\boxed\{([A-J])\}', re.IGNORECASE)

def extract_answer(response: str) -> Optional[str]:
    """Extract answer letter from model response."""
    # Try boxed format first (common in math responses)
    match = BOXED_PATTERN.search(response)
    if match:
        return match.group(1).upper()
    
    # Try answer pattern
    lines = response.strip().split('\n')
    for line in reversed(lines[-5:]):  # Check last 5 lines
        match = ANSWER_PATTERN.search(line)
        if match:
            return match.group(1).upper()
    
    # Last resort: look for standalone letter
    for line in reversed(lines[-3:]):
        line = line.strip()
        if len(line) == 1 and line.upper() in 'ABCDEFGHIJ':
            return line.upper()
    
    return None

def query_model(model_name: str, question: str, options: List[str], max_tokens: int = 512) -> Tuple[str, Optional[str]]:
    """Query a model and return (raw_response, extracted_answer)."""
    model_config = MODELS[model_name]
    
    # Format prompt
    options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    prompt = f"""Answer the following multiple choice question. Choose the correct answer from the options.

Question: {question}

Options:
{options_str}

Think step by step, then provide your final answer in the format: Answer: [X]"""

    try:
        response = requests.post(
            model_config["url"],
            json={
                "model": model_config["model_id"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        raw_response = result["choices"][0]["message"]["content"]
        extracted = extract_answer(raw_response)
        return raw_response, extracted
    except Exception as e:
        print(f"Error querying {model_name}: {e}")
        return "", None

def main():
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    
    # Sample questions (use more for production)
    num_samples = int(os.environ.get("NUM_SAMPLES", "500"))
    print(f"Processing {num_samples} questions...")
    
    # Get diverse sample by category
    categories = dataset.unique("category")
    samples_per_cat = max(1, num_samples // len(categories))
    
    sampled_indices = []
    for cat in categories:
        cat_indices = [i for i, ex in enumerate(dataset) if ex["category"] == cat]
        sampled_indices.extend(cat_indices[:samples_per_cat])
    
    sampled_indices = sampled_indices[:num_samples]
    
    # Load embedding model
    print("Loading embedding model...")
    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    # Process questions
    experience_data = []
    cache_file = "mmlu_pro_cache.json"
    
    # Load existing cache if available
    cached_responses = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cached_responses = {entry["question_id"]: entry for entry in json.load(f)}
        print(f"Loaded {len(cached_responses)} cached responses")
    
    for idx in tqdm(sampled_indices, desc="Processing"):
        example = dataset[idx]
        question_id = f"mmlu_pro_{example['question_id']}"
        
        # Check cache
        if question_id in cached_responses:
            entry = cached_responses[question_id]
        else:
            question = example["question"]
            options = example["options"]
            correct_answer = chr(65 + example["answer_index"]) if isinstance(example["answer_index"], int) else example["answer"]
            
            # Query all models
            model_responses = {}
            for model_name in MODELS:
                raw_response, extracted = query_model(model_name, question, options)
                is_correct = extracted == correct_answer if extracted else False
                model_responses[model_name] = {
                    "raw_response": raw_response,
                    "extracted_answer": extracted,
                    "is_correct": is_correct
                }
            
            entry = {
                "question_id": question_id,
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "category": example["category"],
                "model_responses": model_responses
            }
            cached_responses[question_id] = entry
            
            # Save cache periodically
            if len(cached_responses) % 50 == 0:
                with open(cache_file, "w") as f:
                    json.dump(list(cached_responses.values()), f)
        
        # Generate embedding
        embedding = embed_model.encode(entry["question"], normalize_embeddings=True).tolist()
        
        # Build experience record
        model_scores = {}
        for model_name, resp in entry["model_responses"].items():
            model_scores[model_name] = 1.0 if resp["is_correct"] else 0.0
        
        experience_data.append({
            "embedding": embedding,
            "model_scores": model_scores,
            "metadata": {
                "question_id": entry["question_id"],
                "category": entry["category"],
                "correct_answer": entry["correct_answer"],
                "query_text_preview": entry["question"][:100] + "..." if len(entry["question"]) > 100 else entry["question"]
            }
        })
    
    # Save final cache
    with open(cache_file, "w") as f:
        json.dump(list(cached_responses.values()), f)
    
    # Save experience database
    output_file = "mmlu_pro_experience_db.json"
    with open(output_file, "w") as f:
        json.dump(experience_data, f)
    
    print(f"\nSaved {len(experience_data)} entries to {output_file}")
    
    # Print statistics
    model_correct = {m: 0 for m in MODELS}
    model_wins = {m: 0 for m in MODELS}
    
    for entry in experience_data:
        scores = entry["model_scores"]
        best = max(scores.keys(), key=lambda k: scores[k])
        model_wins[best] += 1
        for m, s in scores.items():
            if s == 1.0:
                model_correct[m] += 1
    
    print("\nModel Statistics:")
    print("  Accuracy:")
    for m in MODELS:
        print(f"    {m}: {model_correct[m]/len(experience_data)*100:.1f}%")
    print("  Best model per query:")
    for m in MODELS:
        print(f"    {m}: {model_wins[m]} ({model_wins[m]/len(experience_data)*100:.1f}%)")

if __name__ == "__main__":
    main()


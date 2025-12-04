#!/usr/bin/env python3
"""
Convert OOD Evaluation Cache to Experience Database Format

This script converts the cached inference results (ood_evaluation_cache.json)
into the experience database format expected by the cluster router.

Usage:
    python convert_cache_to_experience_db.py

Output:
    experience_db.json - Ready for use with cluster router
"""

import json
import os
from typing import Dict, List, Any
from collections import defaultdict

# Model names mapping (from cache to router)
MODEL_NAMES = ["math", "coder", "general"]

def load_cache(cache_path: str = "ood_evaluation_cache.json") -> Dict[str, Any]:
    """Load the OOD evaluation cache."""
    with open(cache_path, "r") as f:
        return json.load(f)

def convert_to_experience_db(cache: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert cache format to experience database format.
    
    Cache format (list of entries):
    [
        {
            "dataset": "arc-challenge",
            "question_id": "...",
            "question": "...",
            "ground_truth": "A",
            "model_responses": {
                "math": {"raw_response": "...", "extracted_answer": "A", "is_correct": true},
                "coder": {...},
                "general": {...}
            }
        }
    ]
    
    Experience DB format:
    [
        {
            "query_text": "...",
            "model_scores": {"math": 1.0, "coder": 0.0, "general": 1.0},
            "metadata": {"dataset": "...", "question_id": "...", "correct_answer": "A"}
        }
    ]
    """
    experience_db = []
    stats = defaultdict(lambda: {"total": 0, "correct": defaultdict(int)})
    
    for entry in cache:
        if not isinstance(entry, dict):
            continue
        
        # Get dataset and question info
        dataset_name = entry.get("dataset", "unknown")
        question_id = entry.get("question_id", "")
        
        # Get question text - try different field names
        query_text = entry.get("question", "") or entry.get("full_prompt", "")
        if not query_text:
            continue
        
        # Get model responses and scores
        model_responses = entry.get("model_responses", {})
        if not model_responses:
            continue
        
        model_scores = {}
        has_valid_response = False
        
        for model_name in MODEL_NAMES:
            if model_name in model_responses:
                response_data = model_responses[model_name]
                if isinstance(response_data, dict):
                    # Score is 1.0 if correct, 0.0 if incorrect
                    correct = response_data.get("is_correct", False)
                    model_scores[model_name] = 1.0 if correct else 0.0
                    has_valid_response = True
                    
                    # Track stats
                    stats[dataset_name]["total"] += 1
                    if correct:
                        stats[dataset_name]["correct"][model_name] += 1
        
        if not has_valid_response:
            continue
        
        # Create experience entry
        exp_entry = {
            "query_text": query_text,
            "model_scores": model_scores,
            "metadata": {
                "dataset": dataset_name,
                "question_id": question_id,
                "correct_answer": entry.get("ground_truth", "")
            }
        }
        
        experience_db.append(exp_entry)
    
    # Print statistics
    print("\n=== Conversion Statistics ===")
    print(f"Total entries: {len(experience_db)}")
    print(f"\nPer-dataset breakdown:")
    
    for dataset, data in sorted(stats.items()):
        total = data["total"] // len(MODEL_NAMES)  # Each question counted per model
        print(f"\n{dataset}:")
        print(f"  Questions: {total}")
        for model in MODEL_NAMES:
            correct = data["correct"][model]
            acc = correct / total * 100 if total > 0 else 0
            print(f"  {model}: {correct}/{total} ({acc:.1f}%)")
    
    return experience_db

def main():
    # Load cache
    cache_path = "ood_evaluation_cache.json"
    if not os.path.exists(cache_path):
        print(f"Error: {cache_path} not found")
        print("Run this script from the tutorial/cluster_router directory")
        return
    
    print(f"Loading cache from {cache_path}...")
    cache = load_cache(cache_path)
    
    # Convert
    print("Converting to experience database format...")
    experience_db = convert_to_experience_db(cache)
    
    # Save
    output_path = "experience_db.json"
    with open(output_path, "w") as f:
        json.dump(experience_db, f, indent=2)
    
    print(f"\nSaved experience database to {output_path}")
    print(f"Total entries: {len(experience_db)}")
    
    # Also create a smaller sample for testing
    sample_size = min(100, len(experience_db))
    sample_db = experience_db[:sample_size]
    sample_path = "experience_db_sample.json"
    with open(sample_path, "w") as f:
        json.dump(sample_db, f, indent=2)
    print(f"Saved sample ({sample_size} entries) to {sample_path}")

if __name__ == "__main__":
    main()


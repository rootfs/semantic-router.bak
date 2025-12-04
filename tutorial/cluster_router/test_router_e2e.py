#!/usr/bin/env python3
"""
E2E Test: Validate Router's Cluster Routing Algorithm

Uses cached MMLU-Pro inference results to:
1. Send queries to the router (via Envoy)
2. Compare router's model selection to oracle (best correct model)
3. Report accuracy metrics
"""

import json
import requests
from collections import Counter
from typing import Dict, List, Tuple
import random

# Router endpoint (through Envoy)
ROUTER_URL = "http://localhost:8801/v1/chat/completions"

# Load cached MMLU-Pro data
CACHE_FILE = "/tmp/mmlu_cache.json"

def load_test_data() -> List[Dict]:
    """Load MMLU-Pro cached inference results."""
    with open(CACHE_FILE) as f:
        data = json.load(f)
    return data['questions']

def get_oracle_model(question: Dict) -> str:
    """Get the best model for this question (oracle selection)."""
    responses = question['model_responses']
    
    # Find all correct models
    correct_models = [
        m for m in ['math', 'coder', 'general']
        if responses.get(m, {}).get('is_correct', False)
    ]
    
    if not correct_models:
        return None  # No model was correct
    
    # Return priority-ordered (math > coder > general for ties)
    for m in ['math', 'coder', 'general']:
        if m in correct_models:
            return m
    return correct_models[0]

def query_router(question_text: str) -> Tuple[str, float]:
    """Query the router and get the selected model."""
    try:
        response = requests.post(
            ROUTER_URL,
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": question_text}],
                "max_tokens": 10  # We don't need actual response, just routing
            },
            timeout=30
        )
        
        # Even if vLLM backend fails, check response headers for routing info
        # The model might be in the error message
        if response.status_code == 200:
            result = response.json()
            # Model should be in the response
            model = result.get('model', 'unknown')
            return model, 1.0
        else:
            # Check if we got a model error (means routing worked)
            try:
                error = response.json()
                error_msg = error.get('error', {}).get('message', '')
                if 'model' in error_msg.lower():
                    # Extract model name from "The model `X` does not exist"
                    import re
                    match = re.search(r'model `(\w+)`', error_msg)
                    if match:
                        return match.group(1), 1.0
            except:
                pass
            return 'unknown', 0.0
            
    except Exception as e:
        print(f"Error querying router: {e}")
        return 'error', 0.0

def main():
    print("=" * 60)
    print("E2E Router Accuracy Test")
    print("=" * 60)
    
    # Load test data
    questions = load_test_data()
    print(f"Loaded {len(questions)} questions from MMLU-Pro cache")
    
    # Sample subset for testing (to avoid overwhelming the router)
    num_test = min(200, len(questions))
    random.seed(42)
    test_questions = random.sample(questions, num_test)
    print(f"Testing with {num_test} random questions")
    print()
    
    # Track results
    results = {
        'correct': 0,          # Router selected a correct model
        'incorrect': 0,        # Router selected wrong model
        'no_oracle': 0,        # All models were wrong
        'router_error': 0,     # Router failed
    }
    model_selections = Counter()
    oracle_selections = Counter()
    
    # Category-wise tracking
    category_results = {}
    
    for i, q in enumerate(test_questions):
        if i % 50 == 0:
            print(f"Processing {i}/{num_test}...")
        
        oracle = get_oracle_model(q)
        router_model, _ = query_router(q['question'])
        
        category = q.get('category', 'unknown')
        if category not in category_results:
            category_results[category] = {'correct': 0, 'total': 0}
        
        model_selections[router_model] += 1
        
        if oracle is None:
            results['no_oracle'] += 1
        elif router_model == 'error' or router_model == 'unknown':
            results['router_error'] += 1
        else:
            oracle_selections[oracle] += 1
            category_results[category]['total'] += 1
            
            # Check if router's selection is correct
            is_correct = q['model_responses'].get(router_model, {}).get('is_correct', False)
            
            if is_correct:
                results['correct'] += 1
                category_results[category]['correct'] += 1
            else:
                results['incorrect'] += 1
    
    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    total_evaluated = results['correct'] + results['incorrect']
    if total_evaluated > 0:
        accuracy = results['correct'] / total_evaluated * 100
        print(f"\nRouter Accuracy: {accuracy:.1f}% ({results['correct']}/{total_evaluated})")
    
    print(f"\nBreakdown:")
    print(f"  Correct selections:   {results['correct']}")
    print(f"  Incorrect selections: {results['incorrect']}")
    print(f"  No correct model:     {results['no_oracle']}")
    print(f"  Router errors:        {results['router_error']}")
    
    print(f"\nRouter Model Selections:")
    for model, count in model_selections.most_common():
        print(f"  {model}: {count} ({count/num_test*100:.1f}%)")
    
    print(f"\nOracle Model Distribution:")
    for model, count in oracle_selections.most_common():
        pct = count / sum(oracle_selections.values()) * 100 if oracle_selections else 0
        print(f"  {model}: {count} ({pct:.1f}%)")
    
    print(f"\nCategory-wise Accuracy:")
    for cat in sorted(category_results.keys()):
        stats = category_results[cat]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {cat}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Compare to single-model baselines
    print(f"\nSingle Model Baselines (on evaluated questions):")
    for model in ['math', 'coder', 'general']:
        correct = sum(
            1 for q in test_questions 
            if q['model_responses'].get(model, {}).get('is_correct', False)
            and get_oracle_model(q) is not None
        )
        total = total_evaluated
        if total > 0:
            print(f"  {model}: {correct/total*100:.1f}% ({correct}/{total})")

if __name__ == "__main__":
    main()


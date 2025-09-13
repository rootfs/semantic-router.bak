#!/usr/bin/env python3
"""
Consolidated reasoning evaluation and config generation for MMLU-Pro.
Evaluates models with/without reasoning and generates optimized router config.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from scipy import stats
import numpy as np

ANSWER_PATTERN = re.compile(r"(?:answer(?:\sis)?:?\s*)([A-J])", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidated reasoning evaluation and config generation")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1", help="API endpoint (vLLM or OpenAI)")
    parser.add_argument("--api-key", type=str, default="", help="API key (auto-detects OPENAI_API_KEY env var)")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI API (sets endpoint to https://api.openai.com/v1)")
    parser.add_argument("--models", type=str, nargs="*", help="Models to evaluate (auto-discover if empty)")
    parser.add_argument("--samples-per-category", type=int, default=5, help="Questions per category")
    parser.add_argument("--concurrent-requests", type=int, default=4, help="Concurrent requests")
    parser.add_argument("--output-config", type=str, default="config.yaml", help="Output config file")
    parser.add_argument("--config-template", type=str, default="", help="Path to config template YAML file")
    parser.add_argument("--significance-level", type=float, default=0.05, help="Statistical significance level (p-value threshold)")
    return parser.parse_args()


def get_models(endpoint: str, api_key: str) -> List[str]:
    """Get available models from endpoint."""
    try:
        # For OpenAI API, we need a valid API key
        if "api.openai.com" in endpoint and not api_key:
            print("OpenAI API requires an API key. Set OPENAI_API_KEY or use --api-key")
            return []
        
        client = OpenAI(base_url=endpoint, api_key=api_key or "dummy")
        models = client.models.list()
        model_list = [m.id for m in models.data]
        
        # Filter OpenAI models to reasoning-capable ones
        if "api.openai.com" in endpoint:
            reasoning_models = [m for m in model_list if any(x in m.lower() for x in ["gpt-4", "o1"])]
            if reasoning_models:
                print(f"Found {len(reasoning_models)} reasoning-capable OpenAI models: {reasoning_models}")
                return reasoning_models
            else:
                print(f"No reasoning-capable models found. Available: {model_list}")
                return model_list
        
        return model_list
    except Exception as e:
        print(f"Failed to get models: {e}")
        return []


def build_reasoning_params(model: str, reasoning: bool) -> Optional[Dict[str, Any]]:
    """Build reasoning parameters for model."""
    lower = model.lower()
    
    # OpenAI models (o1 series has built-in reasoning, GPT-4 uses reasoning parameter)
    if "o1" in lower:
        # o1 models always use reasoning, no parameter needed
        return None
    elif "gpt-4" in lower or ("gpt" in lower and "api.openai.com" in lower):
        # OpenAI GPT models use top-level reasoning parameter
        return {"reasoning": reasoning}
    
    # vLLM models - all use extra_body with specific structures
    elif "deepseek" in lower and ("v31" in lower or "v3" in lower):
        # DeepSeek v3.1 uses chat_template_kwargs.thinking
        return {"chat_template_kwargs": {"thinking": reasoning}}
    elif "qwen3" in lower:
        # Qwen3 uses chat_template_kwargs.enable_thinking
        return {"chat_template_kwargs": {"enable_thinking": reasoning}}
    elif "gpt-oss" in lower:
        # GPT-OSS uses reasoning_effort
        return {"reasoning_effort": "high" if reasoning else "low"}
    
    return None


def format_prompt(question: str, options: List[str]) -> str:
    """Format MMLU-Pro prompt."""
    letters = "ABCDEFGHIJ"
    formatted = "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(options) if opt.lower() != "n/a")
    return f"Question: {question}\n\nOptions:\n{formatted}\n\nProvide your answer in the format 'Answer: [letter]'."


def extract_answer(response: str) -> Optional[str]:
    """Extract answer from model response."""
    if not response:
        return None
    match = ANSWER_PATTERN.search(response)
    if match:
        return match.group(1).upper()
    for char in reversed(response):
        if char.upper() in "ABCDEFGHIJ":
            return char.upper()
    return None


def call_model(client: OpenAI, model: str, prompt: str, extra_body: Optional[Dict] = None, debug_print: bool = False) -> Dict[str, Any]:
    """Call model and return result."""
    try:
        start = time.time()
        
        # Build request parameters
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 6144,  # Increased for reasoning mode support (allows full reasoning chains)
            "temperature": 0.0,
        }
        
        # Add reasoning parameters based on model type
        if extra_body:
            if "reasoning" in extra_body:
                # OpenAI reasoning parameter goes at top level
                params["reasoning"] = extra_body["reasoning"]
                if debug_print:
                    print(f"🔧 OpenAI reasoning param: reasoning={extra_body['reasoning']}")
            else:
                # All vLLM parameters (including chat_template_kwargs) go in extra_body
                # The Python OpenAI client doesn't support chat_template_kwargs at top level
                params["extra_body"] = extra_body
                if debug_print:
                    print(f"🔧 vLLM extra_body: {extra_body}")
        elif debug_print:
            print(f"🔧 No reasoning params for {model}")
        
        response = client.chat.completions.create(**params)
        
        text = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        
        return {
            "response": text,
            "success": True,
            "time": time.time() - start,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }
    except Exception as e:
        return {"response": str(e), "success": False, "time": 0, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


def evaluate_model(model: str, endpoint: str, api_key: str, df: pd.DataFrame, concurrent: int) -> pd.DataFrame:
    """Evaluate model with NR and NR_REASONING modes."""
    client = OpenAI(base_url=endpoint, api_key=api_key or "dummy")
    print(f"Evaluating {model} with {len(df)} questions...")
    
    # Special handling for o1 models (always use reasoning, no separate modes)
    if "o1" in model.lower():
        print("Note: o1 models always use reasoning. Running single mode evaluation.")
        modes = [("NR_REASONING", True)]  # Only one mode for o1
    else:
        modes = [("NR", False), ("NR_REASONING", True)]
    
    # Create all tasks (question × mode combinations)
    tasks = []
    for _, row in df.iterrows():
        prompt = format_prompt(row["question"], row["options"])
        for mode, reasoning in modes:
            extra_body = build_reasoning_params(model, reasoning)
            tasks.append((row, prompt, mode, extra_body))
    
    print(f"Total tasks: {len(tasks)} ({len(df)} questions × {len(modes)} modes)")
    
    # Track debug printing with a counter
    debug_counter = {"count": 0}
    
    def process_task(task):
        row, prompt, mode, extra_body = task
        # Enable debug printing for first 2 tasks only
        debug_print = debug_counter["count"] < 2
        if debug_print:
            debug_counter["count"] += 1
        result = call_model(client, model, prompt, extra_body, debug_print=debug_print)
        predicted = extract_answer(result["response"]) if result["success"] else None
        
        return {
            "question_id": row["question_id"],
            "category": row["category"],
            "correct_answer": row["answer"],
            "predicted_answer": predicted,
            "is_correct": predicted == row["answer"] if predicted else False,
            "mode": mode,
            "success": result["success"],
            "response_time": result["time"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        }
    
    # Process tasks concurrently
    results = []
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [executor.submit(process_task, task) for task in tasks]
        try:
            for future in tqdm(futures, desc=f"Evaluating {model}"):
                results.append(future.result())
        except KeyboardInterrupt:
            print("\n⚠️  Evaluation interrupted by user. Cancelling remaining tasks...")
            # Cancel remaining futures
            for future in futures:
                future.cancel()
            # Collect results from completed futures
            for future in futures:
                if future.done() and not future.cancelled():
                    try:
                        results.append(future.result())
                    except Exception:
                        pass  # Skip failed results
            if not results:
                print("❌ No results to save.")
                raise
            print(f"✅ Saved {len(results)} partial results.")
    
    return pd.DataFrame(results)


def analyze_reasoning_statistical(results_df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze reasoning effectiveness by category using statistical significance."""
    decisions = {}
    
    for category in results_df["category"].unique():
        cat_df = results_df[(results_df["category"] == category) & (results_df["success"])]
        
        nr_df = cat_df[cat_df["mode"] == "NR"]
        nr_reasoning_df = cat_df[cat_df["mode"] == "NR_REASONING"]
        
        if nr_df.empty or nr_reasoning_df.empty:
            decisions[category] = {
                "use_reasoning": False,
                "reasoning_effort": "low",
                "reason": "No data",
                "nr_accuracy": 0.0,
                "nr_reasoning_accuracy": 0.0,
                "improvement": 0.0,
                "token_overhead": 0.0,
                "p_value": 1.0,
                "sample_size": 0,
                "statistically_significant": False,
            }
            continue
        
        # Get binary results for statistical testing
        nr_results = nr_df["is_correct"].values
        nr_reasoning_results = nr_reasoning_df["is_correct"].values
        
        nr_acc = nr_results.mean()
        nr_reasoning_acc = nr_reasoning_results.mean()
        nr_tokens = nr_df["total_tokens"].dropna().mean() or 0
        nr_reasoning_tokens = nr_reasoning_df["total_tokens"].dropna().mean() or 0
        
        improvement = nr_reasoning_acc - nr_acc
        overhead = nr_reasoning_tokens / nr_tokens if nr_tokens > 0 else float('inf')
        
        # Perform statistical significance test
        # Use McNemar's test for paired binary data (same questions, different modes)
        # If sample sizes are equal, we can pair them; otherwise use Fisher's exact test
        
        if len(nr_results) == len(nr_reasoning_results):
            # Paired test - McNemar's test
            # Create contingency table for paired responses
            both_correct = np.sum((nr_results == 1) & (nr_reasoning_results == 1))
            nr_only_correct = np.sum((nr_results == 1) & (nr_reasoning_results == 0))
            reasoning_only_correct = np.sum((nr_results == 0) & (nr_reasoning_results == 1))
            both_wrong = np.sum((nr_results == 0) & (nr_reasoning_results == 0))
            
            # McNemar's test focuses on discordant pairs
            if nr_only_correct + reasoning_only_correct > 0:
                # Use continuity correction for small samples
                mcnemar_stat = (abs(nr_only_correct - reasoning_only_correct) - 1)**2 / (nr_only_correct + reasoning_only_correct)
                p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
            else:
                p_value = 1.0  # No discordant pairs
        else:
            # Unpaired test - Fisher's exact test
            # Create 2x2 contingency table
            nr_correct = np.sum(nr_results)
            nr_total = len(nr_results)
            reasoning_correct = np.sum(nr_reasoning_results)
            reasoning_total = len(nr_reasoning_results)
            
            contingency_table = [
                [nr_correct, nr_total - nr_correct],
                [reasoning_correct, reasoning_total - reasoning_correct]
            ]
            
            _, p_value = stats.fisher_exact(contingency_table, alternative='two-sided')
        
        # Determine if statistically significant
        statistically_significant = p_value < 0.05  # Using standard 0.05 threshold
        
        # Use reasoning if statistically significant AND improvement > 0
        use_reasoning = statistically_significant and improvement > 0
        
        # Determine effort level based on improvement magnitude
        if improvement >= 0.15:  # 15%+ improvement
            effort = "high"
        elif improvement >= 0.10:  # 10%+ improvement
            effort = "medium"
        else:
            effort = "low"
        
        # Create reason string
        if not statistically_significant:
            reason = f"Not statistically significant (p={p_value:.3f})"
        elif improvement <= 0:
            reason = f"Significant but negative improvement ({improvement:.1%}, p={p_value:.3f})"
        else:
            reason = f"Statistically significant improvement ({improvement:.1%}, p={p_value:.3f})"
        
        decisions[category] = {
            "use_reasoning": bool(use_reasoning),
            "reasoning_effort": effort,
            "reason": reason,
            "nr_accuracy": float(nr_acc),
            "nr_reasoning_accuracy": float(nr_reasoning_acc),
            "improvement": float(improvement),
            "token_overhead": float(overhead),
            "p_value": float(p_value),
            "sample_size": len(nr_results),
            "statistically_significant": bool(statistically_significant),
        }
    
    return decisions


def load_config_template(template_path: str = "") -> Dict:
    """Load configuration template from YAML file."""
    if not template_path:
        template_path = os.path.join(os.path.dirname(__file__), "config_template.yaml")
    
    try:
        with open(template_path, "r") as f:
            config = yaml.safe_load(f)
            print(f"✅ Loaded config template from: {template_path}")
            return config
    except FileNotFoundError:
        print(f"⚠️  Warning: Template file not found at {template_path}")
        print("Using fallback hardcoded template")
        # Fallback to hardcoded template
        return {
            "bert_model": {"model_id": "sentence-transformers/all-MiniLM-L12-v2", "threshold": 0.6, "use_cpu": True},
            "semantic_cache": {"enabled": True, "similarity_threshold": 0.8, "max_entries": 1000, "ttl_seconds": 3600},
            "tools": {"enabled": True, "top_k": 3, "similarity_threshold": 0.2, "tools_db_path": "config/tools_db.json", "fallback_to_empty": True},
            "prompt_guard": {"enabled": True, "use_modernbert": True, "model_id": "models/jailbreak_classifier_modernbert-base_model", "threshold": 0.7, "use_cpu": True, "jailbreak_mapping_path": "models/jailbreak_classifier_modernbert-base_model/jailbreak_type_mapping.json"},
            "classifier": {
                "category_model": {"model_id": "models/category_classifier_modernbert-base_model", "use_modernbert": True, "threshold": 0.6, "use_cpu": True, "category_mapping_path": "models/category_classifier_modernbert-base_model/category_mapping.json"},
                "pii_model": {"model_id": "models/pii_classifier_modernbert-base_presidio_token_model", "use_modernbert": True, "threshold": 0.7, "use_cpu": True, "pii_mapping_path": "models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json"}
            },
            "default_model": "",
            "default_reasoning_effort": "high",
            "categories": []
        }


def generate_config(results_df: pd.DataFrame, reasoning_decisions: Dict, model: str, template_path: str = "") -> Dict:
    """Generate router configuration from template."""
    # Load template
    config = load_config_template(template_path)
    
    # Set model-specific values
    config["default_model"] = model
    
    # Add categories with reasoning decisions
    for category, decision in reasoning_decisions.items():
        # Get best accuracy for model scoring
        cat_df = results_df[(results_df["category"] == category) & (results_df["success"])]
        best_acc = float(cat_df["is_correct"].mean()) if not cat_df.empty else 0.0
        
        config["categories"].append({
            "name": category,
            "use_reasoning": decision["use_reasoning"],
            "reasoning_description": f"Data-driven decision: {decision['reason']}",
            "reasoning_effort": decision["reasoning_effort"],
            "model_scores": [{"model": model, "score": float(best_acc)}]
        })
    
    return config


def main():
    args = parse_args()
    
    # Handle OpenAI API setup
    if args.use_openai:
        args.endpoint = "https://api.openai.com/v1"
        print("Using OpenAI API endpoint")
    
    # Auto-detect API key from environment if not provided
    if not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY", "")
        if args.api_key:
            print("Using API key from OPENAI_API_KEY environment variable")
    
    # Validate API key for OpenAI
    if "api.openai.com" in args.endpoint and not args.api_key:
        print("❌ OpenAI API requires an API key. Please:")
        print("   1. Set OPENAI_API_KEY environment variable, or")
        print("   2. Use --api-key parameter")
        return
    
    print(f"Endpoint: {args.endpoint}")
    
    # Get models
    if not args.models:
        print("Auto-discovering models...")
        args.models = get_models(args.endpoint, args.api_key)
        if not args.models:
            print("No models found!")
            return
        print(f"Found models: {args.models}")
    else:
        print(f"Using specified models: {args.models}")
    
    # Load dataset
    print("Loading MMLU-Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    df = pd.DataFrame(dataset)
    
    # Sample questions per category
    if args.samples_per_category:
        sampled = []
        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            sample_size = min(args.samples_per_category, len(cat_df))
            sampled.append(cat_df.sample(sample_size, random_state=42))
        df = pd.concat(sampled)
    
    print(f"Evaluating {len(df)} questions across {df['category'].nunique()} categories")
    
    # Evaluate each model
    all_results = []
    try:
        for model in args.models:
            results_df = evaluate_model(model, args.endpoint, args.api_key, df, args.concurrent_requests)
            results_df["model"] = model
            all_results.append(results_df)
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user.")
        if not all_results:
            print("❌ No complete model evaluations to save.")
            return
        print(f"✅ Proceeding with {len(all_results)} completed model evaluations.")
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Analyze reasoning effectiveness using statistical significance
    print("\nAnalyzing reasoning effectiveness using statistical significance...")
    reasoning_decisions = analyze_reasoning_statistical(combined_df)
    
    # Print analysis
    reasoning_enabled = sum(1 for d in reasoning_decisions.values() if d["use_reasoning"])
    print(f"\nReasoning Analysis ({reasoning_enabled}/{len(reasoning_decisions)} categories enabled):")
    for category, decision in reasoning_decisions.items():
        status = "✓ ENABLED" if decision["use_reasoning"] else "✗ DISABLED"
        acc_change = f"{decision['nr_accuracy']:.1%} → {decision['nr_reasoning_accuracy']:.1%}"
        tokens = f"{decision['token_overhead']:.1f}x tokens"
        p_val = f"p={decision['p_value']:.3f}"
        print(f"  {category}: {status}")
        print(f"    Accuracy: {acc_change} ({decision['improvement']:+.1%})")
        print(f"    Statistical: {p_val} ({'significant' if decision['statistically_significant'] else 'not significant'})")
        print(f"    Cost: {tokens}")
        print(f"    Reason: {decision['reason']}")
        print()
    
    # Generate config (use first model as default)
    config = generate_config(combined_df, reasoning_decisions, args.models[0], args.config_template)
    
    # Save config
    os.makedirs(os.path.dirname(args.output_config) if os.path.dirname(args.output_config) else ".", exist_ok=True)
    with open(args.output_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Save detailed results
    results_file = args.output_config.replace(".yaml", "_results.csv")
    combined_df.to_csv(results_file, index=False)
    
    # Save reasoning analysis
    analysis_file = args.output_config.replace(".yaml", "_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(reasoning_decisions, f, indent=2)
    
    print(f"\n✅ Config saved to: {args.output_config}")
    print(f"📊 Results saved to: {results_file}")
    print(f"📈 Analysis saved to: {analysis_file}")


if __name__ == "__main__":
    main()

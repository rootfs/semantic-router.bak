#!/usr/bin/env python3
"""
Efficiency-Based Reasoning Mode Evaluation for LLM Router Configuration

A streamlined evaluation framework that determines optimal reasoning mode
configurations for large language model routers using efficiency-based
analysis aligned with RL optimizer training objectives.

This module implements a simple, transparent framework that evaluates
the effectiveness of reasoning modes across different task categories,
generating optimized router configurations based on accuracy per token
efficiency rather than complex statistical thresholds.

Accuracy-Biased Efficiency Methodology:
    - Accuracy-weighted efficiency ratio with model-specific token adjustments
    - Multi-tier accuracy improvement thresholds (critical 10%, significant 5%)
    - Statistical similarity testing using two-proportion z-test
    - Model-aware token penalties (Qwen 1.2x, GPT/Claude 0.9x)
    - Prioritizes accuracy improvements while considering computational costs

Key Features:
    - Simple, interpretable decision making
    - Aligned with RL optimizer training objectives
    - Transparent reasoning for each category decision
    - Focus on practical efficiency considerations
    - Production-ready configuration generation

Usage:
    python reasoning_eval_consolidated.py \\
        --endpoint http://localhost:8000/v1 \\
        --samples-per-category 25 \\
        --output-config optimized_config.yaml

Outputs:
    - YAML configuration file with efficiency-based reasoning decisions
    - CSV file containing detailed evaluation results
    - JSON file with comprehensive efficiency analysis
"""

import argparse
import json
import math
import os
# Removed random import - not needed for efficiency-based approach
import re
import time
from concurrent.futures import ThreadPoolExecutor
# Removed deepcopy import - not needed for efficiency-based approach
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from openai import OpenAI
from scipy import stats
from tqdm import tqdm

# Regular expression pattern for extracting multiple choice answers from model responses
ANSWER_PATTERN = re.compile(r"(?:answer(?:\sis)?:?\s*)([A-J])", re.IGNORECASE)

# Statistical significance thresholds
ALPHA_STRICT = 0.05  # Traditional significance level
ALPHA_RELAXED = 0.10  # Relaxed significance level for borderline cases
EFFECT_SIZE_MEDIUM = 0.2  # Cohen's h threshold for medium effect size
BAYESIAN_THRESHOLD = 0.8  # Bayesian probability threshold for strong evidence
MIN_IMPROVEMENT = 0.10  # Minimum improvement threshold for Bayesian pathway

# Efficiency-based analysis parameters (more accuracy-biased)
ACCURACY_WEIGHT = 2.0         # Weight multiplier for accuracy in efficiency calculation
TOKEN_COST_WEIGHT = 0.0005    # Reduced weight factor for token penalty (was 0.001)
NORMALIZATION_FACTOR = 5      # Reduced scaling factor (was 10) 
BASELINE_OFFSET = 0.05        # Reduced baseline offset (was 0.1)

# Model-specific token efficiency adjustments
MODEL_TOKEN_MULTIPLIERS = {
    'qwen': 1.2,      # Qwen models are more verbose, apply higher penalty
    'deepseek': 1.1,  # DeepSeek models moderately verbose
    'gpt': 0.9,       # GPT models typically more concise
    'claude': 0.9,    # Claude models typically more concise
    'default': 1.0    # Default multiplier for unknown models
}

# Accuracy improvement thresholds (more permissive)
CRITICAL_ACCURACY_THRESHOLD = 0.099  # If accuracy improves by ≥10%, strongly favor reasoning (was 0.5)
SIGNIFICANT_ACCURACY_THRESHOLD = 0.049  # If accuracy improves by ≥5%, consider reasoning
MIN_ACCURACY_FLOOR = 0.3             # Never choose a mode with <30% accuracy if alternative exists


# Removed unused NSGA-II optimization classes and functions - using simplified efficiency-based approach


def load_category_decisions_from_results(
    results_file: str,
    use_statistical_analysis: bool = True,
    solution_type: str = "balanced",
) -> Dict[str, Dict]:
    """
    Load per-category reasoning decisions from existing statistical results.

    This function analyzes existing evaluation results to determine per-category
    reasoning effectiveness without needing to run new evaluations.

    Args:
        results_file: Path to CSV file with evaluation results
        use_statistical_analysis: If True, use statistical significance testing.
                                 If False, use simple threshold-based analysis (better for NSGA-II)

    Returns:
        Dictionary mapping category names to reasoning decisions
    """
    try:
        df = pd.read_csv(results_file)
        print(f"📊 Loading category decisions from: {results_file}")

        if use_statistical_analysis:
            # Statistical analysis function was removed - use efficiency-based analysis instead
            print("⚠️  Statistical analysis function removed, falling back to efficiency-based analysis")
            use_statistical_analysis = False

        if not use_statistical_analysis:
            # Use simple threshold-based analysis (better for NSGA-II optimization)
            # Adapt threshold based on solution type
            if solution_type == "max_accuracy":
                threshold = (
                    0.03  # 3% threshold - more liberal for accuracy optimization
                )
                threshold_desc = "3% (max accuracy)"
            elif solution_type == "max_efficiency":
                threshold = (
                    0.07  # 7% threshold - more conservative for efficiency optimization
                )
                threshold_desc = "7% (max efficiency)"
            else:  # balanced
                threshold = 0.05  # 5% threshold - balanced approach
                threshold_desc = "5% (balanced)"

            print(
                f"   Using threshold-based analysis ({threshold_desc} improvement threshold)"
            )
            category_decisions = {}
            categories = df["category"].unique()

            for category in categories:
                cat_df = df[df["category"] == category]
                nr_df = cat_df[cat_df["mode"] == "NR"]
                nr_reasoning_df = cat_df[cat_df["mode"] == "NR_REASONING"]

                if nr_df.empty or nr_reasoning_df.empty:
                    category_decisions[category] = {
                        "use_reasoning": False,
                        "accuracy": 0.0,
                        "reason": "No data available",
                        "no_reasoning_accuracy": 0.0,
                        "reasoning_accuracy": 0.0,
                        "improvement": 0.0,
                        "no_reasoning_success_rate": 0.0,
                        "reasoning_success_rate": 0.0,
                    }
                    continue

                nr_acc = nr_df["is_correct"].mean()
                reasoning_acc = nr_reasoning_df["is_correct"].mean()
                improvement = reasoning_acc - nr_acc

                # Use solution-type specific threshold
                use_reasoning = improvement > threshold

                if use_reasoning:
                    reason = f"Reasoning improves accuracy by {improvement:.1%} (above {threshold:.0%} threshold)"
                    best_accuracy = reasoning_acc
                else:
                    reason = f"Reasoning improvement ({improvement:.1%}) below {threshold:.0%} threshold"
                    best_accuracy = nr_acc

                category_decisions[category] = {
                    "use_reasoning": bool(use_reasoning),
                    "accuracy": float(best_accuracy),
                    "reason": reason,
                    "no_reasoning_accuracy": float(nr_acc),
                    "reasoning_accuracy": float(reasoning_acc),
                    "improvement": float(improvement),
                    "no_reasoning_success_rate": 1.0,  # Assume success from existing results
                    "reasoning_success_rate": 1.0,
                }

        print(f"✅ Loaded decisions for {len(category_decisions)} categories")
        reasoning_enabled = sum(
            1 for d in category_decisions.values() if d["use_reasoning"]
        )
        print(
            f"   Reasoning enabled for {reasoning_enabled}/{len(category_decisions)} categories"
        )

        return category_decisions

    except Exception as e:
        print(f"❌ Error loading category decisions: {e}")
        return {}


# Removed generate_nsga2_configs_from_existing_results function - using simplified efficiency-based approach


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the reasoning evaluation framework.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Statistical reasoning mode evaluation for LLM router configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --endpoint http://localhost:8000/v1 --samples-per-category 25
  %(prog)s --use-openai --models gpt-4o --samples-per-category 50
  %(prog)s --show-methodology
        """,
    )

    # API Configuration
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="API endpoint URL (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key (auto-detects OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API (sets endpoint to https://api.openai.com/v1)",
    )

    # Model Configuration
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="Models to evaluate (auto-discover if not specified)",
    )

    # Evaluation Parameters
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=5,
        help="Number of questions per category (default: %(default)s)",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=8,
        help="Number of concurrent API requests (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens per response (default: %(default)s)",
    )

    # Output Configuration
    parser.add_argument(
        "--output-config",
        type=str,
        default="config.yaml",
        help="Output configuration file path (default: %(default)s)",
    )
    parser.add_argument(
        "--config-template",
        type=str,
        default="",
        help="Path to configuration template YAML file",
    )

    # Statistical Parameters
    parser.add_argument(
        "--significance-level",
        type=float,
        default=ALPHA_STRICT,
        help=f"Statistical significance level (default: {ALPHA_STRICT})",
    )

    # Analysis Parameters
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for accuracy similarity test (default: 0.95)",
    )
    parser.add_argument(
        "--critical-accuracy-threshold",
        type=float,
        default=CRITICAL_ACCURACY_THRESHOLD,
        help=f"If accuracy improves by more than this amount, always use reasoning (default: {CRITICAL_ACCURACY_THRESHOLD})",
    )
    parser.add_argument(
        "--significant-accuracy-threshold",
        type=float,
        default=SIGNIFICANT_ACCURACY_THRESHOLD,
        help=f"If accuracy improves by more than this amount, favor reasoning with efficiency check (default: {SIGNIFICANT_ACCURACY_THRESHOLD})",
    )
    parser.add_argument(
        "--min-accuracy-floor",
        type=float,
        default=MIN_ACCURACY_FLOOR,
        help=f"Never choose a mode with accuracy below this threshold if alternative exists (default: {MIN_ACCURACY_FLOOR})",
    )

    # Documentation
    parser.add_argument(
        "--show-methodology",
        action="store_true",
        help="Display detailed methodology explanation and exit",
    )

    return parser.parse_args()


def get_models(endpoint: str, api_key: str) -> List[str]:
    """
    Discover available models from the specified API endpoint.

    Args:
        endpoint: API endpoint URL
        api_key: API authentication key

    Returns:
        List of available model identifiers

    Raises:
        None: Errors are logged and empty list is returned
    """
    try:
        # Validate OpenAI API requirements
        if "api.openai.com" in endpoint and not api_key:
            print(
                "ERROR: OpenAI API requires authentication. Please set OPENAI_API_KEY environment variable or use --api-key"
            )
            return []

        # Initialize API client
        client = OpenAI(base_url=endpoint, api_key=api_key or "dummy", timeout=300.0)
        models_response = client.models.list()
        model_list = [model.id for model in models_response.data]

        # Apply OpenAI-specific filtering for reasoning-capable models
        if "api.openai.com" in endpoint:
            reasoning_capable = ["gpt-4", "o1"]
            filtered_models = [
                model
                for model in model_list
                if any(capability in model.lower() for capability in reasoning_capable)
            ]

            if filtered_models:
                print(
                    f"Discovered {len(filtered_models)} reasoning-capable models: {filtered_models}"
                )
                return filtered_models
            else:
                print(
                    f"WARNING: No reasoning-capable models found. Available models: {model_list}"
                )
                return model_list

        print(f"Discovered {len(model_list)} models from endpoint")
        return model_list

    except Exception as e:
        print(f"ERROR: Failed to discover models from {endpoint}: {e}")
        return []


def build_reasoning_params(model: str, reasoning_mode: str) -> Optional[Dict[str, Any]]:
    """
    Construct model-specific reasoning parameters for API requests.
    
    This function matches the exact pattern from vllm_client_integration.py
    to ensure compatibility with your existing vLLM setup.

    Args:
        model: Model identifier string
        reasoning_mode: Reasoning mode ('off', 'low', 'medium', 'high')

    Returns:
        Dictionary of reasoning parameters, or None if no parameters needed
    """
    # Convert reasoning_mode to boolean for vLLM models
    reasoning_bool = reasoning_mode != "off"
    
    model_lower = model.lower()

    # DeepSeek v3.1 family (matches vllm_client_integration.py pattern)
    if (("ds" in model_lower) or ("deepseek" in model_lower)) and (
        "v31" in model_lower or "v3.1" in model_lower or "v3" in model_lower
    ):
        if reasoning_bool:
            return {"chat_template_kwargs": {"thinking": True}}
        else:
            return {"chat_template_kwargs": {"thinking": False}}

    # Qwen3 family (matches vllm_client_integration.py pattern)  
    if "qwen3" in model_lower:
        if reasoning_bool:
            return {"chat_template_kwargs": {"enable_thinking": True}}
        else:
            return {"chat_template_kwargs": {"enable_thinking": False}}

    # GPT-OSS family (matches vllm_client_integration.py pattern)
    if "gpt-oss" in model_lower or "openai/gpt-oss" in model_lower or "gpt_oss" in model_lower:
        if reasoning_bool:
            return {"reasoning_effort": "high"}
        else:
            return {"reasoning_effort": "low"}

    # OpenAI models with reasoning parameter
    elif "gpt" in model_lower or "o1" in model_lower:
        return {"reasoning": reasoning_bool}

    # Model does not support reasoning parameters
    return None


def format_prompt(question: str, options: List[str]) -> str:
    """Format MMLU-Pro prompt."""
    letters = "ABCDEFGHIJ"
    formatted = "\n".join(
        f"{letters[i]}) {opt}" for i, opt in enumerate(options) if opt.lower() != "n/a"
    )
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


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 6144,
    extra_body: Optional[Dict] = None,
    debug_print: bool = False,
) -> Dict[str, Any]:
    """Call model and return result."""
    try:
        start = time.time()

        # Build base request parameters
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        # Add reasoning parameters based on model type
        if extra_body:
            if "reasoning" in extra_body:
                # OpenAI reasoning parameter goes at top level
                params["reasoning"] = extra_body["reasoning"]
                if debug_print:
                    print(
                        f"🔧 OpenAI reasoning param: reasoning={extra_body['reasoning']}"
                    )
            else:
                # vLLM parameters: use extra_body since OpenAI client doesn't accept vLLM-specific params
                params["extra_body"] = extra_body
                if debug_print:
                    print(f"🔧 vLLM extra_body: {extra_body}")
        elif debug_print:
            print(f"🔧 No reasoning params for {model}")

        # Debug: print the final params to see what's being sent
        if debug_print:
            print(f"🔧 Final params: {params}")

        response = client.chat.completions.create(**params)

        text = response.choices[0].message.content
        usage = getattr(response, "usage", None)

        return {
            "response": text,
            "success": True,
            "time": time.time() - start,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": (
                getattr(usage, "completion_tokens", None) if usage else None
            ),
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
        }
    except Exception as e:
        # Print the first few errors to help debug
        if not hasattr(call_model, '_error_count'):
            call_model._error_count = 0
        if call_model._error_count < 3:
            print(f"❌ API Error #{call_model._error_count + 1}: {str(e)}")
            call_model._error_count += 1
        return {
            "response": str(e),
            "success": False,
            "time": 0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }


def evaluate_model(
    model: str, endpoint: str, api_key: str, df: pd.DataFrame, concurrent: int, max_tokens: int = 8192
) -> pd.DataFrame:
    """Evaluate model with NR and NR_REASONING modes."""
    client = OpenAI(base_url=endpoint, api_key=api_key or "dummy", timeout=300.0)
    print(f"Evaluating {model} with {len(df)} questions...")

    # Special handling for o1 models (always use reasoning, no separate modes)
    if "o1" in model.lower():
        print("Note: o1 models always use reasoning. Running single mode evaluation.")
        modes = [("NR_REASONING", "high")]  # Only one mode for o1
    else:
        modes = [("NR", "off"), ("NR_REASONING", "high")]

    # Create all tasks (question × mode combinations)
    tasks = []
    for _, row in df.iterrows():
        prompt = format_prompt(row["question"], row["options"])
        for mode, reasoning_mode in modes:
            extra_body = build_reasoning_params(model, reasoning_mode)
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
        result = call_model(client, model, prompt, max_tokens, extra_body, debug_print=debug_print)
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


# Removed evaluate_optimization_config function - not needed for simplified efficiency-based approach


def calculate_efficiency_ratio(accuracy: float, tokens: int, model_name: str = "") -> float:
    """
    Calculate efficiency ratio with accuracy bias and model-specific adjustments.
    
    This version is more biased towards accuracy improvements and accounts for
    different model verbosity patterns (e.g., Qwen is more verbose than GPT).
    
    Formula: (accuracy^ACCURACY_WEIGHT) / (sqrt(tokens * model_multiplier) * TOKEN_COST_WEIGHT * NORMALIZATION_FACTOR + BASELINE_OFFSET)
    
    Args:
        accuracy: Accuracy score (0.0 to 1.0)
        tokens: Total token count
        model_name: Model name for model-specific adjustments
        
    Returns:
        Efficiency ratio (higher is better, more accuracy-biased)
    """
    import math
    if tokens <= 0 or accuracy < 0:
        return 0.0
    
    # Get model-specific token multiplier
    model_multiplier = MODEL_TOKEN_MULTIPLIERS['default']
    model_lower = model_name.lower()
    
    for model_key, multiplier in MODEL_TOKEN_MULTIPLIERS.items():
        if model_key != 'default' and model_key in model_lower:
            model_multiplier = multiplier
            break
    
    # Apply accuracy weighting (higher weight = more bias towards accuracy)
    weighted_accuracy = math.pow(accuracy, 1.0 / ACCURACY_WEIGHT) if accuracy > 0 else 0
    
    # Apply model-specific token adjustment
    adjusted_tokens = tokens * model_multiplier
    sqrt_tokens = math.sqrt(adjusted_tokens)
    
    # Calculate efficiency with reduced token penalty
    efficiency_ratio = weighted_accuracy / (sqrt_tokens * TOKEN_COST_WEIGHT * NORMALIZATION_FACTOR + BASELINE_OFFSET)
    return efficiency_ratio


def check_accuracy_similarity(acc1: float, acc2: float, n1: int, n2: int, confidence_level: float = 0.95) -> Tuple[bool, float, str]:
    """
    Check if two accuracies are statistically similar using confidence intervals.
    
    Args:
        acc1: First accuracy
        acc2: Second accuracy  
        n1: Sample size for first accuracy
        n2: Sample size for second accuracy
        confidence_level: Confidence level for comparison
        
    Returns:
        (is_similar, p_value, explanation)
    """
    if n1 <= 0 or n2 <= 0:
        return True, 1.0, "Insufficient data"
    
    # Use two-proportion z-test for simplicity
    from scipy import stats
    import numpy as np
    
    # Convert accuracies to counts
    count1 = int(acc1 * n1)
    count2 = int(acc2 * n2)
    
    # Two-proportion z-test
    if count1 + count2 == 0:
        return True, 1.0, "No correct answers in either mode"
    
    # Calculate pooled proportion
    pooled_p = (count1 + count2) / (n1 + n2)
    
    if pooled_p == 0 or pooled_p == 1:
        # Edge case: all correct or all wrong
        is_similar = (acc1 == acc2)
        return is_similar, 1.0 if is_similar else 0.0, "Edge case: extreme accuracies"
    
    # Standard error for difference in proportions
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
    
    if se == 0:
        is_similar = (acc1 == acc2)
        return is_similar, 1.0 if is_similar else 0.0, "No variation in results"
    
    # Z-statistic
    z = (acc1 - acc2) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Consider similar if p > 0.05 (not significantly different)
    alpha = 1 - confidence_level
    is_similar = p_value > alpha
    
    explanation = f"Two-proportion z-test: z={z:.3f}, p={p_value:.3f}"
    
    return is_similar, p_value, explanation


def analyze_reasoning_efficiency_based(results_df: pd.DataFrame, critical_threshold: float = CRITICAL_ACCURACY_THRESHOLD, significant_threshold: float = SIGNIFICANT_ACCURACY_THRESHOLD, min_floor: float = MIN_ACCURACY_FLOOR, model_name: str = "") -> Dict[str, Dict]:
    """
    Analyze reasoning effectiveness using efficiency-based methodology from RL optimizer.
    
    Decision Logic:
    1. Calculate efficiency ratio for both modes: accuracy / (sqrt(tokens) * 0.001 * 10 + 0.1)
    2. Check if accuracies are statistically similar (using two-proportion z-test)
    3. If similar accuracy: choose mode with lower token usage
    4. If different accuracy: choose mode with higher efficiency ratio
    
    Args:
        results_df: DataFrame with evaluation results containing:
            - category: Question category
            - mode: 'NR' or 'NR_REASONING' 
            - is_correct: Boolean correctness
            - total_tokens: Token usage
            - success: Whether evaluation succeeded
            
    Returns:
        Dict[str, Dict]: Analysis results for each category containing:
            - use_reasoning: Boolean decision
            - reason: Simple explanation of why this mode was chosen
            - nr_accuracy: Non-reasoning accuracy
            - nr_reasoning_accuracy: Reasoning accuracy
            - nr_tokens: Average tokens for non-reasoning
            - nr_reasoning_tokens: Average tokens for reasoning
            - nr_efficiency: Non-reasoning efficiency ratio
            - nr_reasoning_efficiency: Reasoning efficiency ratio
            - efficiency_improvement: Relative efficiency improvement
            - accuracy_similar: Whether accuracies are statistically similar
            - p_value: P-value from accuracy similarity test
            - sample_size: Number of questions evaluated
    """
    decisions = {}
    
    print(f"\n🔍 Analyzing reasoning effectiveness using efficiency-based methodology...")
    
    for category in results_df["category"].unique():
        cat_df = results_df[
            (results_df["category"] == category) & (results_df["success"])
        ]
        
        nr_df = cat_df[cat_df["mode"] == "NR"]
        nr_reasoning_df = cat_df[cat_df["mode"] == "NR_REASONING"]
        
        if nr_df.empty or nr_reasoning_df.empty:
            decisions[category] = {
                "use_reasoning": False,
                "reason": "No data available for comparison",
                "nr_accuracy": 0.0,
                "nr_reasoning_accuracy": 0.0,
                "nr_tokens": 0.0,
                "nr_reasoning_tokens": 0.0,
                "nr_efficiency": 0.0,
                "nr_reasoning_efficiency": 0.0,
                "efficiency_improvement": 0.0,
                "accuracy_similar": True,
                "p_value": 1.0,
                "sample_size": 0,
            }
            continue
        
        # Calculate basic metrics
        nr_acc = nr_df["is_correct"].mean()
        nr_reasoning_acc = nr_reasoning_df["is_correct"].mean()
        nr_tokens = nr_df["total_tokens"].dropna().mean() or 0
        nr_reasoning_tokens = nr_reasoning_df["total_tokens"].dropna().mean() or 0
        
        n1 = len(nr_df)
        n2 = len(nr_reasoning_df)
        
        # Calculate efficiency ratios using accuracy-biased formula with model-specific adjustments
        nr_efficiency = calculate_efficiency_ratio(nr_acc, nr_tokens, model_name)
        nr_reasoning_efficiency = calculate_efficiency_ratio(nr_reasoning_acc, nr_reasoning_tokens, model_name)
        
        # Check if accuracies are statistically similar
        accuracy_similar, p_value, similarity_explanation = check_accuracy_similarity(
            nr_acc, nr_reasoning_acc, n1, n2
        )
        
        # Make decision based on efficiency with accuracy improvement overrides
        accuracy_improvement = nr_reasoning_acc - nr_acc
        
        # Override 1: Critical accuracy improvement threshold (strongly favor reasoning)
        if accuracy_improvement >= critical_threshold:
            use_reasoning = True
            reason = f"Critical accuracy improvement: {nr_acc:.1%} → {nr_reasoning_acc:.1%} (+{accuracy_improvement:.1%}) overrides token costs"
        
        # Override 2: Significant accuracy improvement (favor reasoning but consider efficiency)
        elif accuracy_improvement >= significant_threshold and nr_reasoning_efficiency > nr_efficiency * 0.7:
            use_reasoning = True
            reason = f"Significant accuracy improvement: {nr_acc:.1%} → {nr_reasoning_acc:.1%} (+{accuracy_improvement:.1%}) with acceptable efficiency"
        
        # Override 3: Accuracy floor - never choose mode with very low accuracy if alternative exists
        elif nr_acc < min_floor and nr_reasoning_acc >= min_floor:
            use_reasoning = True
            reason = f"Accuracy floor override: non-reasoning too low ({nr_acc:.1%} < {min_floor:.0%}), reasoning acceptable ({nr_reasoning_acc:.1%})"
        elif nr_reasoning_acc < min_floor and nr_acc >= min_floor:
            use_reasoning = False
            reason = f"Accuracy floor override: reasoning too low ({nr_reasoning_acc:.1%} < {min_floor:.0%}), non-reasoning acceptable ({nr_acc:.1%})"
        
        # Standard efficiency-based decision logic
        elif accuracy_similar:
            # Accuracies are similar - choose mode with lower token usage
            use_reasoning = nr_reasoning_tokens < nr_tokens
            if use_reasoning:
                reason = f"Similar accuracy ({nr_acc:.1%} vs {nr_reasoning_acc:.1%}), reasoning uses fewer tokens ({nr_reasoning_tokens:.0f} vs {nr_tokens:.0f})"
            else:
                reason = f"Similar accuracy ({nr_acc:.1%} vs {nr_reasoning_acc:.1%}), non-reasoning uses fewer tokens ({nr_tokens:.0f} vs {nr_reasoning_tokens:.0f})"
        else:
            # Accuracies are different - choose mode with higher efficiency ratio
            use_reasoning = nr_reasoning_efficiency > nr_efficiency
            efficiency_diff = ((nr_reasoning_efficiency - nr_efficiency) / nr_efficiency * 100) if nr_efficiency > 0 else 0
            if use_reasoning:
                reason = f"Reasoning has higher efficiency: {nr_reasoning_efficiency:.3f} vs {nr_efficiency:.3f} ({efficiency_diff:+.1f}%)"
            else:
                reason = f"Non-reasoning has higher efficiency: {nr_efficiency:.3f} vs {nr_reasoning_efficiency:.3f} ({-efficiency_diff:+.1f}%)"
        
        # Calculate efficiency improvement
        efficiency_improvement = ((nr_reasoning_efficiency - nr_efficiency) / nr_efficiency) if nr_efficiency > 0 else 0
        
        decisions[category] = {
            "use_reasoning": bool(use_reasoning),
            "reason": reason,
            "nr_accuracy": float(nr_acc),
            "nr_reasoning_accuracy": float(nr_reasoning_acc),
            "nr_tokens": float(nr_tokens),
            "nr_reasoning_tokens": float(nr_reasoning_tokens),
            "nr_efficiency": float(nr_efficiency),
            "nr_reasoning_efficiency": float(nr_reasoning_efficiency),
            "efficiency_improvement": float(efficiency_improvement),
            "accuracy_similar": bool(accuracy_similar),
            "p_value": float(p_value),
            "sample_size": min(n1, n2),  # Use smaller sample size for conservative estimate
        }
        
        print(f"  📊 {category}:")
        print(f"    Accuracy: {nr_acc:.1%} (NR) vs {nr_reasoning_acc:.1%} (Reasoning) - {'Similar' if accuracy_similar else 'Different'} (p={p_value:.3f})")
        print(f"    Tokens: {nr_tokens:.0f} (NR) vs {nr_reasoning_tokens:.0f} (Reasoning)")
        print(f"    Efficiency: {nr_efficiency:.3f} (NR) vs {nr_reasoning_efficiency:.3f} (Reasoning)")
        print(f"    Decision: {'✓ REASONING' if use_reasoning else '✗ NON-REASONING'}")
        print(f"    Reason: {reason}")
        
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
        # Fallback to hardcoded template with complete structure
        return {
            "bert_model": {
                "model_id": "sentence-transformers/all-MiniLM-L12-v2",
                "threshold": 0.6,
                "use_cpu": True,
            },
            "semantic_cache": {
                "enabled": False,
                "backend_type": "memory",
                "similarity_threshold": 0.8,
                "max_entries": 1000,
                "ttl_seconds": 3600,
                "eviction_policy": "fifo",
            },
            "tools": {
                "enabled": True,
                "top_k": 3,
                "similarity_threshold": 0.2,
                "tools_db_path": "config/tools_db.json",
                "fallback_to_empty": True,
            },
            "prompt_guard": {
                "enabled": False,
                "use_modernbert": True,
                "model_id": "models/jailbreak_classifier_modernbert-base_model",
                "threshold": 0.7,
                "use_cpu": True,
                "jailbreak_mapping_path": "models/jailbreak_classifier_modernbert-base_model/jailbreak_type_mapping.json",
            },
            "classifier": {
                "category_model": {
                    "model_id": "models/category_classifier_modernbert-base_model",
                    "use_modernbert": True,
                    "threshold": 0.6,
                    "use_cpu": True,
                    "category_mapping_path": "models/category_classifier_modernbert-base_model/category_mapping.json",
                },
                "pii_model": {
                    "model_id": "models/pii_classifier_modernbert-base_presidio_token_model",
                    "use_modernbert": True,
                    "threshold": 0.7,
                    "use_cpu": True,
                    "pii_mapping_path": "models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json",
                },
            },
            "vllm_endpoints": [
                {
                    "name": "endpoint1",
                    "address": "127.0.0.1",
                    "port": 8000,
                    "models": [""],
                    "weight": 1,
                    "health_check_path": "/health",
                }
            ],
            "model_config": {},
            "default_model": "",
            "default_reasoning_effort": "high",
            "categories": [],
            "reasoning_families": {
                "deepseek": {"type": "chat_template_kwargs", "parameter": "thinking"},
                "qwen3": {
                    "type": "chat_template_kwargs",
                    "parameter": "enable_thinking",
                },
                "gpt-oss": {
                    "type": "reasoning_effort",
                    "parameter": "reasoning_effort",
                },
                "gpt": {"type": "reasoning_effort", "parameter": "reasoning_effort"},
            },
            "api": {
                "batch_classification": {
                    "max_batch_size": 100,
                    "concurrency_threshold": 5,
                    "max_concurrency": 8,
                    "metrics": {
                        "enabled": True,
                        "detailed_goroutine_tracking": True,
                        "high_resolution_timing": False,
                        "sample_rate": 1.0,
                        "duration_buckets": [
                            0.001,
                            0.005,
                            0.01,
                            0.025,
                            0.05,
                            0.1,
                            0.25,
                            0.5,
                            1,
                            2.5,
                            5,
                            10,
                            30,
                        ],
                        "size_buckets": [1, 2, 5, 10, 20, 50, 100, 200],
                    },
                }
            },
        }


def generate_config(
    results_df: pd.DataFrame,
    reasoning_decisions: Dict,
    model: str,
    template_path: str = "",
) -> Dict:
    """
    Generate router configuration from template with complete model and endpoint setup.

    Args:
        results_df: Evaluation results DataFrame
        reasoning_decisions: Statistical analysis decisions for each category
        model: Primary model identifier
        template_path: Path to configuration template file

    Returns:
        Complete router configuration dictionary
    """
    # Load template
    config = load_config_template(template_path)

    # Set model-specific values
    config["default_model"] = model

    # Configure vLLM endpoints
    if "vllm_endpoints" in config and config["vllm_endpoints"]:
        # Update the first endpoint with the actual model
        config["vllm_endpoints"][0]["models"] = [model]

    # Configure model-specific settings
    if "model_config" not in config:
        config["model_config"] = {}

    # Determine reasoning family based on model name
    model_lower = model.lower()
    if "qwen3" in model_lower:
        reasoning_family = "qwen3"
    elif "deepseek" in model_lower and ("v31" in model_lower):
        reasoning_family = "deepseek"
    elif "gpt-oss" in model_lower:
        reasoning_family = "gpt-oss"
    elif "gpt" in model_lower:
        reasoning_family = "gpt"
    else:
        reasoning_family = "gpt"  # Default fallback

    # Add model configuration
    config["model_config"][model] = {
        "reasoning_family": reasoning_family,
        "preferred_endpoints": ["endpoint1"],
        "pii_policy": {"allow_by_default": True},
    }

    # Add categories with reasoning decisions
    config["categories"] = []
    for category, decision in reasoning_decisions.items():
        # Get best accuracy for model scoring (use reasoning accuracy if enabled, otherwise baseline)
        cat_df = results_df[
            (results_df["category"] == category) & (results_df["success"])
        ]
        if decision["use_reasoning"]:
            best_acc = decision["nr_reasoning_accuracy"]
        else:
            best_acc = decision["nr_accuracy"]

        config["categories"].append(
            {
                "name": category,
                "model_scores": [
                    {
                        "model": model,
                        "score": float(best_acc),
                        "use_reasoning": decision["use_reasoning"],
                    }
                ],
            }
        )

    return config


def main():
    args = parse_args()

    # Handle methodology display request
    if args.show_methodology:
        print_methodology_summary()
        return

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

    print(
        f"Evaluating {len(df)} questions across {df['category'].nunique()} categories"
    )

    # Evaluate each model
    all_results = []
    try:
        for model in args.models:
            results_df = evaluate_model(
                model, args.endpoint, args.api_key, df, args.concurrent_requests, args.max_tokens
            )
            results_df["model"] = model
            all_results.append(results_df)
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user.")
        if not all_results:
            print("❌ No complete model evaluations to save.")
            return
        print(f"✅ Proceeding with {len(all_results)} completed model evaluations.")

    # Efficiency-Based Analysis
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Analyze reasoning effectiveness using efficiency-based methodology
    print("\nAnalyzing reasoning effectiveness using efficiency-based methodology...")
    reasoning_decisions = analyze_reasoning_efficiency_based(
        combined_df, 
        critical_threshold=args.critical_accuracy_threshold,
        significant_threshold=args.significant_accuracy_threshold,
        min_floor=args.min_accuracy_floor,
        model_name=args.models[0] if args.models else ""
    )

    # Print analysis
    reasoning_enabled = sum(
        1 for d in reasoning_decisions.values() if d["use_reasoning"]
    )
    print(
        f"\n✅ Efficiency-Based Analysis Complete ({reasoning_enabled}/{len(reasoning_decisions)} categories enabled):"
    )
    print()
    for category, decision in reasoning_decisions.items():
        status = "✓ REASONING ENABLED" if decision["use_reasoning"] else "✗ NON-REASONING SELECTED"
        acc_change = f"{decision['nr_accuracy']:.1%} → {decision['nr_reasoning_accuracy']:.1%}"
        token_change = f"{decision['nr_tokens']:.0f} → {decision['nr_reasoning_tokens']:.0f}"
        efficiency_change = f"{decision['nr_efficiency']:.3f} → {decision['nr_reasoning_efficiency']:.3f}"
        similarity_status = "Similar" if decision['accuracy_similar'] else "Different"
        
        print(f"  📊 {category}: {status}")
        print(f"    Accuracy: {acc_change} ({similarity_status}, p={decision['p_value']:.3f})")
        print(f"    Tokens: {token_change}")
        print(f"    Efficiency: {efficiency_change} ({decision['efficiency_improvement']:+.1%})")
        print(f"    Decision: {decision['reason']}")
        print()

    # Generate config (use first model as default)
    config = generate_config(
        combined_df, reasoning_decisions, args.models[0], args.config_template
    )

    # Save config
    os.makedirs(
        (
            os.path.dirname(args.output_config)
            if os.path.dirname(args.output_config)
            else "."
        ),
        exist_ok=True,
    )
    with open(args.output_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Save detailed results
    results_file = args.output_config.replace(".yaml", "_results.csv")
    combined_df.to_csv(results_file, index=False)

    # Save reasoning analysis
    analysis_file = args.output_config.replace(".yaml", "_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(reasoning_decisions, f, indent=2)

    # Print results table
    print_results_table(combined_df, reasoning_decisions)

    print(f"\n✅ Config saved to: {args.output_config}")
    print(f"📊 Results saved to: {results_file}")
    print(f"📈 Analysis saved to: {analysis_file}")


def print_results_table(results_df: pd.DataFrame, reasoning_decisions: Dict[str, Dict]) -> None:
    """
    Print a formatted table showing per-category accuracy and token usage comparison.
    
    Args:
        results_df: DataFrame with evaluation results
        reasoning_decisions: Dictionary with reasoning decisions for each category
    """
    print('\n📊 MMLU-Pro Category Performance Analysis')
    print('=' * 90)
    print()

    # Filter successful results only
    successful_df = results_df[results_df['success'] == True]
    
    # Create summary table
    categories = sorted(successful_df['category'].unique())
    table_data = []

    for category in categories:
        if category in reasoning_decisions:
            decision = reasoning_decisions[category]
            
            nr_acc = decision['nr_accuracy']
            nr_tokens = decision['nr_tokens']
            reasoning_acc = decision['nr_reasoning_accuracy']
            reasoning_tokens = decision['nr_reasoning_tokens']
            
            # Calculate differences
            acc_diff = reasoning_acc - nr_acc
            token_ratio = reasoning_tokens / nr_tokens if nr_tokens > 0 else 0
            
            table_data.append({
                'Category': category.title(),
                'NR_Accuracy': f'{nr_acc:.0%}',
                'NR_Tokens': f'{nr_tokens:.0f}',
                'Reasoning_Accuracy': f'{reasoning_acc:.0%}',
                'Reasoning_Tokens': f'{reasoning_tokens:.0f}',
                'Acc_Diff': f'{acc_diff:+.0%}' if acc_diff != 0 else '0%',
                'Token_Ratio': f'{token_ratio:.1f}x',
                'Decision': '✓ Reasoning' if decision['use_reasoning'] else '✗ Non-Reasoning'
            })

    # Print formatted table
    print(f'{"Category":<15} {"NR Mode":<20} {"Reasoning Mode":<20} {"Differences":<20} {"Decision":<15}')
    print(f'{"":.<15} {"Acc    Tokens":<20} {"Acc    Tokens":<20} {"Acc   Tokens":<20} {"Selected":<15}')
    print('-' * 95)

    for row in table_data:
        category = row['Category'][:14]  # Truncate if too long
        nr_info = f'{row["NR_Accuracy"]:>3} {row["NR_Tokens"]:>8}'
        reasoning_info = f'{row["Reasoning_Accuracy"]:>3} {row["Reasoning_Tokens"]:>8}'
        diff_info = f'{row["Acc_Diff"]:>4} {row["Token_Ratio"]:>7}'
        decision = row['Decision']
        
        print(f'{category:<15} {nr_info:<20} {reasoning_info:<20} {diff_info:<20} {decision:<15}')

    print('-' * 95)

    # Summary statistics
    nr_data = successful_df[successful_df['mode'] == 'NR']
    reasoning_data = successful_df[successful_df['mode'] == 'NR_REASONING']

    overall_nr_acc = nr_data['is_correct'].mean()
    overall_nr_tokens = nr_data['total_tokens'].mean()
    overall_reasoning_acc = reasoning_data['is_correct'].mean()
    overall_reasoning_tokens = reasoning_data['total_tokens'].mean()

    print(f'{"OVERALL":<15} {overall_nr_acc:>3.0%} {overall_nr_tokens:>8.0f}     {overall_reasoning_acc:>3.0%} {overall_reasoning_tokens:>8.0f}     {overall_reasoning_acc-overall_nr_acc:+4.0%} {overall_reasoning_tokens/overall_nr_tokens:>6.1f}x')

    print()
    print('📈 KEY INSIGHTS:')
    print(f'• Reasoning mode uses {overall_reasoning_tokens/overall_nr_tokens:.1f}x more tokens on average')
    print(f'• Overall accuracy: NR {overall_nr_acc:.0%} vs Reasoning {overall_reasoning_acc:.0%}')

    # Count categories where reasoning helps/hurts
    reasoning_helps = sum(1 for row in table_data if '+' in row['Acc_Diff'] and row['Acc_Diff'] != '0%')
    reasoning_hurts = sum(1 for row in table_data if '-' in row['Acc_Diff'])
    reasoning_same = sum(1 for row in table_data if row['Acc_Diff'] == '0%')
    reasoning_selected = sum(1 for row in table_data if '✓' in row['Decision'])

    print(f'• Categories where reasoning helps accuracy: {reasoning_helps}/14')
    print(f'• Categories where reasoning hurts accuracy: {reasoning_hurts}/14') 
    print(f'• Categories with same accuracy: {reasoning_same}/14')
    print(f'• Categories with reasoning selected: {reasoning_selected}/14')


def print_methodology_summary() -> None:
    """
    Display comprehensive documentation of the efficiency-based methodology.

    Provides detailed technical documentation of the efficiency-based framework
    used for reasoning mode evaluation, aligned with RL optimizer objectives.
    """
    print(
        """
╔══════════════════════════════════════════════════════════════════════════════╗
║           EFFICIENCY-BASED METHODOLOGY FOR REASONING MODE EVALUATION        ║
╚══════════════════════════════════════════════════════════════════════════════╝

OBJECTIVE:
    Determine optimal reasoning mode configurations for LLM router categories
    using efficiency-based analysis that balances accuracy and token usage,
    aligned with the RL category optimizer's training objectives.

EFFICIENCY-BASED FRAMEWORK:

    1. EFFICIENCY RATIO CALCULATION
       • Formula: accuracy / (sqrt(tokens) * 0.001 * 10 + 0.1)
       • Same metric used by RL category optimizer for training
       • Higher values indicate better accuracy per token efficiency
       • Square root normalization accounts for diminishing returns

    2. STATISTICAL SIMILARITY TESTING
       • Two-proportion z-test for accuracy comparison
       • Accounts for sample sizes and statistical uncertainty
       • Conservative approach: similar if p > 0.05

    3. DECISION LOGIC
       • If accuracies are statistically similar: choose mode with lower tokens
       • If accuracies are different: choose mode with higher efficiency ratio
       • Transparent, interpretable decision making

DECISION FRAMEWORK:

    STEP 1: Evaluate both modes (NR and NR_REASONING) on MMLU-Pro
    STEP 2: Calculate efficiency ratios for both modes
    STEP 3: Test accuracy similarity using two-proportion z-test
    STEP 4: Apply decision rule:
        - Similar accuracy → choose lower token usage mode
        - Different accuracy → choose higher efficiency ratio mode

ADVANTAGES OVER COMPLEX STATISTICAL METHODS:

    Traditional statistical approaches using multiple pathways suffer from:
    • Over-complexity that obscures decision rationale
    • Multiple testing issues and pathway dependencies
    • Difficult parameter tuning and interpretation
    • Disconnect from actual optimization objectives

    This efficiency-based framework provides:
    • Direct alignment with RL optimizer training objectives
    • Simple, interpretable decision logic
    • Focus on practical efficiency considerations
    • Transparent reasoning for each category decision

INTERPRETATION GUIDELINES:

    Efficiency Ratio:           Higher is better (accuracy per sqrt(token))
    p-value > 0.05:            Accuracies are statistically similar
    p-value ≤ 0.05:            Accuracies are significantly different
    Token Usage:               Lower is better when accuracy is similar
    Decision Transparency:     Clear explanation for each category

IMPLEMENTATION PARAMETERS:

    CONFIDENCE_LEVEL = 0.95     Confidence level for similarity test
    TOKEN_COST_WEIGHT = 0.001   Weight factor in efficiency calculation
    NORMALIZATION_FACTOR = 10   Scaling factor in efficiency calculation
    BASELINE_OFFSET = 0.1       Baseline offset in efficiency calculation

USAGE:
    python reasoning_eval_consolidated.py \\
        --endpoint http://localhost:8000/v1 \\
        --samples-per-category 25 \\
        --confidence-level 0.95

EXAMPLE OUTPUT:
    📊 math: ✓ REASONING ENABLED
      Accuracy: 65.0% → 78.0% (Different, p=0.023)
      Tokens: 1200 → 2400
      Efficiency: 2.145 → 2.287 (+6.6%)
      Decision: Reasoning has higher efficiency

    📊 history: ✗ NON-REASONING SELECTED  
      Accuracy: 72.0% → 74.0% (Similar, p=0.234)
      Tokens: 800 → 1600
      Efficiency: 2.544 → 1.848 (-27.4%)
      Decision: Similar accuracy, non-reasoning uses fewer tokens

"""
    )


if __name__ == "__main__":
    main()

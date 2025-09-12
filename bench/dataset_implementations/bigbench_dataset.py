"""
BIG-bench (Beyond the Imitation Game) dataset implementation.

This module implements the DatasetInterface for selected BIG-bench tasks
that are suitable for multiple-choice evaluation.
"""

import random
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from datasets import load_dataset

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_interface import (
    DatasetInterface, 
    Question, 
    DatasetInfo, 
    PromptFormatter
)


class BIGBenchDataset(DatasetInterface):
    """BIG-bench dataset implementation with selected tasks."""
    
    # Selected tasks that work well for multiple-choice evaluation
    SUPPORTED_TASKS = {
        # Reasoning tasks
        "logical_deduction_three_objects": "Logical Reasoning",
        "logical_deduction_five_objects": "Logical Reasoning", 
        "logical_deduction_seven_objects": "Logical Reasoning",
        "causal_judgement": "Causal Reasoning",
        "formal_fallacies": "Logical Reasoning",
        
        # Mathematics
        "elementary_math_qa": "Mathematics",
        "math_qa": "Mathematics",
        "arithmetic": "Mathematics",
        
        # Science
        "conceptual_combinations": "Science",
        "strange_stories": "Psychology",
        
        # Language understanding
        "disambiguation_qa": "Language Understanding",
        "implicatures": "Language Understanding",
        "presuppositions_as_nli": "Language Understanding",
        
        # Commonsense reasoning
        "physical_intuition": "Commonsense Reasoning",
        "navigate": "Spatial Reasoning",
        
        # Knowledge
        "known_unknowns": "Knowledge",
        "misconceptions": "Knowledge"
    }
    
    def __init__(self, tasks: Optional[List[str]] = None):
        """Initialize BIG-bench dataset.
        
        Args:
            tasks: List of specific tasks to include. If None, includes all supported tasks.
        """
        if tasks is None:
            self.tasks = list(self.SUPPORTED_TASKS.keys())
        else:
            # Validate tasks
            invalid_tasks = [t for t in tasks if t not in self.SUPPORTED_TASKS]
            if invalid_tasks:
                raise ValueError(f"Unsupported tasks: {invalid_tasks}. "
                               f"Supported tasks: {list(self.SUPPORTED_TASKS.keys())}")
            self.tasks = tasks
        
        self._dataset_cache = None
        self._categories_cache = None
    
    @property
    def dataset_name(self) -> str:
        if len(self.tasks) == len(self.SUPPORTED_TASKS):
            return "BIG-bench"
        return f"BIG-bench ({len(self.tasks)} tasks)"
    
    @property
    def supports_cot(self) -> bool:
        return False  # Most BIG-bench tasks don't have explicit CoT
    
    def _load_task_data(self, task_name: str) -> pd.DataFrame:
        """Load data for a specific BIG-bench task."""
        try:
            # Try to load from Hugging Face datasets
            dataset = load_dataset("google/big_bench", task_name, split="train")
            df = pd.DataFrame(dataset)
            df['task_name'] = task_name
            df['category'] = self.SUPPORTED_TASKS[task_name]
            return df
        except Exception as e:
            print(f"Warning: Could not load BIG-bench task {task_name}: {e}")
            return pd.DataFrame()
    
    def _load_raw_dataset(self):
        """Load raw BIG-bench dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache
        
        all_dataframes = []
        
        for task_name in self.tasks:
            print(f"Loading BIG-bench task: {task_name}")
            task_df = self._load_task_data(task_name)
            if not task_df.empty:
                all_dataframes.append(task_df)
        
        if all_dataframes:
            self._dataset_cache = pd.concat(all_dataframes, ignore_index=True)
        else:
            print("Warning: No BIG-bench tasks were successfully loaded")
            self._dataset_cache = pd.DataFrame()
        
        return self._dataset_cache
    
    def _extract_options_and_answer(self, row: Dict[str, Any]) -> Tuple[List[str], str]:
        """Extract options and correct answer from a BIG-bench row."""
        options = []
        correct_answer = None
        
        # Different BIG-bench tasks have different formats
        # Try to handle the most common formats
        
        # Format 1: 'choices' field with list of options
        if 'choices' in row and isinstance(row['choices'], list):
            options = [str(choice) for choice in row['choices']]
            if 'answer' in row:
                answer_idx = row['answer']
                if isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                    # Convert to letter format
                    correct_answer = chr(ord('A') + answer_idx)
        
        # Format 2: Multiple choice with A, B, C, D options
        elif any(f'option_{letter}' in row for letter in ['a', 'b', 'c', 'd']):
            for letter in ['a', 'b', 'c', 'd']:
                option_key = f'option_{letter}'
                if option_key in row and pd.notna(row[option_key]):
                    options.append(str(row[option_key]))
            if 'answer' in row:
                correct_answer = str(row['answer']).upper()
        
        # Format 3: targets field (for some tasks)
        elif 'targets' in row and isinstance(row['targets'], list):
            if len(row['targets']) > 0:
                # Use targets as options, first one as correct
                options = [str(target) for target in row['targets']]
                correct_answer = 'A'  # Assume first option is correct
        
        # Format 4: Simple true/false or yes/no
        elif 'answer' in row:
            answer = str(row['answer']).lower()
            if answer in ['true', 'false']:
                options = ['True', 'False']
                correct_answer = 'A' if answer == 'true' else 'B'
            elif answer in ['yes', 'no']:
                options = ['Yes', 'No']
                correct_answer = 'A' if answer == 'yes' else 'B'
        
        return options, correct_answer
    
    def load_dataset(
        self, 
        categories: Optional[List[str]] = None,
        samples_per_category: Optional[int] = None,
        seed: int = 42
    ) -> Tuple[List[Question], DatasetInfo]:
        """Load BIG-bench dataset."""
        df = self._load_raw_dataset()
        
        if df.empty:
            # Return empty dataset if loading failed
            return [], DatasetInfo(
                name=self.dataset_name,
                description="BIG-bench dataset (failed to load)",
                categories=[],
                total_questions=0,
                format_type="multiple_choice",
                difficulty_level="mixed"
            )
        
        # Convert to Question objects
        questions = []
        for _, row in df.iterrows():
            # Get question text
            question_text = ""
            for field in ['input', 'question', 'inputs', 'context']:
                if field in row and pd.notna(row[field]):
                    question_text = str(row[field])
                    break
            
            if not question_text:
                continue  # Skip if no question text found
            
            # Extract options and answer
            options, correct_answer = self._extract_options_and_answer(row.to_dict())
            
            if not options or not correct_answer:
                continue  # Skip if we can't extract proper multiple choice format
            
            question = Question(
                question_id=str(row.get('id', f"bigbench_{len(questions)}")),
                category=str(row['category']),
                question=question_text,
                options=options,
                correct_answer=correct_answer,
                cot_content=None,  # Most BIG-bench tasks don't have CoT
                metadata={
                    'source': 'BIG-bench',
                    'task_name': row['task_name'],
                    'difficulty': 'mixed'
                }
            )
            questions.append(question)
        
        # Get all unique categories
        all_categories = sorted(list(set(q.category for q in questions)))
        self._categories_cache = all_categories
        
        # Filter by categories if specified
        if categories:
            questions = [q for q in questions if q.category in categories]
            if not questions:
                valid_categories = ", ".join(all_categories)
                raise ValueError(
                    f"No data found for specified categories. "
                    f"Valid categories are: {valid_categories}"
                )
        
        # Sample if requested
        if samples_per_category:
            random.seed(seed)
            np.random.seed(seed)
            
            # Group by category
            category_questions = {}
            for q in questions:
                if q.category not in category_questions:
                    category_questions[q.category] = []
                category_questions[q.category].append(q)
            
            # Sample from each category
            sampled_questions = []
            for category, cat_questions in category_questions.items():
                if len(cat_questions) > samples_per_category:
                    sampled = random.sample(cat_questions, samples_per_category)
                    sampled_questions.extend(sampled)
                else:
                    sampled_questions.extend(cat_questions)
            
            questions = sampled_questions
        
        # Create dataset info
        dataset_info = DatasetInfo(
            name=self.dataset_name,
            description="Beyond the Imitation Game benchmark (selected tasks)",
            categories=list(set(q.category for q in questions)),
            total_questions=len(questions),
            format_type="multiple_choice",
            difficulty_level="mixed"
        )
        
        return questions, dataset_info
    
    def get_available_categories(self) -> List[str]:
        """Get all available BIG-bench categories."""
        if self._categories_cache is None:
            # Load dataset to get categories
            self.load_dataset()
        return self._categories_cache or []
    
    def format_prompt(
        self, 
        question: Question, 
        prompt_style: str = "plain"
    ) -> str:
        """Format BIG-bench question into prompt."""
        if prompt_style == "plain":
            return PromptFormatter.format_plain_prompt(
                question.question, 
                question.options
            )
        elif prompt_style == "cot":
            return PromptFormatter.format_cot_prompt(
                question.question, 
                question.options
            )
        elif prompt_style == "explicit_cot":
            # BIG-bench typically doesn't have CoT content, so fall back to regular CoT
            return PromptFormatter.format_cot_prompt(
                question.question, 
                question.options
            )
        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")
    
    @classmethod
    def get_supported_tasks(cls) -> Dict[str, str]:
        """Get dictionary of supported tasks and their categories."""
        return cls.SUPPORTED_TASKS.copy()


# Convenience classes for specific task groups
class BIGBenchReasoningDataset(BIGBenchDataset):
    """BIG-bench reasoning tasks only."""
    def __init__(self):
        reasoning_tasks = [
            "logical_deduction_three_objects",
            "logical_deduction_five_objects", 
            "logical_deduction_seven_objects",
            "causal_judgement",
            "formal_fallacies"
        ]
        super().__init__(tasks=reasoning_tasks)


class BIGBenchMathDataset(BIGBenchDataset):
    """BIG-bench mathematics tasks only."""
    def __init__(self):
        math_tasks = [
            "elementary_math_qa",
            "math_qa",
            "arithmetic"
        ]
        super().__init__(tasks=math_tasks)

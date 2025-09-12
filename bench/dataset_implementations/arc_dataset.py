"""
ARC (AI2 Reasoning Challenge) dataset implementation.

This module implements the DatasetInterface for ARC dataset with both
Easy and Challenge variants.
"""

import os
import random
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_interface import DatasetInfo, DatasetInterface, PromptFormatter, Question


class ARCDataset(DatasetInterface):
    """ARC (AI2 Reasoning Challenge) dataset implementation."""

    def __init__(self, variant: str = "both"):
        """Initialize ARC dataset.

        Args:
            variant: Which ARC variant to use ("easy", "challenge", or "both")
        """
        self.variant = variant.lower()
        if self.variant not in ["easy", "challenge", "both"]:
            raise ValueError("variant must be 'easy', 'challenge', or 'both'")

        self._dataset_cache = None
        self._categories_cache = None

    @property
    def dataset_name(self) -> str:
        if self.variant == "both":
            return "ARC"
        return f"ARC-{self.variant.title()}"

    @property
    def supports_cot(self) -> bool:
        return False  # ARC doesn't have built-in CoT content

    def _load_raw_dataset(self):
        """Load raw ARC dataset from Hugging Face."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        datasets_to_load = []

        if self.variant in ["easy", "both"]:
            easy_dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
            easy_df = pd.DataFrame(easy_dataset)
            easy_df["difficulty"] = "Easy"
            easy_df["arc_variant"] = "ARC-Easy"
            datasets_to_load.append(easy_df)

        if self.variant in ["challenge", "both"]:
            challenge_dataset = load_dataset(
                "allenai/ai2_arc", "ARC-Challenge", split="test"
            )
            challenge_df = pd.DataFrame(challenge_dataset)
            challenge_df["difficulty"] = "Challenge"
            challenge_df["arc_variant"] = "ARC-Challenge"
            datasets_to_load.append(challenge_df)

        if len(datasets_to_load) == 1:
            self._dataset_cache = datasets_to_load[0]
        else:
            self._dataset_cache = pd.concat(datasets_to_load, ignore_index=True)

        return self._dataset_cache

    def _extract_subject_from_question(self, question: str, choices: List[str]) -> str:
        """Extract subject category from question content."""
        question_lower = question.lower()

        # Physics keywords
        physics_keywords = [
            "force",
            "energy",
            "motion",
            "gravity",
            "mass",
            "velocity",
            "acceleration",
            "friction",
            "magnet",
            "electric",
            "current",
            "voltage",
            "circuit",
            "light",
            "sound",
            "wave",
            "heat",
            "temperature",
            "pressure",
            "density",
        ]

        # Chemistry keywords
        chemistry_keywords = [
            "atom",
            "molecule",
            "element",
            "compound",
            "chemical",
            "reaction",
            "acid",
            "base",
            "ph",
            "solution",
            "mixture",
            "gas",
            "liquid",
            "solid",
            "metal",
            "oxygen",
            "hydrogen",
            "carbon",
            "ion",
            "bond",
        ]

        # Biology keywords
        biology_keywords = [
            "cell",
            "organism",
            "plant",
            "animal",
            "bacteria",
            "virus",
            "gene",
            "dna",
            "protein",
            "enzyme",
            "photosynthesis",
            "respiration",
            "evolution",
            "ecosystem",
            "habitat",
            "species",
            "population",
            "food",
            "nutrition",
        ]

        # Earth Science keywords
        earth_keywords = [
            "earth",
            "planet",
            "rock",
            "mineral",
            "soil",
            "weather",
            "climate",
            "atmosphere",
            "ocean",
            "river",
            "mountain",
            "volcano",
            "earthquake",
            "fossil",
            "geology",
            "solar system",
            "moon",
            "sun",
            "star",
        ]

        # Count keyword matches
        physics_count = sum(
            1 for keyword in physics_keywords if keyword in question_lower
        )
        chemistry_count = sum(
            1 for keyword in chemistry_keywords if keyword in question_lower
        )
        biology_count = sum(
            1 for keyword in biology_keywords if keyword in question_lower
        )
        earth_count = sum(1 for keyword in earth_keywords if keyword in question_lower)

        # Determine category based on highest count
        counts = {
            "Physics": physics_count,
            "Chemistry": chemistry_count,
            "Biology": biology_count,
            "Earth Science": earth_count,
        }

        max_count = max(counts.values())
        if max_count == 0:
            return "General Science"

        # Return category with highest count
        for category, count in counts.items():
            if count == max_count:
                return category

        return "General Science"

    def load_dataset(
        self,
        categories: Optional[List[str]] = None,
        samples_per_category: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[List[Question], DatasetInfo]:
        """Load ARC dataset."""
        df = self._load_raw_dataset()

        # Convert to Question objects and infer categories
        questions = []
        for _, row in df.iterrows():
            # Extract choices - ARC format has choices as dict with labels
            choices_dict = row["choices"]
            if isinstance(choices_dict, dict):
                # Extract text choices in order
                labels = choices_dict.get("label", [])
                texts = choices_dict.get("text", [])
                options = [text for text in texts if text]  # Filter out empty choices
            else:
                options = []

            # Infer category from question content
            category = self._extract_subject_from_question(row["question"], options)

            question = Question(
                question_id=str(row.get("id", f"arc_{len(questions)}")),
                category=category,
                question=str(row["question"]),
                options=options,
                correct_answer=str(row["answerKey"]),
                cot_content=None,  # ARC doesn't have CoT
                metadata={
                    "source": "ARC",
                    "difficulty": row["difficulty"],
                    "arc_variant": row["arc_variant"],
                },
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
            description=f"AI2 Reasoning Challenge ({self.variant})",
            categories=list(set(q.category for q in questions)),
            total_questions=len(questions),
            format_type="multiple_choice",
            difficulty_level="elementary" if self.variant == "easy" else "mixed",
        )

        return questions, dataset_info

    def get_available_categories(self) -> List[str]:
        """Get all available ARC categories."""
        if self._categories_cache is None:
            # Load dataset to get categories
            self.load_dataset()
        return self._categories_cache or []

    def format_prompt(self, question: Question, prompt_style: str = "plain") -> str:
        """Format ARC question into prompt."""
        if prompt_style == "plain":
            return PromptFormatter.format_plain_prompt(
                question.question, question.options
            )
        elif prompt_style == "cot":
            return PromptFormatter.format_cot_prompt(
                question.question, question.options
            )
        elif prompt_style == "explicit_cot":
            # ARC doesn't have CoT content, so fall back to regular CoT
            return PromptFormatter.format_cot_prompt(
                question.question, question.options
            )
        else:
            raise ValueError(f"Unknown prompt style: {prompt_style}")


# Convenience classes for specific variants
class ARCEasyDataset(ARCDataset):
    """ARC-Easy dataset."""

    def __init__(self):
        super().__init__(variant="easy")


class ARCChallengeDataset(ARCDataset):
    """ARC-Challenge dataset."""

    def __init__(self):
        super().__init__(variant="challenge")

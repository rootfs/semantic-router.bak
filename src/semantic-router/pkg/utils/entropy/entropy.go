// Package entropy provides entropy-based uncertainty analysis for classification decisions.
//
// This package implements Shannon entropy calculation and sophisticated reasoning strategies
// to determine when to enable reasoning mode based on classification uncertainty.
//
// The core concept is that high entropy (uncertainty) in classification probabilities
// indicates the model is unsure, suggesting that reasoning mode should be enabled
// to improve accuracy through more careful analysis.
//
// Entropy-Based Routing Strategy:
//
// 1. Very High Uncertainty (normalized entropy ≥ 0.8):
//   - Uniform-like probability distributions
//   - Conservative approach: enable reasoning by default
//   - Strategy: "conservative_default"
//
// 2. High Uncertainty (normalized entropy 0.6-0.8):
//   - Model is confused between 2-3 top categories
//   - Strategy: "top_two_categories_weighted" - use weighted decision from top categories
//   - If top categories disagree on reasoning, enable reasoning for safety
//
// 3. Medium Uncertainty (normalized entropy 0.4-0.6):
//   - Model has preference but not strong confidence
//   - Strategy: "trust_top_category" - trust the top category's reasoning preference
//   - Apply category-specific reasoning rules
//
// 4. Low Uncertainty (normalized entropy 0.2-0.4):
//   - Model is fairly confident in its prediction
//   - Strategy: "trust_classification" - trust the model's decision
//   - Use category reasoning map directly
//
// 5. Very Low Uncertainty (normalized entropy < 0.2):
//   - Model is very confident (near-certain prediction)
//   - Strategy: "trust_classification" - full trust in model decision
//   - Minimal reasoning overhead
package entropy

import (
	"container/heap"
	"errors"
	"fmt"
	"math"
)

// EntropyResult contains the results of entropy-based analysis
type EntropyResult struct {
	Entropy           float64 // Shannon entropy of the probability distribution
	NormalizedEntropy float64 // Entropy normalized to [0,1] range
	Certainty         float64 // Inverse of normalized entropy (1 - normalized_entropy)
	UncertaintyLevel  string  // Human-readable uncertainty level
}

// CategoryProbability represents a category and its probability
type CategoryProbability struct {
	Category    string  // Category name
	Probability float32 // Probability for this category
}

// CalculateEntropy calculates Shannon entropy from a probability distribution.
//
// Shannon entropy measures the uncertainty or "surprise" in a probability distribution:
// H(X) = -∑ P(x) * log2(P(x))
//
// Properties:
// - Entropy is 0 for certain predictions (one probability = 1, others = 0)
// - Entropy is maximized for uniform distributions
// - Higher entropy indicates more uncertainty/confusion in the model
//
// Args:
//   - probabilities: Probability distribution (should sum to 1.0)
//
// Returns:
//   - Shannon entropy in bits (log base 2)
//   - Returns 0.0 for empty input
//
// Example:
//   - [1.0, 0.0, 0.0] → entropy ≈ 0.0 (certain)
//   - [0.5, 0.5, 0.0] → entropy ≈ 1.0 (uncertain between 2)
//   - [0.33, 0.33, 0.34] → entropy ≈ 1.58 (maximum uncertainty for 3 classes)
func CalculateEntropy(probabilities []float32) float64 {
	if len(probabilities) == 0 {
		return 0.0
	}

	entropy := 0.0
	for _, prob := range probabilities {
		if prob > 0 {
			entropy -= float64(prob) * math.Log2(float64(prob))
		}
	}

	return entropy
}

// CalculateNormalizedEntropy calculates normalized entropy (0 to 1 scale)
func CalculateNormalizedEntropy(probabilities []float32) float64 {
	if len(probabilities) <= 1 {
		return 0.0
	}

	entropy := CalculateEntropy(probabilities)
	maxEntropy := math.Log2(float64(len(probabilities)))

	if maxEntropy == 0 {
		return 0.0
	}

	return entropy / maxEntropy
}

// AnalyzeEntropy performs comprehensive entropy analysis
func AnalyzeEntropy(probabilities []float32) EntropyResult {
	entropy := CalculateEntropy(probabilities)
	normalizedEntropy := CalculateNormalizedEntropy(probabilities)
	certainty := 1.0 - normalizedEntropy

	// Determine uncertainty level
	var uncertaintyLevel string
	if normalizedEntropy >= 0.8 {
		uncertaintyLevel = "very_high"
	} else if normalizedEntropy >= 0.6 {
		uncertaintyLevel = "high"
	} else if normalizedEntropy >= 0.4 {
		uncertaintyLevel = "medium"
	} else if normalizedEntropy >= 0.2 {
		uncertaintyLevel = "low"
	} else {
		uncertaintyLevel = "very_low"
	}

	return EntropyResult{
		Entropy:           entropy,
		NormalizedEntropy: normalizedEntropy,
		Certainty:         certainty,
		UncertaintyLevel:  uncertaintyLevel,
	}
}

// ReasoningDecision contains the result of entropy-based reasoning decision
type ReasoningDecision struct {
	UseReasoning     bool                  // Whether to use reasoning
	Confidence       float64               // Confidence in the decision
	DecisionReason   string                // Human-readable reason for the decision
	FallbackStrategy string                // Strategy used for uncertain cases
	TopCategories    []CategoryProbability // Top predicted categories
}

// Entropy-specific error types for better error handling and debugging
var (
	// ErrEmptyProbabilities indicates that the probability array is empty
	ErrEmptyProbabilities = errors.New("entropy: probability array is empty")

	// ErrMismatchedArrays indicates that probability and category arrays have different lengths
	ErrMismatchedArrays = errors.New("entropy: probability and category arrays have mismatched lengths")

	// ErrInvalidProbabilities indicates that the probability distribution is invalid
	ErrInvalidProbabilities = errors.New("entropy: invalid probability distribution")

	// ErrInvalidTopN indicates that the requested number of top categories is invalid
	ErrInvalidTopN = errors.New("entropy: invalid topN parameter")

	// ErrCategoryMappingMissing indicates that category mapping is not available
	ErrCategoryMappingMissing = errors.New("entropy: category mapping is missing or incomplete")

	// ErrClassificationFailed indicates that the underlying classification failed
	ErrClassificationFailed = errors.New("entropy: classification operation failed")
)

// EntropyError wraps entropy-related errors with additional context
type EntropyError struct {
	Operation string // The operation that failed (e.g., "calculate_entropy", "analyze_entropy")
	Cause     error  // The underlying error
	Context   string // Additional context about the error
}

func (e *EntropyError) Error() string {
	if e.Context != "" {
		return fmt.Sprintf("entropy %s failed: %v (context: %s)", e.Operation, e.Cause, e.Context)
	}
	return fmt.Sprintf("entropy %s failed: %v", e.Operation, e.Cause)
}

func (e *EntropyError) Unwrap() error {
	return e.Cause
}

// NewEntropyError creates a new entropy error with context
func NewEntropyError(operation string, cause error, context string) *EntropyError {
	return &EntropyError{
		Operation: operation,
		Cause:     cause,
		Context:   context,
	}
}

// MakeEntropyBasedReasoningDecision implements the entropy-based reasoning strategy.
//
// This is the core function that analyzes classification uncertainty and decides
// whether to enable reasoning mode. The decision is based on:
//
// 1. Entropy Level Analysis:
//   - Calculates normalized entropy to measure uncertainty
//   - Maps entropy to uncertainty levels (very_low to very_high)
//
// 2. Category-Specific Reasoning Rules:
//   - Uses categoryReasoningMap to determine if categories prefer reasoning
//   - Balances model confidence with category-specific preferences
//
// 3. Fallback Strategies:
//   - Conservative: Enable reasoning when uncertain
//   - Weighted: Use probability-weighted decisions from top categories
//   - Trust: Follow category reasoning preferences when confident
//
// Decision Flow:
//
// Very High Uncertainty (≥0.8):
//
//	→ Enable reasoning (conservative safety approach)
//	→ Strategy: "conservative_default"
//
// High Uncertainty (0.6-0.8):
//
//	→ Weighted decision from top 2 categories
//	→ If categories disagree on reasoning: enable reasoning
//	→ Strategy: "top_two_categories_weighted"
//
// Medium Uncertainty (0.4-0.6):
//
//	→ Trust top category's reasoning preference
//	→ Strategy: "trust_top_category"
//
// Low/Very Low Uncertainty (<0.4):
//
//	→ Trust classification, use category reasoning map
//	→ Strategy: "trust_classification"
//
// Args:
//   - probabilities: Model's probability distribution
//   - categoryNames: Names corresponding to probability indices
//   - categoryReasoningMap: Category → reasoning preference mapping
//   - baseConfidenceThreshold: Classification confidence threshold (unused in entropy logic)
//
// Returns:
//   - ReasoningDecision with reasoning recommendation and explanation
func MakeEntropyBasedReasoningDecision(
	probabilities []float32,
	categoryNames []string,
	categoryReasoningMap map[string]bool,
	baseConfidenceThreshold float64,
) ReasoningDecision {

	if len(probabilities) == 0 || len(categoryNames) == 0 {
		return ReasoningDecision{
			UseReasoning:     false,
			Confidence:       0.0,
			DecisionReason:   "no_classification_data",
			FallbackStrategy: "default_no_reasoning",
		}
	}

	// Analyze entropy
	entropyResult := AnalyzeEntropy(probabilities)

	// Get top predicted categories with probabilities
	topCategories := getTopCategories(probabilities, categoryNames, 3)

	// Check if we have valid top categories
	if len(topCategories) == 0 {
		return ReasoningDecision{
			UseReasoning:     false,
			Confidence:       0.0,
			DecisionReason:   "no_valid_categories",
			FallbackStrategy: "empty_categories_fallback",
			TopCategories:    []CategoryProbability{},
		}
	}

	// Get the top prediction
	topCategory := topCategories[0]
	topConfidence := float64(topCategory.Probability)

	// Entropy-based decision logic
	switch entropyResult.UncertaintyLevel {
	case "very_high":
		// Very uncertain - use conservative default (enable reasoning for safety)
		return ReasoningDecision{
			UseReasoning:     true,
			Confidence:       0.3,
			DecisionReason:   "very_high_uncertainty_conservative_default",
			FallbackStrategy: "high_uncertainty_reasoning_enabled",
			TopCategories:    topCategories,
		}

	case "high":
		// High uncertainty - use weighted decision from top 2 categories
		weightedReasoningScore := 0.0
		totalWeight := 0.0
		for i := 0; i < min(2, len(topCategories)); i++ {
			cat := topCategories[i]
			if useReasoning, exists := categoryReasoningMap[cat.Category]; exists {
				weight := float64(cat.Probability)
				if useReasoning {
					weightedReasoningScore += weight
				}
				totalWeight += weight
			}
		}

		useReasoning := false
		confidence := 0.5
		if totalWeight > 0 {
			reasoningRatio := weightedReasoningScore / totalWeight
			useReasoning = reasoningRatio > 0.5
			confidence = math.Abs(reasoningRatio-0.5) + 0.5 // Convert to confidence
		}

		return ReasoningDecision{
			UseReasoning:     useReasoning,
			Confidence:       confidence,
			DecisionReason:   "high_uncertainty_weighted_decision",
			FallbackStrategy: "top_two_categories_weighted",
			TopCategories:    topCategories,
		}

	case "medium":
		// Medium uncertainty - trust top category if above threshold, otherwise weighted
		if topConfidence >= baseConfidenceThreshold {
			if useReasoning, exists := categoryReasoningMap[topCategory.Category]; exists {
				return ReasoningDecision{
					UseReasoning:     useReasoning,
					Confidence:       topConfidence * 0.8, // Reduce confidence due to medium uncertainty
					DecisionReason:   "medium_uncertainty_top_category_above_threshold",
					FallbackStrategy: "trust_top_category",
					TopCategories:    topCategories,
				}
			}
		}

		// Fall back to weighted decision
		return makeWeightedDecision(topCategories, categoryReasoningMap, "medium_uncertainty_weighted")

	case "low", "very_low":
		// Low uncertainty - trust the classification completely
		if useReasoning, exists := categoryReasoningMap[topCategory.Category]; exists {
			confidenceMultiplier := 0.9
			if entropyResult.UncertaintyLevel == "very_low" {
				confidenceMultiplier = 0.95
			}

			return ReasoningDecision{
				UseReasoning:     useReasoning,
				Confidence:       topConfidence * confidenceMultiplier,
				DecisionReason:   entropyResult.UncertaintyLevel + "_uncertainty_trust_classification",
				FallbackStrategy: "trust_top_category",
				TopCategories:    topCategories,
			}
		}

		// Category not in reasoning map - default to no reasoning
		return ReasoningDecision{
			UseReasoning:     false,
			Confidence:       topConfidence * 0.8,
			DecisionReason:   "category_not_in_reasoning_map",
			FallbackStrategy: "unknown_category_default",
			TopCategories:    topCategories,
		}

	default:
		// Unknown uncertainty level - conservative default
		return ReasoningDecision{
			UseReasoning:     false,
			Confidence:       0.5,
			DecisionReason:   "unknown_uncertainty_level",
			FallbackStrategy: "conservative_default",
			TopCategories:    topCategories,
		}
	}
}

// categoryHeap implements a min-heap for efficient top-N category selection
type categoryHeap []CategoryProbability

func (h categoryHeap) Len() int           { return len(h) }
func (h categoryHeap) Less(i, j int) bool { return h[i].Probability < h[j].Probability } // min-heap
func (h categoryHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *categoryHeap) Push(x interface{}) {
	*h = append(*h, x.(CategoryProbability))
}

func (h *categoryHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

// getTopCategories returns the top N categories by probability using an efficient heap-based approach.
//
// This implementation uses a min-heap to maintain the top N categories in O(n log k) time,
// where n is the total number of categories and k is the number of top categories requested.
// This is much more efficient than the previous O(n²) sorting approach for large category counts.
//
// Performance characteristics:
//   - Time complexity: O(n log k) where k = min(n, topN)
//   - Space complexity: O(k) for the heap
//   - Memory efficient: doesn't create full sorted array
//   - Optimal for k << n (common case: top 3-5 from 100+ categories)
//
// Algorithm:
//  1. Use min-heap to maintain top k elements
//  2. For each category, if heap size < k, add to heap
//  3. If heap size = k and current > heap.min, replace heap.min
//  4. Extract all elements from heap and reverse for descending order
//
// Args:
//   - probabilities: Array of probability values
//   - categoryNames: Corresponding category names
//   - topN: Number of top categories to return
//
// Returns:
//   - Slice of top N categories sorted by probability (descending)
//   - Empty slice for invalid inputs (mismatched lengths, topN <= 0)
func getTopCategories(probabilities []float32, categoryNames []string, topN int) []CategoryProbability {
	// Input validation
	if len(probabilities) != len(categoryNames) || topN <= 0 {
		return []CategoryProbability{}
	}

	// Handle edge case: fewer categories than requested
	actualN := min(topN, len(probabilities))
	if actualN == 0 {
		return []CategoryProbability{}
	}

	// Use heap-based approach for efficiency
	h := &categoryHeap{}
	heap.Init(h)

	for i, prob := range probabilities {
		category := CategoryProbability{
			Category:    categoryNames[i],
			Probability: prob,
		}

		if h.Len() < actualN {
			// Heap not full, add element
			heap.Push(h, category)
		} else if prob > (*h)[0].Probability {
			// Current element is larger than heap minimum, replace it
			heap.Pop(h)
			heap.Push(h, category)
		}
	}

	// Extract all elements from heap
	result := make([]CategoryProbability, h.Len())
	for i := h.Len() - 1; i >= 0; i-- {
		result[i] = heap.Pop(h).(CategoryProbability)
	}

	// Result is now in descending order (largest first)
	return result
}

// Helper function to make weighted decision from top categories
func makeWeightedDecision(topCategories []CategoryProbability, categoryReasoningMap map[string]bool, reason string) ReasoningDecision {
	weightedReasoningScore := 0.0
	totalWeight := 0.0

	for _, cat := range topCategories {
		if useReasoning, exists := categoryReasoningMap[cat.Category]; exists {
			weight := float64(cat.Probability)
			if useReasoning {
				weightedReasoningScore += weight
			}
			totalWeight += weight
		}
	}

	useReasoning := false
	confidence := 0.5
	if totalWeight > 0 {
		reasoningRatio := weightedReasoningScore / totalWeight
		useReasoning = reasoningRatio > 0.5
		confidence = math.Abs(reasoningRatio-0.5) + 0.5
	}

	return ReasoningDecision{
		UseReasoning:     useReasoning,
		Confidence:       confidence,
		DecisionReason:   reason,
		FallbackStrategy: "weighted_decision",
		TopCategories:    topCategories,
	}
}

// Helper function to get minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

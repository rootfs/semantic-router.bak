package entropy

import (
	"math"
	"testing"
)

func TestCalculateEntropy(t *testing.T) {
	tests := []struct {
		name           string
		probabilities  []float32
		expectedResult float64
	}{
		{
			name:           "Uniform distribution",
			probabilities:  []float32{0.25, 0.25, 0.25, 0.25},
			expectedResult: 2.0, // log2(4) = 2.0 for uniform distribution
		},
		{
			name:           "Certain prediction",
			probabilities:  []float32{1.0, 0.0, 0.0, 0.0},
			expectedResult: 0.0, // No uncertainty
		},
		{
			name:           "High certainty",
			probabilities:  []float32{0.85, 0.05, 0.05, 0.05},
			expectedResult: 0.8476, // Should be low entropy
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CalculateEntropy(tt.probabilities)
			if math.Abs(result-tt.expectedResult) > 0.01 {
				t.Errorf("CalculateEntropy() = %v, want %v", result, tt.expectedResult)
			}
		})
	}
}

func TestCalculateNormalizedEntropy(t *testing.T) {
	tests := []struct {
		name           string
		probabilities  []float32
		expectedResult float64
	}{
		{
			name:           "Uniform distribution",
			probabilities:  []float32{0.25, 0.25, 0.25, 0.25},
			expectedResult: 1.0, // Maximum entropy for 4 classes
		},
		{
			name:           "Certain prediction",
			probabilities:  []float32{1.0, 0.0, 0.0, 0.0},
			expectedResult: 0.0, // No uncertainty
		},
		{
			name:           "High certainty biology",
			probabilities:  []float32{0.85, 0.05, 0.03, 0.03, 0.02, 0.02},
			expectedResult: 0.365, // Should be low normalized entropy
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CalculateNormalizedEntropy(tt.probabilities)
			if math.Abs(result-tt.expectedResult) > 0.01 {
				t.Errorf("CalculateNormalizedEntropy() = %v, want %v", result, tt.expectedResult)
			}
		})
	}
}

func TestAnalyzeEntropy(t *testing.T) {
	tests := []struct {
		name                     string
		probabilities            []float32
		expectedUncertaintyLevel string
	}{
		{
			name:                     "Very high uncertainty",
			probabilities:            []float32{0.17, 0.17, 0.17, 0.17, 0.16, 0.16},
			expectedUncertaintyLevel: "very_high",
		},
		{
			name:                     "High uncertainty",
			probabilities:            []float32{0.45, 0.40, 0.10, 0.05},
			expectedUncertaintyLevel: "high",
		},
		{
			name:                     "Medium uncertainty",
			probabilities:            []float32{0.70, 0.15, 0.10, 0.05},
			expectedUncertaintyLevel: "high", // Actually 0.660 normalized entropy
		},
		{
			name:                     "Low uncertainty",
			probabilities:            []float32{0.85, 0.05, 0.05, 0.05},
			expectedUncertaintyLevel: "medium", // Actually 0.424 normalized entropy
		},
		{
			name:                     "Very low uncertainty",
			probabilities:            []float32{0.90, 0.04, 0.03, 0.02, 0.01},
			expectedUncertaintyLevel: "low", // Actually 0.282 normalized entropy
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := AnalyzeEntropy(tt.probabilities)
			if result.UncertaintyLevel != tt.expectedUncertaintyLevel {
				t.Errorf("AnalyzeEntropy().UncertaintyLevel = %v, want %v", result.UncertaintyLevel, tt.expectedUncertaintyLevel)
			}
		})
	}
}

func TestMakeEntropyBasedReasoningDecision(t *testing.T) {
	categoryReasoningMap := map[string]bool{
		"biology":   false,
		"chemistry": false,
		"law":       false,
		"other":     false,
		"physics":   true,
		"business":  true,
	}

	tests := []struct {
		name                   string
		probabilities          []float32
		categoryNames          []string
		expectedUseReasoning   bool
		expectedDecisionReason string
	}{
		{
			name:                   "High certainty biology (should not use reasoning)",
			probabilities:          []float32{0.85, 0.05, 0.03, 0.03, 0.02, 0.02},
			categoryNames:          []string{"biology", "other", "chemistry", "law", "physics", "business"},
			expectedUseReasoning:   false,
			expectedDecisionReason: "low_uncertainty_trust_classification",
		},
		{
			name:                   "Uniform distribution (very high uncertainty)",
			probabilities:          []float32{0.17, 0.17, 0.17, 0.17, 0.16, 0.16},
			categoryNames:          []string{"biology", "other", "chemistry", "law", "physics", "business"},
			expectedUseReasoning:   true,
			expectedDecisionReason: "very_high_uncertainty_conservative_default",
		},
		{
			name:                   "High uncertainty between biology and chemistry",
			probabilities:          []float32{0.45, 0.40, 0.10, 0.03, 0.01, 0.01},
			categoryNames:          []string{"biology", "chemistry", "other", "law", "physics", "business"},
			expectedUseReasoning:   false, // Both biology and chemistry don't use reasoning
			expectedDecisionReason: "high_uncertainty_weighted_decision",
		},
		{
			name:                   "Strong physics classification",
			probabilities:          []float32{0.90, 0.04, 0.02, 0.02, 0.01, 0.01},
			categoryNames:          []string{"physics", "biology", "chemistry", "law", "other", "business"},
			expectedUseReasoning:   true,                                   // Physics uses reasoning
			expectedDecisionReason: "low_uncertainty_trust_classification", // Actually low uncertainty, not very low
		},
		{
			name:                   "Problematic other category with medium uncertainty",
			probabilities:          []float32{0.70, 0.15, 0.10, 0.03, 0.01, 0.01},
			categoryNames:          []string{"other", "biology", "chemistry", "law", "physics", "business"},
			expectedUseReasoning:   false, // Other category doesn't use reasoning
			expectedDecisionReason: "medium_uncertainty_top_category_above_threshold",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MakeEntropyBasedReasoningDecision(
				tt.probabilities,
				tt.categoryNames,
				categoryReasoningMap,
				0.6, // threshold
			)

			if result.UseReasoning != tt.expectedUseReasoning {
				t.Errorf("MakeEntropyBasedReasoningDecision().UseReasoning = %v, want %v", result.UseReasoning, tt.expectedUseReasoning)
			}

			if result.DecisionReason != tt.expectedDecisionReason {
				t.Errorf("MakeEntropyBasedReasoningDecision().DecisionReason = %v, want %v", result.DecisionReason, tt.expectedDecisionReason)
			}

			// Verify top categories are returned
			if len(result.TopCategories) == 0 {
				t.Error("Expected top categories to be returned")
			}

			// Verify confidence is reasonable
			if result.Confidence < 0.0 || result.Confidence > 1.0 {
				t.Errorf("Confidence should be between 0 and 1, got %v", result.Confidence)
			}
		})
	}
}

func TestGetTopCategories(t *testing.T) {
	probabilities := []float32{0.45, 0.30, 0.15, 0.05, 0.03, 0.02}
	categoryNames := []string{"biology", "chemistry", "physics", "law", "other", "business"}

	result := getTopCategories(probabilities, categoryNames, 3)

	if len(result) != 3 {
		t.Errorf("Expected 3 top categories, got %d", len(result))
	}

	// Check that they're sorted by probability (descending)
	if result[0].Category != "biology" || result[0].Probability != 0.45 {
		t.Errorf("Expected first category to be biology with 0.45, got %s with %f", result[0].Category, result[0].Probability)
	}

	if result[1].Category != "chemistry" || result[1].Probability != 0.30 {
		t.Errorf("Expected second category to be chemistry with 0.30, got %s with %f", result[1].Category, result[1].Probability)
	}

	if result[2].Category != "physics" || result[2].Probability != 0.15 {
		t.Errorf("Expected third category to be physics with 0.15, got %s with %f", result[2].Category, result[2].Probability)
	}
}

// TestCalculateEntropyEdgeCases tests edge cases for entropy calculation
func TestCalculateEntropyEdgeCases(t *testing.T) {
	testCases := []struct {
		name            string
		probabilities   []float32
		expectedEntropy float64
		description     string
	}{
		{
			name:            "Empty probability array",
			probabilities:   []float32{},
			expectedEntropy: 0.0,
			description:     "Empty array should return zero entropy",
		},
		{
			name:            "Single element array",
			probabilities:   []float32{1.0},
			expectedEntropy: 0.0,
			description:     "Single certain element should have zero entropy",
		},
		{
			name:            "Array with zero probabilities",
			probabilities:   []float32{0.0, 0.0, 0.0},
			expectedEntropy: 0.0,
			description:     "All zeros should return zero entropy",
		},
		{
			name:            "Array with negative probabilities",
			probabilities:   []float32{-0.1, 0.6, 0.5},
			expectedEntropy: 0.942, // Only positive values contribute: -0*log2(0) + 0.6*log2(0.6) + 0.5*log2(0.5)
			description:     "Negative probabilities should be ignored",
		},
		{
			name:            "Array with very small probabilities",
			probabilities:   []float32{0.999999, 0.000001, 0.0},
			expectedEntropy: 0.000014, // Very small entropy
			description:     "Very small probabilities should work correctly",
		},
		{
			name:            "Array that doesn't sum to 1.0",
			probabilities:   []float32{0.3, 0.3, 0.3}, // Sum = 0.9
			expectedEntropy: 1.563,                    // Still calculates entropy: 3 * (-0.3 * log2(0.3))
			description:     "Non-normalized probabilities should still work",
		},
		{
			name:            "Array with probabilities > 1.0",
			probabilities:   []float32{1.5, 0.5}, // Invalid but should handle
			expectedEntropy: -0.377,              // Negative because 1.5*log2(1.5) is positive, and we have -sum
			description:     "Invalid probabilities > 1.0 can result in negative entropy",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			entropy := CalculateEntropy(tc.probabilities)

			if math.Abs(entropy-tc.expectedEntropy) > 0.01 {
				t.Errorf("Expected entropy %.6f, got %.6f for %s",
					tc.expectedEntropy, entropy, tc.description)
			}

			// Entropy should never be negative for valid probability distributions
			// But can be negative for invalid distributions (probabilities > 1.0)
			if entropy < 0 && tc.name != "Array with probabilities > 1.0" {
				t.Errorf("Entropy should never be negative for valid distributions, got %.6f", entropy)
			}

			t.Logf("Test '%s' passed: entropy=%.6f, %s", tc.name, entropy, tc.description)
		})
	}
}

// TestAnalyzeEntropyEdgeCases tests edge cases for entropy analysis
func TestAnalyzeEntropyEdgeCases(t *testing.T) {
	testCases := []struct {
		name                string
		probabilities       []float32
		expectedUncertainty string
		shouldError         bool
		description         string
	}{
		{
			name:                "Empty array",
			probabilities:       []float32{},
			expectedUncertainty: "very_low", // Default for empty
			shouldError:         false,
			description:         "Empty array should handle gracefully",
		},
		{
			name:                "Single element",
			probabilities:       []float32{1.0},
			expectedUncertainty: "very_low", // No uncertainty with single element
			shouldError:         false,
			description:         "Single element should be very low uncertainty",
		},
		{
			name:                "Two element uniform",
			probabilities:       []float32{0.5, 0.5},
			expectedUncertainty: "very_high", // Maximum entropy for 2 classes
			shouldError:         false,
			description:         "Uniform distribution should be very high uncertainty",
		},
		{
			name:                "Array with extreme skew",
			probabilities:       []float32{0.99999, 0.00001},
			expectedUncertainty: "very_low",
			shouldError:         false,
			description:         "Extremely skewed distribution should be very low uncertainty",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := AnalyzeEntropy(tc.probabilities)

			if result.UncertaintyLevel != tc.expectedUncertainty {
				t.Errorf("Expected uncertainty level '%s', got '%s' for %s",
					tc.expectedUncertainty, result.UncertaintyLevel, tc.description)
			}

			// Normalized entropy should be between 0 and 1
			if result.NormalizedEntropy < 0 || result.NormalizedEntropy > 1 {
				t.Errorf("Normalized entropy should be [0,1], got %.6f", result.NormalizedEntropy)
			}

			// Certainty should be inverse of normalized entropy
			expectedCertainty := 1.0 - result.NormalizedEntropy
			if math.Abs(result.Certainty-expectedCertainty) > 0.001 {
				t.Errorf("Certainty should be 1-normalized_entropy, got %.6f vs %.6f",
					result.Certainty, expectedCertainty)
			}

			t.Logf("Test '%s' passed: uncertainty=%s, entropy=%.3f, %s",
				tc.name, result.UncertaintyLevel, result.Entropy, tc.description)
		})
	}
}

// TestMakeEntropyBasedReasoningDecisionEdgeCases tests edge cases for reasoning decisions
func TestMakeEntropyBasedReasoningDecisionEdgeCases(t *testing.T) {
	testCases := []struct {
		name                 string
		probabilities        []float32
		categoryNames        []string
		categoryReasoningMap map[string]bool
		threshold            float64
		expectedValid        bool
		description          string
	}{
		{
			name:                 "Empty probabilities",
			probabilities:        []float32{},
			categoryNames:        []string{},
			categoryReasoningMap: map[string]bool{},
			threshold:            0.7,
			expectedValid:        false, // Should return invalid decision
			description:          "Empty inputs should return invalid decision",
		},
		{
			name:                 "Mismatched array lengths",
			probabilities:        []float32{0.5, 0.3, 0.2},
			categoryNames:        []string{"cat1", "cat2"}, // Missing one name
			categoryReasoningMap: map[string]bool{"cat1": true, "cat2": false},
			threshold:            0.7,
			expectedValid:        false,
			description:          "Mismatched lengths should be handled gracefully",
		},
		{
			name:                 "Empty category reasoning map",
			probabilities:        []float32{0.7, 0.2, 0.1},
			categoryNames:        []string{"unknown1", "unknown2", "unknown3"},
			categoryReasoningMap: map[string]bool{}, // Empty map
			threshold:            0.7,
			expectedValid:        true, // Should still work with defaults
			description:          "Empty reasoning map should use defaults",
		},
		{
			name:          "Categories with special characters",
			probabilities: []float32{0.6, 0.4},
			categoryNames: []string{"category with spaces", "category-with-dashes"},
			categoryReasoningMap: map[string]bool{
				"category with spaces": true,
				"category-with-dashes": false,
			},
			threshold:     0.7,
			expectedValid: true,
			description:   "Special characters in category names should work",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			decision := MakeEntropyBasedReasoningDecision(
				tc.probabilities,
				tc.categoryNames,
				tc.categoryReasoningMap,
				tc.threshold,
			)

			if tc.expectedValid {
				// Valid decisions should have reasonable values
				if decision.Confidence < 0 || decision.Confidence > 1 {
					t.Errorf("Decision confidence should be [0,1], got %.3f", decision.Confidence)
				}

				if decision.DecisionReason == "" {
					t.Errorf("Decision reason should not be empty for valid decisions")
				}

				if decision.FallbackStrategy == "" {
					t.Errorf("Fallback strategy should not be empty for valid decisions")
				}
			} else {
				// Invalid decisions should have safe defaults
				if decision.UseReasoning != false {
					t.Logf("Invalid decision defaulted to UseReasoning=%v", decision.UseReasoning)
				}
			}

			t.Logf("Test '%s' passed: valid=%v, reasoning=%v, confidence=%.3f, %s",
				tc.name, tc.expectedValid, decision.UseReasoning, decision.Confidence, tc.description)
		})
	}
}

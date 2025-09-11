package entropy

import (
	"math"
	"testing"
)

func TestDefaultAdvancedEntropyConfig(t *testing.T) {
	config := DefaultAdvancedEntropyConfig()

	if !config.EnableRenyi {
		t.Errorf("Expected EnableRenyi to be true")
	}
	if !config.EnableTsallis {
		t.Errorf("Expected EnableTsallis to be true")
	}
	if config.RenyiAlpha != 2.0 {
		t.Errorf("Expected RenyiAlpha 2.0, got %.3f", config.RenyiAlpha)
	}
	if config.TsallisQ != 2.0 {
		t.Errorf("Expected TsallisQ 2.0, got %.3f", config.TsallisQ)
	}

	// Check weights sum to 1.0
	totalWeight := config.ShannonWeight + config.RenyiWeight + config.TsallisWeight
	if math.Abs(totalWeight-1.0) > 0.001 {
		t.Errorf("Expected weights to sum to 1.0, got %.3f", totalWeight)
	}
}

func TestCalculateRenyiEntropy(t *testing.T) {
	testCases := []struct {
		name          string
		probabilities []float32
		alpha         float64
		expectedRange [2]float64 // [min, max] expected range
		description   string
	}{
		{
			name:          "Uniform distribution alpha=2",
			probabilities: []float32{0.25, 0.25, 0.25, 0.25},
			alpha:         2.0,
			expectedRange: [2]float64{1.8, 2.2}, // Should be close to 2.0
			description:   "Rényi entropy with alpha=2 for uniform distribution",
		},
		{
			name:          "Certain prediction alpha=2",
			probabilities: []float32{1.0, 0.0, 0.0, 0.0},
			alpha:         2.0,
			expectedRange: [2]float64{0.0, 0.1}, // Should be close to 0
			description:   "Rényi entropy with alpha=2 for certain prediction",
		},
		{
			name:          "Alpha=1 should equal Shannon",
			probabilities: []float32{0.5, 0.3, 0.2},
			alpha:         1.0,
			expectedRange: [2]float64{1.4, 1.6}, // Shannon entropy for this distribution
			description:   "Rényi entropy with alpha=1 should equal Shannon entropy",
		},
		{
			name:          "Alpha=0 gives log of non-zero count",
			probabilities: []float32{0.5, 0.3, 0.2, 0.0},
			alpha:         0.0,
			expectedRange: [2]float64{1.58, 1.59}, // log₂(3) ≈ 1.585
			description:   "Rényi entropy with alpha=0 counts non-zero probabilities",
		},
		{
			name:          "Alpha=infinity gives min-entropy",
			probabilities: []float32{0.6, 0.3, 0.1},
			alpha:         math.Inf(1),
			expectedRange: [2]float64{0.7, 0.8}, // -log₂(0.6) ≈ 0.737
			description:   "Rényi entropy with alpha=∞ gives min-entropy",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			entropy := CalculateRenyiEntropy(tc.probabilities, tc.alpha)

			if entropy < tc.expectedRange[0] || entropy > tc.expectedRange[1] {
				t.Errorf("Entropy %.6f is outside expected range [%.3f, %.3f] for %s",
					entropy, tc.expectedRange[0], tc.expectedRange[1], tc.description)
			}

			t.Logf("Test '%s' passed: entropy=%.6f, %s", tc.name, entropy, tc.description)
		})
	}
}

func TestCalculateTsallisEntropy(t *testing.T) {
	testCases := []struct {
		name          string
		probabilities []float32
		q             float64
		expectedRange [2]float64 // [min, max] expected range
		description   string
	}{
		{
			name:          "Uniform distribution q=2",
			probabilities: []float32{0.25, 0.25, 0.25, 0.25},
			q:             2.0,
			expectedRange: [2]float64{0.7, 0.8}, // Should be 0.75 for uniform with q=2
			description:   "Tsallis entropy with q=2 for uniform distribution",
		},
		{
			name:          "Certain prediction q=2",
			probabilities: []float32{1.0, 0.0, 0.0, 0.0},
			q:             2.0,
			expectedRange: [2]float64{0.0, 0.1}, // Should be close to 0
			description:   "Tsallis entropy with q=2 for certain prediction",
		},
		{
			name:          "Q=1 should equal Shannon",
			probabilities: []float32{0.5, 0.3, 0.2},
			q:             1.0,
			expectedRange: [2]float64{1.4, 1.6}, // Shannon entropy for this distribution
			description:   "Tsallis entropy with q=1 should equal Shannon entropy",
		},
		{
			name:          "Q=2 gives Gini-Simpson related measure",
			probabilities: []float32{0.4, 0.3, 0.2, 0.1},
			q:             2.0,
			expectedRange: [2]float64{0.6, 0.8}, // 1 - sum(p²)
			description:   "Tsallis entropy with q=2 is related to Gini-Simpson index",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			entropy := CalculateTsallisEntropy(tc.probabilities, tc.q)

			if entropy < tc.expectedRange[0] || entropy > tc.expectedRange[1] {
				t.Errorf("Entropy %.6f is outside expected range [%.3f, %.3f] for %s",
					entropy, tc.expectedRange[0], tc.expectedRange[1], tc.description)
			}

			t.Logf("Test '%s' passed: entropy=%.6f, %s", tc.name, entropy, tc.description)
		})
	}
}

func TestCalculateAdvancedEntropy(t *testing.T) {
	config := DefaultAdvancedEntropyConfig()
	probabilities := []float32{0.4, 0.3, 0.2, 0.1}

	result := CalculateAdvancedEntropy(probabilities, config)

	// Check that all entropy measures are calculated
	if result.Shannon <= 0 {
		t.Errorf("Expected positive Shannon entropy, got %.6f", result.Shannon)
	}
	if result.Renyi <= 0 {
		t.Errorf("Expected positive Rényi entropy, got %.6f", result.Renyi)
	}
	if result.Tsallis <= 0 {
		t.Errorf("Expected positive Tsallis entropy, got %.6f", result.Tsallis)
	}

	// Check that normalized values are in [0, 1] range
	if result.NormalizedShannon < 0 || result.NormalizedShannon > 1 {
		t.Errorf("Normalized Shannon entropy %.6f is outside [0,1] range", result.NormalizedShannon)
	}
	if result.NormalizedRenyi < 0 || result.NormalizedRenyi > 1 {
		t.Errorf("Normalized Rényi entropy %.6f is outside [0,1] range", result.NormalizedRenyi)
	}
	if result.NormalizedTsallis < 0 || result.NormalizedTsallis > 1 {
		t.Errorf("Normalized Tsallis entropy %.6f is outside [0,1] range", result.NormalizedTsallis)
	}

	// Check consensus measures
	if result.ConsensusEntropy < 0 || result.ConsensusEntropy > 1 {
		t.Errorf("Consensus entropy %.6f is outside [0,1] range", result.ConsensusEntropy)
	}

	validUncertaintyLevels := []string{"very_low", "low", "medium", "high", "very_high"}
	isValid := false
	for _, level := range validUncertaintyLevels {
		if result.ConsensusUncertainty == level {
			isValid = true
			break
		}
	}
	if !isValid {
		t.Errorf("Invalid consensus uncertainty level: %s", result.ConsensusUncertainty)
	}

	if result.EntropyDivergence < 0 {
		t.Errorf("Entropy divergence should be non-negative, got %.6f", result.EntropyDivergence)
	}

	t.Logf("Advanced entropy test passed:")
	t.Logf("  Shannon: %.6f (normalized: %.6f)", result.Shannon, result.NormalizedShannon)
	t.Logf("  Rényi: %.6f (normalized: %.6f)", result.Renyi, result.NormalizedRenyi)
	t.Logf("  Tsallis: %.6f (normalized: %.6f)", result.Tsallis, result.NormalizedTsallis)
	t.Logf("  Consensus: %.6f (uncertainty: %s)", result.ConsensusEntropy, result.ConsensusUncertainty)
	t.Logf("  Divergence: %.6f", result.EntropyDivergence)
}

func TestCalculateAdvancedEntropyEmptyInput(t *testing.T) {
	config := DefaultAdvancedEntropyConfig()
	result := CalculateAdvancedEntropy([]float32{}, config)

	if result.Shannon != 0 {
		t.Errorf("Expected Shannon entropy 0 for empty input, got %.6f", result.Shannon)
	}
	if result.Renyi != 0 {
		t.Errorf("Expected Rényi entropy 0 for empty input, got %.6f", result.Renyi)
	}
	if result.Tsallis != 0 {
		t.Errorf("Expected Tsallis entropy 0 for empty input, got %.6f", result.Tsallis)
	}
}

func TestMakeAdvancedEntropyBasedReasoningDecision(t *testing.T) {
	config := DefaultAdvancedEntropyConfig()
	categoryNames := []string{"physics", "biology", "chemistry", "other"}
	categoryReasoningMap := map[string]bool{
		"physics":   false,
		"biology":   true,
		"chemistry": true,
		"other":     false,
	}

	testCases := []struct {
		name          string
		probabilities []float32
		description   string
	}{
		{
			name:          "High certainty physics",
			probabilities: []float32{0.8, 0.1, 0.05, 0.05},
			description:   "Strong physics prediction should use consensus decision",
		},
		{
			name:          "Uniform distribution",
			probabilities: []float32{0.25, 0.25, 0.25, 0.25},
			description:   "High uncertainty should enable reasoning",
		},
		{
			name:          "Biology preference",
			probabilities: []float32{0.1, 0.6, 0.2, 0.1},
			description:   "Strong biology prediction should follow category preference",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			decision := MakeAdvancedEntropyBasedReasoningDecision(
				tc.probabilities,
				categoryNames,
				categoryReasoningMap,
				0.7,
				config,
			)

			// Basic validation
			if decision.Confidence < 0 || decision.Confidence > 1 {
				t.Errorf("Decision confidence %.3f is outside [0,1] range", decision.Confidence)
			}

			if decision.DecisionReason == "" {
				t.Errorf("Decision reason should not be empty")
			}

			if decision.FallbackStrategy == "" {
				t.Errorf("Fallback strategy should not be empty")
			}

			if len(decision.TopCategories) == 0 {
				t.Errorf("Top categories should not be empty")
			}

			t.Logf("Advanced reasoning decision for '%s': reasoning=%v, confidence=%.3f, reason=%s",
				tc.name, decision.UseReasoning, decision.Confidence, decision.DecisionReason)
		})
	}
}

func TestEntropyDivergenceHighDisagreement(t *testing.T) {
	// Create a configuration that would cause high divergence
	config := AdvancedEntropyConfig{
		EnableRenyi:   true,
		EnableTsallis: true,
		RenyiAlpha:    0.5,  // Emphasizes low probabilities
		TsallisQ:      3.0,  // Different behavior from Shannon
		ShannonWeight: 0.33,
		RenyiWeight:   0.33,
		TsallisWeight: 0.34,
	}

	// Use a distribution that might cause disagreement between measures
	probabilities := []float32{0.7, 0.15, 0.1, 0.05}
	categoryNames := []string{"physics", "biology", "chemistry", "other"}
	categoryReasoningMap := map[string]bool{
		"physics":   false,
		"biology":   true,
		"chemistry": true,
		"other":     false,
	}

	decision := MakeAdvancedEntropyBasedReasoningDecision(
		probabilities,
		categoryNames,
		categoryReasoningMap,
		0.7,
		config,
	)

	// The decision should be valid regardless of divergence
	if decision.Confidence < 0 || decision.Confidence > 1 {
		t.Errorf("Decision confidence %.3f is outside [0,1] range", decision.Confidence)
	}

	t.Logf("High divergence test: reasoning=%v, confidence=%.3f, reason=%s",
		decision.UseReasoning, decision.Confidence, decision.DecisionReason)
}

func TestNormalizeTsallisEntropy(t *testing.T) {
	testCases := []struct {
		name           string
		tsallisEntropy float64
		numClasses     int
		q              float64
		expectedRange  [2]float64
		description    string
	}{
		{
			name:           "Normal case",
			tsallisEntropy: 0.5,
			numClasses:     4,
			q:              2.0,
			expectedRange:  [2]float64{0.0, 1.0},
			description:    "Normal Tsallis entropy normalization",
		},
		{
			name:           "Single class",
			tsallisEntropy: 0.0,
			numClasses:     1,
			q:              2.0,
			expectedRange:  [2]float64{0.0, 0.0},
			description:    "Single class should return 0",
		},
		{
			name:           "Zero classes",
			tsallisEntropy: 0.0,
			numClasses:     0,
			q:              2.0,
			expectedRange:  [2]float64{0.0, 0.0},
			description:    "Zero classes should return 0",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			normalized := normalizeTsallisEntropy(tc.tsallisEntropy, tc.numClasses, tc.q)

			if normalized < tc.expectedRange[0] || normalized > tc.expectedRange[1] {
				t.Errorf("Normalized entropy %.6f is outside expected range [%.3f, %.3f] for %s",
					normalized, tc.expectedRange[0], tc.expectedRange[1], tc.description)
			}

			// Should always be in [0, 1] range
			if normalized < 0 || normalized > 1 {
				t.Errorf("Normalized entropy %.6f is outside [0,1] range", normalized)
			}

			t.Logf("Test '%s' passed: normalized=%.6f, %s", tc.name, normalized, tc.description)
		})
	}
}

func TestConsensusUncertaintyLevels(t *testing.T) {
	config := DefaultAdvancedEntropyConfig()

	testCases := []struct {
		consensusEntropy    float64
		expectedUncertainty string
	}{
		{0.95, "very_high"},
		{0.85, "very_high"},
		{0.75, "high"},
		{0.65, "high"},
		{0.55, "medium"},
		{0.45, "medium"},
		{0.35, "low"},
		{0.25, "low"},
		{0.15, "very_low"},
		{0.05, "very_low"},
	}

	for _, tc := range testCases {
		result := AdvancedEntropyResult{
			ConsensusEntropy: tc.consensusEntropy,
		}
		
		uncertainty := determineConsensusUncertaintyLevel(result, config)
		
		if uncertainty != tc.expectedUncertainty {
			t.Errorf("Consensus entropy %.2f: expected '%s', got '%s'",
				tc.consensusEntropy, tc.expectedUncertainty, uncertainty)
		}
	}
}

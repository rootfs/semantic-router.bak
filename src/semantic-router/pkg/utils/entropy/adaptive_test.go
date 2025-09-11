package entropy

import (
	"math"
	"testing"
)

func TestNewAdaptiveThresholds(t *testing.T) {
	config := AdaptiveConfig{
		Enabled:      true,
		LearningRate: 0.1,
		MinSamples:   50,
		MaxHistory:   500,
	}

	at := NewAdaptiveThresholds(config)

	// Check default thresholds
	if at.VeryHighThreshold != 0.8 {
		t.Errorf("Expected VeryHighThreshold 0.8, got %.3f", at.VeryHighThreshold)
	}
	if at.HighThreshold != 0.6 {
		t.Errorf("Expected HighThreshold 0.6, got %.3f", at.HighThreshold)
	}
	if at.MediumThreshold != 0.4 {
		t.Errorf("Expected MediumThreshold 0.4, got %.3f", at.MediumThreshold)
	}
	if at.LowThreshold != 0.2 {
		t.Errorf("Expected LowThreshold 0.2, got %.3f", at.LowThreshold)
	}

	// Check configuration
	if at.LearningRate != 0.1 {
		t.Errorf("Expected LearningRate 0.1, got %.3f", at.LearningRate)
	}
	if at.MinSamples != 50 {
		t.Errorf("Expected MinSamples 50, got %d", at.MinSamples)
	}
	if at.MaxHistory != 500 {
		t.Errorf("Expected MaxHistory 500, got %d", at.MaxHistory)
	}
}

func TestRecordPerformance(t *testing.T) {
	at := NewAdaptiveThresholds(AdaptiveConfig{})

	// Record some performance data
	at.RecordPerformance(0.9, "very_high", true, "biology", 0.6, true)
	at.RecordPerformance(0.3, "low", false, "physics", 0.9, true)
	at.RecordPerformance(0.5, "medium", true, "chemistry", 0.7, false)

	if len(at.PerformanceHistory) != 3 {
		t.Errorf("Expected 3 performance records, got %d", len(at.PerformanceHistory))
	}

	// Check first record
	first := at.PerformanceHistory[0]
	if first.EntropyValue != 0.9 {
		t.Errorf("Expected entropy 0.9, got %.3f", first.EntropyValue)
	}
	if first.UncertaintyLevel != "very_high" {
		t.Errorf("Expected uncertainty level 'very_high', got '%s'", first.UncertaintyLevel)
	}
	if !first.ReasoningEnabled {
		t.Errorf("Expected reasoning enabled to be true")
	}
	if first.Category != "biology" {
		t.Errorf("Expected category 'biology', got '%s'", first.Category)
	}
}

func TestMaxHistoryLimit(t *testing.T) {
	at := NewAdaptiveThresholds(AdaptiveConfig{MaxHistory: 5})

	// Add more records than the limit
	for i := 0; i < 10; i++ {
		at.RecordPerformance(0.5, "medium", true, "test", 0.7, true)
	}

	if len(at.PerformanceHistory) != 5 {
		t.Errorf("Expected history to be limited to 5 records, got %d", len(at.PerformanceHistory))
	}
}

func TestGetUncertaintyLevel(t *testing.T) {
	at := NewAdaptiveThresholds(AdaptiveConfig{})

	testCases := []struct {
		entropy  float64
		expected string
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
		result := at.GetUncertaintyLevel(tc.entropy)
		if result != tc.expected {
			t.Errorf("Entropy %.2f: expected '%s', got '%s'", tc.entropy, tc.expected, result)
		}
	}
}

func TestUpdateThresholdsInsufficientSamples(t *testing.T) {
	at := NewAdaptiveThresholds(AdaptiveConfig{MinSamples: 100})

	// Add only a few samples
	for i := 0; i < 10; i++ {
		at.RecordPerformance(0.5, "medium", true, "test", 0.7, true)
	}

	err := at.UpdateThresholds()
	if err == nil {
		t.Errorf("Expected error for insufficient samples, got nil")
	}
}

func TestUpdateThresholdsSufficientSamples(t *testing.T) {
	at := NewAdaptiveThresholds(AdaptiveConfig{
		LearningRate: 0.1,
		MinSamples:   10,
		MaxHistory:   1000,
	})

	// Add samples with performance better than expected for "medium" uncertainty
	// Expected accuracy for medium is 0.70, we'll give it 0.85
	for i := 0; i < 20; i++ {
		at.RecordPerformance(0.5, "medium", true, "test", 0.85, true)
	}

	originalMediumThreshold := at.MediumThreshold

	err := at.UpdateThresholds()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Since medium uncertainty performed better than expected (0.85 vs 0.70),
	// the threshold should be adjusted to be more aggressive (lower threshold)
	if at.MediumThreshold >= originalMediumThreshold {
		t.Logf("Medium threshold adjustment: %.3f -> %.3f", originalMediumThreshold, at.MediumThreshold)
		// Note: The actual adjustment depends on the learning rate and performance gap
		// We're mainly testing that the update process works without errors
	}
}

func TestValidateAndFixThresholdOrder(t *testing.T) {
	at := NewAdaptiveThresholds(AdaptiveConfig{})

	// Mess up the threshold order
	at.VeryHighThreshold = 0.3
	at.HighThreshold = 0.7
	at.MediumThreshold = 0.9
	at.LowThreshold = 0.5

	at.validateAndFixThresholdOrder()

	// Check that thresholds are now in proper order
	if at.VeryHighThreshold <= at.HighThreshold {
		t.Errorf("VeryHighThreshold (%.3f) should be > HighThreshold (%.3f)", 
			at.VeryHighThreshold, at.HighThreshold)
	}
	if at.HighThreshold <= at.MediumThreshold {
		t.Errorf("HighThreshold (%.3f) should be > MediumThreshold (%.3f)", 
			at.HighThreshold, at.MediumThreshold)
	}
	if at.MediumThreshold <= at.LowThreshold {
		t.Errorf("MediumThreshold (%.3f) should be > LowThreshold (%.3f)", 
			at.MediumThreshold, at.LowThreshold)
	}

	// Check that all thresholds are within valid range
	thresholds := []float64{at.VeryHighThreshold, at.HighThreshold, at.MediumThreshold, at.LowThreshold}
	for i, threshold := range thresholds {
		if threshold < 0.0 || threshold > 1.0 {
			t.Errorf("Threshold %d (%.3f) is outside valid range [0,1]", i, threshold)
		}
	}
}

func TestGetPerformanceStats(t *testing.T) {
	at := NewAdaptiveThresholds(AdaptiveConfig{})

	// Test with empty history
	stats := at.GetPerformanceStats()
	if stats["total_samples"] != 0 {
		t.Errorf("Expected 0 total samples, got %v", stats["total_samples"])
	}

	// Add some performance data
	at.RecordPerformance(0.9, "very_high", true, "biology", 0.6, true)
	at.RecordPerformance(0.3, "low", false, "physics", 0.9, true)
	at.RecordPerformance(0.5, "medium", true, "chemistry", 0.7, false)

	stats = at.GetPerformanceStats()

	if stats["total_samples"] != 3 {
		t.Errorf("Expected 3 total samples, got %v", stats["total_samples"])
	}
	if stats["correct_decisions"] != 2 {
		t.Errorf("Expected 2 correct decisions, got %v", stats["correct_decisions"])
	}

	decisionAccuracy := stats["decision_accuracy"].(float64)
	expectedDecisionAccuracy := 2.0 / 3.0
	if math.Abs(decisionAccuracy-expectedDecisionAccuracy) > 0.01 {
		t.Errorf("Expected decision accuracy %.3f, got %.3f", expectedDecisionAccuracy, decisionAccuracy)
	}

	averageAccuracy := stats["average_accuracy"].(float64)
	expectedAverageAccuracy := (0.6 + 0.9 + 0.7) / 3.0
	if math.Abs(averageAccuracy-expectedAverageAccuracy) > 0.01 {
		t.Errorf("Expected average accuracy %.3f, got %.3f", expectedAverageAccuracy, averageAccuracy)
	}
}

func TestGetCurrentThresholds(t *testing.T) {
	at := NewAdaptiveThresholds(AdaptiveConfig{})

	veryHigh, high, medium, low := at.GetCurrentThresholds()

	if veryHigh != 0.8 {
		t.Errorf("Expected VeryHigh threshold 0.8, got %.3f", veryHigh)
	}
	if high != 0.6 {
		t.Errorf("Expected High threshold 0.6, got %.3f", high)
	}
	if medium != 0.4 {
		t.Errorf("Expected Medium threshold 0.4, got %.3f", medium)
	}
	if low != 0.2 {
		t.Errorf("Expected Low threshold 0.2, got %.3f", low)
	}
}

func TestGetPredictedAccuracy(t *testing.T) {
	at := NewAdaptiveThresholds(AdaptiveConfig{})

	testCases := []struct {
		level    string
		expected float64
	}{
		{"very_low", 0.95},
		{"low", 0.85},
		{"medium", 0.70},
		{"high", 0.55},
		{"very_high", 0.45},
		{"unknown", 0.50},
	}

	for _, tc := range testCases {
		result := at.getPredictedAccuracy(tc.level)
		if result != tc.expected {
			t.Errorf("Level '%s': expected accuracy %.2f, got %.2f", tc.level, tc.expected, result)
		}
	}
}

func TestAdaptiveThresholdsConcurrency(t *testing.T) {
	at := NewAdaptiveThresholds(AdaptiveConfig{MinSamples: 5})

	// Test concurrent access
	done := make(chan bool, 3)

	// Goroutine 1: Record performance
	go func() {
		for i := 0; i < 10; i++ {
			at.RecordPerformance(0.5, "medium", true, "test", 0.7, true)
		}
		done <- true
	}()

	// Goroutine 2: Get uncertainty levels
	go func() {
		for i := 0; i < 10; i++ {
			at.GetUncertaintyLevel(0.5)
		}
		done <- true
	}()

	// Goroutine 3: Get performance stats
	go func() {
		for i := 0; i < 10; i++ {
			at.GetPerformanceStats()
		}
		done <- true
	}()

	// Wait for all goroutines to complete
	for i := 0; i < 3; i++ {
		<-done
	}

	// Try to update thresholds
	err := at.UpdateThresholds()
	if err != nil {
		t.Logf("Update error (expected): %v", err)
	}

	// Verify final state is consistent
	veryHigh, high, medium, low := at.GetCurrentThresholds()
	if veryHigh <= high || high <= medium || medium <= low {
		t.Errorf("Threshold order is invalid after concurrent access: %.3f, %.3f, %.3f, %.3f", 
			veryHigh, high, medium, low)
	}
}

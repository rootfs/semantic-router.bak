// Package entropy - performance benchmarks for entropy-based routing
//
// This file contains comprehensive benchmarks for all entropy-related operations
// to ensure performance regressions are detected early in the CI pipeline.

package entropy

import (
	"math/rand"
	"testing"
	"time"
)

// BenchmarkCalculateEntropy benchmarks Shannon entropy calculation
func BenchmarkCalculateEntropy(b *testing.B) {
	testCases := []struct {
		name  string
		probs []float32
	}{
		{"Small_4_categories", []float32{0.4, 0.3, 0.2, 0.1}},
		{"Medium_10_categories", generateUniformProbs(10)},
		{"Large_100_categories", generateUniformProbs(100)},
		{"Very_large_1000_categories", generateUniformProbs(1000)},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = CalculateEntropy(tc.probs)
			}
		})
	}
}

// BenchmarkCalculateNormalizedEntropy benchmarks normalized entropy calculation
func BenchmarkCalculateNormalizedEntropy(b *testing.B) {
	testCases := []struct {
		name  string
		probs []float32
	}{
		{"Small_4_categories", []float32{0.4, 0.3, 0.2, 0.1}},
		{"Medium_10_categories", generateUniformProbs(10)},
		{"Large_100_categories", generateUniformProbs(100)},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = CalculateNormalizedEntropy(tc.probs)
			}
		})
	}
}

// BenchmarkAnalyzeEntropy benchmarks entropy analysis
func BenchmarkAnalyzeEntropy(b *testing.B) {
	testCases := []struct {
		name  string
		probs []float32
	}{
		{"Small_4_categories", []float32{0.4, 0.3, 0.2, 0.1}},
		{"Medium_10_categories", generateUniformProbs(10)},
		{"Large_100_categories", generateUniformProbs(100)},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = AnalyzeEntropy(tc.probs)
			}
		})
	}
}

// BenchmarkGetTopCategories benchmarks top-N category selection with heap
func BenchmarkGetTopCategories(b *testing.B) {
	testCases := []struct {
		name       string
		probs      []float32
		categories []string
		topN       int
	}{
		{"Small_4_categories_top3", generateRandomProbs(4), generateCategoryNames(4), 3},
		{"Medium_10_categories_top5", generateRandomProbs(10), generateCategoryNames(10), 5},
		{"Large_100_categories_top10", generateRandomProbs(100), generateCategoryNames(100), 10},
		{"Very_large_1000_categories_top10", generateRandomProbs(1000), generateCategoryNames(1000), 10},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = getTopCategories(tc.probs, tc.categories, tc.topN)
			}
		})
	}
}

// BenchmarkMakeEntropyBasedReasoningDecision benchmarks the main decision function
func BenchmarkMakeEntropyBasedReasoningDecision(b *testing.B) {
	categoryReasoningMap := map[string]bool{
		"physics":   false,
		"biology":   true,
		"chemistry": true,
		"other":     false,
	}

	testCases := []struct {
		name       string
		probs      []float32
		categories []string
	}{
		{"Small_4_categories", []float32{0.4, 0.3, 0.2, 0.1}, []string{"physics", "biology", "chemistry", "other"}},
		{"Medium_10_categories", generateRandomProbs(10), generateCategoryNames(10)},
		{"Large_100_categories", generateRandomProbs(100), generateCategoryNames(100)},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MakeEntropyBasedReasoningDecision(
					tc.probs,
					tc.categories,
					categoryReasoningMap,
					0.7,
				)
			}
		})
	}
}

// BenchmarkCalculateRenyiEntropy benchmarks Rényi entropy calculation
func BenchmarkCalculateRenyiEntropy(b *testing.B) {
	probs := generateRandomProbs(10)
	alphaValues := []float64{0.5, 1.0, 2.0, 5.0}

	for _, alpha := range alphaValues {
		b.Run("alpha_"+formatFloat(alpha), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = CalculateRenyiEntropy(probs, alpha)
			}
		})
	}
}

// BenchmarkCalculateTsallisEntropy benchmarks Tsallis entropy calculation
func BenchmarkCalculateTsallisEntropy(b *testing.B) {
	probs := generateRandomProbs(10)
	qValues := []float64{0.5, 1.0, 2.0, 5.0}

	for _, q := range qValues {
		b.Run("q_"+formatFloat(q), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = CalculateTsallisEntropy(probs, q)
			}
		})
	}
}

// BenchmarkCalculateAdvancedEntropy benchmarks advanced entropy calculation
func BenchmarkCalculateAdvancedEntropy(b *testing.B) {
	config := DefaultAdvancedEntropyConfig()
	testCases := []struct {
		name  string
		probs []float32
	}{
		{"Small_4_categories", generateRandomProbs(4)},
		{"Medium_10_categories", generateRandomProbs(10)},
		{"Large_100_categories", generateRandomProbs(100)},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = CalculateAdvancedEntropy(tc.probs, config)
			}
		})
	}
}

// BenchmarkMakeAdvancedEntropyBasedReasoningDecision benchmarks advanced reasoning
func BenchmarkMakeAdvancedEntropyBasedReasoningDecision(b *testing.B) {
	config := DefaultAdvancedEntropyConfig()
	categoryReasoningMap := map[string]bool{
		"physics":   false,
		"biology":   true,
		"chemistry": true,
		"other":     false,
	}

	testCases := []struct {
		name       string
		probs      []float32
		categories []string
	}{
		{"Small_4_categories", generateRandomProbs(4), []string{"physics", "biology", "chemistry", "other"}},
		{"Medium_10_categories", generateRandomProbs(10), generateCategoryNames(10)},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MakeAdvancedEntropyBasedReasoningDecision(
					tc.probs,
					tc.categories,
					categoryReasoningMap,
					0.7,
					config,
				)
			}
		})
	}
}

// BenchmarkAdaptiveThresholds benchmarks adaptive threshold operations
func BenchmarkAdaptiveThresholds(b *testing.B) {
	config := AdaptiveConfig{
		LearningRate: 0.05,
		MinSamples:   10, // Low for benchmarking
		MaxHistory:   100,
	}
	
	at := NewAdaptiveThresholds(config)
	
	// Pre-populate with some data
	for i := 0; i < 50; i++ {
		at.RecordPerformance(
			rand.Float64(),
			"medium",
			rand.Float64() > 0.5,
			"test",
			0.7+rand.Float64()*0.3,
			rand.Float64() > 0.3,
		)
	}

	b.Run("RecordPerformance", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			at.RecordPerformance(
				rand.Float64(),
				"medium",
				true,
				"test",
				0.8,
				true,
			)
		}
	})

	b.Run("GetUncertaintyLevel", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = at.GetUncertaintyLevel(rand.Float64())
		}
	})

	b.Run("UpdateThresholds", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = at.UpdateThresholds()
		}
	})
}

// BenchmarkConfigValidation benchmarks configuration validation
func BenchmarkConfigValidation(b *testing.B) {
	config := DefaultEntropyConfig()
	
	b.Run("ValidateConfig_Valid", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = ValidateConfig(config, false)
		}
	})
	
	// Create invalid config for auto-fix benchmarking
	invalidConfig := config
	invalidConfig.Thresholds.VeryHighThreshold = 0.3
	invalidConfig.Thresholds.HighThreshold = 0.7
	
	b.Run("ValidateConfig_AutoFix", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			testConfig := invalidConfig // Copy for each iteration
			_ = ValidateConfig(testConfig, true)
		}
	})
}

// BenchmarkMonitoring benchmarks monitoring operations
func BenchmarkMonitoring(b *testing.B) {
	config := LoggingConfig{
		EnableDebugLogs:     false,
		EnableDecisionTrace: true,
		LogLevel:           "info",
		LogSampleRate:      1.0,
	}
	
	monitor := NewEntropyMonitor(config, 1000)
	
	// Create sample trace
	trace := DecisionTrace{
		RequestID:           "test-request",
		Timestamp:          time.Now(),
		PredictedClass:     0,
		ClassConfidence:    0.8,
		CategoryName:       "physics",
		ProbabilityDist:    []float64{0.8, 0.1, 0.05, 0.05},
		ShannonEntropy:     0.7,
		NormalizedEntropy:  0.35,
		UncertaintyLevel:   "low",
		UseReasoning:       false,
		ReasoningConfidence: 0.8,
		DecisionReason:     "low_uncertainty_trust_classification",
		TotalLatencyMs:     15.5,
		ClassificationModel: "linear_bert",
		ProbSumValid:       true,
		HasNegativeProbabilities: false,
	}

	b.Run("TraceDecision", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			monitor.TraceDecision(trace)
		}
	})

	// Pre-populate monitor with traces
	for i := 0; i < 100; i++ {
		monitor.TraceDecision(trace)
	}

	b.Run("GetRecentTraces", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = monitor.GetRecentTraces(10)
		}
	})

	b.Run("GetPerformanceStats", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = monitor.GetPerformanceStats()
		}
	})
}

// BenchmarkMemoryUsage benchmarks memory usage for large probability arrays
func BenchmarkMemoryUsage(b *testing.B) {
	sizes := []int{10, 100, 1000, 10000}
	
	for _, size := range sizes {
		b.Run("size_"+formatInt(size), func(b *testing.B) {
			probs := generateRandomProbs(size)
			categories := generateCategoryNames(size)
			
			b.ResetTimer()
			b.ReportAllocs()
			
			for i := 0; i < b.N; i++ {
				_ = CalculateEntropy(probs)
				_ = getTopCategories(probs, categories, 10)
			}
		})
	}
}

// Helper functions for benchmark data generation

func generateUniformProbs(n int) []float32 {
	probs := make([]float32, n)
	prob := float32(1.0) / float32(n)
	for i := range probs {
		probs[i] = prob
	}
	return probs
}

func generateRandomProbs(n int) []float32 {
	probs := make([]float32, n)
	sum := float32(0.0)
	
	// Generate random values
	for i := range probs {
		probs[i] = rand.Float32()
		sum += probs[i]
	}
	
	// Normalize to sum to 1.0
	for i := range probs {
		probs[i] /= sum
	}
	
	return probs
}

func generateCategoryNames(n int) []string {
	names := make([]string, n)
	for i := range names {
		names[i] = "category_" + formatInt(i)
	}
	return names
}

func formatFloat(f float64) string {
	if f == float64(int(f)) {
		return formatInt(int(f))
	}
	return "0_5" // Simple formatting for 0.5
}

func formatInt(i int) string {
	// Simple integer to string conversion for benchmarking
	if i == 0 {
		return "0"
	}
	if i == 1 {
		return "1"
	}
	if i == 2 {
		return "2"
	}
	if i == 4 {
		return "4"
	}
	if i == 5 {
		return "5"
	}
	if i == 10 {
		return "10"
	}
	if i == 100 {
		return "100"
	}
	if i == 1000 {
		return "1000"
	}
	if i == 10000 {
		return "10000"
	}
	return "other"
}

// BenchmarkComparison compares different entropy calculation approaches
func BenchmarkComparison(b *testing.B) {
	probs := generateRandomProbs(100)
	
	b.Run("Shannon_only", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = CalculateEntropy(probs)
		}
	})
	
	b.Run("Shannon_plus_normalized", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = CalculateEntropy(probs)
			_ = CalculateNormalizedEntropy(probs)
		}
	})
	
	b.Run("All_advanced_entropy", func(b *testing.B) {
		config := DefaultAdvancedEntropyConfig()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = CalculateAdvancedEntropy(probs, config)
		}
	})
}

// Performance regression test - fails if performance degrades significantly
func TestPerformanceRegression(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance regression test in short mode")
	}

	// Baseline performance expectations (adjust based on your requirements)
	performanceTests := []struct {
		name            string
		maxNsPerOp      int64 // Maximum nanoseconds per operation
		testFunc        func() interface{}
		description     string
	}{
		{
			name:       "Shannon_entropy_small",
			maxNsPerOp: 1000, // 1 microsecond
			testFunc: func() interface{} {
				return CalculateEntropy([]float32{0.4, 0.3, 0.2, 0.1})
			},
			description: "Shannon entropy for 4 categories should be very fast",
		},
		{
			name:       "Top_categories_medium",
			maxNsPerOp: 15000, // 15 microseconds (more realistic for test environment)
			testFunc: func() interface{} {
				probs := generateRandomProbs(100)
				categories := generateCategoryNames(100)
				return getTopCategories(probs, categories, 10)
			},
			description: "Top-10 selection from 100 categories should use efficient heap",
		},
		{
			name:       "Reasoning_decision_small",
			maxNsPerOp: 10000, // 10 microseconds
			testFunc: func() interface{} {
				return MakeEntropyBasedReasoningDecision(
					[]float32{0.4, 0.3, 0.2, 0.1},
					[]string{"physics", "biology", "chemistry", "other"},
					map[string]bool{"physics": false, "biology": true, "chemistry": true, "other": false},
					0.7,
				)
			},
			description: "Complete reasoning decision should be fast for small inputs",
		},
	}

	for _, test := range performanceTests {
		t.Run(test.name, func(t *testing.T) {
			// Warm up
			for i := 0; i < 100; i++ {
				test.testFunc()
			}

			// Measure performance
			start := time.Now()
			iterations := 1000
			for i := 0; i < iterations; i++ {
				test.testFunc()
			}
			elapsed := time.Since(start)

			avgNsPerOp := elapsed.Nanoseconds() / int64(iterations)

			if avgNsPerOp > test.maxNsPerOp {
				t.Errorf("Performance regression detected for %s: %d ns/op > %d ns/op (%.1fx slower than expected)\n%s",
					test.name, avgNsPerOp, test.maxNsPerOp, float64(avgNsPerOp)/float64(test.maxNsPerOp), test.description)
			} else {
				t.Logf("Performance test passed for %s: %d ns/op <= %d ns/op", test.name, avgNsPerOp, test.maxNsPerOp)
			}
		})
	}
}

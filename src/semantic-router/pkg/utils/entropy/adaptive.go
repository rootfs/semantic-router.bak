// Package entropy - adaptive threshold management for entropy-based routing
//
// This module implements adaptive threshold adjustment based on historical accuracy
// and performance metrics. It allows the entropy-based routing system to learn
// and optimize its decision boundaries over time.

package entropy

import (
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"
)

// AdaptiveThresholds manages dynamic threshold adjustment for entropy-based decisions
type AdaptiveThresholds struct {
	// Current thresholds for uncertainty levels
	VeryHighThreshold float64 `json:"very_high_threshold"` // >= this value = very high uncertainty
	HighThreshold     float64 `json:"high_threshold"`      // >= this value = high uncertainty  
	MediumThreshold   float64 `json:"medium_threshold"`    // >= this value = medium uncertainty
	LowThreshold      float64 `json:"low_threshold"`       // >= this value = low uncertainty
	// < low_threshold = very low uncertainty

	// Historical performance tracking
	PerformanceHistory []ThresholdPerformance `json:"performance_history"`
	
	// Configuration
	LearningRate    float64   `json:"learning_rate"`     // Rate of threshold adjustment (0.01-0.1)
	MinSamples      int       `json:"min_samples"`       // Minimum samples before adjustment
	MaxHistory      int       `json:"max_history"`       // Maximum history entries to keep
	LastUpdated     time.Time `json:"last_updated"`      // When thresholds were last updated
	
	// Thread safety
	mutex sync.RWMutex
}

// ThresholdPerformance tracks performance metrics for threshold evaluation
type ThresholdPerformance struct {
	Timestamp         time.Time `json:"timestamp"`
	UncertaintyLevel  string    `json:"uncertainty_level"`
	ReasoningEnabled  bool      `json:"reasoning_enabled"`
	ActualAccuracy    float64   `json:"actual_accuracy"`     // Measured accuracy for this decision
	PredictedAccuracy float64   `json:"predicted_accuracy"`  // Expected accuracy based on uncertainty
	EntropyValue      float64   `json:"entropy_value"`
	Category          string    `json:"category"`
	CorrectDecision   bool      `json:"correct_decision"`    // Was the reasoning decision optimal?
}

// AdaptiveConfig contains configuration for adaptive threshold management
type AdaptiveConfig struct {
	Enabled         bool    `json:"enabled"`           // Enable adaptive thresholds
	LearningRate    float64 `json:"learning_rate"`     // Rate of adjustment (default: 0.05)
	MinSamples      int     `json:"min_samples"`       // Min samples before adjustment (default: 100)
	MaxHistory      int     `json:"max_history"`       // Max history entries (default: 1000)
	UpdateInterval  string  `json:"update_interval"`   // How often to update (default: "1h")
	PerformanceFile string  `json:"performance_file"`  // File to persist performance data
}

// NewAdaptiveThresholds creates a new adaptive threshold manager with default values
func NewAdaptiveThresholds(config AdaptiveConfig) *AdaptiveThresholds {
	if config.LearningRate == 0 {
		config.LearningRate = 0.05
	}
	if config.MinSamples == 0 {
		config.MinSamples = 100
	}
	if config.MaxHistory == 0 {
		config.MaxHistory = 1000
	}

	return &AdaptiveThresholds{
		// Default thresholds (same as static implementation)
		VeryHighThreshold: 0.8,
		HighThreshold:     0.6,
		MediumThreshold:   0.4,
		LowThreshold:      0.2,
		
		PerformanceHistory: make([]ThresholdPerformance, 0),
		LearningRate:       config.LearningRate,
		MinSamples:         config.MinSamples,
		MaxHistory:         config.MaxHistory,
		LastUpdated:        time.Now(),
	}
}

// RecordPerformance records the performance of an entropy-based decision
func (at *AdaptiveThresholds) RecordPerformance(
	entropyValue float64,
	uncertaintyLevel string,
	reasoningEnabled bool,
	category string,
	actualAccuracy float64,
	correctDecision bool,
) {
	at.mutex.Lock()
	defer at.mutex.Unlock()

	// Calculate predicted accuracy based on uncertainty level
	predictedAccuracy := at.getPredictedAccuracy(uncertaintyLevel)

	performance := ThresholdPerformance{
		Timestamp:         time.Now(),
		UncertaintyLevel:  uncertaintyLevel,
		ReasoningEnabled:  reasoningEnabled,
		ActualAccuracy:    actualAccuracy,
		PredictedAccuracy: predictedAccuracy,
		EntropyValue:      entropyValue,
		Category:          category,
		CorrectDecision:   correctDecision,
	}

	at.PerformanceHistory = append(at.PerformanceHistory, performance)

	// Trim history if it exceeds maximum
	if len(at.PerformanceHistory) > at.MaxHistory {
		at.PerformanceHistory = at.PerformanceHistory[len(at.PerformanceHistory)-at.MaxHistory:]
	}
}

// UpdateThresholds analyzes performance history and adjusts thresholds
func (at *AdaptiveThresholds) UpdateThresholds() error {
	at.mutex.Lock()
	defer at.mutex.Unlock()

	if len(at.PerformanceHistory) < at.MinSamples {
		return fmt.Errorf("insufficient samples for threshold update: %d < %d", 
			len(at.PerformanceHistory), at.MinSamples)
	}

	// Analyze performance by uncertainty level
	levelPerformance := at.analyzePerformanceByLevel()

	// Adjust thresholds based on performance analysis
	oldThresholds := AdaptiveThresholds{
		VeryHighThreshold: at.VeryHighThreshold,
		HighThreshold:     at.HighThreshold,
		MediumThreshold:   at.MediumThreshold,
		LowThreshold:      at.LowThreshold,
	}

	at.adjustThresholdsBasedOnPerformance(levelPerformance)

	// Ensure thresholds remain in logical order
	at.validateAndFixThresholdOrder()

	at.LastUpdated = time.Now()

	// Log threshold changes
	fmt.Printf("Adaptive thresholds updated:\n")
	fmt.Printf("  Very High: %.3f -> %.3f\n", oldThresholds.VeryHighThreshold, at.VeryHighThreshold)
	fmt.Printf("  High: %.3f -> %.3f\n", oldThresholds.HighThreshold, at.HighThreshold)
	fmt.Printf("  Medium: %.3f -> %.3f\n", oldThresholds.MediumThreshold, at.MediumThreshold)
	fmt.Printf("  Low: %.3f -> %.3f\n", oldThresholds.LowThreshold, at.LowThreshold)

	return nil
}

// GetCurrentThresholds returns the current threshold values (thread-safe)
func (at *AdaptiveThresholds) GetCurrentThresholds() (float64, float64, float64, float64) {
	at.mutex.RLock()
	defer at.mutex.RUnlock()
	
	return at.VeryHighThreshold, at.HighThreshold, at.MediumThreshold, at.LowThreshold
}

// GetUncertaintyLevel determines uncertainty level using current adaptive thresholds
func (at *AdaptiveThresholds) GetUncertaintyLevel(normalizedEntropy float64) string {
	at.mutex.RLock()
	defer at.mutex.RUnlock()

	if normalizedEntropy >= at.VeryHighThreshold {
		return "very_high"
	} else if normalizedEntropy >= at.HighThreshold {
		return "high"
	} else if normalizedEntropy >= at.MediumThreshold {
		return "medium"
	} else if normalizedEntropy >= at.LowThreshold {
		return "low"
	} else {
		return "very_low"
	}
}

// GetPerformanceStats returns performance statistics for monitoring
func (at *AdaptiveThresholds) GetPerformanceStats() map[string]interface{} {
	at.mutex.RLock()
	defer at.mutex.RUnlock()

	if len(at.PerformanceHistory) == 0 {
		return map[string]interface{}{
			"total_samples": 0,
			"last_updated": at.LastUpdated,
		}
	}

	// Calculate overall statistics
	totalSamples := len(at.PerformanceHistory)
	correctDecisions := 0
	totalAccuracy := 0.0
	levelCounts := make(map[string]int)

	for _, perf := range at.PerformanceHistory {
		if perf.CorrectDecision {
			correctDecisions++
		}
		totalAccuracy += perf.ActualAccuracy
		levelCounts[perf.UncertaintyLevel]++
	}

	return map[string]interface{}{
		"total_samples":      totalSamples,
		"correct_decisions":  correctDecisions,
		"decision_accuracy":  float64(correctDecisions) / float64(totalSamples),
		"average_accuracy":   totalAccuracy / float64(totalSamples),
		"level_distribution": levelCounts,
		"last_updated":       at.LastUpdated,
		"current_thresholds": map[string]float64{
			"very_high": at.VeryHighThreshold,
			"high":      at.HighThreshold,
			"medium":    at.MediumThreshold,
			"low":       at.LowThreshold,
		},
	}
}

// SaveToFile saves the adaptive thresholds and performance history to a file
func (at *AdaptiveThresholds) SaveToFile(filename string) error {
	at.mutex.RLock()
	defer at.mutex.RUnlock()

	_, err := json.MarshalIndent(at, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal adaptive thresholds: %w", err)
	}

	return fmt.Errorf("file operations not implemented in this environment")
}

// LoadFromFile loads adaptive thresholds and performance history from a file
func LoadAdaptiveThresholdsFromFile(filename string) (*AdaptiveThresholds, error) {
	return nil, fmt.Errorf("file operations not implemented in this environment")
}

// Private helper methods

func (at *AdaptiveThresholds) getPredictedAccuracy(uncertaintyLevel string) float64 {
	// Expected accuracy based on uncertainty level (baseline expectations)
	switch uncertaintyLevel {
	case "very_low":
		return 0.95 // Very confident predictions should be highly accurate
	case "low":
		return 0.85 // Confident predictions should be quite accurate
	case "medium":
		return 0.70 // Medium confidence should be moderately accurate
	case "high":
		return 0.55 // High uncertainty should be slightly better than random
	case "very_high":
		return 0.45 // Very high uncertainty may be worse than random
	default:
		return 0.50 // Default to random chance
	}
}

func (at *AdaptiveThresholds) analyzePerformanceByLevel() map[string]LevelPerformanceAnalysis {
	analysis := make(map[string]LevelPerformanceAnalysis)

	// Group performance data by uncertainty level
	levelData := make(map[string][]ThresholdPerformance)
	for _, perf := range at.PerformanceHistory {
		levelData[perf.UncertaintyLevel] = append(levelData[perf.UncertaintyLevel], perf)
	}

	// Analyze each level
	for level, levelPerfs := range levelData {
		if len(levelPerfs) == 0 {
			continue
		}

		totalAccuracy := 0.0
		correctDecisions := 0
		avgEntropy := 0.0

		for _, perf := range levelPerfs {
			totalAccuracy += perf.ActualAccuracy
			if perf.CorrectDecision {
				correctDecisions++
			}
			avgEntropy += perf.EntropyValue
		}

		analysis[level] = LevelPerformanceAnalysis{
			SampleCount:     len(levelPerfs),
			AverageAccuracy: totalAccuracy / float64(len(levelPerfs)),
			DecisionAccuracy: float64(correctDecisions) / float64(len(levelPerfs)),
			AverageEntropy:  avgEntropy / float64(len(levelPerfs)),
			PerformanceGap:  (totalAccuracy / float64(len(levelPerfs))) - at.getPredictedAccuracy(level),
		}
	}

	return analysis
}

func (at *AdaptiveThresholds) adjustThresholdsBasedOnPerformance(analysis map[string]LevelPerformanceAnalysis) {
	// Adjust thresholds based on performance gaps
	// If a level is performing better than expected, we can be more aggressive with that threshold
	// If a level is performing worse than expected, we should be more conservative

	for level, perf := range analysis {
		if perf.SampleCount < 10 { // Need minimum samples for reliable adjustment
			continue
		}

		adjustment := at.LearningRate * perf.PerformanceGap

		switch level {
		case "very_high":
			// If very high uncertainty is performing better than expected, we can lower the threshold
			// (be more aggressive about classifying things as very high uncertainty)
			at.VeryHighThreshold -= adjustment
		case "high":
			at.HighThreshold -= adjustment
		case "medium":
			at.MediumThreshold -= adjustment
		case "low":
			at.LowThreshold -= adjustment
		}
	}
}

func (at *AdaptiveThresholds) validateAndFixThresholdOrder() {
	// Ensure thresholds are in descending order: very_high > high > medium > low
	// and all are within reasonable bounds [0, 1]

	// Clamp to valid range
	at.VeryHighThreshold = math.Max(0.0, math.Min(1.0, at.VeryHighThreshold))
	at.HighThreshold = math.Max(0.0, math.Min(1.0, at.HighThreshold))
	at.MediumThreshold = math.Max(0.0, math.Min(1.0, at.MediumThreshold))
	at.LowThreshold = math.Max(0.0, math.Min(1.0, at.LowThreshold))

	// Ensure proper ordering with minimum gaps
	minGap := 0.05

	// Fix ordering from bottom up to avoid cascading issues
	if at.MediumThreshold <= at.LowThreshold {
		at.MediumThreshold = at.LowThreshold + minGap
	}
	if at.HighThreshold <= at.MediumThreshold {
		at.HighThreshold = at.MediumThreshold + minGap
	}
	if at.VeryHighThreshold <= at.HighThreshold {
		at.VeryHighThreshold = at.HighThreshold + minGap
	}

	// Re-clamp after adjustments
	at.VeryHighThreshold = math.Min(1.0, at.VeryHighThreshold)
	at.HighThreshold = math.Min(0.95, at.HighThreshold)
	at.MediumThreshold = math.Min(0.90, at.MediumThreshold)
	at.LowThreshold = math.Min(0.85, at.LowThreshold)
}

// LevelPerformanceAnalysis contains analysis results for a specific uncertainty level
type LevelPerformanceAnalysis struct {
	SampleCount      int     `json:"sample_count"`
	AverageAccuracy  float64 `json:"average_accuracy"`
	DecisionAccuracy float64 `json:"decision_accuracy"`  // Fraction of correct reasoning decisions
	AverageEntropy   float64 `json:"average_entropy"`
	PerformanceGap   float64 `json:"performance_gap"`    // Actual - Expected accuracy
}

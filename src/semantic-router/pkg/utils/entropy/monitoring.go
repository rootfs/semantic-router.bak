// Package entropy - monitoring, logging, and tracing for entropy-based routing decisions
//
// This module provides comprehensive monitoring capabilities for entropy-based
// classification systems, including detailed decision path tracing, performance
// monitoring, and structured logging for debugging and analysis.

package entropy

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"
)

// DecisionTrace contains detailed information about an entropy-based decision
type DecisionTrace struct {
	// Request identification
	RequestID    string    `json:"request_id"`
	Timestamp    time.Time `json:"timestamp"`
	InputText    string    `json:"input_text,omitempty"` // Omitted in production for privacy
	
	// Classification results
	PredictedClass    int       `json:"predicted_class"`
	ClassConfidence   float64   `json:"class_confidence"`
	CategoryName      string    `json:"category_name"`
	ProbabilityDist   []float64 `json:"probability_distribution"`
	
	// Entropy analysis
	ShannonEntropy       float64 `json:"shannon_entropy"`
	NormalizedEntropy    float64 `json:"normalized_entropy"`
	UncertaintyLevel     string  `json:"uncertainty_level"`
	EntropyCalculationMs float64 `json:"entropy_calculation_ms"`
	
	// Advanced entropy (if enabled)
	RenyiEntropy         *float64 `json:"renyi_entropy,omitempty"`
	TsallisEntropy       *float64 `json:"tsallis_entropy,omitempty"`
	ConsensusEntropy     *float64 `json:"consensus_entropy,omitempty"`
	EntropyDivergence    *float64 `json:"entropy_divergence,omitempty"`
	
	// Reasoning decision
	UseReasoning         bool    `json:"use_reasoning"`
	ReasoningConfidence  float64 `json:"reasoning_confidence"`
	DecisionReason       string  `json:"decision_reason"`
	FallbackStrategy     string  `json:"fallback_strategy"`
	DecisionLatencyMs    float64 `json:"decision_latency_ms"`
	
	// Top categories
	TopCategories []CategoryTrace `json:"top_categories"`
	
	// Performance metrics
	TotalLatencyMs       float64 `json:"total_latency_ms"`
	ClassificationModel  string  `json:"classification_model"` // "linear_bert", "modern_bert"
	
	// Quality indicators
	ProbSumValid         bool    `json:"prob_sum_valid"`
	HasNegativeProbabilities bool `json:"has_negative_probabilities"`
	
	// Adaptive thresholds (if enabled)
	AdaptiveThresholds   *AdaptiveThresholdTrace `json:"adaptive_thresholds,omitempty"`
}

// CategoryTrace contains information about a category in the decision
type CategoryTrace struct {
	Name        string  `json:"name"`
	Probability float64 `json:"probability"`
	Index       int     `json:"index"`
	Reasoning   bool    `json:"reasoning_preference"`
}

// AdaptiveThresholdTrace contains adaptive threshold information
type AdaptiveThresholdTrace struct {
	VeryHighThreshold float64 `json:"very_high_threshold"`
	HighThreshold     float64 `json:"high_threshold"`
	MediumThreshold   float64 `json:"medium_threshold"`
	LowThreshold      float64 `json:"low_threshold"`
	LastUpdated       time.Time `json:"last_updated"`
	SampleCount       int     `json:"sample_count"`
}

// PerformanceStats contains aggregated performance statistics
type PerformanceStats struct {
	// Time window
	WindowStart time.Time `json:"window_start"`
	WindowEnd   time.Time `json:"window_end"`
	
	// Request counts
	TotalRequests        int64 `json:"total_requests"`
	ReasoningEnabled     int64 `json:"reasoning_enabled"`
	ReasoningDisabled    int64 `json:"reasoning_disabled"`
	
	// Latency statistics
	AvgLatencyMs         float64 `json:"avg_latency_ms"`
	P50LatencyMs         float64 `json:"p50_latency_ms"`
	P95LatencyMs         float64 `json:"p95_latency_ms"`
	P99LatencyMs         float64 `json:"p99_latency_ms"`
	MaxLatencyMs         float64 `json:"max_latency_ms"`
	
	// Entropy distribution
	AvgEntropy           float64 `json:"avg_entropy"`
	EntropyDistribution  map[string]int64 `json:"entropy_distribution"` // uncertainty_level -> count
	
	// Classification quality
	AvgConfidence        float64 `json:"avg_confidence"`
	InvalidProbDists     int64   `json:"invalid_prob_distributions"`
	
	// Model usage
	ModelUsage           map[string]int64 `json:"model_usage"` // model_name -> count
	
	// Error rates
	ClassificationErrors int64 `json:"classification_errors"`
	EntropyErrors        int64 `json:"entropy_errors"`
}

// EntropyMonitor provides monitoring and tracing capabilities
type EntropyMonitor struct {
	config        LoggingConfig
	traces        []DecisionTrace
	tracesMutex   sync.RWMutex
	stats         PerformanceStats
	statsMutex    sync.RWMutex
	latencies     []float64
	latenciesMutex sync.Mutex
	maxTraces     int
	enabled       bool
}

// NewEntropyMonitor creates a new entropy monitoring instance
func NewEntropyMonitor(config LoggingConfig, maxTraces int) *EntropyMonitor {
	return &EntropyMonitor{
		config:    config,
		traces:    make([]DecisionTrace, 0),
		maxTraces: maxTraces,
		enabled:   config.EnableDecisionTrace,
		stats: PerformanceStats{
			WindowStart:         time.Now(),
			EntropyDistribution: make(map[string]int64),
			ModelUsage:         make(map[string]int64),
		},
		latencies: make([]float64, 0),
	}
}

// TraceDecision records a complete decision trace
func (em *EntropyMonitor) TraceDecision(trace DecisionTrace) {
	if !em.enabled {
		return
	}

	// Sample based on configured rate
	if em.config.LogSampleRate < 1.0 {
		// Simple sampling based on timestamp
		timestamp := time.Now().UnixNano()
		sampleValue := float64(timestamp%1000000) / 1000000.0
		if sampleValue > em.config.LogSampleRate {
			return
		}
	}

	em.tracesMutex.Lock()
	defer em.tracesMutex.Unlock()

	// Add trace
	em.traces = append(em.traces, trace)

	// Trim traces if exceeding limit
	if len(em.traces) > em.maxTraces {
		em.traces = em.traces[len(em.traces)-em.maxTraces:]
	}

	// Update statistics
	em.updateStats(trace)

	// Log decision if debug logging is enabled
	if em.config.EnableDebugLogs {
		em.logDecision(trace)
	}
}

// GetRecentTraces returns recent decision traces
func (em *EntropyMonitor) GetRecentTraces(limit int) []DecisionTrace {
	if !em.enabled {
		return []DecisionTrace{}
	}

	em.tracesMutex.RLock()
	defer em.tracesMutex.RUnlock()

	if limit <= 0 || limit > len(em.traces) {
		limit = len(em.traces)
	}

	if limit == 0 {
		return []DecisionTrace{}
	}

	// Return most recent traces
	start := len(em.traces) - limit
	result := make([]DecisionTrace, limit)
	copy(result, em.traces[start:])
	
	return result
}

// GetPerformanceStats returns current performance statistics
func (em *EntropyMonitor) GetPerformanceStats() PerformanceStats {
	em.statsMutex.RLock()
	defer em.statsMutex.RUnlock()

	// Calculate latency percentiles
	em.latenciesMutex.Lock()
	stats := em.stats
	stats.WindowEnd = time.Now()
	
	if len(em.latencies) > 0 {
		// Simple percentile calculation (for production, use a proper percentile library)
		latenciesCopy := make([]float64, len(em.latencies))
		copy(latenciesCopy, em.latencies)
		
		// Sort latencies for percentile calculation
		for i := 0; i < len(latenciesCopy)-1; i++ {
			for j := i + 1; j < len(latenciesCopy); j++ {
				if latenciesCopy[i] > latenciesCopy[j] {
					latenciesCopy[i], latenciesCopy[j] = latenciesCopy[j], latenciesCopy[i]
				}
			}
		}
		
		stats.P50LatencyMs = latenciesCopy[len(latenciesCopy)/2]
		stats.P95LatencyMs = latenciesCopy[int(float64(len(latenciesCopy))*0.95)]
		stats.P99LatencyMs = latenciesCopy[int(float64(len(latenciesCopy))*0.99)]
		stats.MaxLatencyMs = latenciesCopy[len(latenciesCopy)-1]
	}
	em.latenciesMutex.Unlock()

	return stats
}

// LogDecisionPath logs the detailed decision path for debugging
func (em *EntropyMonitor) LogDecisionPath(
	requestID string,
	inputText string,
	steps []DecisionStep,
) {
	if !em.config.EnableDebugLogs {
		return
	}

	log.Printf("[ENTROPY_DECISION_PATH] RequestID: %s", requestID)
	
	// Don't log input text in production for privacy
	if em.config.LogLevel == "debug" {
		log.Printf("[ENTROPY_DECISION_PATH] Input: %s", inputText)
	}

	for i, step := range steps {
		log.Printf("[ENTROPY_DECISION_PATH] Step %d: %s", i+1, step.Description)
		if step.Data != nil {
			if dataJSON, err := json.Marshal(step.Data); err == nil {
				log.Printf("[ENTROPY_DECISION_PATH] Data: %s", string(dataJSON))
			}
		}
		if step.Duration > 0 {
			log.Printf("[ENTROPY_DECISION_PATH] Duration: %.2fms", step.Duration.Seconds()*1000)
		}
	}
}

// ResetStats resets performance statistics
func (em *EntropyMonitor) ResetStats() {
	em.statsMutex.Lock()
	defer em.statsMutex.Unlock()
	em.latenciesMutex.Lock()
	defer em.latenciesMutex.Unlock()

	em.stats = PerformanceStats{
		WindowStart:         time.Now(),
		EntropyDistribution: make(map[string]int64),
		ModelUsage:         make(map[string]int64),
	}
	em.latencies = make([]float64, 0)
}

// DecisionStep represents a step in the decision-making process
type DecisionStep struct {
	Description string        `json:"description"`
	Data        interface{}   `json:"data,omitempty"`
	Duration    time.Duration `json:"duration"`
	Error       error         `json:"error,omitempty"`
}

// CreateDecisionTrace creates a decision trace from classification results
func CreateDecisionTrace(
	requestID string,
	inputText string,
	classificationResult interface{}, // ClassResultWithProbs
	reasoningDecision ReasoningDecision,
	modelType string,
	totalLatency time.Duration,
	config LoggingConfig,
) DecisionTrace {
	trace := DecisionTrace{
		RequestID:           requestID,
		Timestamp:          time.Now(),
		UseReasoning:       reasoningDecision.UseReasoning,
		ReasoningConfidence: reasoningDecision.Confidence,
		DecisionReason:     reasoningDecision.DecisionReason,
		FallbackStrategy:   reasoningDecision.FallbackStrategy,
		TotalLatencyMs:     totalLatency.Seconds() * 1000,
		ClassificationModel: modelType,
	}

	// Only include input text in debug mode for privacy
	if config.LogLevel == "debug" {
		trace.InputText = inputText
	}

	// Convert top categories
	trace.TopCategories = make([]CategoryTrace, len(reasoningDecision.TopCategories))
	for i, cat := range reasoningDecision.TopCategories {
		trace.TopCategories[i] = CategoryTrace{
			Name:        cat.Category,
			Probability: float64(cat.Probability),
			Index:       i, // This would need to be mapped from actual class index
		}
	}

	return trace
}

// LogEntropyDecision logs an entropy-based decision with appropriate detail level
func LogEntropyDecision(
	requestID string,
	decision ReasoningDecision,
	entropy float64,
	uncertaintyLevel string,
	config LoggingConfig,
) {
	switch config.LogLevel {
	case "debug":
		log.Printf("[ENTROPY_DEBUG] RequestID: %s, Entropy: %.6f, Uncertainty: %s, Reasoning: %v, Confidence: %.3f, Reason: %s",
			requestID, entropy, uncertaintyLevel, decision.UseReasoning, decision.Confidence, decision.DecisionReason)
		
		if len(decision.TopCategories) > 0 {
			log.Printf("[ENTROPY_DEBUG] Top categories: %+v", decision.TopCategories)
		}

	case "info":
		if decision.UseReasoning {
			log.Printf("[ENTROPY_INFO] RequestID: %s enabled reasoning (uncertainty: %s, entropy: %.3f)",
				requestID, uncertaintyLevel, entropy)
		}

	case "warn":
		// Only log warnings for unusual conditions
		if entropy < 0 {
			log.Printf("[ENTROPY_WARN] RequestID: %s has negative entropy %.6f", requestID, entropy)
		}
		if decision.FallbackStrategy != "" && decision.FallbackStrategy != "trust_classification" {
			log.Printf("[ENTROPY_WARN] RequestID: %s used fallback strategy: %s", requestID, decision.FallbackStrategy)
		}

	case "error":
		// Only log errors
		if decision.DecisionReason == "classification_error" {
			log.Printf("[ENTROPY_ERROR] RequestID: %s classification failed", requestID)
		}
	}
}

// LogPerformanceAlert logs performance-related alerts
func LogPerformanceAlert(
	alertType string,
	message string,
	metrics map[string]interface{},
	config LoggingConfig,
) {
	if config.LogLevel == "error" && alertType != "error" {
		return
	}
	if config.LogLevel == "warn" && alertType == "info" {
		return
	}

	logLevel := fmt.Sprintf("ENTROPY_%s", alertType)
	if metricsJSON, err := json.Marshal(metrics); err == nil {
		log.Printf("[%s] %s - Metrics: %s", logLevel, message, string(metricsJSON))
	} else {
		log.Printf("[%s] %s", logLevel, message)
	}
}

// Private helper methods

func (em *EntropyMonitor) updateStats(trace DecisionTrace) {
	em.statsMutex.Lock()
	defer em.statsMutex.Unlock()
	em.latenciesMutex.Lock()
	defer em.latenciesMutex.Unlock()

	// Update request counts
	em.stats.TotalRequests++
	if trace.UseReasoning {
		em.stats.ReasoningEnabled++
	} else {
		em.stats.ReasoningDisabled++
	}

	// Update latency stats
	em.latencies = append(em.latencies, trace.TotalLatencyMs)
	if len(em.latencies) > 1000 { // Keep only recent latencies
		em.latencies = em.latencies[len(em.latencies)-1000:]
	}

	// Calculate running average
	totalLatency := 0.0
	for _, lat := range em.latencies {
		totalLatency += lat
	}
	em.stats.AvgLatencyMs = totalLatency / float64(len(em.latencies))

	// Update entropy distribution
	em.stats.EntropyDistribution[trace.UncertaintyLevel]++

	// Update average entropy
	totalEntropy := trace.ShannonEntropy * float64(em.stats.TotalRequests-1) + trace.ShannonEntropy
	em.stats.AvgEntropy = totalEntropy / float64(em.stats.TotalRequests)

	// Update average confidence
	totalConfidence := em.stats.AvgConfidence * float64(em.stats.TotalRequests-1) + trace.ClassConfidence
	em.stats.AvgConfidence = totalConfidence / float64(em.stats.TotalRequests)

	// Update model usage
	em.stats.ModelUsage[trace.ClassificationModel]++

	// Update quality indicators
	if !trace.ProbSumValid || trace.HasNegativeProbabilities {
		em.stats.InvalidProbDists++
	}
}

func (em *EntropyMonitor) logDecision(trace DecisionTrace) {
	// Create a sanitized version for logging
	logTrace := trace
	logTrace.InputText = "" // Remove input text for privacy
	logTrace.ProbabilityDist = nil // Remove detailed probability distribution

	if traceJSON, err := json.Marshal(logTrace); err == nil {
		log.Printf("[ENTROPY_TRACE] %s", string(traceJSON))
	} else {
		log.Printf("[ENTROPY_TRACE] RequestID: %s, Reasoning: %v, Entropy: %.3f, Uncertainty: %s",
			trace.RequestID, trace.UseReasoning, trace.ShannonEntropy, trace.UncertaintyLevel)
	}
}

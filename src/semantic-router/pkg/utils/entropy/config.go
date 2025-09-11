// Package entropy - configuration validation and management for entropy-based routing
//
// This module provides comprehensive configuration validation, schema definition,
// and runtime configuration management for entropy-based classification systems.
// It ensures that all entropy-related configurations are valid, consistent,
// and optimal for the intended use case.

package entropy

import (
	"fmt"
	"math"
)

// EntropyConfig contains all entropy-related configuration options
type EntropyConfig struct {
	// Basic entropy configuration
	Enabled                bool    `json:"enabled" yaml:"enabled"`                                     // Enable entropy-based routing
	DefaultUncertaintyMode string  `json:"default_uncertainty_mode" yaml:"default_uncertainty_mode"` // Default mode when entropy fails
	
	// Threshold configuration
	Thresholds ThresholdConfig `json:"thresholds" yaml:"thresholds"`
	
	// Advanced entropy configuration
	Advanced AdvancedEntropyConfig `json:"advanced" yaml:"advanced"`
	
	// Adaptive threshold configuration
	Adaptive AdaptiveConfig `json:"adaptive" yaml:"adaptive"`
	
	// Performance and monitoring
	Performance PerformanceConfig `json:"performance" yaml:"performance"`
	
	// Logging and debugging
	Logging LoggingConfig `json:"logging" yaml:"logging"`
}

// ThresholdConfig defines uncertainty level thresholds
type ThresholdConfig struct {
	// Static thresholds (used when adaptive is disabled)
	VeryHighThreshold float64 `json:"very_high_threshold" yaml:"very_high_threshold"` // >= this = very high uncertainty
	HighThreshold     float64 `json:"high_threshold" yaml:"high_threshold"`           // >= this = high uncertainty
	MediumThreshold   float64 `json:"medium_threshold" yaml:"medium_threshold"`       // >= this = medium uncertainty
	LowThreshold      float64 `json:"low_threshold" yaml:"low_threshold"`             // >= this = low uncertainty
	
	// Validation constraints
	MinThresholdGap   float64 `json:"min_threshold_gap" yaml:"min_threshold_gap"`     // Minimum gap between thresholds
	MaxThreshold      float64 `json:"max_threshold" yaml:"max_threshold"`             // Maximum allowed threshold value
	MinThreshold      float64 `json:"min_threshold" yaml:"min_threshold"`             // Minimum allowed threshold value
}

// PerformanceConfig defines performance-related settings
type PerformanceConfig struct {
	EnableMetrics        bool    `json:"enable_metrics" yaml:"enable_metrics"`               // Enable Prometheus metrics
	MaxCategoryCount     int     `json:"max_category_count" yaml:"max_category_count"`       // Max categories for heap optimization
	CacheSize            int     `json:"cache_size" yaml:"cache_size"`                       // Cache size for entropy calculations
	TimeoutMs            int     `json:"timeout_ms" yaml:"timeout_ms"`                       // Timeout for entropy calculations
	MaxConcurrentCalcs   int     `json:"max_concurrent_calcs" yaml:"max_concurrent_calcs"`   // Max concurrent entropy calculations
}

// LoggingConfig defines logging and debugging settings
type LoggingConfig struct {
	EnableDebugLogs      bool   `json:"enable_debug_logs" yaml:"enable_debug_logs"`           // Enable debug logging
	EnableDecisionTrace  bool   `json:"enable_decision_trace" yaml:"enable_decision_trace"`   // Trace decision paths
	LogLevel             string `json:"log_level" yaml:"log_level"`                           // Log level (debug, info, warn, error)
	LogSampleRate        float64 `json:"log_sample_rate" yaml:"log_sample_rate"`              // Sampling rate for logs (0.0-1.0)
}

// ConfigValidationError represents a configuration validation error
type ConfigValidationError struct {
	Field   string `json:"field"`   // Field that failed validation
	Value   interface{} `json:"value"`   // Invalid value
	Reason  string `json:"reason"`  // Reason for validation failure
	Suggestion string `json:"suggestion"` // Suggested fix
}

func (e ConfigValidationError) Error() string {
	return fmt.Sprintf("config validation error in field '%s': %s (value: %v) - %s", 
		e.Field, e.Reason, e.Value, e.Suggestion)
}

// ConfigValidationResult contains the results of configuration validation
type ConfigValidationResult struct {
	Valid    bool                     `json:"valid"`    // Whether configuration is valid
	Errors   []ConfigValidationError  `json:"errors"`   // Validation errors
	Warnings []ConfigValidationError  `json:"warnings"` // Validation warnings
	Fixed    bool                     `json:"fixed"`    // Whether auto-fixes were applied
}

// DefaultEntropyConfig returns a sensible default entropy configuration
func DefaultEntropyConfig() EntropyConfig {
	return EntropyConfig{
		Enabled:                true,
		DefaultUncertaintyMode: "conservative", // conservative, aggressive, balanced
		
		Thresholds: ThresholdConfig{
			VeryHighThreshold: 0.8,
			HighThreshold:     0.6,
			MediumThreshold:   0.4,
			LowThreshold:      0.2,
			MinThresholdGap:   0.05,
			MaxThreshold:      1.0,
			MinThreshold:      0.0,
		},
		
		Advanced: DefaultAdvancedEntropyConfig(),
		
		Adaptive: AdaptiveConfig{
			Enabled:         false, // Disabled by default for safety
			LearningRate:    0.05,
			MinSamples:      100,
			MaxHistory:      1000,
			UpdateInterval:  "1h",
			PerformanceFile: "/tmp/entropy_performance.json",
		},
		
		Performance: PerformanceConfig{
			EnableMetrics:      true,
			MaxCategoryCount:   1000,
			CacheSize:         100,
			TimeoutMs:         5000,
			MaxConcurrentCalcs: 10,
		},
		
		Logging: LoggingConfig{
			EnableDebugLogs:     false,
			EnableDecisionTrace: false,
			LogLevel:           "info",
			LogSampleRate:      1.0,
		},
	}
}

// ValidateConfig performs comprehensive validation of entropy configuration
//
// Validation checks include:
//   - Threshold ordering and ranges
//   - Advanced entropy parameter validity
//   - Adaptive learning constraints  
//   - Performance limits and timeouts
//   - Logging configuration consistency
//   - Cross-field dependencies
//
// Args:
//   - config: Configuration to validate
//   - autoFix: Whether to automatically fix correctable issues
//
// Returns:
//   - ConfigValidationResult with validation status and any errors/warnings
func ValidateConfig(config EntropyConfig, autoFix bool) ConfigValidationResult {
	result := ConfigValidationResult{
		Valid:    true,
		Errors:   []ConfigValidationError{},
		Warnings: []ConfigValidationError{},
		Fixed:    false,
	}

	// Validate threshold configuration
	validateThresholds(&config.Thresholds, &result, autoFix)
	
	// Validate advanced entropy configuration
	validateAdvancedConfig(&config.Advanced, &result, autoFix)
	
	// Validate adaptive configuration
	validateAdaptiveConfig(&config.Adaptive, &result, autoFix)
	
	// Validate performance configuration
	validatePerformanceConfig(&config.Performance, &result, autoFix)
	
	// Validate logging configuration
	validateLoggingConfig(&config.Logging, &result, autoFix)
	
	// Validate cross-field dependencies
	validateCrossFieldDependencies(&config, &result, autoFix)

	// Overall validity
	result.Valid = len(result.Errors) == 0

	return result
}

// ValidateThresholds validates threshold configuration and ordering
func ValidateThresholds(thresholds ThresholdConfig) []ConfigValidationError {
	result := ConfigValidationResult{
		Errors: []ConfigValidationError{},
	}
	validateThresholds(&thresholds, &result, false)
	return result.Errors
}

// GetConfigSchema returns a JSON schema for entropy configuration validation
func GetConfigSchema() map[string]interface{} {
	return map[string]interface{}{
		"$schema": "http://json-schema.org/draft-07/schema#",
		"type":    "object",
		"title":   "Entropy Configuration Schema",
		"description": "Configuration schema for entropy-based routing system",
		"properties": map[string]interface{}{
			"enabled": map[string]interface{}{
				"type":        "boolean",
				"description": "Enable entropy-based routing",
				"default":     true,
			},
			"default_uncertainty_mode": map[string]interface{}{
				"type":        "string",
				"description": "Default mode when entropy calculation fails",
				"enum":        []string{"conservative", "aggressive", "balanced"},
				"default":     "conservative",
			},
			"thresholds": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"very_high_threshold": map[string]interface{}{
						"type":    "number",
						"minimum": 0.0,
						"maximum": 1.0,
						"default": 0.8,
					},
					"high_threshold": map[string]interface{}{
						"type":    "number",
						"minimum": 0.0,
						"maximum": 1.0,
						"default": 0.6,
					},
					"medium_threshold": map[string]interface{}{
						"type":    "number",
						"minimum": 0.0,
						"maximum": 1.0,
						"default": 0.4,
					},
					"low_threshold": map[string]interface{}{
						"type":    "number",
						"minimum": 0.0,
						"maximum": 1.0,
						"default": 0.2,
					},
				},
				"required": []string{"very_high_threshold", "high_threshold", "medium_threshold", "low_threshold"},
			},
		},
		"required": []string{"enabled", "thresholds"},
	}
}

// Private validation functions

func validateThresholds(thresholds *ThresholdConfig, result *ConfigValidationResult, autoFix bool) {
	// Check threshold ranges
	thresholdFields := []struct {
		name  string
		value *float64
	}{
		{"very_high_threshold", &thresholds.VeryHighThreshold},
		{"high_threshold", &thresholds.HighThreshold},
		{"medium_threshold", &thresholds.MediumThreshold},
		{"low_threshold", &thresholds.LowThreshold},
	}

	for _, field := range thresholdFields {
		if *field.value < thresholds.MinThreshold || *field.value > thresholds.MaxThreshold {
			if autoFix {
				oldValue := *field.value
				*field.value = math.Max(thresholds.MinThreshold, math.Min(thresholds.MaxThreshold, *field.value))
				result.Fixed = true
				result.Warnings = append(result.Warnings, ConfigValidationError{
					Field:  field.name,
					Value:  oldValue,
					Reason: fmt.Sprintf("threshold outside valid range [%.2f, %.2f], auto-fixed to %.2f", 
						thresholds.MinThreshold, thresholds.MaxThreshold, *field.value),
					Suggestion: "Use thresholds within the valid range",
				})
			} else {
				result.Errors = append(result.Errors, ConfigValidationError{
					Field:  field.name,
					Value:  *field.value,
					Reason: fmt.Sprintf("threshold outside valid range [%.2f, %.2f]", 
						thresholds.MinThreshold, thresholds.MaxThreshold),
					Suggestion: "Set threshold between min and max threshold values",
				})
			}
		}
	}

	// Check threshold ordering: very_high > high > medium > low
	// Fix from bottom up to avoid cascading issues
	if autoFix {
		// Fix ordering from bottom up
		if thresholds.MediumThreshold <= thresholds.LowThreshold {
			thresholds.MediumThreshold = thresholds.LowThreshold + thresholds.MinThresholdGap
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "medium_threshold",
				Value:      thresholds.MediumThreshold,
				Reason:     "medium_threshold must be greater than low_threshold, auto-fixed",
				Suggestion: "Ensure threshold ordering: very_high > high > medium > low",
			})
		}
		if thresholds.HighThreshold <= thresholds.MediumThreshold {
			thresholds.HighThreshold = thresholds.MediumThreshold + thresholds.MinThresholdGap
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "high_threshold",
				Value:      thresholds.HighThreshold,
				Reason:     "high_threshold must be greater than medium_threshold, auto-fixed",
				Suggestion: "Ensure threshold ordering: very_high > high > medium > low",
			})
		}
		if thresholds.VeryHighThreshold <= thresholds.HighThreshold {
			thresholds.VeryHighThreshold = thresholds.HighThreshold + thresholds.MinThresholdGap
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "very_high_threshold",
				Value:      thresholds.VeryHighThreshold,
				Reason:     "very_high_threshold must be greater than high_threshold, auto-fixed",
				Suggestion: "Ensure threshold ordering: very_high > high > medium > low",
			})
		}
	} else {
		// Check ordering and report errors
		if thresholds.VeryHighThreshold <= thresholds.HighThreshold {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "very_high_threshold",
				Value:      thresholds.VeryHighThreshold,
				Reason:     "very_high_threshold must be greater than high_threshold",
				Suggestion: "Increase very_high_threshold or decrease high_threshold",
			})
		}
		if thresholds.HighThreshold <= thresholds.MediumThreshold {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "high_threshold",
				Value:      thresholds.HighThreshold,
				Reason:     "high_threshold must be greater than medium_threshold",
				Suggestion: "Increase high_threshold or decrease medium_threshold",
			})
		}
		if thresholds.MediumThreshold <= thresholds.LowThreshold {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "medium_threshold",
				Value:      thresholds.MediumThreshold,
				Reason:     "medium_threshold must be greater than low_threshold",
				Suggestion: "Increase medium_threshold or decrease low_threshold",
			})
		}
	}

	// Check minimum gap validation
	if thresholds.MinThresholdGap <= 0 || thresholds.MinThresholdGap > 0.2 {
		if autoFix {
			thresholds.MinThresholdGap = 0.05
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "min_threshold_gap",
				Value:      thresholds.MinThresholdGap,
				Reason:     "min_threshold_gap should be between 0 and 0.2, auto-fixed to 0.05",
				Suggestion: "Use a small positive gap between thresholds",
			})
		} else {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "min_threshold_gap",
				Value:      thresholds.MinThresholdGap,
				Reason:     "min_threshold_gap should be between 0 and 0.2",
				Suggestion: "Set a reasonable gap like 0.05",
			})
		}
	}
}

func validateAdvancedConfig(advanced *AdvancedEntropyConfig, result *ConfigValidationResult, autoFix bool) {
	// Validate Rényi alpha parameter
	if advanced.EnableRenyi && (advanced.RenyiAlpha <= 0 || advanced.RenyiAlpha > 10) {
		if autoFix {
			advanced.RenyiAlpha = 2.0
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "renyi_alpha",
				Value:      advanced.RenyiAlpha,
				Reason:     "renyi_alpha should be positive and reasonable (0, 10], auto-fixed to 2.0",
				Suggestion: "Use alpha between 0.5 and 5.0 for most applications",
			})
		} else {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "renyi_alpha",
				Value:      advanced.RenyiAlpha,
				Reason:     "renyi_alpha should be positive and reasonable",
				Suggestion: "Set alpha between 0.5 and 5.0",
			})
		}
	}

	// Validate Tsallis q parameter
	if advanced.EnableTsallis && (advanced.TsallisQ <= 0 || advanced.TsallisQ > 10) {
		if autoFix {
			advanced.TsallisQ = 2.0
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "tsallis_q",
				Value:      advanced.TsallisQ,
				Reason:     "tsallis_q should be positive and reasonable (0, 10], auto-fixed to 2.0",
				Suggestion: "Use q between 0.5 and 5.0 for most applications",
			})
		} else {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "tsallis_q",
				Value:      advanced.TsallisQ,
				Reason:     "tsallis_q should be positive and reasonable",
				Suggestion: "Set q between 0.5 and 5.0",
			})
		}
	}

	// Validate weights sum to 1.0
	totalWeight := advanced.ShannonWeight + advanced.RenyiWeight + advanced.TsallisWeight
	if math.Abs(totalWeight-1.0) > 0.001 {
		if autoFix {
			// Normalize weights
			advanced.ShannonWeight /= totalWeight
			advanced.RenyiWeight /= totalWeight
			advanced.TsallisWeight /= totalWeight
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "entropy_weights",
				Value:      totalWeight,
				Reason:     "entropy weights should sum to 1.0, auto-normalized",
				Suggestion: "Ensure shannon_weight + renyi_weight + tsallis_weight = 1.0",
			})
		} else {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "entropy_weights",
				Value:      totalWeight,
				Reason:     "entropy weights should sum to 1.0",
				Suggestion: "Adjust weights so they sum to exactly 1.0",
			})
		}
	}
}

func validateAdaptiveConfig(adaptive *AdaptiveConfig, result *ConfigValidationResult, autoFix bool) {
	if !adaptive.Enabled {
		return // Skip validation if adaptive is disabled
	}

	// Validate learning rate
	if adaptive.LearningRate <= 0 || adaptive.LearningRate > 1.0 {
		if autoFix {
			adaptive.LearningRate = 0.05
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "learning_rate",
				Value:      adaptive.LearningRate,
				Reason:     "learning_rate should be between 0 and 1.0, auto-fixed to 0.05",
				Suggestion: "Use learning rate between 0.01 and 0.1 for stability",
			})
		} else {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "learning_rate",
				Value:      adaptive.LearningRate,
				Reason:     "learning_rate should be between 0 and 1.0",
				Suggestion: "Set learning rate between 0.01 and 0.1",
			})
		}
	}

	// Validate minimum samples
	if adaptive.MinSamples < 10 || adaptive.MinSamples > 10000 {
		if autoFix {
			adaptive.MinSamples = 100
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "min_samples",
				Value:      adaptive.MinSamples,
				Reason:     "min_samples should be between 10 and 10000, auto-fixed to 100",
				Suggestion: "Use min_samples between 50 and 1000",
			})
		} else {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "min_samples",
				Value:      adaptive.MinSamples,
				Reason:     "min_samples should be between 10 and 10000",
				Suggestion: "Set min_samples between 50 and 1000",
			})
		}
	}
}

func validatePerformanceConfig(performance *PerformanceConfig, result *ConfigValidationResult, autoFix bool) {
	// Validate max category count
	if performance.MaxCategoryCount < 2 || performance.MaxCategoryCount > 100000 {
		if autoFix {
			performance.MaxCategoryCount = 1000
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "max_category_count",
				Value:      performance.MaxCategoryCount,
				Reason:     "max_category_count should be between 2 and 100000, auto-fixed to 1000",
				Suggestion: "Set reasonable category limit based on your use case",
			})
		} else {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "max_category_count",
				Value:      performance.MaxCategoryCount,
				Reason:     "max_category_count should be between 2 and 100000",
				Suggestion: "Set a reasonable category limit",
			})
		}
	}

	// Validate timeout
	if performance.TimeoutMs < 100 || performance.TimeoutMs > 60000 {
		if autoFix {
			performance.TimeoutMs = 5000
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "timeout_ms",
				Value:      performance.TimeoutMs,
				Reason:     "timeout_ms should be between 100 and 60000, auto-fixed to 5000",
				Suggestion: "Set timeout between 1000ms and 10000ms",
			})
		} else {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "timeout_ms",
				Value:      performance.TimeoutMs,
				Reason:     "timeout_ms should be between 100 and 60000",
				Suggestion: "Set reasonable timeout in milliseconds",
			})
		}
	}
}

func validateLoggingConfig(logging *LoggingConfig, result *ConfigValidationResult, autoFix bool) {
	// Validate log level
	validLogLevels := []string{"debug", "info", "warn", "error"}
	isValidLevel := false
	for _, level := range validLogLevels {
		if logging.LogLevel == level {
			isValidLevel = true
			break
		}
	}

	if !isValidLevel {
		if autoFix {
			logging.LogLevel = "info"
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "log_level",
				Value:      logging.LogLevel,
				Reason:     "invalid log level, auto-fixed to 'info'",
				Suggestion: "Use one of: debug, info, warn, error",
			})
		} else {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "log_level",
				Value:      logging.LogLevel,
				Reason:     "invalid log level",
				Suggestion: "Use one of: debug, info, warn, error",
			})
		}
	}

	// Validate sample rate
	if logging.LogSampleRate < 0 || logging.LogSampleRate > 1.0 {
		if autoFix {
			logging.LogSampleRate = 1.0
			result.Fixed = true
			result.Warnings = append(result.Warnings, ConfigValidationError{
				Field:      "log_sample_rate",
				Value:      logging.LogSampleRate,
				Reason:     "log_sample_rate should be between 0.0 and 1.0, auto-fixed to 1.0",
				Suggestion: "Set sample rate between 0.0 and 1.0",
			})
		} else {
			result.Errors = append(result.Errors, ConfigValidationError{
				Field:      "log_sample_rate",
				Value:      logging.LogSampleRate,
				Reason:     "log_sample_rate should be between 0.0 and 1.0",
				Suggestion: "Set sample rate between 0.0 and 1.0",
			})
		}
	}
}

func validateCrossFieldDependencies(config *EntropyConfig, result *ConfigValidationResult, autoFix bool) {
	// Check adaptive vs static threshold consistency
	if config.Adaptive.Enabled && config.Advanced.EnableRenyi && config.Advanced.RenyiAlpha == 1.0 {
		result.Warnings = append(result.Warnings, ConfigValidationError{
			Field:      "cross_field_dependency",
			Value:      "adaptive + renyi_alpha=1.0",
			Reason:     "using adaptive thresholds with Rényi alpha=1.0 is redundant (equals Shannon)",
			Suggestion: "Consider using alpha != 1.0 for Rényi entropy or disable adaptive",
		})
	}

	// Check performance vs advanced entropy settings
	if config.Advanced.EnableRenyi && config.Advanced.EnableTsallis && config.Performance.TimeoutMs < 2000 {
		result.Warnings = append(result.Warnings, ConfigValidationError{
			Field:      "cross_field_dependency",
			Value:      "advanced_entropy + low_timeout",
			Reason:     "using multiple entropy measures with low timeout may cause timeouts",
			Suggestion: "Increase timeout_ms or disable some advanced entropy measures",
		})
	}
}

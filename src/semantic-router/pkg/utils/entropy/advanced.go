// Package entropy - advanced entropy measures for enhanced uncertainty analysis
//
// This module implements additional entropy measures beyond Shannon entropy,
// including Rényi entropy and Tsallis entropy. These measures can provide
// different perspectives on uncertainty and may be more suitable for certain
// types of probability distributions or decision-making scenarios.

package entropy

import (
	"math"
)

// AdvancedEntropyResult contains results from multiple entropy measures
type AdvancedEntropyResult struct {
	Shannon      float64 `json:"shannon"`       // Shannon entropy (base case)
	Renyi        float64 `json:"renyi"`         // Rényi entropy with specified alpha
	Tsallis      float64 `json:"tsallis"`       // Tsallis entropy with specified q
	Alpha        float64 `json:"alpha"`         // Alpha parameter for Rényi entropy
	Q            float64 `json:"q"`             // Q parameter for Tsallis entropy
	
	// Normalized versions (0-1 scale)
	NormalizedShannon  float64 `json:"normalized_shannon"`
	NormalizedRenyi    float64 `json:"normalized_renyi"`
	NormalizedTsallis  float64 `json:"normalized_tsallis"`
	
	// Consensus measures
	ConsensusEntropy      float64 `json:"consensus_entropy"`       // Weighted combination
	ConsensusUncertainty  string  `json:"consensus_uncertainty"`   // Consensus uncertainty level
	EntropyDivergence     float64 `json:"entropy_divergence"`      // How much different measures disagree
}

// AdvancedEntropyConfig configures advanced entropy calculations
type AdvancedEntropyConfig struct {
	EnableRenyi     bool    `json:"enable_renyi"`      // Enable Rényi entropy calculation
	EnableTsallis   bool    `json:"enable_tsallis"`    // Enable Tsallis entropy calculation
	RenyiAlpha      float64 `json:"renyi_alpha"`       // Alpha parameter for Rényi (default: 2.0)
	TsallisQ        float64 `json:"tsallis_q"`         // Q parameter for Tsallis (default: 2.0)
	
	// Consensus weighting (should sum to 1.0)
	ShannonWeight   float64 `json:"shannon_weight"`    // Weight for Shannon entropy (default: 0.5)
	RenyiWeight     float64 `json:"renyi_weight"`      // Weight for Rényi entropy (default: 0.3)
	TsallisWeight   float64 `json:"tsallis_weight"`    // Weight for Tsallis entropy (default: 0.2)
}

// DefaultAdvancedEntropyConfig returns a sensible default configuration
func DefaultAdvancedEntropyConfig() AdvancedEntropyConfig {
	return AdvancedEntropyConfig{
		EnableRenyi:   true,
		EnableTsallis: true,
		RenyiAlpha:    2.0,  // Quadratic entropy (emphasizes high probabilities)
		TsallisQ:      2.0,  // Similar to Rényi with alpha=2
		ShannonWeight: 0.5,  // Primary weight on Shannon
		RenyiWeight:   0.3,  // Secondary weight on Rényi
		TsallisWeight: 0.2,  // Tertiary weight on Tsallis
	}
}

// CalculateAdvancedEntropy computes multiple entropy measures for a probability distribution
//
// Shannon Entropy: H(X) = -∑ p(x) * log₂(p(x))
// - Standard information-theoretic entropy
// - Treats all probabilities equally
// - Most commonly used and well-understood
//
// Rényi Entropy: H_α(X) = (1/(1-α)) * log₂(∑ p(x)^α)
// - Generalization of Shannon entropy (α=1 gives Shannon)
// - α > 1: emphasizes high probability events
// - α < 1: emphasizes low probability events
// - α=2: quadratic entropy, computationally efficient
// - α→∞: min-entropy (focuses on maximum probability)
//
// Tsallis Entropy: S_q(X) = (1/(q-1)) * (1 - ∑ p(x)^q)
// - Non-extensive entropy, useful for complex systems
// - q=1: approaches Shannon entropy
// - q > 1: sub-extensive (emphasizes rare events)
// - q < 1: super-extensive (emphasizes common events)
// - q=2: similar behavior to Rényi with α=2
//
// Args:
//   - probabilities: Probability distribution (should sum to 1.0)
//   - config: Configuration for advanced entropy calculation
//
// Returns:
//   - AdvancedEntropyResult with all entropy measures and analysis
func CalculateAdvancedEntropy(probabilities []float32, config AdvancedEntropyConfig) AdvancedEntropyResult {
	if len(probabilities) == 0 {
		return AdvancedEntropyResult{}
	}

	// Calculate Shannon entropy (baseline)
	shannon := CalculateEntropy(probabilities)
	normalizedShannon := CalculateNormalizedEntropy(probabilities)

	result := AdvancedEntropyResult{
		Shannon:           shannon,
		NormalizedShannon: normalizedShannon,
		Alpha:             config.RenyiAlpha,
		Q:                 config.TsallisQ,
	}

	// Calculate Rényi entropy if enabled
	if config.EnableRenyi {
		renyi := calculateRenyiEntropy(probabilities, config.RenyiAlpha)
		result.Renyi = renyi
		result.NormalizedRenyi = normalizeRenyiEntropy(renyi, len(probabilities), config.RenyiAlpha)
	}

	// Calculate Tsallis entropy if enabled
	if config.EnableTsallis {
		tsallis := calculateTsallisEntropy(probabilities, config.TsallisQ)
		result.Tsallis = tsallis
		result.NormalizedTsallis = normalizeTsallisEntropy(tsallis, len(probabilities), config.TsallisQ)
	}

	// Calculate consensus measures
	result.ConsensusEntropy = calculateConsensusEntropy(result, config)
	result.ConsensusUncertainty = determineConsensusUncertaintyLevel(result, config)
	result.EntropyDivergence = calculateEntropyDivergence(result, config)

	return result
}

// CalculateRenyiEntropy computes Rényi entropy with specified alpha parameter
//
// Rényi entropy formula: H_α(X) = (1/(1-α)) * log₂(∑ p(x)^α)
//
// Special cases:
//   - α = 0: H₀(X) = log₂(n) where n is the number of non-zero probabilities
//   - α = 1: H₁(X) = Shannon entropy (limit as α → 1)
//   - α = 2: H₂(X) = -log₂(∑ p(x)²) (collision entropy)
//   - α = ∞: H∞(X) = -log₂(max(p(x))) (min-entropy)
func CalculateRenyiEntropy(probabilities []float32, alpha float64) float64 {
	return calculateRenyiEntropy(probabilities, alpha)
}

// CalculateTsallisEntropy computes Tsallis entropy with specified q parameter
//
// Tsallis entropy formula: S_q(X) = (1/(q-1)) * (1 - ∑ p(x)^q)
//
// Special cases:
//   - q = 1: S₁(X) = Shannon entropy (limit as q → 1)
//   - q = 2: S₂(X) = 1 - ∑ p(x)² (Gini-Simpson index)
//   - q → 0: S₀(X) → (n-1)/1 where n is number of non-zero probabilities
func CalculateTsallisEntropy(probabilities []float32, q float64) float64 {
	return calculateTsallisEntropy(probabilities, q)
}

// MakeAdvancedEntropyBasedReasoningDecision uses multiple entropy measures for reasoning decisions
//
// This function extends the basic entropy-based reasoning by considering multiple entropy
// measures and their consensus. It can provide more robust decisions by:
//
// 1. Detecting cases where different entropy measures disagree (high divergence)
// 2. Using consensus uncertainty level from multiple measures
// 3. Applying measure-specific reasoning strategies
//
// Decision Strategy:
//   - High entropy divergence: Use conservative approach (enable reasoning)
//   - Low entropy divergence: Use consensus uncertainty level
//   - Rényi emphasis: Better for distributions with dominant categories
//   - Tsallis emphasis: Better for complex, multi-modal distributions
func MakeAdvancedEntropyBasedReasoningDecision(
	probabilities []float32,
	categoryNames []string,
	categoryReasoningMap map[string]bool,
	baseConfidenceThreshold float64,
	config AdvancedEntropyConfig,
) ReasoningDecision {
	// Calculate advanced entropy measures
	advancedResult := CalculateAdvancedEntropy(probabilities, config)

	// Check for high entropy divergence (different measures disagree significantly)
	if advancedResult.EntropyDivergence > 0.3 {
		return ReasoningDecision{
			UseReasoning:     true,
			Confidence:       0.3, // Low confidence due to disagreement
			DecisionReason:   "high_entropy_divergence_conservative_approach",
			FallbackStrategy: "entropy_disagreement_safety",
			TopCategories:    getTopCategories(probabilities, categoryNames, 3),
		}
	}

	// Use consensus uncertainty level for decision
	return makeReasoningDecisionFromConsensusLevel(
		advancedResult.ConsensusUncertainty,
		probabilities,
		categoryNames,
		categoryReasoningMap,
		advancedResult.ConsensusEntropy,
	)
}

// Private helper functions

func calculateRenyiEntropy(probabilities []float32, alpha float64) float64 {
	if len(probabilities) == 0 {
		return 0.0
	}

	// Handle special cases
	if alpha == 1.0 {
		// Limit as α → 1 gives Shannon entropy
		return CalculateEntropy(probabilities)
	}

	if alpha == 0.0 {
		// H₀(X) = log₂(number of non-zero probabilities)
		nonZeroCount := 0
		for _, prob := range probabilities {
			if prob > 0 {
				nonZeroCount++
			}
		}
		return math.Log2(float64(nonZeroCount))
	}

	if math.IsInf(alpha, 1) {
		// H∞(X) = -log₂(max probability) (min-entropy)
		maxProb := float32(0.0)
		for _, prob := range probabilities {
			if prob > maxProb {
				maxProb = prob
			}
		}
		if maxProb > 0 {
			return -math.Log2(float64(maxProb))
		}
		return 0.0
	}

	// General case: H_α(X) = (1/(1-α)) * log₂(∑ p(x)^α)
	sum := 0.0
	for _, prob := range probabilities {
		if prob > 0 {
			sum += math.Pow(float64(prob), alpha)
		}
	}

	if sum <= 0 {
		return 0.0
	}

	return (1.0 / (1.0 - alpha)) * math.Log2(sum)
}

func calculateTsallisEntropy(probabilities []float32, q float64) float64 {
	if len(probabilities) == 0 {
		return 0.0
	}

	// Handle special case q = 1 (approaches Shannon entropy)
	if math.Abs(q-1.0) < 1e-10 {
		return CalculateEntropy(probabilities)
	}

	// General case: S_q(X) = (1/(q-1)) * (1 - ∑ p(x)^q)
	sum := 0.0
	for _, prob := range probabilities {
		if prob > 0 {
			sum += math.Pow(float64(prob), q)
		}
	}

	return (1.0 / (q - 1.0)) * (1.0 - sum)
}

func normalizeRenyiEntropy(renyiEntropy float64, numClasses int, alpha float64) float64 {
	if numClasses <= 1 {
		return 0.0
	}

	// Maximum Rényi entropy for uniform distribution
	uniformProb := 1.0 / float64(numClasses)
	maxSum := float64(numClasses) * math.Pow(uniformProb, alpha)
	
	var maxRenyi float64
	if alpha == 1.0 {
		maxRenyi = math.Log2(float64(numClasses)) // Shannon entropy limit
	} else {
		maxRenyi = (1.0 / (1.0 - alpha)) * math.Log2(maxSum)
	}

	if maxRenyi <= 0 {
		return 0.0
	}

	normalized := renyiEntropy / maxRenyi
	
	// Clamp to [0, 1] range
	if normalized < 0 {
		return 0.0
	}
	if normalized > 1 {
		return 1.0
	}
	
	return normalized
}

func normalizeTsallisEntropy(tsallisEntropy float64, numClasses int, q float64) float64 {
	if numClasses <= 1 {
		return 0.0
	}

	// Maximum Tsallis entropy for uniform distribution
	uniformProb := 1.0 / float64(numClasses)
	maxSum := float64(numClasses) * math.Pow(uniformProb, q)
	maxTsallis := (1.0 / (q - 1.0)) * (1.0 - maxSum)

	if maxTsallis <= 0 {
		return 0.0
	}

	normalized := tsallisEntropy / maxTsallis
	
	// Clamp to [0, 1] range
	if normalized < 0 {
		return 0.0
	}
	if normalized > 1 {
		return 1.0
	}
	
	return normalized
}

func calculateConsensusEntropy(result AdvancedEntropyResult, config AdvancedEntropyConfig) float64 {
	// Normalize weights to sum to 1.0
	totalWeight := config.ShannonWeight
	if config.EnableRenyi {
		totalWeight += config.RenyiWeight
	}
	if config.EnableTsallis {
		totalWeight += config.TsallisWeight
	}

	if totalWeight == 0 {
		return result.NormalizedShannon
	}

	consensus := (config.ShannonWeight / totalWeight) * result.NormalizedShannon

	if config.EnableRenyi {
		consensus += (config.RenyiWeight / totalWeight) * result.NormalizedRenyi
	}

	if config.EnableTsallis {
		consensus += (config.TsallisWeight / totalWeight) * result.NormalizedTsallis
	}

	return consensus
}

func determineConsensusUncertaintyLevel(result AdvancedEntropyResult, config AdvancedEntropyConfig) string {
	consensus := result.ConsensusEntropy

	// Use same thresholds as standard entropy analysis
	if consensus >= 0.8 {
		return "very_high"
	} else if consensus >= 0.6 {
		return "high"
	} else if consensus >= 0.4 {
		return "medium"
	} else if consensus >= 0.2 {
		return "low"
	} else {
		return "very_low"
	}
}

func calculateEntropyDivergence(result AdvancedEntropyResult, config AdvancedEntropyConfig) float64 {
	// Calculate how much different entropy measures disagree
	entropies := []float64{result.NormalizedShannon}
	
	if config.EnableRenyi {
		entropies = append(entropies, result.NormalizedRenyi)
	}
	
	if config.EnableTsallis {
		entropies = append(entropies, result.NormalizedTsallis)
	}

	if len(entropies) <= 1 {
		return 0.0
	}

	// Calculate standard deviation of normalized entropy values
	mean := 0.0
	for _, entropy := range entropies {
		mean += entropy
	}
	mean /= float64(len(entropies))

	variance := 0.0
	for _, entropy := range entropies {
		diff := entropy - mean
		variance += diff * diff
	}
	variance /= float64(len(entropies))

	return math.Sqrt(variance)
}

func makeReasoningDecisionFromConsensusLevel(
	consensusLevel string,
	probabilities []float32,
	categoryNames []string,
	categoryReasoningMap map[string]bool,
	consensusEntropy float64,
) ReasoningDecision {
	// Get top categories for decision making
	topCategories := getTopCategories(probabilities, categoryNames, 3)
	
	if len(topCategories) == 0 {
		return ReasoningDecision{
			UseReasoning:     false,
			Confidence:       0.0,
			DecisionReason:   "no_valid_categories",
			FallbackStrategy: "consensus_empty_categories_fallback",
			TopCategories:    []CategoryProbability{},
		}
	}

	topCategory := topCategories[0]
	topConfidence := float64(topCategory.Probability)

	// Apply consensus-based decision logic
	switch consensusLevel {
	case "very_high":
		return ReasoningDecision{
			UseReasoning:     true,
			Confidence:       0.3,
			DecisionReason:   "consensus_very_high_uncertainty_conservative",
			FallbackStrategy: "consensus_conservative_default",
			TopCategories:    topCategories,
		}

	case "high":
		// Weighted decision from top categories
		return makeWeightedDecision(topCategories, categoryReasoningMap, "consensus_high_uncertainty_weighted")

	case "medium":
		// Trust top category reasoning preference
		useReasoning := categoryReasoningMap[topCategory.Category]
		return ReasoningDecision{
			UseReasoning:     useReasoning,
			Confidence:       topConfidence,
			DecisionReason:   "consensus_medium_uncertainty_trust_top_category",
			FallbackStrategy: "consensus_trust_top_category",
			TopCategories:    topCategories,
		}

	case "low", "very_low":
		// Trust classification
		useReasoning := categoryReasoningMap[topCategory.Category]
		return ReasoningDecision{
			UseReasoning:     useReasoning,
			Confidence:       topConfidence,
			DecisionReason:   "consensus_low_uncertainty_trust_classification",
			FallbackStrategy: "consensus_trust_classification",
			TopCategories:    topCategories,
		}

	default:
		// Fallback to conservative approach
		return ReasoningDecision{
			UseReasoning:     true,
			Confidence:       0.5,
			DecisionReason:   "consensus_unknown_uncertainty_level",
			FallbackStrategy: "consensus_unknown_level_fallback",
			TopCategories:    topCategories,
		}
	}
}

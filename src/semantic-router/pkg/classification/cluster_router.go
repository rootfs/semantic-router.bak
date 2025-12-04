package classification

import (
	"fmt"
	"sync"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ClusterRouter implements cluster-based intelligent model routing.
// It uses K-means clustering on query embeddings to learn patterns
// of which models perform best for which types of queries.
//
// Training Flow:
//  1. Collect experience data: (query_embedding, model_performance_scores)
//  2. Run K-means clustering on embeddings to find query clusters
//  3. For each cluster, compute average model performance
//  4. Select best model per cluster considering performance and cost
//
// Routing Flow:
//  1. Generate embedding for new query
//  2. Find nearest cluster center (cosine similarity)
//  3. Return the best model for that cluster
type ClusterRouter struct {
	config        *config.ClusterRouterConfig
	embeddingDim  int
	initialized   bool
	mu            sync.RWMutex
	defaultModel  string
	availableModels []string
}

// NewClusterRouter creates a new cluster router instance
func NewClusterRouter(cfg *config.ClusterRouterConfig, defaultModel string, availableModels []string) *ClusterRouter {
	return &ClusterRouter{
		config:         cfg,
		defaultModel:   defaultModel,
		availableModels: availableModels,
	}
}

// Train trains the cluster router from experience data
//
// Parameters:
//   - experienceData: Array of experience records containing embeddings and model scores
//
// Returns:
//   - error: Non-nil if training fails
func (r *ClusterRouter) Train(experienceData []candle_binding.ExperienceRecord) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(experienceData) == 0 {
		return fmt.Errorf("experience data cannot be empty")
	}

	// Validate experience data
	embeddingDim := len(experienceData[0].Embedding)
	if embeddingDim == 0 {
		return fmt.Errorf("first experience record has empty embedding")
	}

	for i, record := range experienceData {
		if len(record.Embedding) != embeddingDim {
			return fmt.Errorf("experience record %d has inconsistent embedding dimension (%d vs %d)",
				i, len(record.Embedding), embeddingDim)
		}
		if len(record.ModelScores) == 0 {
			return fmt.Errorf("experience record %d has no model scores", i)
		}
	}

	// Build candle binding config
	bindingConfig := candle_binding.ClusterRouterConfig{
		NClusters:     r.config.NClusters,
		MaxIterations: r.config.MaxIterations,
		Alpha:         r.config.Alpha,
		UseCPU:        r.config.UseCPU,
	}

	// Initialize the cluster router
	err := candle_binding.InitClusterRouter(experienceData, r.config.ModelCosts, bindingConfig)
	if err != nil {
		return fmt.Errorf("failed to train cluster router: %w", err)
	}

	r.embeddingDim = embeddingDim
	r.initialized = true

	logging.Infof("Cluster router trained with %d experience records, %d clusters, embedding_dim=%d",
		len(experienceData), r.config.NClusters, embeddingDim)

	return nil
}

// Route routes a query embedding to the best model
//
// Parameters:
//   - queryEmbedding: The embedding vector for the query
//
// Returns:
//   - string: The name of the best model for this query
//   - float32: Confidence score (cosine similarity to nearest cluster)
//   - error: Non-nil if routing fails
func (r *ClusterRouter) Route(queryEmbedding []float32) (string, float32, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return r.defaultModel, 0.0, fmt.Errorf("cluster router not initialized")
	}

	if len(queryEmbedding) != r.embeddingDim {
		return r.defaultModel, 0.0, fmt.Errorf("embedding dimension mismatch (%d vs %d)",
			len(queryEmbedding), r.embeddingDim)
	}

	result, err := candle_binding.RouteQuery(queryEmbedding)
	if err != nil {
		logging.Warnf("Cluster routing failed: %v, falling back to default model", err)
		return r.defaultModel, 0.0, nil
	}

	// Validate the returned model is in our available models list
	modelValid := false
	for _, m := range r.availableModels {
		if m == result.ModelName {
			modelValid = true
			break
		}
	}

	if !modelValid {
		logging.Warnf("Cluster router returned unknown model '%s', falling back to default", result.ModelName)
		return r.defaultModel, result.Confidence, nil
	}

	logging.Debugf("Cluster router: query -> cluster %d -> model '%s' (confidence: %.3f)",
		result.ClusterID, result.ModelName, result.Confidence)

	return result.ModelName, result.Confidence, nil
}

// RouteWithText routes a query by first generating its embedding
//
// Parameters:
//   - text: The query text
//   - embeddingModel: The embedding model to use ("qwen3" or "gemma")
//   - embeddingDim: Target embedding dimension
//
// Returns:
//   - string: The name of the best model for this query
//   - float32: Confidence score
//   - error: Non-nil if routing fails
func (r *ClusterRouter) RouteWithText(text string, embeddingModel string, embeddingDim int) (string, float32, error) {
	// Generate embedding for the query
	embOutput, err := candle_binding.GetEmbeddingWithModelType(text, embeddingModel, embeddingDim)
	if err != nil {
		return r.defaultModel, 0.0, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	return r.Route(embOutput.Embedding)
}

// IsInitialized returns whether the router has been trained
func (r *ClusterRouter) IsInitialized() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.initialized
}

// GetClusterCount returns the number of clusters
func (r *ClusterRouter) GetClusterCount() int {
	return candle_binding.GetClusterCount()
}

// GetClusterModelScores returns model performance scores for a specific cluster
func (r *ClusterRouter) GetClusterModelScores(clusterID int) (map[string]float32, error) {
	return candle_binding.GetClusterModelScores(clusterID)
}

// GetConfig returns the router configuration
func (r *ClusterRouter) GetConfig() *config.ClusterRouterConfig {
	return r.config
}

// GetDefaultModel returns the default model used when routing fails
func (r *ClusterRouter) GetDefaultModel() string {
	return r.defaultModel
}

// GetAvailableModels returns the list of available models
func (r *ClusterRouter) GetAvailableModels() []string {
	return r.availableModels
}


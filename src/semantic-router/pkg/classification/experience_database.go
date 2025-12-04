package classification

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ExperienceEntry represents a single entry in the experience database
// It stores the query embedding along with how each model performed on that query
type ExperienceEntry struct {
	// QueryText is the original query text (optional, for debugging)
	QueryText string `json:"query_text,omitempty"`

	// Embedding is the query embedding vector
	// If not provided, it will be generated from QueryText
	Embedding []float32 `json:"embedding,omitempty"`

	// ModelScores maps model names to their performance scores
	// Score is typically 1.0 for correct, 0.0 for incorrect
	// Can also be a continuous score (e.g., BLEU score)
	ModelScores map[string]float32 `json:"model_scores"`

	// Metadata stores additional information (optional)
	Metadata map[string]string `json:"metadata,omitempty"`
}

// ExperienceDatabase manages the experience data for cluster-based routing
// It handles loading, saving, and converting experience entries for the cluster router
type ExperienceDatabase struct {
	entries []ExperienceEntry
	mu      sync.RWMutex
	config  *config.ClusterRouterConfig
}

// NewExperienceDatabase creates a new experience database
func NewExperienceDatabase(cfg *config.ClusterRouterConfig) *ExperienceDatabase {
	return &ExperienceDatabase{
		entries: []ExperienceEntry{},
		config:  cfg,
	}
}

// LoadFromFile loads experience entries from a JSON file
//
// The JSON format is:
//
//	[
//	    {
//	        "query_text": "What is 2+2?",
//	        "embedding": [0.1, 0.2, ...],  // optional if query_text provided
//	        "model_scores": {
//	            "math": 1.0,
//	            "coder": 0.0,
//	            "general": 1.0
//	        },
//	        "metadata": {"category": "math"}  // optional
//	    },
//	    ...
//	]
func (db *ExperienceDatabase) LoadFromFile(filePath string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read experience database file: %w", err)
	}

	var entries []ExperienceEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return fmt.Errorf("failed to parse experience database JSON: %w", err)
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	// Process entries - generate embeddings if needed
	embeddingModel := db.config.EmbeddingModel
	if embeddingModel == "" {
		embeddingModel = "qwen3"
	}
	embeddingDim := db.config.EmbeddingDim
	if embeddingDim == 0 {
		embeddingDim = 768
	}

	validEntries := 0
	for i := range entries {
		entry := &entries[i]

		// Generate embedding if not provided
		if len(entry.Embedding) == 0 && entry.QueryText != "" {
			output, err := candle_binding.GetEmbeddingWithModelType(entry.QueryText, embeddingModel, embeddingDim)
			if err != nil {
				logging.Warnf("Failed to generate embedding for entry %d: %v", i, err)
				continue
			}
			entry.Embedding = output.Embedding
		}

		// Validate entry has required fields
		if len(entry.Embedding) == 0 {
			logging.Warnf("Entry %d has no embedding and no query_text, skipping", i)
			continue
		}
		if len(entry.ModelScores) == 0 {
			logging.Warnf("Entry %d has no model scores, skipping", i)
			continue
		}

		db.entries = append(db.entries, *entry)
		validEntries++
	}

	logging.Infof("Loaded %d experience entries from %s (%d total in file)",
		validEntries, filePath, len(entries))

	return nil
}

// SaveToFile saves the experience database to a JSON file
func (db *ExperienceDatabase) SaveToFile(filePath string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	data, err := json.MarshalIndent(db.entries, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal experience database: %w", err)
	}

	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write experience database file: %w", err)
	}

	logging.Infof("Saved %d experience entries to %s", len(db.entries), filePath)
	return nil
}

// AddEntry adds a new experience entry to the database
//
// Parameters:
//   - queryText: The query text
//   - modelScores: Map of model names to their performance scores
//   - metadata: Optional metadata (can be nil)
func (db *ExperienceDatabase) AddEntry(queryText string, modelScores map[string]float32, metadata map[string]string) error {
	if queryText == "" && len(modelScores) == 0 {
		return fmt.Errorf("query text and model scores cannot both be empty")
	}

	// Generate embedding
	embeddingModel := db.config.EmbeddingModel
	if embeddingModel == "" {
		embeddingModel = "qwen3"
	}
	embeddingDim := db.config.EmbeddingDim
	if embeddingDim == 0 {
		embeddingDim = 768
	}

	output, err := candle_binding.GetEmbeddingWithModelType(queryText, embeddingModel, embeddingDim)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	entry := ExperienceEntry{
		QueryText:   queryText,
		Embedding:   output.Embedding,
		ModelScores: modelScores,
		Metadata:    metadata,
	}

	db.mu.Lock()
	db.entries = append(db.entries, entry)
	db.mu.Unlock()

	return nil
}

// AddEntryWithEmbedding adds a new experience entry with a pre-computed embedding
func (db *ExperienceDatabase) AddEntryWithEmbedding(embedding []float32, modelScores map[string]float32, metadata map[string]string) error {
	if len(embedding) == 0 {
		return fmt.Errorf("embedding cannot be empty")
	}
	if len(modelScores) == 0 {
		return fmt.Errorf("model scores cannot be empty")
	}

	entry := ExperienceEntry{
		Embedding:   embedding,
		ModelScores: modelScores,
		Metadata:    metadata,
	}

	db.mu.Lock()
	db.entries = append(db.entries, entry)
	db.mu.Unlock()

	return nil
}

// GetEntries returns all experience entries
func (db *ExperienceDatabase) GetEntries() []ExperienceEntry {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Return a copy to prevent external modification
	result := make([]ExperienceEntry, len(db.entries))
	copy(result, db.entries)
	return result
}

// GetEntryCount returns the number of entries in the database
func (db *ExperienceDatabase) GetEntryCount() int {
	db.mu.RLock()
	defer db.mu.RUnlock()
	return len(db.entries)
}

// ToExperienceRecords converts the database entries to candle_binding.ExperienceRecord format
// This is used for training the cluster router
func (db *ExperienceDatabase) ToExperienceRecords() []candle_binding.ExperienceRecord {
	db.mu.RLock()
	defer db.mu.RUnlock()

	records := make([]candle_binding.ExperienceRecord, 0, len(db.entries))
	for _, entry := range db.entries {
		if len(entry.Embedding) > 0 && len(entry.ModelScores) > 0 {
			records = append(records, candle_binding.ExperienceRecord{
				Embedding:   entry.Embedding,
				ModelScores: entry.ModelScores,
			})
		}
	}
	return records
}

// Clear removes all entries from the database
func (db *ExperienceDatabase) Clear() {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.entries = []ExperienceEntry{}
}

// GetStatistics returns statistics about the experience database
func (db *ExperienceDatabase) GetStatistics() map[string]interface{} {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Count model occurrences and total scores
	modelCounts := make(map[string]int)
	modelTotalScores := make(map[string]float32)

	for _, entry := range db.entries {
		for model, score := range entry.ModelScores {
			modelCounts[model]++
			modelTotalScores[model] += score
		}
	}

	// Calculate average scores
	modelAvgScores := make(map[string]float32)
	for model, total := range modelTotalScores {
		if count := modelCounts[model]; count > 0 {
			modelAvgScores[model] = total / float32(count)
		}
	}

	return map[string]interface{}{
		"total_entries":    len(db.entries),
		"model_counts":     modelCounts,
		"model_avg_scores": modelAvgScores,
	}
}

// LoadExperienceDBAndTrainRouter is a convenience function that loads an experience database
// from a file and trains the cluster router
func LoadExperienceDBAndTrainRouter(dbPath string, cfg *config.ClusterRouterConfig, router *ClusterRouter) error {
	if dbPath == "" {
		return fmt.Errorf("experience database path not configured")
	}

	// Create and load experience database
	expDB := NewExperienceDatabase(cfg)
	if err := expDB.LoadFromFile(dbPath); err != nil {
		return fmt.Errorf("failed to load experience database: %w", err)
	}

	if expDB.GetEntryCount() == 0 {
		return fmt.Errorf("experience database is empty")
	}

	// Convert to experience records
	records := expDB.ToExperienceRecords()

	// Train the router
	if err := router.Train(records); err != nil {
		return fmt.Errorf("failed to train cluster router: %w", err)
	}

	logging.Infof("Cluster router trained with %d experience records from %s",
		len(records), dbPath)

	return nil
}


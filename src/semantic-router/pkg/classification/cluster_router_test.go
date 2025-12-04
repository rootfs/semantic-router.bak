package classification

import (
	"encoding/json"
	"os"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestClusterRouterWithExperienceDB validates the cluster router using the tutorial experience database
// Note: This test requires the embedding model to be initialized, so it may skip if not available
func TestClusterRouterWithExperienceDB(t *testing.T) {
	// Skip this test in normal CI - it requires the embedding model
	// To run: go test -v -run TestClusterRouterWithExperienceDB -tags=integration
	t.Skip("Requires embedding model initialization - use TestClusterRouterTrainAndRoute for validation")
}

// TestClusterRouterTrainAndRoute tests the full training and routing pipeline
func TestClusterRouterTrainAndRoute(t *testing.T) {
	// Skip if already initialized (singleton limitation)
	if candle_binding.IsClusterRouterInitialized() {
		t.Skip("Cluster router already initialized from previous test")
	}

	// Create synthetic experience data for testing
	// This avoids needing the actual embedding model
	embeddingDim := 768
	numSamples := 20

	var experienceData []candle_binding.ExperienceRecord

	// Create math-like queries (high scores for math model)
	for i := 0; i < numSamples/2; i++ {
		embedding := make([]float32, embeddingDim)
		// Math queries cluster around [1, 0, 0, ...]
		embedding[0] = 0.9 + float32(i)*0.01
		embedding[1] = 0.1
		embedding[2] = 0.1

		experienceData = append(experienceData, candle_binding.ExperienceRecord{
			Embedding: embedding,
			ModelScores: map[string]float32{
				"math":    1.0, // Math model wins
				"coder":   0.5,
				"general": 0.3,
			},
		})
	}

	// Create coder-like queries (high scores for coder model)
	for i := 0; i < numSamples/2; i++ {
		embedding := make([]float32, embeddingDim)
		// Coder queries cluster around [0, 1, 0, ...]
		embedding[0] = 0.1
		embedding[1] = 0.9 + float32(i)*0.01
		embedding[2] = 0.1

		experienceData = append(experienceData, candle_binding.ExperienceRecord{
			Embedding: embedding,
			ModelScores: map[string]float32{
				"math":    0.3,
				"coder":   1.0, // Coder model wins
				"general": 0.5,
			},
		})
	}

	// Model costs
	modelCosts := map[string]float32{
		"math":    7.0,
		"coder":   7.0,
		"general": 14.0,
	}

	// Config
	routerConfig := candle_binding.ClusterRouterConfig{
		NClusters:     2, // Should separate math vs coder
		MaxIterations: 50,
		Alpha:         1.0, // Performance only
		UseCPU:        true,
		TopK:          1,
		Beta:          9.0,
	}

	// Initialize router
	err := candle_binding.InitClusterRouter(experienceData, modelCosts, routerConfig)
	if err != nil {
		t.Fatalf("Failed to initialize cluster router: %v", err)
	}

	t.Logf("Cluster router initialized with %d clusters", candle_binding.GetClusterCount())

	// Test routing a math-like query
	mathQuery := make([]float32, embeddingDim)
	mathQuery[0] = 0.95
	mathQuery[1] = 0.1
	mathQuery[2] = 0.1

	mathResult, err := candle_binding.RouteQuery(mathQuery)
	if err != nil {
		t.Fatalf("Failed to route math query: %v", err)
	}

	t.Logf("Math query routed to: %s (confidence: %.3f, cluster: %d)",
		mathResult.ModelName, mathResult.Confidence, mathResult.ClusterID)

	// Test routing a coder-like query
	coderQuery := make([]float32, embeddingDim)
	coderQuery[0] = 0.1
	coderQuery[1] = 0.95
	coderQuery[2] = 0.1

	coderResult, err := candle_binding.RouteQuery(coderQuery)
	if err != nil {
		t.Fatalf("Failed to route coder query: %v", err)
	}

	t.Logf("Coder query routed to: %s (confidence: %.3f, cluster: %d)",
		coderResult.ModelName, coderResult.Confidence, coderResult.ClusterID)

	// Verify routing makes sense
	if mathResult.ModelName != "math" {
		t.Logf("Warning: Math query routed to %s instead of math", mathResult.ModelName)
	}
	if coderResult.ModelName != "coder" {
		t.Logf("Warning: Coder query routed to %s instead of coder", coderResult.ModelName)
	}

	// Test export
	exportPath := "/tmp/cluster_router_test"
	err = candle_binding.ExportClusterRouter(exportPath)
	if err != nil {
		t.Fatalf("Failed to export cluster router: %v", err)
	}
	t.Logf("Exported cluster router to %s", exportPath)

	// Verify export files exist
	if _, err := os.Stat(exportPath + "/metadata.json"); os.IsNotExist(err) {
		t.Error("metadata.json not found after export")
	}
	if _, err := os.Stat(exportPath + "/cluster_centers.bin"); os.IsNotExist(err) {
		t.Error("cluster_centers.bin not found after export")
	}

	// Read and log metadata
	metadataBytes, err := os.ReadFile(exportPath + "/metadata.json")
	if err == nil {
		var metadata map[string]interface{}
		json.Unmarshal(metadataBytes, &metadata)
		t.Logf("Exported metadata: n_clusters=%v, embedding_dim=%v",
			metadata["n_clusters"], metadata["embedding_dim"])
	}
}

// TestExperienceDatabase tests the experience database operations
func TestExperienceDatabase(t *testing.T) {
	cfg := &config.ClusterRouterConfig{
		EmbeddingModel: "qwen3",
		EmbeddingDim:   768,
	}

	db := NewExperienceDatabase(cfg)

	// Test adding entry with embedding
	embedding := make([]float32, 768)
	embedding[0] = 1.0

	err := db.AddEntryWithEmbedding(
		embedding,
		map[string]float32{"math": 1.0, "coder": 0.5},
		map[string]string{"source": "test"},
	)
	if err != nil {
		t.Fatalf("Failed to add entry: %v", err)
	}

	if db.GetEntryCount() != 1 {
		t.Errorf("Expected 1 entry, got %d", db.GetEntryCount())
	}

	// Test get entries
	entries := db.GetEntries()
	if len(entries) != 1 {
		t.Errorf("Expected 1 entry, got %d", len(entries))
	}

	// Test save and load
	tmpFile := "/tmp/test_experience_db.json"
	err = db.SaveToFile(tmpFile)
	if err != nil {
		t.Fatalf("Failed to save: %v", err)
	}

	db2 := NewExperienceDatabase(cfg)
	err = db2.LoadFromFile(tmpFile)
	if err != nil {
		t.Fatalf("Failed to load: %v", err)
	}

	if db2.GetEntryCount() != 1 {
		t.Errorf("Expected 1 entry after load, got %d", db2.GetEntryCount())
	}

	// Test statistics
	stats := db.GetStatistics()
	if stats["total_entries"].(int) != 1 {
		t.Errorf("Expected total_entries=1, got %v", stats["total_entries"])
	}

	// Cleanup
	os.Remove(tmpFile)
}


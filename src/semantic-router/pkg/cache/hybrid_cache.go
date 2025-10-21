//go:build !windows && cgo
// +build !windows,cgo

package cache

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// HybridCache combines in-memory HNSW index with external Milvus storage
// Architecture:
//   - In-memory: HNSW index with ALL embeddings (for fast O(log n) search)
//   - Milvus: ALL documents (fetched by ID after search)
// This provides fast search while supporting millions of entries without storing docs in memory
type HybridCache struct {
	// In-memory components (search only)
	hnswIndex  *HNSWIndex
	embeddings [][]float32
	idMap      map[int]string // Entry index → Milvus ID

	// External storage (all documents)
	milvusCache *MilvusCache

	// Configuration
	similarityThreshold float32
	maxMemoryEntries    int // Max entries in HNSW index
	ttlSeconds          int
	enabled             bool

	// Statistics
	hitCount   int64
	missCount  int64
	evictCount int64

	// Concurrency control
	mu sync.RWMutex
}

// HybridCacheOptions contains configuration for the hybrid cache
type HybridCacheOptions struct {
	// Core settings
	Enabled             bool
	SimilarityThreshold float32
	TTLSeconds          int

	// HNSW settings
	MaxMemoryEntries   int // Max entries in HNSW (default: 100,000)
	HNSWM              int // HNSW M parameter
	HNSWEfConstruction int // HNSW efConstruction parameter

	// Milvus settings
	MilvusConfigPath string
}

// NewHybridCache creates a new hybrid cache instance
func NewHybridCache(options HybridCacheOptions) (*HybridCache, error) {
	observability.Infof("Initializing hybrid cache: enabled=%t, maxMemoryEntries=%d, threshold=%.3f",
		options.Enabled, options.MaxMemoryEntries, options.SimilarityThreshold)

	if !options.Enabled {
		observability.Debugf("Hybrid cache disabled, returning inactive instance")
		return &HybridCache{
			enabled: false,
		}, nil
	}

	// Initialize Milvus backend
	milvusOptions := MilvusCacheOptions{
		Enabled:             true,
		SimilarityThreshold: options.SimilarityThreshold,
		TTLSeconds:          options.TTLSeconds,
		ConfigPath:          options.MilvusConfigPath,
	}

	milvusCache, err := NewMilvusCache(milvusOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Milvus backend: %w", err)
	}

	// Set defaults
	if options.MaxMemoryEntries <= 0 {
		options.MaxMemoryEntries = 100000 // Default: 100K entries in memory
	}
	if options.HNSWM <= 0 {
		options.HNSWM = 16
	}
	if options.HNSWEfConstruction <= 0 {
		options.HNSWEfConstruction = 200
	}

	// Initialize HNSW index
	hnswIndex := newHNSWIndex(options.HNSWM, options.HNSWEfConstruction)

	cache := &HybridCache{
		hnswIndex:           hnswIndex,
		embeddings:          make([][]float32, 0, options.MaxMemoryEntries),
		idMap:               make(map[int]string),
		milvusCache:         milvusCache,
		similarityThreshold: options.SimilarityThreshold,
		maxMemoryEntries:    options.MaxMemoryEntries,
		ttlSeconds:          options.TTLSeconds,
		enabled:             true,
	}

	observability.Infof("Hybrid cache initialized: HNSW(M=%d, ef=%d), maxMemory=%d",
		options.HNSWM, options.HNSWEfConstruction, options.MaxMemoryEntries)

	return cache, nil
}

// IsEnabled returns whether the cache is active
func (h *HybridCache) IsEnabled() bool {
	return h.enabled
}

// AddPendingRequest stores a request awaiting its response
func (h *HybridCache) AddPendingRequest(requestID string, model string, query string, requestBody []byte) error {
	start := time.Now()

	if !h.enabled {
		return nil
	}

	// Generate embedding
	embedding, err := candle_binding.GetEmbedding(query, 0)
	if err != nil {
		metrics.RecordCacheOperation("hybrid", "add_pending", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Store in Milvus (write-through)
	if err := h.milvusCache.AddPendingRequest(requestID, model, query, requestBody); err != nil {
		metrics.RecordCacheOperation("hybrid", "add_pending", "error", time.Since(start).Seconds())
		return fmt.Errorf("milvus add pending failed: %w", err)
	}

	// Add to in-memory HNSW index
	h.mu.Lock()
	defer h.mu.Unlock()

	// Check if we need to evict
	if len(h.embeddings) >= h.maxMemoryEntries {
		h.evictOneUnsafe()
	}

	// Add to HNSW
	entryIndex := len(h.embeddings)
	h.embeddings = append(h.embeddings, embedding)
	h.idMap[entryIndex] = requestID
	h.addNodeHybrid(entryIndex, embedding)

	observability.Debugf("HybridCache.AddPendingRequest: added to HNSW index=%d, milvusID=%s",
		entryIndex, requestID)

	metrics.RecordCacheOperation("hybrid", "add_pending", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("hybrid", len(h.embeddings))

	return nil
}

// UpdateWithResponse completes a pending request with its response
func (h *HybridCache) UpdateWithResponse(requestID string, responseBody []byte) error {
	start := time.Now()

	if !h.enabled {
		return nil
	}

	// Update in Milvus
	if err := h.milvusCache.UpdateWithResponse(requestID, responseBody); err != nil {
		metrics.RecordCacheOperation("hybrid", "update_response", "error", time.Since(start).Seconds())
		return fmt.Errorf("milvus update failed: %w", err)
	}

	// HNSW index already has the embedding, no update needed there

	observability.Debugf("HybridCache.UpdateWithResponse: updated milvusID=%s", requestID)
	metrics.RecordCacheOperation("hybrid", "update_response", "success", time.Since(start).Seconds())

	return nil
}

// AddEntry stores a complete request-response pair
func (h *HybridCache) AddEntry(requestID string, model string, query string, requestBody, responseBody []byte) error {
	start := time.Now()

	if !h.enabled {
		return nil
	}

	// Generate embedding
	embedding, err := candle_binding.GetEmbedding(query, 0)
	if err != nil {
		metrics.RecordCacheOperation("hybrid", "add_entry", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Store in Milvus (write-through)
	if err := h.milvusCache.AddEntry(requestID, model, query, requestBody, responseBody); err != nil {
		metrics.RecordCacheOperation("hybrid", "add_entry", "error", time.Since(start).Seconds())
		return fmt.Errorf("milvus add entry failed: %w", err)
	}

	// Add to in-memory HNSW index
	h.mu.Lock()
	defer h.mu.Unlock()

	// Check if we need to evict
	if len(h.embeddings) >= h.maxMemoryEntries {
		h.evictOneUnsafe()
	}

	// Add to HNSW
	entryIndex := len(h.embeddings)
	h.embeddings = append(h.embeddings, embedding)
	h.idMap[entryIndex] = requestID
	h.addNodeHybrid(entryIndex, embedding)

	observability.Debugf("HybridCache.AddEntry: added to HNSW index=%d, milvusID=%s",
		entryIndex, requestID)
	observability.LogEvent("hybrid_cache_entry_added", map[string]interface{}{
		"backend": "hybrid",
		"query":   query,
		"model":   model,
		"in_hnsw": true,
	})

	metrics.RecordCacheOperation("hybrid", "add_entry", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("hybrid", len(h.embeddings))

	return nil
}

// AddEntriesBatch stores multiple request-response pairs efficiently
func (h *HybridCache) AddEntriesBatch(entries []CacheEntry) error {
	start := time.Now()

	if !h.enabled {
		return nil
	}

	if len(entries) == 0 {
		return nil
	}

	observability.Debugf("HybridCache.AddEntriesBatch: adding %d entries in batch", len(entries))

	// Generate all embeddings first
	embeddings := make([][]float32, len(entries))
	for i, entry := range entries {
		embedding, err := candle_binding.GetEmbedding(entry.Query, 0)
		if err != nil {
			metrics.RecordCacheOperation("hybrid", "add_entries_batch", "error", time.Since(start).Seconds())
			return fmt.Errorf("failed to generate embedding for entry %d: %w", i, err)
		}
		embeddings[i] = embedding
	}

	// Store all in Milvus at once (write-through)
	if err := h.milvusCache.AddEntriesBatch(entries); err != nil {
		metrics.RecordCacheOperation("hybrid", "add_entries_batch", "error", time.Since(start).Seconds())
		return fmt.Errorf("milvus batch add failed: %w", err)
	}

	// Add all to in-memory HNSW index
	h.mu.Lock()
	defer h.mu.Unlock()

	for i, entry := range entries {
		// Check if we need to evict
		if len(h.embeddings) >= h.maxMemoryEntries {
			h.evictOneUnsafe()
		}

		// Add to HNSW
		entryIndex := len(h.embeddings)
		h.embeddings = append(h.embeddings, embeddings[i])
		h.idMap[entryIndex] = entry.RequestID
		h.addNodeHybrid(entryIndex, embeddings[i])
	}

	elapsed := time.Since(start)
	observability.Debugf("HybridCache.AddEntriesBatch: added %d entries in %v (%.0f entries/sec)",
		len(entries), elapsed, float64(len(entries))/elapsed.Seconds())
	observability.LogEvent("hybrid_cache_entries_added", map[string]interface{}{
		"backend": "hybrid",
		"count":   len(entries),
		"in_hnsw": true,
	})

	metrics.RecordCacheOperation("hybrid", "add_entries_batch", "success", elapsed.Seconds())
	metrics.UpdateCacheEntries("hybrid", len(h.embeddings))

	return nil
}

// Flush forces Milvus to persist all buffered data to disk
func (h *HybridCache) Flush() error {
	if !h.enabled {
		return nil
	}
	
	return h.milvusCache.Flush()
}

// FindSimilar searches for semantically similar cached requests
func (h *HybridCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	start := time.Now()

	if !h.enabled {
		return nil, false, nil
	}

	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}
	observability.Debugf("HybridCache.FindSimilar: searching for model='%s', query='%s'",
		model, queryPreview)

	// Generate query embedding
	queryEmbedding, err := candle_binding.GetEmbedding(query, 0)
	if err != nil {
		metrics.RecordCacheOperation("hybrid", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Phase 1: Search HNSW index for candidates (with similarity scores!)
	h.mu.RLock()
	candidates := h.searchKNNHybrid(queryEmbedding, 10, 50)
	
	// Filter by similarity threshold BEFORE fetching from Milvus
	var qualifiedCandidates []searchResult
	for _, candidate := range candidates {
		if candidate.similarity >= h.similarityThreshold {
			qualifiedCandidates = append(qualifiedCandidates, candidate)
		}
	}
	
	// Map qualified candidates to Milvus IDs
	type candidateWithID struct {
		milvusID   string
		similarity float32
		index      int
	}
	
	candidatesWithIDs := make([]candidateWithID, 0, len(qualifiedCandidates))
	for _, candidate := range qualifiedCandidates {
		if milvusID, ok := h.idMap[candidate.index]; ok {
			candidatesWithIDs = append(candidatesWithIDs, candidateWithID{
				milvusID:   milvusID,
				similarity: candidate.similarity,
				index:      candidate.index,
			})
		}
	}
	h.mu.RUnlock()

	if len(candidatesWithIDs) == 0 {
		atomic.AddInt64(&h.missCount, 1)
		if len(candidates) > 0 {
			observability.Debugf("HybridCache.FindSimilar: %d candidates found but none above threshold %.3f",
				len(candidates), h.similarityThreshold)
		} else {
			observability.Debugf("HybridCache.FindSimilar: no candidates found in HNSW")
		}
		metrics.RecordCacheOperation("hybrid", "find_similar", "miss", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	observability.Debugf("HybridCache.FindSimilar: HNSW returned %d candidates, %d above threshold",
		len(candidates), len(candidatesWithIDs))

	// Phase 2: Fetch from Milvus (only for candidates above threshold!)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Try candidates in order (already sorted by similarity from HNSW)
	for _, candidate := range candidatesWithIDs {
		// Fetch document from Milvus by ID (optimized - no redundant search!)
		fetchCtx, fetchCancel := context.WithTimeout(ctx, 2*time.Second)
		responseBody, err := h.milvusCache.GetByID(fetchCtx, candidate.milvusID)
		fetchCancel()
		
		if err != nil {
			observability.Debugf("HybridCache.FindSimilar: Milvus GetByID failed for %s: %v", 
				candidate.milvusID, err)
			continue
		}

		if responseBody != nil {
			atomic.AddInt64(&h.hitCount, 1)
			observability.Debugf("HybridCache.FindSimilar: MILVUS HIT - similarity=%.4f (threshold=%.3f)",
				candidate.similarity, h.similarityThreshold)
			observability.LogEvent("hybrid_cache_hit", map[string]interface{}{
				"backend":     "hybrid",
				"source":      "milvus",
				"similarity":  candidate.similarity,
				"threshold":   h.similarityThreshold,
				"model":       model,
				"latency_ms":  time.Since(start).Milliseconds(),
			})
			metrics.RecordCacheOperation("hybrid", "find_similar", "hit_milvus", time.Since(start).Seconds())
			metrics.RecordCacheHit()
			return responseBody, true, nil
		}
	}

	// No match found above threshold
	atomic.AddInt64(&h.missCount, 1)
	observability.Debugf("HybridCache.FindSimilar: CACHE MISS - no match above threshold")
	observability.LogEvent("hybrid_cache_miss", map[string]interface{}{
		"backend":        "hybrid",
		"threshold":      h.similarityThreshold,
		"model":          model,
		"candidates":     len(candidatesWithIDs),
	})
	metrics.RecordCacheOperation("hybrid", "find_similar", "miss", time.Since(start).Seconds())
	metrics.RecordCacheMiss()

	// Suppress context error to avoid noise
	_ = ctx

	return nil, false, nil
}

// Close releases all resources
func (h *HybridCache) Close() error {
	if !h.enabled {
		return nil
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// Close Milvus connection
	if h.milvusCache != nil {
		if err := h.milvusCache.Close(); err != nil {
			observability.Debugf("HybridCache.Close: Milvus close error: %v", err)
		}
	}

	// Clear in-memory structures
	h.embeddings = nil
	h.idMap = nil
	h.hnswIndex = nil

	metrics.UpdateCacheEntries("hybrid", 0)

	return nil
}

// GetStats returns cache statistics
func (h *HybridCache) GetStats() CacheStats {
	h.mu.RLock()
	defer h.mu.RUnlock()

	hits := atomic.LoadInt64(&h.hitCount)
	misses := atomic.LoadInt64(&h.missCount)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	return CacheStats{
		TotalEntries: len(h.embeddings),
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}
}

// Helper methods

// evictOneUnsafe removes one entry from HNSW index (must hold write lock)
func (h *HybridCache) evictOneUnsafe() {
	if len(h.embeddings) == 0 {
		return
	}

	// Simple FIFO eviction: remove oldest entry
	victimIdx := 0

	// Could use LRU/LFU here by tracking access times/counts
	// For now, just evict the first entry

	// Get milvusID before removing from map (for logging)
	milvusID := h.idMap[victimIdx]

	// Remove from structures
	delete(h.idMap, victimIdx)

	// Note: We don't remove from Milvus (data persists there)
	// We also don't rebuild HNSW (mark as stale)
	h.hnswIndex.markStale()

	atomic.AddInt64(&h.evictCount, 1)

	observability.LogEvent("hybrid_cache_evicted", map[string]interface{}{
		"backend":     "hybrid",
		"milvus_id":   milvusID,
		"hnsw_index":  victimIdx,
		"max_entries": h.maxMemoryEntries,
	})
}

// searchResult holds a candidate with its similarity score
type searchResult struct {
	index      int
	similarity float32
}

// dotProduct calculates dot product between two vectors
func dotProduct(a, b []float32) float32 {
	var sum float32
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}
	for i := 0; i < minLen; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// hybridHNSWAdapter adapts the HNSW index to work with [][]float32 instead of []CacheEntry
type hybridHNSWAdapter struct {
	embeddings [][]float32
}

func (h *hybridHNSWAdapter) getEmbedding(idx int) []float32 {
	if idx < 0 || idx >= len(h.embeddings) {
		return nil
	}
	return h.embeddings[idx]
}

func (h *hybridHNSWAdapter) distance(idx1, idx2 int) float32 {
	emb1 := h.getEmbedding(idx1)
	emb2 := h.getEmbedding(idx2)
	if emb1 == nil || emb2 == nil {
		return 0
	}
	return dotProduct(emb1, emb2)
}

// addNodeHybrid adds a node to the HNSW index (hybrid version)
func (h *HybridCache) addNodeHybrid(entryIndex int, embedding []float32) {
	// Lock is already held by caller (mu.Lock())

	level := h.selectLevelHybrid()
	node := &HNSWNode{
		entryIndex: entryIndex,
		neighbors:  make(map[int][]int),
		maxLayer:   level,
	}

	for i := 0; i <= level; i++ {
		node.neighbors[i] = make([]int, 0)
	}

	h.hnswIndex.nodes = append(h.hnswIndex.nodes, node)

	if h.hnswIndex.entryPoint == -1 {
		h.hnswIndex.entryPoint = entryIndex
		h.hnswIndex.maxLayer = level
		return
	}

	// Find nearest neighbors at each layer
	adapter := &hybridHNSWAdapter{embeddings: h.embeddings}
	
	// Start from top layer
	currNearest := h.hnswIndex.entryPoint
	for lc := h.hnswIndex.maxLayer; lc > level; lc-- {
		// Search for nearest at this layer
		candidates := []int{currNearest}
		for i := range h.hnswIndex.nodes {
			hn := h.hnswIndex.nodes[i]
			if hn.entryIndex == currNearest && hn.neighbors[lc] != nil {
				for _, neighbor := range hn.neighbors[lc] {
					if neighbor >= 0 && neighbor < len(h.embeddings) {
						candidates = append(candidates, neighbor)
					}
				}
				break
			}
		}

		// Find closest
		bestDist := adapter.distance(entryIndex, currNearest)
		for _, candidate := range candidates {
			dist := adapter.distance(entryIndex, candidate)
			if dist > bestDist {
				bestDist = dist
				currNearest = candidate
			}
		}
	}

	// Insert at appropriate layers
	for lc := level; lc >= 0; lc-- {
		// Find neighbors at this layer
		neighbors := h.searchLayerHybrid(embedding, h.hnswIndex.efConstruction, lc, []int{currNearest})
		
		m := h.hnswIndex.M
		if lc == 0 {
			m = h.hnswIndex.Mmax0
		}
		
		selectedNeighbors := h.selectNeighborsHybrid(neighbors, m)
		
		// Add bidirectional links
		for _, neighborID := range selectedNeighbors {
			node.neighbors[lc] = append(node.neighbors[lc], neighborID)
			
			// Add reverse link
			for i := range h.hnswIndex.nodes {
				if h.hnswIndex.nodes[i].entryIndex == neighborID {
					if h.hnswIndex.nodes[i].neighbors[lc] == nil {
						h.hnswIndex.nodes[i].neighbors[lc] = make([]int, 0)
					}
					h.hnswIndex.nodes[i].neighbors[lc] = append(h.hnswIndex.nodes[i].neighbors[lc], entryIndex)
					break
				}
			}
		}
	}

	if level > h.hnswIndex.maxLayer {
		h.hnswIndex.maxLayer = level
		h.hnswIndex.entryPoint = entryIndex
	}
}

// selectLevelHybrid randomly selects a level for a new node
func (h *HybridCache) selectLevelHybrid() int {
	// Use exponential decay to select level
	// Most nodes at layer 0, fewer at higher layers
	level := 0
	for level < 16 { // Max 16 layers
		if randFloat() > h.hnswIndex.ml {
			break
		}
		level++
	}
	return level
}

// randFloat returns a random float between 0 and 1
func randFloat() float64 {
	// Simple random using time-based seed
	return float64(time.Now().UnixNano()%1000) / 1000.0
}

// searchLayerHybrid searches for nearest neighbors at a specific layer (hybrid version)
func (h *HybridCache) searchLayerHybrid(query []float32, ef int, layer int, entryPoints []int) []int {
	visited := make(map[int]bool)
	candidates := newMinHeap()
	results := newMaxHeap()

	for _, ep := range entryPoints {
		if ep < 0 || ep >= len(h.embeddings) {
			continue
		}
		dist := -dotProduct(query, h.embeddings[ep])
		candidates.push(ep, dist)
		results.push(ep, dist)
		visited[ep] = true
	}

	for len(candidates.data) > 0 {
		currentIdx, currentDist := candidates.pop()
		if len(results.data) > 0 && currentDist > -results.data[0].dist {
			break
		}

		// Find the node with this entryIndex
		var currentNode *HNSWNode
		for i := range h.hnswIndex.nodes {
			if h.hnswIndex.nodes[i].entryIndex == currentIdx {
				currentNode = h.hnswIndex.nodes[i]
				break
			}
		}

		if currentNode == nil || currentNode.neighbors[layer] == nil {
			continue
		}

		for _, neighborID := range currentNode.neighbors[layer] {
			if visited[neighborID] || neighborID < 0 || neighborID >= len(h.embeddings) {
				continue
			}
			visited[neighborID] = true

			dist := -dotProduct(query, h.embeddings[neighborID])
			
			if len(results.data) < ef || dist < -results.data[0].dist {
				candidates.push(neighborID, dist)
				results.push(neighborID, dist)
				
				if len(results.data) > ef {
					results.pop()
				}
			}
		}
	}

	// Extract IDs
	resultIDs := make([]int, 0, len(results.data))
	for len(results.data) > 0 {
		idx, _ := results.pop()
		resultIDs = append([]int{idx}, resultIDs...)
	}

	return resultIDs
}

// selectNeighborsHybrid selects the best neighbors from candidates (hybrid version)
func (h *HybridCache) selectNeighborsHybrid(candidates []int, m int) []int {
	if len(candidates) <= m {
		return candidates
	}
	
	// Simple selection: take first M candidates
	return candidates[:m]
}

// searchKNNHybrid searches for k nearest neighbors (hybrid version)
func (h *HybridCache) searchKNNHybrid(query []float32, k int, ef int) []searchResult {
	// Lock is already held by caller (mu.RLock())

	if h.hnswIndex.entryPoint == -1 || len(h.embeddings) == 0 {
		return nil
	}

	// Search from top layer (just get indices for navigation)
	currNearest := []int{h.hnswIndex.entryPoint}
	
	for lc := h.hnswIndex.maxLayer; lc > 0; lc-- {
		currNearest = h.searchLayerHybrid(query, 1, lc, currNearest)
	}

	// Search at layer 0 with ef - this returns candidates with distances
	candidateIndices := h.searchLayerHybrid(query, ef, 0, currNearest)

	// Convert to searchResults with similarity scores
	results := make([]searchResult, 0, len(candidateIndices))
	for _, idx := range candidateIndices {
		if idx >= 0 && idx < len(h.embeddings) {
			similarity := dotProduct(query, h.embeddings[idx])
			results = append(results, searchResult{
				index:      idx,
				similarity: similarity,
			})
		}
	}

	// Results are already sorted by similarity (from searchLayer)
	// Return top k
	if len(results) > k {
		return results[:k]
	}
	return results
}


//go:build !windows && cgo
// +build !windows,cgo

package cache

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	usearch "github.com/unum-cloud/usearch/golang"
	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// USearchHNSWIndex wraps USearch for fast ANN search
type USearchHNSWIndex struct {
	index          *usearch.Index
	mu             sync.RWMutex
	dimensions     uint
	efConstruction int
	efSearch       int
	M              int
}

// InMemoryCache provides a high-performance semantic cache using BERT embeddings in memory
type InMemoryCache struct {
	entries             []CacheEntry
	mu                  sync.RWMutex
	similarityThreshold float32
	maxEntries          int
	ttlSeconds          int
	enabled             bool
	hitCount            int64
	missCount           int64
	lastCleanupTime     *time.Time
	evictionPolicy      EvictionPolicy
	hnswIndex           *USearchHNSWIndex
	useHNSW             bool
	hnswEfSearch        int // Search-time ef parameter
}

// InMemoryCacheOptions contains configuration parameters for the in-memory cache
type InMemoryCacheOptions struct {
	SimilarityThreshold float32
	MaxEntries          int
	TTLSeconds          int
	Enabled             bool
	EvictionPolicy      EvictionPolicyType
	UseHNSW             bool // Enable HNSW index for faster search
	HNSWM               int  // Number of bi-directional links (default: 16)
	HNSWEfConstruction  int  // Size of dynamic candidate list during construction (default: 200)
	HNSWEfSearch        int  // Size of dynamic candidate list during search (default: 50)
	VectorDimensions    int  // Embedding dimensions (will be auto-detected if 0)
}

// NewInMemoryCache initializes a new in-memory semantic cache instance
func NewInMemoryCache(options InMemoryCacheOptions) *InMemoryCache {
	observability.Debugf("Initializing in-memory cache with USearch: enabled=%t, maxEntries=%d, ttlSeconds=%d, threshold=%.3f, eviction_policy=%s, useHNSW=%t",
		options.Enabled, options.MaxEntries, options.TTLSeconds, options.SimilarityThreshold, options.EvictionPolicy, options.UseHNSW)

	var evictionPolicy EvictionPolicy
	switch options.EvictionPolicy {
	case LRUEvictionPolicyType:
		evictionPolicy = &LRUPolicy{}
	case LFUEvictionPolicyType:
		evictionPolicy = &LFUPolicy{}
	default: // FIFOEvictionPolicyType
		evictionPolicy = &FIFOPolicy{}
	}

	// Set HNSW search ef parameter
	efSearch := options.HNSWEfSearch
	if efSearch <= 0 {
		efSearch = 50 // Default value
	}

	cache := &InMemoryCache{
		entries:             []CacheEntry{},
		similarityThreshold: options.SimilarityThreshold,
		maxEntries:          options.MaxEntries,
		ttlSeconds:          options.TTLSeconds,
		enabled:             options.Enabled,
		evictionPolicy:      evictionPolicy,
		useHNSW:             options.UseHNSW,
		hnswEfSearch:        efSearch,
	}

	// Initialize USearch HNSW index if enabled
	if options.UseHNSW {
		M := options.HNSWM
		if M <= 0 {
			M = 16 // Default value
		}
		efConstruction := options.HNSWEfConstruction
		if efConstruction <= 0 {
			efConstruction = 200 // Default value
		}

		// Note: dimensions will be set on first add
		cache.hnswIndex = newUSearchHNSWIndex(0, M, efConstruction, efSearch)
		observability.Debugf("USearch HNSW index initialized: M=%d, efConstruction=%d, efSearch=%d", M, efConstruction, efSearch)
	}

	return cache
}

// IsEnabled returns the current cache activation status
func (c *InMemoryCache) IsEnabled() bool {
	return c.enabled
}

// AddPendingRequest stores a request that is awaiting its response
func (c *InMemoryCache) AddPendingRequest(requestID string, model string, query string, requestBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Generate semantic embedding for the query
	embedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		metrics.RecordCacheOperation("memory", "add_pending", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Remove expired entries to maintain cache hygiene
	c.cleanupExpiredEntries()

	// Check if eviction is needed before adding the new entry
	if c.maxEntries > 0 && len(c.entries) >= c.maxEntries {
		c.evictOne()
	}

	// Create cache entry for the pending request
	now := time.Now()
	entry := CacheEntry{
		RequestID:    requestID,
		RequestBody:  requestBody,
		Model:        model,
		Query:        query,
		Embedding:    embedding,
		Timestamp:    now,
		LastAccessAt: now,
		HitCount:     0,
	}

	c.entries = append(c.entries, entry)
	entryIndex := len(c.entries) - 1

	// Add to USearch HNSW index if enabled
	if c.useHNSW && c.hnswIndex != nil {
		if err := c.hnswIndex.addVector(entryIndex, embedding); err != nil {
			observability.Debugf("Warning: failed to add vector to USearch index: %v", err)
		}
	}

	observability.Debugf("InMemoryCache.AddPendingRequest: added pending entry (total entries: %d, embedding_dim: %d, useHNSW: %t)",
		len(c.entries), len(embedding), c.useHNSW)

	// Record metrics
	metrics.RecordCacheOperation("memory", "add_pending", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("memory", len(c.entries))

	return nil
}

// UpdateWithResponse completes a pending request by adding the response
func (c *InMemoryCache) UpdateWithResponse(requestID string, responseBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Clean up expired entries during the update
	c.cleanupExpiredEntries()

	// Locate the pending request and complete it
	for i, entry := range c.entries {
		if entry.RequestID == requestID && entry.ResponseBody == nil {
			// Complete the cache entry with the response
			c.entries[i].ResponseBody = responseBody
			c.entries[i].Timestamp = time.Now()
			c.entries[i].LastAccessAt = time.Now()
			observability.Debugf("InMemoryCache.UpdateWithResponse: updated entry with response (response_size: %d bytes)",
				len(responseBody))

			// Record successful completion
			metrics.RecordCacheOperation("memory", "update_response", "success", time.Since(start).Seconds())
			return nil
		}
	}

	// No matching pending request found
	metrics.RecordCacheOperation("memory", "update_response", "error", time.Since(start).Seconds())
	return fmt.Errorf("no pending request found for request ID: %s", requestID)
}

// AddEntry stores a complete request-response pair in the cache
func (c *InMemoryCache) AddEntry(requestID string, model string, query string, requestBody, responseBody []byte) error {
	start := time.Now()

	if !c.enabled {
		return nil
	}

	// Generate semantic embedding for the query
	embedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		metrics.RecordCacheOperation("memory", "add_entry", "error", time.Since(start).Seconds())
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Clean up expired entries before adding new one
	c.cleanupExpiredEntries()

	// Check if eviction is needed before adding the new entry
	if c.maxEntries > 0 && len(c.entries) >= c.maxEntries {
		c.evictOne()
	}

	now := time.Now()
	entry := CacheEntry{
		RequestID:    requestID,
		RequestBody:  requestBody,
		ResponseBody: responseBody,
		Model:        model,
		Query:        query,
		Embedding:    embedding,
		Timestamp:    now,
		LastAccessAt: now,
		HitCount:     0,
	}

	c.entries = append(c.entries, entry)
	entryIndex := len(c.entries) - 1

	// Add to USearch HNSW index if enabled
	if c.useHNSW && c.hnswIndex != nil {
		if err := c.hnswIndex.addVector(entryIndex, embedding); err != nil {
			observability.Debugf("Warning: failed to add vector to USearch index: %v", err)
		}
	}

	observability.Debugf("InMemoryCache.AddEntry: added complete entry (total entries: %d, request_size: %d, response_size: %d, useHNSW: %t)",
		len(c.entries), len(requestBody), len(responseBody), c.useHNSW)
	observability.LogEvent("cache_entry_added", map[string]interface{}{
		"backend": "memory",
		"query":   query,
		"model":   model,
		"useHNSW": c.useHNSW,
	})

	// Record success metrics
	metrics.RecordCacheOperation("memory", "add_entry", "success", time.Since(start).Seconds())
	metrics.UpdateCacheEntries("memory", len(c.entries))

	return nil
}

// FindSimilar searches for semantically similar cached requests
func (c *InMemoryCache) FindSimilar(model string, query string) ([]byte, bool, error) {
	start := time.Now()

	if !c.enabled {
		observability.Debugf("InMemoryCache.FindSimilar: cache disabled")
		return nil, false, nil
	}
	queryPreview := query
	if len(query) > 50 {
		queryPreview = query[:50] + "..."
	}
	observability.Debugf("InMemoryCache.FindSimilar: searching for model='%s', query='%s' (len=%d chars)",
		model, queryPreview, len(query))

	// Generate semantic embedding for similarity comparison
	queryEmbedding, err := candle_binding.GetEmbedding(query, 0) // Auto-detect dimension
	if err != nil {
		metrics.RecordCacheOperation("memory", "find_similar", "error", time.Since(start).Seconds())
		return nil, false, fmt.Errorf("failed to generate embedding: %w", err)
	}

	c.mu.RLock()
	var (
		bestIndex      = -1
		bestEntry      CacheEntry
		bestSimilarity float32
		entriesChecked int
		expiredCount   int
	)
	// Capture the lookup time after acquiring the read lock so TTL checks aren't skewed by embedding work or lock wait
	now := time.Now()

	// Use USearch HNSW index for fast search if enabled
	if c.useHNSW && c.hnswIndex != nil && c.hnswIndex.size() > 0 {
		// Search using USearch index with configured ef parameter
		candidateIndices, distances, err := c.hnswIndex.search(queryEmbedding, 10)
		if err != nil {
			observability.Debugf("Warning: USearch search failed, falling back to linear: %v", err)
			c.mu.RUnlock()
			return c.findSimilarLinear(model, query, queryEmbedding, start)
		}

		// Convert distances to similarities (USearch returns distances, we need similarity scores)
		// For cosine distance: similarity = 1 - distance
		// But USearch actually returns squared L2 distance for IP (Inner Product)
		// We need to handle this based on the metric type
		
		// Filter candidates by model and expiration, then find best match
		for i, entryIndex := range candidateIndices {
			if entryIndex < 0 || entryIndex >= len(c.entries) {
				continue
			}

			entry := c.entries[entryIndex]

			// Skip incomplete entries
			if entry.ResponseBody == nil {
				continue
			}

			// Only consider entries for the same model
			if entry.Model != model {
				continue
			}

			// Skip entries that have expired before considering them
			if c.isExpired(entry, now) {
				expiredCount++
				continue
			}

			// Calculate similarity from distance
			// USearch with IP (Inner Product) metric returns negative dot product as distance
			// So similarity = -distance for IP metric
			similarity := -distances[i]

			entriesChecked++
			if bestIndex == -1 || similarity > bestSimilarity {
				bestSimilarity = similarity
				bestIndex = entryIndex
			}
		}

		observability.Debugf("InMemoryCache.FindSimilar: USearch HNSW search checked %d candidates", len(candidateIndices))
	} else {
		// Fallback to linear search
		c.mu.RUnlock()
		return c.findSimilarLinear(model, query, queryEmbedding, start)
	}

	// Snapshot the best entry before releasing the read lock
	if bestIndex >= 0 {
		bestEntry = c.entries[bestIndex]
	}

	// Unlock the read lock since we need the write lock to update the access info
	c.mu.RUnlock()

	// Log if any expired entries were skipped
	if expiredCount > 0 {
		observability.Debugf("InMemoryCache: excluded %d expired entries during search (TTL: %ds)",
			expiredCount, c.ttlSeconds)
		observability.LogEvent("cache_expired_entries_found", map[string]interface{}{
			"backend":       "memory",
			"expired_count": expiredCount,
			"ttl_seconds":   c.ttlSeconds,
		})
	}

	// Handle case where no suitable entries exist
	if bestIndex < 0 {
		atomic.AddInt64(&c.missCount, 1)
		observability.Debugf("InMemoryCache.FindSimilar: no entries found with responses")
		metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	// Check if the best match meets the similarity threshold
	if bestSimilarity >= c.similarityThreshold {
		atomic.AddInt64(&c.hitCount, 1)

		c.mu.Lock()
		c.updateAccessInfo(bestIndex, bestEntry)
		c.mu.Unlock()

		observability.Debugf("InMemoryCache.FindSimilar: CACHE HIT - similarity=%.4f >= threshold=%.4f, response_size=%d bytes",
			bestSimilarity, c.similarityThreshold, len(bestEntry.ResponseBody))
		observability.LogEvent("cache_hit", map[string]interface{}{
			"backend":    "memory",
			"similarity": bestSimilarity,
			"threshold":  c.similarityThreshold,
			"model":      model,
		})
		metrics.RecordCacheOperation("memory", "find_similar", "hit", time.Since(start).Seconds())
		metrics.RecordCacheHit()
		return bestEntry.ResponseBody, true, nil
	}

	atomic.AddInt64(&c.missCount, 1)
	observability.Debugf("InMemoryCache.FindSimilar: CACHE MISS - best_similarity=%.4f < threshold=%.4f (checked %d entries)",
		bestSimilarity, c.similarityThreshold, entriesChecked)
	observability.LogEvent("cache_miss", map[string]interface{}{
		"backend":         "memory",
		"best_similarity": bestSimilarity,
		"threshold":       c.similarityThreshold,
		"model":           model,
		"entries_checked": entriesChecked,
	})
	metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
	metrics.RecordCacheMiss()
	return nil, false, nil
}

// findSimilarLinear performs linear search as a fallback
func (c *InMemoryCache) findSimilarLinear(model string, query string, queryEmbedding []float32, start time.Time) ([]byte, bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var (
		bestIndex      = -1
		bestEntry      CacheEntry
		bestSimilarity float32
		entriesChecked int
		expiredCount   int
	)
	now := time.Now()

	for entryIndex, entry := range c.entries {
		// Skip incomplete entries
		if entry.ResponseBody == nil {
			continue
		}

		// Only consider entries for the same model
		if entry.Model != model {
			continue
		}

		// Skip entries that have expired before considering them
		if c.isExpired(entry, now) {
			expiredCount++
			continue
		}

		// Compute semantic similarity using dot product
		var dotProduct float32
		for i := 0; i < len(queryEmbedding) && i < len(entry.Embedding); i++ {
			dotProduct += queryEmbedding[i] * entry.Embedding[i]
		}

		entriesChecked++
		if bestIndex == -1 || dotProduct > bestSimilarity {
			bestSimilarity = dotProduct
			bestIndex = entryIndex
		}
	}

	observability.Debugf("InMemoryCache.FindSimilar: Linear search used (HNSW disabled or empty)")

	// Snapshot the best entry
	if bestIndex >= 0 {
		bestEntry = c.entries[bestIndex]
	}

	// Handle miss case
	if bestIndex < 0 || bestSimilarity < c.similarityThreshold {
		atomic.AddInt64(&c.missCount, 1)
		metrics.RecordCacheOperation("memory", "find_similar", "miss", time.Since(start).Seconds())
		metrics.RecordCacheMiss()
		return nil, false, nil
	}

	// Handle hit case
	atomic.AddInt64(&c.hitCount, 1)
	c.updateAccessInfo(bestIndex, bestEntry)
	metrics.RecordCacheOperation("memory", "find_similar", "hit", time.Since(start).Seconds())
	metrics.RecordCacheHit()
	return bestEntry.ResponseBody, true, nil
}

// Close releases all resources held by the cache
func (c *InMemoryCache) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Clear all entries to free memory
	c.entries = nil

	// Destroy USearch index
	if c.hnswIndex != nil {
		c.hnswIndex.destroy()
		c.hnswIndex = nil
	}

	// Zero cache entries metrics
	metrics.UpdateCacheEntries("memory", 0)

	return nil
}

// GetStats provides current cache performance metrics
func (c *InMemoryCache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	hits := atomic.LoadInt64(&c.hitCount)
	misses := atomic.LoadInt64(&c.missCount)
	total := hits + misses

	var hitRatio float64
	if total > 0 {
		hitRatio = float64(hits) / float64(total)
	}

	stats := CacheStats{
		TotalEntries: len(c.entries),
		HitCount:     hits,
		MissCount:    misses,
		HitRatio:     hitRatio,
	}

	if c.lastCleanupTime != nil {
		stats.LastCleanupTime = c.lastCleanupTime
	}

	return stats
}

// cleanupExpiredEntries removes entries that have exceeded their TTL and updates the cache entry count metric to keep metrics in sync.
// Caller must hold a write lock
func (c *InMemoryCache) cleanupExpiredEntries() {
	if c.ttlSeconds <= 0 {
		return
	}

	now := time.Now()
	validEntries := make([]CacheEntry, 0, len(c.entries))

	for _, entry := range c.entries {
		// Retain entries that are still within their TTL based on last access
		if !c.isExpired(entry, now) {
			validEntries = append(validEntries, entry)
		}
	}

	if len(validEntries) == len(c.entries) {
		return
	}

	expiredCount := len(c.entries) - len(validEntries)
	observability.Debugf("InMemoryCache: TTL cleanup removed %d expired entries (remaining: %d)",
		expiredCount, len(validEntries))
	observability.LogEvent("cache_cleanup", map[string]interface{}{
		"backend":         "memory",
		"expired_count":   expiredCount,
		"remaining_count": len(validEntries),
		"ttl_seconds":     c.ttlSeconds,
	})
	c.entries = validEntries
	cleanupTime := time.Now()
	c.lastCleanupTime = &cleanupTime

	// Rebuild USearch HNSW index if entries were removed
	if expiredCount > 0 && c.useHNSW && c.hnswIndex != nil {
		c.rebuildUSearchIndex()
	}

	// Update metrics after cleanup
	metrics.UpdateCacheEntries("memory", len(c.entries))
}

// isExpired checks if a cache entry has expired based on its last access time
func (c *InMemoryCache) isExpired(entry CacheEntry, now time.Time) bool {
	if c.ttlSeconds <= 0 {
		return false
	}

	return now.Sub(entry.LastAccessAt) >= time.Duration(c.ttlSeconds)*time.Second
}

// updateAccessInfo updates the access information for the given entry index
func (c *InMemoryCache) updateAccessInfo(entryIndex int, target CacheEntry) {
	// fast path
	if entryIndex < len(c.entries) && c.entries[entryIndex].RequestID == target.RequestID {
		c.entries[entryIndex].LastAccessAt = time.Now()
		c.entries[entryIndex].HitCount++
		return
	}

	// fallback to linear search
	for i := range c.entries {
		if c.entries[i].RequestID == target.RequestID {
			c.entries[i].LastAccessAt = time.Now()
			c.entries[i].HitCount++
			break
		}
	}
}

// evictOne removes one entry based on the configured eviction policy
func (c *InMemoryCache) evictOne() {
	if len(c.entries) == 0 {
		return
	}

	victimIdx := c.evictionPolicy.SelectVictim(c.entries)
	if victimIdx < 0 || victimIdx >= len(c.entries) {
		return
	}

	evictedRequestID := c.entries[victimIdx].RequestID

	// If using USearch HNSW, we need to rebuild the index after eviction
	// USearch supports removal but for simplicity we'll rebuild
	if c.useHNSW && c.hnswIndex != nil {
		c.hnswIndex.markForRebuild()
	}

	c.entries[victimIdx] = c.entries[len(c.entries)-1]
	c.entries = c.entries[:len(c.entries)-1]

	observability.LogEvent("cache_evicted", map[string]any{
		"backend":     "memory",
		"request_id":  evictedRequestID,
		"max_entries": c.maxEntries,
	})
}

// ===== USearch HNSW Index Implementation =====

// newUSearchHNSWIndex creates a new USearch-based HNSW index
func newUSearchHNSWIndex(dimensions uint, m, efConstruction, efSearch int) *USearchHNSWIndex {
	return &USearchHNSWIndex{
		dimensions:     dimensions,
		efConstruction: efConstruction,
		efSearch:       efSearch,
		M:              m,
	}
}

// initializeIndex creates the USearch index with the correct dimensions
func (h *USearchHNSWIndex) initializeIndex(dimensions uint, initialCapacity uint) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.index != nil {
		return nil // Already initialized
	}

	h.dimensions = dimensions

	// Create USearch configuration
	conf := usearch.DefaultConfig(dimensions)
	conf.Connectivity = uint(h.M)
	conf.ExpansionAdd = uint(h.efConstruction)
	conf.Metric = usearch.InnerProduct // Inner Product (cosine similarity for normalized vectors)

	// Create index
	index, err := usearch.NewIndex(conf)
	if err != nil {
		return fmt.Errorf("failed to create USearch index: %w", err)
	}

	// Reserve capacity to avoid resizing
	if initialCapacity > 0 {
		if err := index.Reserve(initialCapacity); err != nil {
			observability.Debugf("Warning: failed to reserve capacity: %v", err)
		}
	}

	// Configure threading
	numCPU := uint(runtime.NumCPU())
	_ = index.ChangeThreadsAdd(numCPU)
	_ = index.ChangeThreadsSearch(numCPU)
	
	// Set search expansion parameter
	_ = index.ChangeExpansionSearch(uint(h.efSearch))

	h.index = index
	observability.Debugf("USearch index initialized: dimensions=%d, M=%d, efConstruction=%d, efSearch=%d, capacity=%d, metric=IP",
		dimensions, h.M, h.efConstruction, h.efSearch, initialCapacity)

	return nil
}

// addVector adds a vector to the USearch index
func (h *USearchHNSWIndex) addVector(entryIndex int, vector []float32) error {
	// Initialize index on first add with reasonable capacity
	if h.index == nil {
		initialCapacity := uint(10000) // Pre-allocate for 10k vectors
		if err := h.initializeIndex(uint(len(vector)), initialCapacity); err != nil {
			return err
		}
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	// Use entry index directly as USearch key to avoid map lookups
	key := usearch.Key(entryIndex)

	// Add to USearch index
	if err := h.index.Add(key, vector); err != nil {
		return fmt.Errorf("failed to add vector to USearch: %w", err)
	}

	return nil
}

// search performs k-nearest neighbor search
func (h *USearchHNSWIndex) search(query []float32, k int) ([]int, []float32, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.index == nil {
		return nil, nil, fmt.Errorf("index not initialized")
	}

	// Perform search
	keys, distances, err := h.index.Search(query, uint(k))
	if err != nil {
		return nil, nil, fmt.Errorf("USearch search failed: %w", err)
	}

	// Convert keys directly to entry indices (no map lookup needed)
	indices := make([]int, len(keys))
	for i, key := range keys {
		indices[i] = int(key)
	}

	return indices, distances, nil
}

// size returns the number of vectors in the index
func (h *USearchHNSWIndex) size() int {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.index == nil {
		return 0
	}

	len, _ := h.index.Len()
	return int(len)
}

// markForRebuild marks the index as needing a rebuild
func (h *USearchHNSWIndex) markForRebuild() {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.index != nil {
		h.index.Destroy()
		h.index = nil
	}
}

// destroy releases USearch resources
func (h *USearchHNSWIndex) destroy() {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.index != nil {
		h.index.Destroy()
		h.index = nil
	}
}

// rebuildUSearchIndex rebuilds the USearch index from scratch
// Caller must hold a write lock
func (c *InMemoryCache) rebuildUSearchIndex() {
	if c.hnswIndex == nil {
		return
	}

	observability.Debugf("InMemoryCache: Rebuilding USearch index with %d entries", len(c.entries))

	// Clear the existing index
	c.hnswIndex.markForRebuild()

	// Count valid entries for capacity hint
	validCount := 0
	for _, entry := range c.entries {
		if len(entry.Embedding) > 0 {
			validCount++
		}
	}

	// Pre-initialize with correct capacity if we have entries
	if validCount > 0 && len(c.entries) > 0 && len(c.entries[0].Embedding) > 0 {
		if err := c.hnswIndex.initializeIndex(uint(len(c.entries[0].Embedding)), uint(validCount)); err != nil {
			observability.Debugf("Warning: failed to initialize USearch index during rebuild: %v", err)
		}
	}

	// Rebuild by adding all entries
	for i, entry := range c.entries {
		if len(entry.Embedding) > 0 {
			if err := c.hnswIndex.addVector(i, entry.Embedding); err != nil {
				observability.Debugf("Warning: failed to add vector %d to USearch during rebuild: %v", i, err)
			}
		}
	}

	observability.Debugf("InMemoryCache: USearch index rebuilt with %d vectors", c.hnswIndex.size())
}


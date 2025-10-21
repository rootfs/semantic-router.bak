//go:build !windows && cgo
// +build !windows,cgo

package cache

import (
	"math"
	"time"
)

// HNSWNode represents a node in the HNSW graph
// Used by HybridCache for custom HNSW implementation
type HNSWNode struct {
	entryIndex int           // Index into entries slice
	neighbors  map[int][]int // Layer -> neighbor indices
	maxLayer   int           // Highest layer this node appears in
}

// HNSWIndex implements Hierarchical Navigable Small World graph for fast ANN search
// Custom implementation used by HybridCache
type HNSWIndex struct {
	nodes          []*HNSWNode
	entryPoint     int     // Index of the top-level entry point
	maxLayer       int     // Maximum layer in the graph
	efConstruction int     // Size of dynamic candidate list during construction
	M              int     // Number of bi-directional links per node
	Mmax           int     // Maximum number of connections per node (=M)
	Mmax0          int     // Maximum number of connections for layer 0 (=M*2)
	ml             float64 // Normalization factor for level assignment
}

// newHNSWIndex creates a new HNSW index
func newHNSWIndex(m, efConstruction int) *HNSWIndex {
	return &HNSWIndex{
		nodes:          []*HNSWNode{},
		entryPoint:     -1,
		maxLayer:       -1,
		efConstruction: efConstruction,
		M:              m,
		Mmax:           m,
		Mmax0:          m * 2,
		ml:             1.0 / math.Log(float64(m)),
	}
}

// markStale marks the index as needing a rebuild
func (h *HNSWIndex) markStale() {
	// Simple approach: clear the index
	h.nodes = []*HNSWNode{}
	h.entryPoint = -1
	h.maxLayer = -1
}

// Helper priority queue implementations for HNSW

type heapItem struct {
	index int
	dist  float32
}

type minHeap struct {
	data []heapItem
}

func newMinHeap() *minHeap {
	return &minHeap{data: []heapItem{}}
}

func (h *minHeap) push(index int, dist float32) {
	h.data = append(h.data, heapItem{index, dist})
	h.bubbleUp(len(h.data) - 1)
}

func (h *minHeap) pop() (int, float32) {
	if len(h.data) == 0 {
		return -1, 0
	}
	result := h.data[0]
	h.data[0] = h.data[len(h.data)-1]
	h.data = h.data[:len(h.data)-1]
	if len(h.data) > 0 {
		h.bubbleDown(0)
	}
	return result.index, result.dist
}

func (h *minHeap) peekDist() float32 {
	if len(h.data) == 0 {
		return math.MaxFloat32
	}
	return h.data[0].dist
}

func (h *minHeap) len() int {
	return len(h.data)
}

func (h *minHeap) items() []int {
	result := make([]int, len(h.data))
	for i, item := range h.data {
		result[i] = item.index
	}
	return result
}

func (h *minHeap) bubbleUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.data[i].dist >= h.data[parent].dist {
			break
		}
		h.data[i], h.data[parent] = h.data[parent], h.data[i]
		i = parent
	}
}

func (h *minHeap) bubbleDown(i int) {
	for {
		left := 2*i + 1
		right := 2*i + 2
		smallest := i

		if left < len(h.data) && h.data[left].dist < h.data[smallest].dist {
			smallest = left
		}
		if right < len(h.data) && h.data[right].dist < h.data[smallest].dist {
			smallest = right
		}
		if smallest == i {
			break
		}
		h.data[i], h.data[smallest] = h.data[smallest], h.data[i]
		i = smallest
	}
}

type maxHeap struct {
	data []heapItem
}

func newMaxHeap() *maxHeap {
	return &maxHeap{data: []heapItem{}}
}

func (h *maxHeap) push(index int, dist float32) {
	h.data = append(h.data, heapItem{index, dist})
	h.bubbleUp(len(h.data) - 1)
}

func (h *maxHeap) pop() (int, float32) {
	if len(h.data) == 0 {
		return -1, 0
	}
	result := h.data[0]
	h.data[0] = h.data[len(h.data)-1]
	h.data = h.data[:len(h.data)-1]
	if len(h.data) > 0 {
		h.bubbleDown(0)
	}
	return result.index, result.dist
}

func (h *maxHeap) len() int {
	return len(h.data)
}

func (h *maxHeap) bubbleUp(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.data[i].dist <= h.data[parent].dist {
			break
		}
		h.data[i], h.data[parent] = h.data[parent], h.data[i]
		i = parent
	}
}

func (h *maxHeap) bubbleDown(i int) {
	for {
		left := 2*i + 1
		right := 2*i + 2
		largest := i

		if left < len(h.data) && h.data[left].dist > h.data[largest].dist {
			largest = left
		}
		if right < len(h.data) && h.data[right].dist > h.data[largest].dist {
			largest = right
		}
		if largest == i {
			break
		}
		h.data[i], h.data[largest] = h.data[largest], h.data[i]
		i = largest
	}
}

// Helper function for selecting random level
func randFloat() float64 {
	return float64(time.Now().UnixNano()%1000000) / 1000000.0
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}


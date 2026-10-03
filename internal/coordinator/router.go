package coordinator

import (
	"container/heap"
	"fmt"
	"sort"
	"sync"

	"vektordb/internal/bridge"
)

// QueryRouter coordinates distributed fan-out queries and priority-queue result merging.
type QueryRouter struct {
	cluster *ShardCluster
}

// NewQueryRouter creates a router attached to the shard cluster.
func NewQueryRouter(cluster *ShardCluster) *QueryRouter {
	return &QueryRouter{cluster: cluster}
}

// Search fans out the search request across all shards concurrently,
// then merges candidate matches into a single global top-K priority queue.
func (r *QueryRouter) Search(query []float32, k int, efSearch int) ([]bridge.SearchResult, error) {
	shards := r.cluster.Shards()
	if len(shards) == 0 {
		return []bridge.SearchResult{}, nil
	}

	if k <= 0 {
		return []bridge.SearchResult{}, nil
	}

	type shardResponse struct {
		matches []bridge.SearchResult
		err     error
	}

	resChan := make(chan shardResponse, len(shards))
	var wg sync.WaitGroup

	// Parallel query fan-out across all shards
	for _, shard := range shards {
		wg.Add(1)
		go func(s *Shard) {
			defer wg.Done()
			matches, err := s.Search(query, k, efSearch)
			resChan <- shardResponse{matches: matches, err: err}
		}(shard)
	}

	wg.Wait()
	close(resChan)

	var allCandidates [][]bridge.SearchResult
	for res := range resChan {
		if res.err != nil {
			return nil, fmt.Errorf("shard query failed: %w", res.err)
		}
		allCandidates = append(allCandidates, res.matches)
	}

	// Merge all shard candidates using a priority queue
	return MergeTopK(allCandidates, k), nil
}

// ============================================================================
// Priority Queue / Max-Heap for Top-K Nearest Neighbors (Smallest Distances)
// ============================================================================

type maxMatchHeap []bridge.SearchResult

func (h maxMatchHeap) Len() int           { return len(h) }
func (h maxMatchHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance } // Largest distance at top
func (h maxMatchHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *maxMatchHeap) Push(x interface{}) {
	*h = append(*h, x.(bridge.SearchResult))
}

func (h *maxMatchHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[0 : n-1]
	return item
}

// MergeTopK merges multiple sorted candidate lists from shards into the global top-K nearest neighbors.
func MergeTopK(shardResults [][]bridge.SearchResult, k int) []bridge.SearchResult {
	if k <= 0 {
		return []bridge.SearchResult{}
	}

	h := &maxMatchHeap{}
	heap.Init(h)

	// Keep track of visited IDs to deduplicate if any duplicates exist
	seen := make(map[uint64]bool)

	for _, list := range shardResults {
		for _, item := range list {
			if seen[item.ID] {
				continue
			}
			seen[item.ID] = true

			if h.Len() < k {
				heap.Push(h, item)
			} else if item.Distance < (*h)[0].Distance {
				// Candidate is closer than current farthest in top-k
				heap.Pop(h)
				heap.Push(h, item)
			}
		}
	}

	// Extract from heap and sort in ascending distance order
	result := make([]bridge.SearchResult, h.Len())
	for i := len(result) - 1; i >= 0; i-- {
		result[i] = heap.Pop(h).(bridge.SearchResult)
	}

	// Ensure strictly ascending sort
	sort.Slice(result, func(i, j int) bool {
		return result[i].Distance < result[j].Distance
	})

	return result
}

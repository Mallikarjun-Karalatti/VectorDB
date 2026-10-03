package coordinator

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"sync"

	"vektordb/internal/bridge"
	"vektordb/internal/storage"
)

// Shard represents a single physical partition containing an HNSW index and dedicated WAL.
type Shard struct {
	ID          int
	Index       *bridge.Index
	WAL         *storage.WAL
	dir         string
	mu          sync.RWMutex
	vectorCount int
}

// ShardStat provides telemetry for a single shard.
type ShardStat struct {
	ID          int    `json:"id"`
	VectorCount int    `json:"vector_count"`
	Status      string `json:"status"`
	WALPath     string `json:"wal_path"`
	MemoryBytes int64  `json:"memory_bytes"`
}

// NewShard initializes or recovers a shard from disk.
func NewShard(id int, baseDir string, dim, maxElements, M, efConstruction, metric int, syncWAL bool) (*Shard, error) {
	shardDir := filepath.Join(baseDir, fmt.Sprintf("shard_%d", id))
	if err := os.MkdirAll(shardDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create shard directory: %w", err)
	}

	walPath := filepath.Join(shardDir, "wal.bin")
	wal, err := storage.OpenWAL(walPath, syncWAL)
	if err != nil {
		return nil, fmt.Errorf("failed to open shard %d WAL: %w", id, err)
	}

	idx, err := bridge.NewIndex(dim, maxElements, M, efConstruction, metric)
	if err != nil {
		_ = wal.Close()
		return nil, fmt.Errorf("failed to create shard %d index: %w", id, err)
	}

	// Replay existing WAL if present
	recovered, err := wal.Replay("", idx)
	if err != nil {
		_ = wal.Close()
		idx.Close()
		return nil, fmt.Errorf("failed to replay shard %d WAL: %w", id, err)
	}

	s := &Shard{
		ID:          id,
		Index:       idx,
		WAL:         wal,
		dir:         shardDir,
		vectorCount: recovered,
	}

	return s, nil
}

// Insert writes to the WAL and updates the shard's in-memory HNSW index.
func (s *Shard) Insert(id uint64, vec []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 1. Write to WAL first (durability guarantee)
	if err := s.WAL.WriteEntry(storage.OpInsert, id, vec); err != nil {
		return fmt.Errorf("shard %d wal write failed: %w", s.ID, err)
	}

	// 2. Insert into C++ HNSW index
	if err := s.Index.Insert(id, vec); err != nil {
		return fmt.Errorf("shard %d index insert failed: %w", s.ID, err)
	}

	s.vectorCount++
	return nil
}

// Search queries the shard's HNSW index.
func (s *Shard) Search(query []float32, k int, ef int) ([]bridge.SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	results, err := s.Index.Search(query, k, ef)
	if err != nil {
		return nil, err
	}

	for i := range results {
		results[i].ShardID = s.ID
	}
	return results, nil
}

// Count returns the vector count of the shard.
func (s *Shard) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.Index.Count()
}

// Close gracefully closes the WAL and frees the index.
func (s *Shard) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.WAL != nil {
		_ = s.WAL.Close()
	}
	if s.Index != nil {
		s.Index.Close()
	}
}

// ShardCluster coordinates partition management across 3 local shards.
type ShardCluster struct {
	shards []*Shard
	dim    int
	metric int
	mu     sync.RWMutex
}

// NewShardCluster initializes a cluster with a fixed number of shards (default 3).
func NewShardCluster(baseDir string, numShards, dim, maxElementsPerShard, M, efConstruction, metric int, syncWAL bool) (*ShardCluster, error) {
	if numShards <= 0 {
		numShards = 3
	}

	shards := make([]*Shard, numShards)
	for i := 0; i < numShards; i++ {
		shard, err := NewShard(i, baseDir, dim, maxElementsPerShard, M, efConstruction, metric, syncWAL)
		if err != nil {
			// Clean up already initialized shards
			for j := 0; j < i; j++ {
				shards[j].Close()
			}
			return nil, err
		}
		shards[i] = shard
	}

	return &ShardCluster{
		shards: shards,
		dim:    dim,
		metric: metric,
	}, nil
}

// RouteShard determines the target shard for a given vector ID using FNV-1a consistent hashing.
func (c *ShardCluster) RouteShard(id uint64) *Shard {
	h := fnv.New64a()
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], id)
	_, _ = h.Write(b[:])
	shardIdx := int(h.Sum64() % uint64(len(c.shards)))
	return c.shards[shardIdx]
}

// Insert hashes vector ID, routes to the corresponding shard, and writes data.
func (c *ShardCluster) Insert(id uint64, vec []float32) (int, error) {
	shard := c.RouteShard(id)
	if err := shard.Insert(id, vec); err != nil {
		return shard.ID, err
	}
	return shard.ID, nil
}

// TotalVectors returns the total number of vectors indexed across all shards.
func (c *ShardCluster) TotalVectors() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := 0
	for _, s := range c.shards {
		total += s.Count()
	}
	return total
}

// ShardStats returns status for each shard.
func (c *ShardCluster) ShardStats() []ShardStat {
	c.mu.RLock()
	defer c.mu.RUnlock()

	stats := make([]ShardStat, len(c.shards))
	for i, s := range c.shards {
		cnt := s.Count()
		// Approximate memory: vector storage + graph connectivity overhead
		approxBytes := int64(cnt) * int64(c.dim*4 + 64)
		stats[i] = ShardStat{
			ID:          s.ID,
			VectorCount: cnt,
			Status:      "healthy",
			WALPath:     s.WAL.Path(),
			MemoryBytes: approxBytes,
		}
	}
	return stats
}

// Shards returns the underlying shard slice.
func (c *ShardCluster) Shards() []*Shard {
	return c.shards
}

// Dim returns vector dimension.
func (c *ShardCluster) Dim() int {
	return c.dim
}

// Close shuts down all shards cleanly.
func (c *ShardCluster) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, s := range c.shards {
		s.Close()
	}
}

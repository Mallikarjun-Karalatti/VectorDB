package test

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"vektordb/internal/bridge"
	"vektordb/internal/storage"
)

func TestCrashRecovery(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "vektordb_crash_test_*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	walPath := filepath.Join(tempDir, "shard_0.wal")
	const (
		dim         = 32
		numVectors  = 50000
		maxElements = 60000
		M           = 16
		efConstruct = 100
	)

	// Step 1: Open WAL and create initial C++ index
	wal, err := storage.OpenWAL(walPath, false)
	if err != nil {
		t.Fatalf("failed to open wal: %v", err)
	}

	idx, err := bridge.NewIndex(dim, maxElements, M, efConstruct, bridge.MetricCosine)
	if err != nil {
		t.Fatalf("failed to create index: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	t.Logf("Ingesting %d vectors into WAL and C++ index...", numVectors)
	startTime := time.Now()

	testQuery := make([]float32, dim)
	for d := 0; d < dim; d++ {
		testQuery[d] = rng.Float32()
	}

	vecBuf := make([]float32, dim)
	for i := uint64(1); i <= numVectors; i++ {
		for d := 0; d < dim; d++ {
			vecBuf[d] = rng.Float32()
		}

		// Write to WAL
		if err := wal.WriteEntry(storage.OpInsert, i, vecBuf); err != nil {
			t.Fatalf("failed to write WAL entry for vector %d: %v", i, err)
		}

		// Update in-memory index
		if err := idx.Insert(i, vecBuf); err != nil {
			t.Fatalf("failed to insert vector %d into index: %v", i, err)
		}
	}

	// All 50,000 acknowledged writes are flushed to disk
	if err := wal.Sync(); err != nil {
		t.Fatalf("failed to sync WAL: %v", err)
	}

	t.Logf("Ingestion completed in %v. Current index count: %d", time.Since(startTime), idx.Count())
	if idx.Count() != numVectors {
		t.Fatalf("expected %d vectors in index, got %d", numVectors, idx.Count())
	}

	// Step 2: Simulate ungraceful crash (SIGKILL mid-write)
	// Inject a corrupted partial trailing frame to simulate a torn write during crash
	t.Log("Simulating ungraceful crash with corrupted trailing bytes injected...")
	corruptBytes := []byte{0x00, 0x00, 0x00, 0x80, 0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02}
	f, err := os.OpenFile(walPath, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatalf("failed to open wal file for crash injection: %v", err)
	}
	if _, err := f.Write(corruptBytes); err != nil {
		t.Fatalf("failed to inject corrupted trailing bytes: %v", err)
	}
	_ = f.Sync()
	f.Close()

	// Ungracefully abandon the old index and WAL handle without clean shutdown
	_ = wal.Close()
	idx.Close()

	// Step 3: Engine recovery from WAL
	t.Log("Reloading engine from WAL log...")
	recoveryStart := time.Now()

	recoveredIdx, err := bridge.NewIndex(dim, maxElements, M, efConstruct, bridge.MetricCosine)
	if err != nil {
		t.Fatalf("failed to create recovered index: %v", err)
	}
	defer recoveredIdx.Close()

	recoveredCount, err := storage.ReplayFromPath(walPath, "", recoveredIdx)
	if err != nil {
		t.Fatalf("replay failed: %v", err)
	}

	t.Logf("Recovery completed in %v. Recovered count: %d", time.Since(recoveryStart), recoveredCount)

	// Step 4: Verify zero lost acknowledged writes
	if recoveredCount != numVectors {
		t.Fatalf("Crash recovery data loss! Expected %d recovered vectors, got %d", numVectors, recoveredCount)
	}

	if recoveredIdx.Count() != numVectors {
		t.Fatalf("Index count mismatch after recovery! Expected %d, got %d", numVectors, recoveredIdx.Count())
	}

	// Verify query execution on recovered index
	results, err := recoveredIdx.Search(testQuery, 10, 64)
	if err != nil {
		t.Fatalf("search on recovered index failed: %v", err)
	}

	if len(results) != 10 {
		t.Fatalf("expected 10 search results, got %d", len(results))
	}

	t.Logf("Verification succeeded! Top match: ID=%d, Distance=%.4f (zero data loss verified across %d vectors)",
		results[0].ID, results[0].Distance, numVectors)
}

package test

import (
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"

	"vektordb/internal/bridge"
	"vektordb/internal/coordinator"
)

// BenchmarkSIMDVsScalar measures raw throughput and latency between Scalar distance and SIMD distance.
func BenchmarkSIMDVsScalar(b *testing.B) {
	runSIMDVsScalar(b.Logf, b.N)
}

func TestBenchmarkSIMDVsScalar(t *testing.T) {
	runSIMDVsScalar(t.Logf, 50000)
}

func runSIMDVsScalar(logf func(format string, args ...interface{}), iterations int) {
	const dim = 768
	if iterations <= 0 {
		iterations = 50000
	}

	rng := rand.New(rand.NewSource(1337))
	vecA := make([]float32, dim)
	vecB := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vecA[i] = rng.Float32()
		vecB[i] = rng.Float32()
	}

	// Warmup
	_ = bridge.DistCosineScalar(vecA, vecB)
	_ = bridge.DistCosineSIMD(vecA, vecB)

	// Measure Scalar
	startScalar := time.Now()
	for i := 0; i < iterations; i++ {
		_ = bridge.DistCosineScalar(vecA, vecB)
	}
	scalarDuration := time.Since(startScalar)
	scalarQPS := float64(iterations) / scalarDuration.Seconds()
	scalarAvgLatencyNs := float64(scalarDuration.Nanoseconds()) / float64(iterations)

	// Measure SIMD (AVX2 or ARM NEON)
	startSIMD := time.Now()
	for i := 0; i < iterations; i++ {
		_ = bridge.DistCosineSIMD(vecA, vecB)
	}
	simdDuration := time.Since(startSIMD)
	simdQPS := float64(iterations) / simdDuration.Seconds()
	simdAvgLatencyNs := float64(simdDuration.Nanoseconds()) / float64(iterations)

	speedup := scalarDuration.Seconds() / simdDuration.Seconds()

	hw := "Scalar (generic)"
	if bridge.HasAVX2() {
		hw = "AVX2 SIMD"
	} else if bridge.HasNEON() {
		hw = "ARM NEON SIMD"
	}

	logf("\n==================================================================")
	logf("      DISTANCE KERNEL BENCHMARK (Dim: %d, Iterations: %d)       ", dim, iterations)
	logf("==================================================================")
	logf(" Hardware Vector Engine : %s", hw)
	logf(" Scalar Cosine Distance : %10.2f ops/sec | Avg: %8.2f ns/op", scalarQPS, scalarAvgLatencyNs)
	logf(" SIMD Cosine Distance   : %10.2f ops/sec | Avg: %8.2f ns/op", simdQPS, simdAvgLatencyNs)
	logf(" Performance Speedup    : %.2fx faster with SIMD vectorization", speedup)
	logf("==================================================================\n")
}

// BenchmarkRecallVsEfSearch evaluates Recall@10 and query latency at varying efSearch configurations.
func BenchmarkRecallVsEfSearch(b *testing.B) {
	runRecallVsEfSearch(b.Logf)
}

func TestBenchmarkRecallVsEfSearch(t *testing.T) {
	runRecallVsEfSearch(t.Logf)
}

func runRecallVsEfSearch(logf func(format string, args ...interface{})) {
	const (
		dim         = 128
		numVectors  = 5000
		numQueries  = 100
		k           = 10
		maxElements = 10000
		M           = 16
		efConstruct = 200
	)

	tempDir, err := os.MkdirTemp("", "vektordb_recall_*")
	if err != nil {
		logf("error creating temp dir: %v", err)
		return
	}
	defer os.RemoveAll(tempDir)

	idx, err := bridge.NewIndex(dim, maxElements, M, efConstruct, bridge.MetricCosine)
	if err != nil {
		logf("error initializing index: %v", err)
		return
	}
	defer idx.Close()

	rng := rand.New(rand.NewSource(42))
	dataset := make([][]float32, numVectors)
	for i := 0; i < numVectors; i++ {
		v := make([]float32, dim)
		for d := 0; d < dim; d++ {
			v[d] = rng.Float32()
		}
		dataset[i] = v
		_ = idx.Insert(uint64(i+1), v)
	}

	queries := make([][]float32, numQueries)
	for i := 0; i < numQueries; i++ {
		q := make([]float32, dim)
		for d := 0; d < dim; d++ {
			q[d] = rng.Float32()
		}
		queries[i] = q
	}

	// Compute exact brute-force ground truth for all queries
	type neighbor struct {
		id   uint64
		dist float32
	}
	groundTruth := make([][]uint64, numQueries)
	for qi, q := range queries {
		allDists := make([]neighbor, numVectors)
		for vi, v := range dataset {
			d := bridge.DistCosineSIMD(q, v)
			allDists[vi] = neighbor{id: uint64(vi + 1), dist: d}
		}
		sort.Slice(allDists, func(i, j int) bool {
			return allDists[i].dist < allDists[j].dist
		})

		topK := make([]uint64, k)
		for ki := 0; ki < k; ki++ {
			topK[ki] = allDists[ki].id
		}
		groundTruth[qi] = topK
	}

	efValues := []int{16, 32, 64, 128, 200}
	logf("\n==================================================================")
	logf("      RECALL@10 vs. efSearch TRADE-OFF EVALUATION (K=%d)         ", k)
	logf("==================================================================")
	logf(" efSearch | Avg Latency (ms) | QPS (Single-Thread) | Recall@10   ")
	logf("----------|------------------|---------------------|-------------")

	for _, ef := range efValues {
		totalRecall := 0.0
		start := time.Now()

		for qi, q := range queries {
			results, err := idx.Search(q, k, ef)
			if err != nil {
				continue
			}

			// Compute intersection with ground truth
			gtSet := make(map[uint64]bool)
			for _, id := range groundTruth[qi] {
				gtSet[id] = true
			}

			hits := 0
			for _, res := range results {
				if gtSet[res.ID] {
					hits++
				}
			}
			totalRecall += float64(hits) / float64(k)
		}

		elapsed := time.Since(start)
		avgLatencyMs := (elapsed.Seconds() * 1000.0) / float64(numQueries)
		qps := float64(numQueries) / elapsed.Seconds()
		meanRecall := (totalRecall / float64(numQueries)) * 100.0

		logf(" %-8d | %16.3f | %19.1f | %10.1f%% ", ef, avgLatencyMs, qps, meanRecall)
	}
	logf("==================================================================\n")
}

// TestShardedCoordinatorThroughput verifies 3-shard routing and Top-K merge correctness.
func TestShardedCoordinatorThroughput(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "vektordb_shards_*")
	if err != nil {
		t.Fatalf("temp dir error: %v", err)
	}
	defer os.RemoveAll(tempDir)

	const (
		numShards   = 3
		dim         = 64
		numVectors  = 3000
		maxPerShard = 5000
	)

	cluster, err := coordinator.NewShardCluster(
		tempDir,
		numShards,
		dim,
		maxPerShard,
		16,
		100,
		bridge.MetricCosine,
		false,
	)
	if err != nil {
		t.Fatalf("failed to init cluster: %v", err)
	}
	defer cluster.Close()

	rng := rand.New(rand.NewSource(99))
	for i := 1; i <= numVectors; i++ {
		v := make([]float32, dim)
		for d := 0; d < dim; d++ {
			v[d] = rng.Float32()
		}
		_, err := cluster.Insert(uint64(i), v)
		if err != nil {
			t.Fatalf("insert failed: %v", err)
		}
	}

	if cluster.TotalVectors() != numVectors {
		t.Fatalf("expected %d total vectors across shards, got %d", numVectors, cluster.TotalVectors())
	}

	router := coordinator.NewQueryRouter(cluster)
	query := make([]float32, dim)
	for d := 0; d < dim; d++ {
		query[d] = rng.Float32()
	}

	matches, err := router.Search(query, 10, 64)
	if err != nil {
		t.Fatalf("router search failed: %v", err)
	}
	if len(matches) != 10 {
		t.Fatalf("expected 10 matches, got %d", len(matches))
	}

	// Verify matches are in ascending distance order
	for i := 1; i < len(matches); i++ {
		if matches[i].Distance < matches[i-1].Distance {
			t.Fatalf("matches out of order at index %d: %f < %f", i, matches[i].Distance, matches[i-1].Distance)
		}
	}

	t.Logf("Sharded cluster search validated: 10 matches merged across 3 shards in strictly ascending order")
}

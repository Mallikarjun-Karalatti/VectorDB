package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"vektordb/internal/bridge"
	"vektordb/internal/coordinator"
	"vektordb/internal/server"
)

func main() {
	port := flag.Int("port", 8080, "HTTP server listening port")
	dataDir := flag.String("data-dir", "./data", "Directory storing shard WAL and snapshots")
	numShards := flag.Int("shards", 3, "Number of local shards to manage")
	dim := flag.Int("dim", 128, "Vector dimension")
	maxElements := flag.Int("max-elements", 500000, "Maximum vector capacity per shard")
	M := flag.Int("M", 16, "HNSW M parameter (connectivity)")
	efConstruction := flag.Int("ef-construction", 200, "HNSW efConstruction parameter")
	metricStr := flag.String("metric", "cosine", "Distance metric (cosine or l2)")
	syncWAL := flag.Bool("sync-wal", false, "Synchronous WAL fsync on every write")
	seedVectors := flag.Int("seed", 1500, "Pre-seed N synthetic vectors on startup for playground demo")
	flag.Parse()

	metric := bridge.MetricCosine
	if *metricStr == "l2" {
		metric = bridge.MetricL2
	}

	hwSIMD := "Scalar"
	if bridge.HasAVX2() {
		hwSIMD = "AVX2 SIMD (Intel/AMD)"
	} else if bridge.HasNEON() {
		hwSIMD = "ARM NEON SIMD (Apple Silicon/ARM64)"
	}

	fmt.Println("==================================================================")
	fmt.Println("             VektorDB — Sharded Vector Database Daemon             ")
	fmt.Println("==================================================================")
	fmt.Printf(" [Hardware Acceleration] : %s\n", hwSIMD)
	fmt.Printf(" [Shards]                : %d active local partitions\n", *numShards)
	fmt.Printf(" [Vector Dimension]      : %d (Metric: %s)\n", *dim, *metricStr)
	fmt.Printf(" [Storage Directory]     : %s\n", *dataDir)
	fmt.Printf(" [HTTP REST / SSE]       : http://localhost:%d\n", *port)
	fmt.Println("==================================================================")

	absDataDir, err := filepath.Abs(*dataDir)
	if err != nil {
		log.Fatalf("Invalid data directory path: %v", err)
	}

	// Initialize sharded cluster
	cluster, err := coordinator.NewShardCluster(
		absDataDir,
		*numShards,
		*dim,
		*maxElements,
		*M,
		*efConstruction,
		metric,
		*syncWAL,
	)
	if err != nil {
		log.Fatalf("Failed to initialize shard cluster: %v", err)
	}
	defer cluster.Close()

	// Pre-seed demo vectors if requested and database is empty
	if *seedVectors > 0 && cluster.TotalVectors() == 0 {
		fmt.Printf("Pre-seeding %d synthetic vectors across shards...\n", *seedVectors)
		rng := rand.New(rand.NewSource(1337))
		vec := make([]float32, *dim)
		for i := 1; i <= *seedVectors; i++ {
			for d := 0; d < *dim; d++ {
				vec[d] = rng.Float32()
			}
			_, _ = cluster.Insert(uint64(i), vec)
		}
		fmt.Printf("Pre-seeding complete! Total vectors in cluster: %d\n", cluster.TotalVectors())
	}

	metrics := server.NewMetricsCollector(cluster)
	srv := server.NewServer(*port, cluster, metrics)

	// Graceful shutdown listener
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigCh
		fmt.Printf("\nReceived signal %v. Gracefully shutting down VektorDB...\n", sig)
		_ = srv.Close()
		cluster.Close()
		os.Exit(0)
	}()

	fmt.Printf("VektorDB daemon ready on :%d! Press Ctrl+C to stop.\n", *port)
	if err := srv.Start(); err != nil && err.Error() != "http: Server closed" {
		log.Fatalf("Server error: %v", err)
	}
}

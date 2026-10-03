package server

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"sync"
	"time"

	"vektordb/internal/bridge"
	"vektordb/internal/coordinator"
)

// Server encapsulates the HTTP REST and SSE daemon.
type Server struct {
	cluster   *coordinator.ShardCluster
	router    *coordinator.QueryRouter
	metrics   *MetricsCollector
	port      int
	server    *http.Server
	benchmarkRunning sync.Mutex
}

// IngestionRequest represents vector payload.
type IngestionRequest struct {
	ID       uint64                 `json:"id"`
	Vector   []float32              `json:"vector"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// SearchRequest represents KNN query payload.
type SearchRequest struct {
	Query    []float32 `json:"query"`
	K        int       `json:"k"`
	EfSearch int       `json:"ef_search"`
}

// SearchResponse represents query results with execution timing.
type SearchResponse struct {
	Matches         []bridge.SearchResult `json:"matches"`
	TookMs          float64               `json:"took_ms"`
	TotalCandidates int                   `json:"total_candidates"`
}

// NewServer creates a new HTTP server.
func NewServer(port int, cluster *coordinator.ShardCluster, metrics *MetricsCollector) *Server {
	router := coordinator.NewQueryRouter(cluster)
	return &Server{
		cluster: cluster,
		router:  router,
		metrics: metrics,
		port:    port,
	}
}

// Start boots the HTTP server.
func (s *Server) Start() error {
	mux := http.NewServeMux()

	// REST API endpoints
	mux.HandleFunc("/api/v1/collections/vectors", s.handleVectors)
	mux.HandleFunc("/api/v1/collections/vectors/batch", s.handleVectorsBatch)
	mux.HandleFunc("/api/v1/collections/search", s.handleSearch)
	mux.HandleFunc("/api/v1/metrics", s.handleMetrics)
	mux.HandleFunc("/api/v1/stream/metrics", s.handleStreamMetrics)
	mux.HandleFunc("/api/v1/mode", s.handleSetMode)
	mux.HandleFunc("/api/v1/benchmark/run", s.handleRunBenchmark)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	// Serve compiled frontend UI if available
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Pass API routes
		if len(r.URL.Path) >= 4 && r.URL.Path[:4] == "/api" {
			http.NotFound(w, r)
			return
		}
		http.FileServer(http.Dir("web/dist")).ServeHTTP(w, r)
	})

	handler := s.corsMiddleware(mux)

	s.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", s.port),
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
	}

	return s.server.ListenAndServe()
}

// Close terminates the HTTP server.
func (s *Server) Close() error {
	if s.server != nil {
		return s.server.Close()
	}
	return nil
}

// POST /api/v1/collections/vectors
func (s *Server) handleVectors(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req IngestionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid json payload: %v", err), http.StatusBadRequest)
		return
	}

	if req.ID == 0 {
		http.Error(w, "vector id must be a non-zero uint64", http.StatusBadRequest)
		return
	}

	if len(req.Vector) != s.cluster.Dim() {
		http.Error(w, fmt.Sprintf("vector dimension mismatch: expected %d, got %d", s.cluster.Dim(), len(req.Vector)), http.StatusBadRequest)
		return
	}

	shardID, err := s.cluster.Insert(req.ID, req.Vector)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to insert vector: %v", err), http.StatusInternalServerError)
		return
	}

	s.metrics.RecordInsert()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status":   "indexed",
		"id":       req.ID,
		"shard_id": shardID,
	})
}

// POST /api/v1/collections/vectors/batch
func (s *Server) handleVectorsBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var reqs []IngestionRequest
	if err := json.NewDecoder(r.Body).Decode(&reqs); err != nil {
		http.Error(w, fmt.Sprintf("invalid json payload: %v", err), http.StatusBadRequest)
		return
	}

	indexed := 0
	for _, req := range reqs {
		if req.ID == 0 || len(req.Vector) != s.cluster.Dim() {
			continue
		}
		if _, err := s.cluster.Insert(req.ID, req.Vector); err == nil {
			s.metrics.RecordInsert()
			indexed++
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "batch_indexed",
		"count":   indexed,
		"total":   len(reqs),
	})
}

// POST /api/v1/collections/search
func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid json payload: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.Query) != s.cluster.Dim() {
		http.Error(w, fmt.Sprintf("query dimension mismatch: expected %d, got %d", s.cluster.Dim(), len(req.Query)), http.StatusBadRequest)
		return
	}

	if req.K <= 0 {
		req.K = 10
	}
	if req.EfSearch <= 0 {
		req.EfSearch = 64
	}

	start := time.Now()
	matches, err := s.router.Search(req.Query, req.K, req.EfSearch)
	elapsedMs := float64(time.Since(start).Nanoseconds()) / 1e6

	if err != nil {
		http.Error(w, fmt.Sprintf("search error: %v", err), http.StatusInternalServerError)
		return
	}

	s.metrics.RecordQuery(elapsedMs)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(SearchResponse{
		Matches:         matches,
		TookMs:          elapsedMs,
		TotalCandidates: len(matches),
	})
}

// GET /api/v1/metrics
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	snap := s.metrics.Snapshot()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(snap)
}

// GET /api/v1/stream/metrics (Server-Sent Events)
func (s *Server) handleStreamMetrics(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	// Initial immediate flush
	snap := s.metrics.Snapshot()
	data, _ := json.Marshal(snap)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			snap := s.metrics.Snapshot()
			data, err := json.Marshal(snap)
			if err != nil {
				return
			}
			_, writeErr := fmt.Fprintf(w, "data: %s\n\n", data)
			if writeErr != nil {
				return
			}
			flusher.Flush()
		}
	}
}

// POST /api/v1/mode
func (s *Server) handleSetMode(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Mode string `json:"mode"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	s.metrics.SetSimulatedMode(req.Mode)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "updated", "mode": req.Mode})
}

// POST /api/v1/benchmark/run
func (s *Server) handleRunBenchmark(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		DurationSeconds int `json:"duration_seconds"`
		Concurrency     int `json:"concurrency"`
	}
	_ = json.NewDecoder(r.Body).Decode(&req)
	if req.DurationSeconds <= 0 {
		req.DurationSeconds = 5
	}
	if req.Concurrency <= 0 {
		req.Concurrency = 8
	}

	go func() {
		dim := s.cluster.Dim()
		stopTime := time.Now().Add(time.Duration(req.DurationSeconds) * time.Second)
		var wg sync.WaitGroup

		for w := 0; w < req.Concurrency; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				rng := rand.New(rand.NewSource(time.Now().UnixNano()))
				query := make([]float32, dim)
				for time.Now().Before(stopTime) {
					for d := 0; d < dim; d++ {
						query[d] = rng.Float32()
					}
					start := time.Now()
					_, _ = s.router.Search(query, 10, 64)
					tookMs := float64(time.Since(start).Nanoseconds()) / 1e6
					s.metrics.RecordQuery(tookMs)
				}
			}()
		}
		wg.Wait()
	}()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status":           "benchmark_started",
		"duration_seconds": req.DurationSeconds,
		"concurrency":      req.Concurrency,
	})
}

// corsMiddleware attaches CORS headers to every HTTP response.
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

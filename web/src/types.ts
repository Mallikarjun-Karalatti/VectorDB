export interface ShardStat {
  id: number;
  vector_count: number;
  status: string;
  wal_path: string;
  memory_bytes: number;
}

export interface TelemetrySnapshot {
  status: string;
  total_vectors: number;
  active_shards: number;
  total_shards: number;
  memory_rss_bytes: number;
  memory_rss_formatted: string;
  qps: number;
  p50_latency_ms: number;
  p99_latency_ms: number;
  hardware_simd: string;
  simulated_mode: string;
  shards: ShardStat[];
  timestamp: string;
}

export interface SearchResult {
  id: number;
  distance: number;
  shard_id?: number;
}

export interface SearchResponse {
  matches: SearchResult[];
  took_ms: number;
  total_candidates: number;
}

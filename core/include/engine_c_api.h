#ifndef ENGINE_C_API_H
#define ENGINE_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vektor_index vektor_index_t;

// Metric types:
// 0 = L2 / Euclidean distance
// 1 = Cosine / Inner Product distance (1 - cos(theta))
#define VEKTOR_METRIC_L2 0
#define VEKTOR_METRIC_COSINE 1

/**
 * Initialize a new in-memory HNSW index.
 * @param dim Dimensionality of the vectors (e.g. 128, 768)
 * @param max_elements Maximum number of elements that can be stored
 * @param M Number of bi-directional links created for every new element (e.g. 16, 32)
 * @param ef_construction Size of the dynamic candidate list for construction (e.g. 200)
 * @param metric VEKTOR_METRIC_L2 (0) or VEKTOR_METRIC_COSINE (1)
 * @return Pointer to opaque vektor_index_t, or NULL on error
 */
vektor_index_t* vektor_init(int dim, int max_elements, int M, int ef_construction, int metric);

/**
 * Insert or update a vector in the index. Thread-safe.
 * @param index Index handle
 * @param id Unique 64-bit unsigned integer ID
 * @param vec Array of float32 values of size 'dim'
 * @return 0 on success, non-zero on error
 */
int vektor_add(vektor_index_t* index, uint64_t id, const float* vec);

/**
 * Search the index for the K nearest neighbors. Thread-safe.
 * @param index Index handle
 * @param query Query vector of size 'dim'
 * @param k Number of nearest neighbors to return
 * @param ef_search Size of dynamic candidate list during search (e.g. 50, 100)
 * @param out_ids Output array of size at least k (will be populated with IDs)
 * @param out_distances Output array of size at least k (will be populated with distances)
 * @return Number of neighbors found (<= k), or -1 on error
 */
int vektor_search(vektor_index_t* index, const float* query, int k, int ef_search, uint64_t* out_ids, float* out_distances);

/**
 * Serialize the index to disk.
 * @param index Index handle
 * @param path File path
 */
void vektor_save(vektor_index_t* index, const char* path);

/**
 * Deserialize an index from disk.
 * @param path File path
 * @param dim Vector dimensionality
 * @param max_elements Maximum capacity
 * @param metric Metric type
 * @return Pointer to opaque vektor_index_t, or NULL on error
 */
vektor_index_t* vektor_load(const char* path, int dim, int max_elements, int metric);

/**
 * Destroy the index and release all allocated memory.
 * @param index Index handle
 */
void vektor_free(vektor_index_t* index);

/**
 * Get current number of vectors in the index.
 */
size_t vektor_count(vektor_index_t* index);

/**
 * Get vector dimensionality.
 */
int vektor_dim(vektor_index_t* index);

// ============================================================================
// Distance Kernels for Benchmarking & Direct Evaluation
// ============================================================================

float vektor_dist_l2_scalar(const float* a, const float* b, int dim);
float vektor_dist_cosine_scalar(const float* a, const float* b, int dim);

float vektor_dist_l2_simd(const float* a, const float* b, int dim);
float vektor_dist_cosine_simd(const float* a, const float* b, int dim);

int vektor_has_avx2(void);
int vektor_has_neon(void);

#ifdef __cplusplus
}
#endif

#endif // ENGINE_C_API_H

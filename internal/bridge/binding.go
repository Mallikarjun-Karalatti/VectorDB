package bridge

/*
#cgo CFLAGS: -I${SRCDIR}/../../core/include
#cgo LDFLAGS: -L${SRCDIR}/../../core/build -lvektor_core -Wl,-rpath,${SRCDIR}/../../core/build
#include "engine_c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

const (
	MetricL2     = int(C.VEKTOR_METRIC_L2)     // 0
	MetricCosine = int(C.VEKTOR_METRIC_COSINE) // 1
)

var (
	ErrIndexClosed     = errors.New("vektor index is closed or uninitialized")
	ErrInvalidDim      = errors.New("vector dimension mismatch")
	ErrNullVector      = errors.New("vector cannot be empty or nil")
	ErrOperationFailed = errors.New("vektor core operation failed")
)

// SearchResult represents a single KNN match.
type SearchResult struct {
	ID       uint64  `json:"id"`
	Distance float32 `json:"distance"`
	ShardID  int     `json:"shard_id,omitempty"`
}

// Index wraps a pointer to the C++ vektor_index_t HNSW engine.
type Index struct {
	ptr    *C.vektor_index_t
	dim    int
	metric int
	mu     sync.RWMutex
	closed bool
}

// NewIndex initializes a new in-memory HNSW index.
func NewIndex(dim, maxElements, M, efConstruction, metric int) (*Index, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("invalid dim: %d", dim)
	}
	if maxElements <= 0 {
		return nil, fmt.Errorf("invalid maxElements: %d", maxElements)
	}

	ptr := C.vektor_init(
		C.int(dim),
		C.int(maxElements),
		C.int(M),
		C.int(efConstruction),
		C.int(metric),
	)
	if ptr == nil {
		return nil, errors.New("failed to initialize C++ vektor index")
	}

	idx := &Index{
		ptr:    ptr,
		dim:    dim,
		metric: metric,
	}

	runtime.SetFinalizer(idx, (*Index).Close)
	return idx, nil
}

// LoadIndex loads a saved HNSW index from disk.
func LoadIndex(path string, dim, maxElements, metric int) (*Index, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	ptr := C.vektor_load(cPath, C.int(dim), C.int(maxElements), C.int(metric))
	if ptr == nil {
		return nil, fmt.Errorf("failed to load C++ index from %s", path)
	}

	idx := &Index{
		ptr:    ptr,
		dim:    dim,
		metric: metric,
	}

	runtime.SetFinalizer(idx, (*Index).Close)
	return idx, nil
}

// Insert adds or updates a vector associated with id.
func (idx *Index) Insert(id uint64, vec []float32) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.closed || idx.ptr == nil {
		return ErrIndexClosed
	}
	if len(vec) != idx.dim {
		return fmt.Errorf("%w: expected %d, got %d", ErrInvalidDim, idx.dim, len(vec))
	}

	cVec := (*C.float)(unsafe.Pointer(&vec[0]))
	rc := C.vektor_add(idx.ptr, C.uint64_t(id), cVec)
	if rc != 0 {
		return ErrOperationFailed
	}
	return nil
}

// Search queries the index for the top-k nearest neighbors.
func (idx *Index) Search(query []float32, k int, ef int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.closed || idx.ptr == nil {
		return nil, ErrIndexClosed
	}
	if len(query) != idx.dim {
		return nil, fmt.Errorf("%w: expected %d, got %d", ErrInvalidDim, idx.dim, len(query))
	}
	if k <= 0 {
		return []SearchResult{}, nil
	}

	outIDs := make([]uint64, k)
	outDists := make([]float32, k)

	cQuery := (*C.float)(unsafe.Pointer(&query[0]))
	cIDs := (*C.uint64_t)(unsafe.Pointer(&outIDs[0]))
	cDists := (*C.float)(unsafe.Pointer(&outDists[0]))

	n := int(C.vektor_search(idx.ptr, cQuery, C.int(k), C.int(ef), cIDs, cDists))
	if n < 0 {
		return nil, ErrOperationFailed
	}

	results := make([]SearchResult, n)
	for i := 0; i < n; i++ {
		results[i] = SearchResult{
			ID:       outIDs[i],
			Distance: outDists[i],
		}
	}
	return results, nil
}

// Save serializes the index to a file on disk.
func (idx *Index) Save(path string) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.closed || idx.ptr == nil {
		return ErrIndexClosed
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	C.vektor_save(idx.ptr, cPath)
	return nil
}

// Count returns the number of vectors stored in the index.
func (idx *Index) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.closed || idx.ptr == nil {
		return 0
	}
	return int(C.vektor_count(idx.ptr))
}

// Dim returns the dimensionality of vectors in the index.
func (idx *Index) Dim() int {
	return idx.dim
}

// Metric returns the distance metric (0 = L2, 1 = Cosine).
func (idx *Index) Metric() int {
	return idx.metric
}

// Close destroys the underlying C++ index and frees memory.
func (idx *Index) Close() {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if !idx.closed && idx.ptr != nil {
		C.vektor_free(idx.ptr)
		idx.ptr = nil
		idx.closed = true
	}
}

// ============================================================================
// Distance Benchmarking Primitives
// ============================================================================

func DistL2Scalar(a, b []float32) float32 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	return float32(C.vektor_dist_l2_scalar((*C.float)(&a[0]), (*C.float)(&b[0]), C.int(len(a))))
}

func DistCosineScalar(a, b []float32) float32 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	return float32(C.vektor_dist_cosine_scalar((*C.float)(&a[0]), (*C.float)(&b[0]), C.int(len(a))))
}

func DistL2SIMD(a, b []float32) float32 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	return float32(C.vektor_dist_l2_simd((*C.float)(&a[0]), (*C.float)(&b[0]), C.int(len(a))))
}

func DistCosineSIMD(a, b []float32) float32 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	return float32(C.vektor_dist_cosine_simd((*C.float)(&a[0]), (*C.float)(&b[0]), C.int(len(a))))
}

func HasAVX2() bool {
	return C.vektor_has_avx2() != 0
}

func HasNEON() bool {
	return C.vektor_has_neon() != 0
}

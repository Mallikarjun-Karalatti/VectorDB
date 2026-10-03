#include "engine_c_api.h"
#include <cmath>
#include <cstdlib>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

extern "C" {

int vektor_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int vektor_has_neon(void) {
#if defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

// ============================================================================
// Scalar Implementations
// ============================================================================

float vektor_dist_l2_scalar(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float vektor_dist_cosine_scalar(const float* a, const float* b, int dim) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) {
        return 1.0f;
    }
    float cos_sim = dot / denom;
    if (cos_sim > 1.0f) cos_sim = 1.0f;
    if (cos_sim < -1.0f) cos_sim = -1.0f;
    return 1.0f - cos_sim;
}

// ============================================================================
// SIMD (AVX2 / ARM NEON / Auto-vectorized OpenMP SIMD) Implementations
// ============================================================================

#if defined(__AVX2__)

static inline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum = _mm_add_ps(vlow, vhigh);
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    return _mm_cvtss_f32(vsum);
}

float vektor_dist_l2_simd(const float* a, const float* b, int dim) {
    int i = 0;
    __m256 acc = _mm256_setzero_ps();

    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(diff, diff, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
#endif
    }

    float sum = hsum256_ps(acc);

    // Scalar tail
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

float vektor_dist_cosine_simd(const float* a, const float* b, int dim) {
    int i = 0;
    __m256 acc_dot = _mm256_setzero_ps();
    __m256 acc_na  = _mm256_setzero_ps();
    __m256 acc_nb  = _mm256_setzero_ps();

    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        acc_dot = _mm256_fmadd_ps(va, vb, acc_dot);
        acc_na  = _mm256_fmadd_ps(va, va, acc_na);
        acc_nb  = _mm256_fmadd_ps(vb, vb, acc_nb);
#else
        acc_dot = _mm256_add_ps(acc_dot, _mm256_mul_ps(va, vb));
        acc_na  = _mm256_add_ps(acc_na,  _mm256_mul_ps(va, va));
        acc_nb  = _mm256_add_ps(acc_nb,  _mm256_mul_ps(vb, vb));
#endif
    }

    float dot    = hsum256_ps(acc_dot);
    float norm_a = hsum256_ps(acc_na);
    float norm_b = hsum256_ps(acc_nb);

    // Scalar tail
    for (; i < dim; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) {
        return 1.0f;
    }
    float cos_sim = dot / denom;
    if (cos_sim > 1.0f) cos_sim = 1.0f;
    if (cos_sim < -1.0f) cos_sim = -1.0f;
    return 1.0f - cos_sim;
}

#elif defined(__ARM_NEON)

float vektor_dist_l2_simd(const float* a, const float* b, int dim) {
    int i = 0;
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    for (; i <= dim - 8; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t diff0 = vsubq_f32(va0, vb0);
        acc0 = vfmaq_f32(acc0, diff0, diff0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t diff1 = vsubq_f32(va1, vb1);
        acc1 = vfmaq_f32(acc1, diff1, diff1);
    }

    for (; i <= dim - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        acc0 = vfmaq_f32(acc0, diff, diff);
    }

    float32x4_t total_acc = vaddq_f32(acc0, acc1);
#if defined(__aarch64__)
    float sum = vaddvq_f32(total_acc);
#else
    float32x2_t vsum = vadd_f32(vget_low_f32(total_acc), vget_high_f32(total_acc));
    float sum = vget_lane_f32(vpadd_f32(vsum, vsum), 0);
#endif

    // Tail
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

float vektor_dist_cosine_simd(const float* a, const float* b, int dim) {
    int i = 0;
    float32x4_t acc_dot = vdupq_n_f32(0.0f);
    float32x4_t acc_na  = vdupq_n_f32(0.0f);
    float32x4_t acc_nb  = vdupq_n_f32(0.0f);

    for (; i <= dim - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc_dot = vfmaq_f32(acc_dot, va, vb);
        acc_na  = vfmaq_f32(acc_na, va, va);
        acc_nb  = vfmaq_f32(acc_nb, vb, vb);
    }

#if defined(__aarch64__)
    float dot    = vaddvq_f32(acc_dot);
    float norm_a = vaddvq_f32(acc_na);
    float norm_b = vaddvq_f32(acc_nb);
#else
    float32x2_t dsum = vadd_f32(vget_low_f32(acc_dot), vget_high_f32(acc_dot));
    float dot = vget_lane_f32(vpadd_f32(dsum, dsum), 0);

    float32x2_t nasum = vadd_f32(vget_low_f32(acc_na), vget_high_f32(acc_na));
    float norm_a = vget_lane_f32(vpadd_f32(nasum, nasum), 0);

    float32x2_t nbsum = vadd_f32(vget_low_f32(acc_nb), vget_high_f32(acc_nb));
    float norm_b = vget_lane_f32(vpadd_f32(nbsum, nbsum), 0);
#endif

    // Tail
    for (; i < dim; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) {
        return 1.0f;
    }
    float cos_sim = dot / denom;
    if (cos_sim > 1.0f) cos_sim = 1.0f;
    if (cos_sim < -1.0f) cos_sim = -1.0f;
    return 1.0f - cos_sim;
}

#else

// OpenMP SIMD / Auto-vectorized fallback
float vektor_dist_l2_simd(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float vektor_dist_cosine_simd(const float* a, const float* b, int dim) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    #pragma omp simd reduction(+:dot, norm_a, norm_b)
    for (int i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) {
        return 1.0f;
    }
    float cos_sim = dot / denom;
    if (cos_sim > 1.0f) cos_sim = 1.0f;
    if (cos_sim < -1.0f) cos_sim = -1.0f;
    return 1.0f - cos_sim;
}

#endif

} // extern "C"

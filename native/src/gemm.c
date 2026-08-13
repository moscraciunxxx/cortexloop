#include "internal.h"

#include <string.h>

#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#endif

void cl_gemm_i8_dotprod(int m, int n, int k, const int8_t *a, const int8_t *b, const float *a_scale,
                        const float *b_scale, const float *bias, float *c) {
#if !defined(__ARM_FEATURE_DOTPROD)
    cl_gemm_i8_scalar(m, n, k, a, b, a_scale, b_scale, bias, c);
#else
    for (int i = 0; i < m; i++) {
        const int8_t *ap = a + (size_t)i * (size_t)k;
        const float as = a_scale[i];
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);
            const int8_t *b0 = b + (size_t)(j + 0) * (size_t)k;
            const int8_t *b1 = b + (size_t)(j + 1) * (size_t)k;
            const int8_t *b2 = b + (size_t)(j + 2) * (size_t)k;
            const int8_t *b3 = b + (size_t)(j + 3) * (size_t)k;
            int t = 0;
            for (; t + 16 <= k; t += 16) {
                int8x16_t va = vld1q_s8(ap + t);
                acc0 = vdotq_s32(acc0, va, vld1q_s8(b0 + t));
                acc1 = vdotq_s32(acc1, va, vld1q_s8(b1 + t));
                acc2 = vdotq_s32(acc2, va, vld1q_s8(b2 + t));
                acc3 = vdotq_s32(acc3, va, vld1q_s8(b3 + t));
            }
            int32_t s0 = vaddvq_s32(acc0);
            int32_t s1 = vaddvq_s32(acc1);
            int32_t s2 = vaddvq_s32(acc2);
            int32_t s3 = vaddvq_s32(acc3);
            for (; t < k; t++) {
                int32_t av = ap[t];
                s0 += av * b0[t];
                s1 += av * b1[t];
                s2 += av * b2[t];
                s3 += av * b3[t];
            }
            float *out = c + (size_t)i * (size_t)n + (size_t)j;
            out[0] = (float)s0 * as * b_scale[j] + (bias ? bias[j] : 0.f);
            out[1] = (float)s1 * as * b_scale[j + 1] + (bias ? bias[j + 1] : 0.f);
            out[2] = (float)s2 * as * b_scale[j + 2] + (bias ? bias[j + 2] : 0.f);
            out[3] = (float)s3 * as * b_scale[j + 3] + (bias ? bias[j + 3] : 0.f);
        }
        for (; j < n; j++) {
            const int8_t *bp = b + (size_t)j * (size_t)k;
            int32x4_t acc = vdupq_n_s32(0);
            int t = 0;
            for (; t + 16 <= k; t += 16)
                acc = vdotq_s32(acc, vld1q_s8(ap + t), vld1q_s8(bp + t));
            int32_t s = vaddvq_s32(acc);
            for (; t < k; t++)
                s += (int32_t)ap[t] * (int32_t)bp[t];
            c[(size_t)i * (size_t)n + (size_t)j] =
                (float)s * as * b_scale[j] + (bias ? bias[j] : 0.f);
        }
    }
#endif
}

void cl_gemm_i8_i8mm(int m, int n, int k, const int8_t *a, const int8_t *b, const float *a_scale,
                     const float *b_scale, const float *bias, float *c) {
#if !defined(__ARM_FEATURE_MATMUL_INT8)
    cl_gemm_i8_dotprod(m, n, k, a, b, a_scale, b_scale, bias, c);
#else
    /* I8MM SMMLA accumulates a 2x2 tile of int32 from two 2x8 int8 panels. */
    int i = 0;
    for (; i + 2 <= m; i += 2) {
        const int8_t *a0 = a + (size_t)i * (size_t)k;
        const int8_t *a1 = a + (size_t)(i + 1) * (size_t)k;
        int j = 0;
        for (; j + 2 <= n; j += 2) {
            const int8_t *b0 = b + (size_t)j * (size_t)k;
            const int8_t *b1 = b + (size_t)(j + 1) * (size_t)k;
            int32x4_t acc = vdupq_n_s32(0);
            int t = 0;
            for (; t + 8 <= k; t += 8) {
                int8x8_t a0v = vld1_s8(a0 + t);
                int8x8_t a1v = vld1_s8(a1 + t);
                int8x8_t b0v = vld1_s8(b0 + t);
                int8x8_t b1v = vld1_s8(b1 + t);
                int8x16_t av = vcombine_s8(a0v, a1v);
                int8x16_t bv = vcombine_s8(b0v, b1v);
                acc = vmmlaq_s32(acc, av, bv);
            }
            int32_t tile[4];
            vst1q_s32(tile, acc);
            int32_t s00 = tile[0], s01 = tile[1], s10 = tile[2], s11 = tile[3];
            for (; t < k; t++) {
                s00 += (int32_t)a0[t] * (int32_t)b0[t];
                s01 += (int32_t)a0[t] * (int32_t)b1[t];
                s10 += (int32_t)a1[t] * (int32_t)b0[t];
                s11 += (int32_t)a1[t] * (int32_t)b1[t];
            }
            c[(size_t)i * (size_t)n + (size_t)j] =
                (float)s00 * a_scale[i] * b_scale[j] + (bias ? bias[j] : 0.f);
            c[(size_t)i * (size_t)n + (size_t)j + 1] =
                (float)s01 * a_scale[i] * b_scale[j + 1] + (bias ? bias[j + 1] : 0.f);
            c[(size_t)(i + 1) * (size_t)n + (size_t)j] =
                (float)s10 * a_scale[i + 1] * b_scale[j] + (bias ? bias[j] : 0.f);
            c[(size_t)(i + 1) * (size_t)n + (size_t)j + 1] =
                (float)s11 * a_scale[i + 1] * b_scale[j + 1] + (bias ? bias[j + 1] : 0.f);
        }
        if (j < n) {
            for (int ii = 0; ii < 2; ii++) {
                cl_gemm_i8_dotprod(1, n - j, k, a + (size_t)(i + ii) * (size_t)k,
                                   b + (size_t)j * (size_t)k, a_scale + i + ii, b_scale + j,
                                   bias ? bias + j : NULL,
                                   c + (size_t)(i + ii) * (size_t)n + (size_t)j);
            }
        }
    }
    if (i < m) {
        cl_gemm_i8_dotprod(m - i, n, k, a + (size_t)i * (size_t)k, b, a_scale + i, b_scale, bias,
                           c + (size_t)i * (size_t)n);
    }
#endif
}

cl_gemm_fn cl_active_gemm(void) {
    switch (g_backend) {
    case CL_BACKEND_SCALAR:
        return cl_gemm_i8_scalar;
    case CL_BACKEND_NEON_I8MM:
        return cl_gemm_i8_i8mm;
    case CL_BACKEND_KLEIDIAI:
        /* Application CNN path still uses I8MM/DotProd; KleidiAI is the INT4 GEMM engine. */
        return cl_gemm_i8_i8mm;
    case CL_BACKEND_NEON_DOTPROD:
    default:
        return cl_gemm_i8_dotprod;
    }
}

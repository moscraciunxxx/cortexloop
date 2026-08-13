#include "internal.h"

#ifndef CORTEXLOOP_WITH_KLEIDIAI

int cl_kleidiai_available(void) { return 0; }

int cl_kleidiai_gemm_f32(int m, int n, int k, const float *a, const float *b, float *c,
                         double *pack_us, double *compute_us) {
    (void)m;
    (void)n;
    (void)k;
    (void)a;
    (void)b;
    (void)c;
    if (pack_us)
        *pack_us = 0;
    if (compute_us)
        *compute_us = 0;
    return -1;
}

#else

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"

#define INT4_MIN (-8)
#define INT4_MAX (7)

static size_t roundup(size_t a, size_t b) { return ((a + b - 1) / b) * b; }

static void quant_nxk_qs4cx_f32(size_t n, size_t k, const float *rhs_f32, uint8_t *rhs_qs4cx,
                                float *rhs_scales_f32) {
    const size_t stride = roundup(k, 2) / 2;
    std::memset(rhs_qs4cx, 0, n * stride);
    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        const float *src = rhs_f32 + n_idx * k;
        float max0 = -std::numeric_limits<float>::max();
        float min0 = std::numeric_limits<float>::max();
        for (size_t t = 0; t < k; ++t) {
            max0 = std::max(max0, src[t]);
            min0 = std::min(min0, src[t]);
        }
        const float rmin0 = std::min(0.0f, min0);
        const float rmax0 = std::max(0.0f, max0);
        const float scale0 = (rmin0 == rmax0) ? 1.f : (float)(INT4_MAX - INT4_MIN) / (rmax0 - rmin0);
        const float recip = scale0 ? 1.0f / scale0 : 0.0f;
        for (size_t t = 0; t < k; ++t) {
            int v = (int)std::round(src[t] * scale0);
            v = std::max(v, INT4_MIN);
            v = std::min(v, INT4_MAX);
            const uint8_t u = (uint8_t)(v + 8);
            const size_t addr = (t / 2) + n_idx * stride;
            if ((t % 2) == 0)
                rhs_qs4cx[addr] |= u;
            else
                rhs_qs4cx[addr] |= (uint8_t)(u << 4);
        }
        rhs_scales_f32[n_idx] = recip;
    }
}

int cl_kleidiai_available(void) { return 1; }

int cl_kleidiai_gemm_f32(int m, int n, int k, const float *a, const float *b, float *c,
                         double *pack_us, double *compute_us) {
    if (m <= 0 || n <= 0 || k <= 0 || !a || !b || !c)
        return -1;

    const bool use_i8mm = (cl_detect_features() & CL_FEAT_I8MM) != 0;
    const size_t mr = use_i8mm ? kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm()
                               : kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod();
    const size_t nr = use_i8mm ? kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm()
                               : kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod();
    const size_t kr = use_i8mm ? kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm()
                               : kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod();
    const size_t sr = use_i8mm ? kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm()
                               : kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod();

    static std::vector<uint8_t> rhs_packed;
    static int cached_n = -1, cached_k = -1;
    static const float *cached_b = nullptr;

    double t0 = cl_now_us();
    if (cached_n != n || cached_k != k || cached_b != b) {
        const size_t rhs_qs4_size = (size_t)n * (roundup((size_t)k, 2) / 2);
        std::vector<uint8_t> rhs_qs4(rhs_qs4_size);
        std::vector<float> rhs_scales((size_t)n);
        quant_nxk_qs4cx_f32((size_t)n, (size_t)k, b, rhs_qs4.data(), rhs_scales.data());
        const size_t rhs_packed_size =
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0((size_t)n, (size_t)k, nr, kr, sr);
        rhs_packed.resize(rhs_packed_size);
        struct kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0_params params;
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;
        kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(1, (size_t)n, (size_t)k, nr, kr, sr, rhs_qs4.data(),
                                               nullptr, rhs_scales.data(), rhs_packed.data(), 0,
                                               &params);
        cached_n = n;
        cached_k = k;
        cached_b = b;
    }

    const size_t lhs_packed_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32((size_t)m, (size_t)k, mr, kr, sr);
    std::vector<uint8_t> lhs_packed(lhs_packed_size);
    kai_run_lhs_quant_pack_qai8dxp_f32((size_t)m, (size_t)k, mr, kr, sr, 0, a,
                                       (size_t)k * sizeof(float), lhs_packed.data());
    double t1 = cl_now_us();

    const size_t dst_stride = (size_t)n * sizeof(float);
    if (use_i8mm) {
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm(
            (size_t)m, (size_t)n, (size_t)k, lhs_packed.data(), rhs_packed.data(), c, dst_stride,
            sizeof(float), -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    } else {
        kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(
            (size_t)m, (size_t)n, (size_t)k, lhs_packed.data(), rhs_packed.data(), c, dst_stride,
            sizeof(float), -std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    }
    double t2 = cl_now_us();
    if (pack_us)
        *pack_us = t1 - t0;
    if (compute_us)
        *compute_us = t2 - t1;
    return 0;
}

#endif

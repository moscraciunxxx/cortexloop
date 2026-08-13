#include "internal.h"

void cl_gemm_i8_scalar(int m, int n, int k, const int8_t *a, const int8_t *b, const float *a_scale,
                       const float *b_scale, const float *bias, float *c) {
    for (int i = 0; i < m; i++) {
        const int8_t *ap = a + (size_t)i * (size_t)k;
        for (int j = 0; j < n; j++) {
            const int8_t *bp = b + (size_t)j * (size_t)k;
            int32_t acc = 0;
            for (int t = 0; t < k; t++)
                acc += (int32_t)ap[t] * (int32_t)bp[t];
            float v = (float)acc * a_scale[i] * b_scale[j];
            if (bias)
                v += bias[j];
            c[(size_t)i * (size_t)n + (size_t)j] = v;
        }
    }
}

static inline uint8_t cl_clamp_u8(int v) {
    if (v < 0)
        return 0;
    if (v > 255)
        return 255;
    return (uint8_t)v;
}

void cl_resize_bilinear_u8_scalar(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw,
                                  int dh) {
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
        return;
    const float x_scale = (float)sw / (float)dw;
    const float y_scale = (float)sh / (float)dh;
    for (int y = 0; y < dh; y++) {
        float fy = ((float)y + 0.5f) * y_scale - 0.5f;
        if (fy < 0)
            fy = 0;
        int y0 = (int)fy;
        int y1 = y0 + 1;
        if (y0 >= sh)
            y0 = sh - 1;
        if (y1 >= sh)
            y1 = sh - 1;
        float wy = fy - (float)y0;
        for (int x = 0; x < dw; x++) {
            float fx = ((float)x + 0.5f) * x_scale - 0.5f;
            if (fx < 0)
                fx = 0;
            int x0 = (int)fx;
            int x1 = x0 + 1;
            if (x0 >= sw)
                x0 = sw - 1;
            if (x1 >= sw)
                x1 = sw - 1;
            float wx = fx - (float)x0;
            for (int c = 0; c < 3; c++) {
                float p00 = src[(y0 * sw + x0) * 3 + c];
                float p10 = src[(y0 * sw + x1) * 3 + c];
                float p01 = src[(y1 * sw + x0) * 3 + c];
                float p11 = src[(y1 * sw + x1) * 3 + c];
                float top = p00 + (p10 - p00) * wx;
                float bot = p01 + (p11 - p01) * wx;
                dst[(y * dw + x) * 3 + c] = cl_clamp_u8((int)(top + (bot - top) * wy + 0.5f));
            }
        }
    }
}

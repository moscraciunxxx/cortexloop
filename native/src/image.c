#include "internal.h"

#include <math.h>
#include <string.h>

#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#endif

static inline uint8_t cl_clamp_u8(int v) {
    if (v < 0)
        return 0;
    if (v > 255)
        return 255;
    return (uint8_t)v;
}

void cl_resize_bilinear_u8_neon(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh) {
#if !(defined(__aarch64__) || defined(__arm64__))
    cl_resize_bilinear_u8_scalar(src, sw, sh, dst, dw, dh);
    return;
#else
    if (sw < 2 || sh < 2 || dw < 1 || dh < 1) {
        cl_resize_bilinear_u8_scalar(src, sw, sh, dst, dw, dh);
        return;
    }
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
        const uint8_t *row0 = src + (size_t)y0 * (size_t)sw * 3;
        const uint8_t *row1 = src + (size_t)y1 * (size_t)sw * 3;
        int x = 0;
        for (; x + 4 <= dw; x += 4) {
            float fx[4];
            int x0[4], x1[4];
            float wx[4];
            for (int i = 0; i < 4; i++) {
                fx[i] = ((float)(x + i) + 0.5f) * x_scale - 0.5f;
                if (fx[i] < 0)
                    fx[i] = 0;
                x0[i] = (int)fx[i];
                x1[i] = x0[i] + 1;
                if (x0[i] >= sw)
                    x0[i] = sw - 1;
                if (x1[i] >= sw)
                    x1[i] = sw - 1;
                wx[i] = fx[i] - (float)x0[i];
            }
            uint8_t *out = dst + (size_t)(y * dw + x) * 3;
            for (int i = 0; i < 4; i++) {
                const uint8_t *p00 = row0 + x0[i] * 3;
                const uint8_t *p10 = row0 + x1[i] * 3;
                const uint8_t *p01 = row1 + x0[i] * 3;
                const uint8_t *p11 = row1 + x1[i] * 3;
                float32x4_t a = {p00[0], p00[1], p00[2], 0};
                float32x4_t b = {p10[0], p10[1], p10[2], 0};
                float32x4_t c = {p01[0], p01[1], p01[2], 0};
                float32x4_t d = {p11[0], p11[1], p11[2], 0};
                float32x4_t top = vmlaq_n_f32(a, vsubq_f32(b, a), wx[i]);
                float32x4_t bot = vmlaq_n_f32(c, vsubq_f32(d, c), wx[i]);
                float32x4_t pix = vmlaq_n_f32(top, vsubq_f32(bot, top), wy);
                float tmp[4];
                vst1q_f32(tmp, pix);
                out[i * 3 + 0] = cl_clamp_u8((int)(tmp[0] + 0.5f));
                out[i * 3 + 1] = cl_clamp_u8((int)(tmp[1] + 0.5f));
                out[i * 3 + 2] = cl_clamp_u8((int)(tmp[2] + 0.5f));
            }
        }
        for (; x < dw; x++) {
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
            uint8_t *out = dst + (size_t)(y * dw + x) * 3;
            for (int c = 0; c < 3; c++) {
                float p00 = row0[x0 * 3 + c];
                float p10 = row0[x1 * 3 + c];
                float p01 = row1[x0 * 3 + c];
                float p11 = row1[x1 * 3 + c];
                float top = p00 + (p10 - p00) * wx;
                float bot = p01 + (p11 - p01) * wx;
                out[c] = cl_clamp_u8((int)(top + (bot - top) * wy + 0.5f));
            }
        }
    }
#endif
}

void cl_im2col_u8(const uint8_t *src, int h, int w, int c, int kh, int kw, int stride, int pad,
                  int8_t *col, int *out_h, int *out_w) {
    const int oh = (h + 2 * pad - kh) / stride + 1;
    const int ow = (w + 2 * pad - kw) / stride + 1;
    *out_h = oh;
    *out_w = ow;
    const int k = c * kh * kw;
    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            int8_t *dst = col + ((size_t)oy * (size_t)ow + (size_t)ox) * (size_t)k;
            int idx = 0;
            for (int ky = 0; ky < kh; ky++) {
                int iy = oy * stride + ky - pad;
                for (int kx = 0; kx < kw; kx++) {
                    int ix = ox * stride + kx - pad;
                    if (iy < 0 || iy >= h || ix < 0 || ix >= w) {
                        for (int cc = 0; cc < c; cc++)
                            dst[idx++] = 0;
                    } else {
                        const uint8_t *p = src + ((size_t)iy * (size_t)w + (size_t)ix) * (size_t)c;
                        for (int cc = 0; cc < c; cc++)
                            dst[idx++] = (int8_t)((int)p[cc] - 128);
                    }
                }
            }
        }
    }
}

void cl_quant_rows_f32(const float *src, int rows, int cols, int8_t *dst, float *scales) {
    for (int r = 0; r < rows; r++) {
        const float *row = src + (size_t)r * (size_t)cols;
        float amax = 0.f;
        for (int i = 0; i < cols; i++) {
            float a = row[i] < 0 ? -row[i] : row[i];
            if (a > amax)
                amax = a;
        }
        float scale = amax > 0.f ? amax / 127.f : 1.f;
        scales[r] = scale;
        float inv = 1.f / scale;
        int8_t *d = dst + (size_t)r * (size_t)cols;
        int i = 0;
#if defined(__aarch64__) || defined(__arm64__)
        float32x4_t vinv = vdupq_n_f32(inv);
        float32x4_t vlo = vdupq_n_f32(-127.f);
        float32x4_t vhi = vdupq_n_f32(127.f);
        for (; i + 16 <= cols; i += 16) {
            float32x4_t a0 = vmulq_f32(vld1q_f32(row + i), vinv);
            float32x4_t a1 = vmulq_f32(vld1q_f32(row + i + 4), vinv);
            float32x4_t a2 = vmulq_f32(vld1q_f32(row + i + 8), vinv);
            float32x4_t a3 = vmulq_f32(vld1q_f32(row + i + 12), vinv);
            a0 = vmaxq_f32(vlo, vminq_f32(vhi, vrndnq_f32(a0)));
            a1 = vmaxq_f32(vlo, vminq_f32(vhi, vrndnq_f32(a1)));
            a2 = vmaxq_f32(vlo, vminq_f32(vhi, vrndnq_f32(a2)));
            a3 = vmaxq_f32(vlo, vminq_f32(vhi, vrndnq_f32(a3)));
            int32x4_t i0 = vcvtq_s32_f32(a0);
            int32x4_t i1 = vcvtq_s32_f32(a1);
            int32x4_t i2 = vcvtq_s32_f32(a2);
            int32x4_t i3 = vcvtq_s32_f32(a3);
            int16x8_t s0 = vcombine_s16(vmovn_s32(i0), vmovn_s32(i1));
            int16x8_t s1 = vcombine_s16(vmovn_s32(i2), vmovn_s32(i3));
            vst1q_s8(d + i, vcombine_s8(vmovn_s16(s0), vmovn_s16(s1)));
        }
#endif
        for (; i < cols; i++) {
            int v = (int)(row[i] * inv + (row[i] >= 0 ? 0.5f : -0.5f));
            if (v > 127)
                v = 127;
            if (v < -127)
                v = -127;
            d[i] = (int8_t)v;
        }
    }
}

void cl_relu(float *x, int n) {
    int i = 0;
#if defined(__aarch64__) || defined(__arm64__)
    float32x4_t z = vdupq_n_f32(0.f);
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(x + i, vmaxq_f32(vld1q_f32(x + i), z));
        vst1q_f32(x + i + 4, vmaxq_f32(vld1q_f32(x + i + 4), z));
        vst1q_f32(x + i + 8, vmaxq_f32(vld1q_f32(x + i + 8), z));
        vst1q_f32(x + i + 12, vmaxq_f32(vld1q_f32(x + i + 12), z));
    }
#endif
    for (; i < n; i++)
        if (x[i] < 0)
            x[i] = 0;
}

void cl_gap(const float *src, int h, int w, int c, float *dst) {
    const float inv = 1.f / (float)(h * w);
    for (int ch = 0; ch < c; ch++)
        dst[ch] = 0.f;
    for (int i = 0; i < h * w; i++) {
        const float *p = src + (size_t)i * (size_t)c;
        for (int ch = 0; ch < c; ch++)
            dst[ch] += p[ch];
    }
    for (int ch = 0; ch < c; ch++)
        dst[ch] *= inv;
}

void cl_softmax(const float *logits, int n, float *probs) {
    float m = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > m)
            m = logits[i];
    float s = 0.f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - m);
        s += probs[i];
    }
    float inv = 1.f / s;
    for (int i = 0; i < n; i++)
        probs[i] *= inv;
}

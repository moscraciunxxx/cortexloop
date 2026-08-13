#include "internal.h"

#include <stdio.h>
#include <string.h>

static const char *k_magic = "CL01";

static int read_exact(FILE *f, void *dst, size_t n) { return fread(dst, 1, n, f) == n; }

static int8_t *read_i8(FILE *f, size_t n) {
    int8_t *p = (int8_t *)cl_xmalloc(n);
    if (!read_exact(f, p, n)) {
        cl_xfree(p);
        return NULL;
    }
    return p;
}

static float *read_f32(FILE *f, size_t n) {
    float *p = (float *)cl_xmalloc(n * sizeof(float));
    if (!read_exact(f, p, n * sizeof(float))) {
        cl_xfree(p);
        return NULL;
    }
    return p;
}

int cl_load_model(const char *path) {
    cl_free_model();
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;
    char magic[4];
    int32_t n_classes = 0, in_w = 0, in_h = 0, in_c = 0;
    if (!read_exact(f, magic, 4) || memcmp(magic, k_magic, 4) != 0) {
        fclose(f);
        return -2;
    }
    if (!read_exact(f, &n_classes, 4) || !read_exact(f, &in_w, 4) || !read_exact(f, &in_h, 4) ||
        !read_exact(f, &in_c, 4)) {
        fclose(f);
        return -2;
    }
    if (n_classes != CL_N_CLASSES || in_w != CL_IN_W || in_h != CL_IN_H || in_c != CL_IN_C) {
        fclose(f);
        return -3;
    }

    int32_t c1n, c1c, c1kh, c1kw, c1s, c1p;
    if (!read_exact(f, &c1n, 4) || !read_exact(f, &c1c, 4) || !read_exact(f, &c1kh, 4) ||
        !read_exact(f, &c1kw, 4) || !read_exact(f, &c1s, 4) || !read_exact(f, &c1p, 4)) {
        fclose(f);
        return -2;
    }
    g_model.conv1.n = c1n;
    g_model.conv1.c = c1c;
    g_model.conv1.kh = c1kh;
    g_model.conv1.kw = c1kw;
    g_model.conv1.stride = c1s;
    g_model.conv1.pad = c1p;
    size_t c1w = (size_t)c1n * (size_t)c1c * (size_t)c1kh * (size_t)c1kw;
    g_model.conv1.w = read_i8(f, c1w);
    g_model.conv1.scale = read_f32(f, (size_t)c1n);
    g_model.conv1.bias = read_f32(f, (size_t)c1n);

    int32_t c2n, c2c, c2kh, c2kw, c2s, c2p;
    if (!read_exact(f, &c2n, 4) || !read_exact(f, &c2c, 4) || !read_exact(f, &c2kh, 4) ||
        !read_exact(f, &c2kw, 4) || !read_exact(f, &c2s, 4) || !read_exact(f, &c2p, 4)) {
        fclose(f);
        return -2;
    }
    g_model.conv2.n = c2n;
    g_model.conv2.c = c2c;
    g_model.conv2.kh = c2kh;
    g_model.conv2.kw = c2kw;
    g_model.conv2.stride = c2s;
    g_model.conv2.pad = c2p;
    size_t c2w = (size_t)c2n * (size_t)c2c * (size_t)c2kh * (size_t)c2kw;
    g_model.conv2.w = read_i8(f, c2w);
    g_model.conv2.scale = read_f32(f, (size_t)c2n);
    g_model.conv2.bias = read_f32(f, (size_t)c2n);

    int32_t fcout, fcin;
    if (!read_exact(f, &fcout, 4) || !read_exact(f, &fcin, 4)) {
        fclose(f);
        return -2;
    }
    g_model.fc.out = fcout;
    g_model.fc.in = fcin;
    g_model.fc.w = read_i8(f, (size_t)fcout * (size_t)fcin);
    g_model.fc.scale = read_f32(f, (size_t)fcout);
    g_model.fc.bias = read_f32(f, (size_t)fcout);
    fclose(f);

    if (!g_model.conv1.w || !g_model.conv2.w || !g_model.fc.w)
        return -2;
    g_model.loaded = 1;
    return 0;
}

void cl_free_model(void) {
    cl_xfree(g_model.conv1.w);
    cl_xfree(g_model.conv1.scale);
    cl_xfree(g_model.conv1.bias);
    cl_xfree(g_model.conv2.w);
    cl_xfree(g_model.conv2.scale);
    cl_xfree(g_model.conv2.bias);
    cl_xfree(g_model.fc.w);
    cl_xfree(g_model.fc.scale);
    cl_xfree(g_model.fc.bias);
    cl_xfree(g_model.scratch);
    memset(&g_model, 0, sizeof(g_model));
}

size_t cl_model_nbytes(void) {
    if (!g_model.loaded)
        return 0;
    size_t n = 0;
    n += (size_t)g_model.conv1.n * g_model.conv1.c * g_model.conv1.kh * g_model.conv1.kw;
    n += (size_t)g_model.conv2.n * g_model.conv2.c * g_model.conv2.kh * g_model.conv2.kw;
    n += (size_t)g_model.fc.out * g_model.fc.in;
    n += (size_t)(g_model.conv1.n + g_model.conv2.n + g_model.fc.out) * (sizeof(float) * 2);
    return n;
}

void cl_conv_forward(const cl_conv *layer, const uint8_t *src, int h, int w, int c, float *dst,
                     int *out_h, int *out_w, int use_u8_src, const float *src_f32) {
    int oh, ow;
    const int kdim = layer->c * layer->kh * layer->kw;
    const int spatial = ((h + 2 * layer->pad - layer->kh) / layer->stride + 1) *
                        ((w + 2 * layer->pad - layer->kw) / layer->stride + 1);
    int8_t *col = (int8_t *)cl_scratch((size_t)spatial * (size_t)kdim + 64);
    if (use_u8_src) {
        cl_im2col_u8(src, h, w, c, layer->kh, layer->kw, layer->stride, layer->pad, col, &oh, &ow);
    } else {
        oh = (h + 2 * layer->pad - layer->kh) / layer->stride + 1;
        ow = (w + 2 * layer->pad - layer->kw) / layer->stride + 1;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int8_t *d = col + ((size_t)oy * (size_t)ow + (size_t)ox) * (size_t)kdim;
                int idx = 0;
                for (int ky = 0; ky < layer->kh; ky++) {
                    int iy = oy * layer->stride + ky - layer->pad;
                    for (int kx = 0; kx < layer->kw; kx++) {
                        int ix = ox * layer->stride + kx - layer->pad;
                        if (iy < 0 || iy >= h || ix < 0 || ix >= w) {
                            for (int cc = 0; cc < layer->c; cc++)
                                d[idx++] = 0;
                        } else {
                            const float *p =
                                src_f32 + ((size_t)iy * (size_t)w + (size_t)ix) * (size_t)layer->c;
                            for (int cc = 0; cc < layer->c; cc++) {
                                float v = p[cc];
                                int q = (int)(v + (v >= 0 ? 0.5f : -0.5f));
                                if (q > 127)
                                    q = 127;
                                if (q < -127)
                                    q = -127;
                                d[idx++] = (int8_t)q;
                            }
                        }
                    }
                }
            }
        }
    }
    *out_h = oh;
    *out_w = ow;
    float *a_scale = (float *)cl_xmalloc((size_t)(oh * ow) * sizeof(float));
    if (use_u8_src) {
        for (int i = 0; i < oh * ow; i++)
            a_scale[i] = 1.f / 128.f;
    } else {
        /* Activations were rounded to int8 with scale 1 after ReLU in float. */
        for (int i = 0; i < oh * ow; i++)
            a_scale[i] = 1.f;
    }
    cl_gemm_fn gemm = cl_active_gemm();
    gemm(oh * ow, layer->n, kdim, col, layer->w, a_scale, layer->scale, layer->bias, dst);
    cl_xfree(a_scale);
}

static void conv2_from_float(const cl_conv *layer, const float *src, int h, int w, int c,
                             float *dst, int *out_h, int *out_w) {
    const int oh = (h + 2 * layer->pad - layer->kh) / layer->stride + 1;
    const int ow = (w + 2 * layer->pad - layer->kw) / layer->stride + 1;
    const int kdim = layer->c * layer->kh * layer->kw;
    const int spatial = oh * ow;
    int8_t *col = (int8_t *)cl_xmalloc((size_t)spatial * (size_t)kdim);
    float *a_scale = (float *)cl_xmalloc((size_t)spatial * sizeof(float));
    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            float patch[512];
            int idx = 0;
            for (int ky = 0; ky < layer->kh; ky++) {
                int iy = oy * layer->stride + ky - layer->pad;
                for (int kx = 0; kx < layer->kw; kx++) {
                    int ix = ox * layer->stride + kx - layer->pad;
                    for (int cc = 0; cc < c; cc++) {
                        if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                            patch[idx++] = 0.f;
                        else
                            patch[idx++] =
                                src[((size_t)iy * (size_t)w + (size_t)ix) * (size_t)c + (size_t)cc];
                    }
                }
            }
            int row = oy * ow + ox;
            cl_quant_rows_f32(patch, 1, kdim, col + (size_t)row * (size_t)kdim, a_scale + row);
        }
    }
    cl_gemm_fn gemm = cl_active_gemm();
    gemm(spatial, layer->n, kdim, col, layer->w, a_scale, layer->scale, layer->bias, dst);
    *out_h = oh;
    *out_w = ow;
    cl_xfree(col);
    cl_xfree(a_scale);
}

cl_perception cl_cnn_forward(const uint8_t *rgb32) {
    cl_perception r;
    memset(&r, 0, sizeof(r));
    int h1, w1, h2, w2;
    float *c1 = (float *)cl_xmalloc((size_t)16 * 16 * g_model.conv1.n * sizeof(float));
    cl_conv_forward(&g_model.conv1, rgb32, CL_IN_H, CL_IN_W, 3, c1, &h1, &w1, 1, NULL);
    cl_relu(c1, h1 * w1 * g_model.conv1.n);

    float *c2 = (float *)cl_xmalloc((size_t)8 * 8 * g_model.conv2.n * sizeof(float));
    conv2_from_float(&g_model.conv2, c1, h1, w1, g_model.conv1.n, c2, &h2, &w2);
    cl_relu(c2, h2 * w2 * g_model.conv2.n);

    float gap[32];
    cl_gap(c2, h2, w2, g_model.conv2.n, gap);

    int8_t q[32];
    float a_scale[1];
    cl_quant_rows_f32(gap, 1, g_model.fc.in, q, a_scale);
    cl_gemm_fn gemm = cl_active_gemm();
    gemm(1, g_model.fc.out, g_model.fc.in, q, g_model.fc.w, a_scale, g_model.fc.scale,
         g_model.fc.bias, r.logits);

    float probs[CL_N_CLASSES];
    cl_softmax(r.logits, CL_N_CLASSES, probs);
    int best = 0;
    for (int i = 1; i < CL_N_CLASSES; i++)
        if (probs[i] > probs[best])
            best = i;
    r.class_id = best;
    r.confidence = probs[best];
    cl_xfree(c1);
    cl_xfree(c2);
    return r;
}

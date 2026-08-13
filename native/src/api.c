#include "internal.h"

#include <stdio.h>
#include <string.h>

static uint32_t g_features;
static int g_inited;

int cl_init(const char *model_path) {
    g_features = cl_detect_features();
    if (g_features & CL_FEAT_I8MM)
        g_backend = CL_BACKEND_NEON_I8MM;
    else if (g_features & CL_FEAT_DOTPROD)
        g_backend = CL_BACKEND_NEON_DOTPROD;
    else
        g_backend = CL_BACKEND_SCALAR;
    int rc = cl_load_model(model_path);
    g_inited = rc == 0;
    return rc;
}

void cl_shutdown(void) {
    cl_free_model();
    g_inited = 0;
}

int cl_ready(void) { return g_inited; }

int cl_backend_available(cl_backend backend) {
    switch (backend) {
    case CL_BACKEND_SCALAR:
        return 1;
    case CL_BACKEND_NEON_DOTPROD:
        return !!(g_features & CL_FEAT_DOTPROD) || !!(g_features & CL_FEAT_NEON);
    case CL_BACKEND_NEON_I8MM:
        return !!(g_features & CL_FEAT_I8MM);
    case CL_BACKEND_KLEIDIAI:
        return cl_kleidiai_available();
    default:
        return 0;
    }
}

int cl_set_backend(cl_backend backend) {
    if (!cl_backend_available(backend))
        return -1;
    g_backend = backend;
    return 0;
}

cl_backend cl_get_backend(void) { return g_backend; }

cl_perception cl_perceive(const uint8_t *rgb, int width, int height) {
    cl_perception r;
    memset(&r, 0, sizeof(r));
    if (!g_inited || !rgb) {
        r.class_id = -1;
        return r;
    }
    uint8_t resized[CL_IN_W * CL_IN_H * 3];
    double t0 = cl_now_us();
    if (g_backend == CL_BACKEND_SCALAR)
        cl_resize_bilinear_u8_scalar(rgb, width, height, resized, CL_IN_W, CL_IN_H);
    else
        cl_resize_bilinear_u8_neon(rgb, width, height, resized, CL_IN_W, CL_IN_H);
    double t1 = cl_now_us();
    r = cl_cnn_forward(resized);
    double t2 = cl_now_us();
    r.preprocess_us = t1 - t0;
    r.infer_us = t2 - t1;
    r.total_us = t2 - t0;
    return r;
}

static void fill_rand_i8(int8_t *p, int n, unsigned *seed) {
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1664525u + 1013904223u;
        p[i] = (int8_t)((int)(*seed >> 16) % 255 - 127);
    }
}

static void fill_rand_f32(float *p, int n, unsigned *seed) {
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1664525u + 1013904223u;
        p[i] = ((*seed >> 8) / 16777216.f) * 2.f - 1.f;
    }
}

int cl_run_bench(int gemm_m, int gemm_n, int gemm_k, int e2e_iters, cl_bench_row *rows,
                 int max_rows) {
    if (gemm_m < 8)
        gemm_m = 8;
    if (gemm_n < 8)
        gemm_n = 8;
    if (gemm_k < 32)
        gemm_k = 32;
    if (e2e_iters < 8)
        e2e_iters = 8;
    if (max_rows < 1)
        return 0;

    int8_t *A = (int8_t *)cl_xmalloc((size_t)gemm_m * gemm_k);
    int8_t *B = (int8_t *)cl_xmalloc((size_t)gemm_n * gemm_k);
    float *C = (float *)cl_xmalloc((size_t)gemm_m * gemm_n * sizeof(float));
    float *as = (float *)cl_xmalloc((size_t)gemm_m * sizeof(float));
    float *bs = (float *)cl_xmalloc((size_t)gemm_n * sizeof(float));
    float *bias = (float *)cl_xmalloc((size_t)gemm_n * sizeof(float));
    unsigned seed = 1;
    fill_rand_i8(A, gemm_m * gemm_k, &seed);
    fill_rand_i8(B, gemm_n * gemm_k, &seed);
    for (int i = 0; i < gemm_m; i++)
        as[i] = 0.02f;
    for (int i = 0; i < gemm_n; i++) {
        bs[i] = 0.02f;
        bias[i] = 0.f;
    }

    uint8_t *frame = (uint8_t *)cl_xmalloc(96 * 64 * 3);
    for (int i = 0; i < 96 * 64 * 3; i++)
        frame[i] = (uint8_t)(i * 13);

    float *Af = NULL;
    float *Bf = NULL;
    float *Cf = NULL;
    if (cl_kleidiai_available()) {
        Af = (float *)cl_xmalloc((size_t)gemm_m * gemm_k * sizeof(float));
        Bf = (float *)cl_xmalloc((size_t)gemm_n * gemm_k * sizeof(float));
        Cf = (float *)cl_xmalloc((size_t)gemm_m * gemm_n * sizeof(float));
        fill_rand_f32(Af, gemm_m * gemm_k, &seed);
        fill_rand_f32(Bf, gemm_n * gemm_k, &seed);
    }

    cl_backend saved = g_backend;
    cl_backend order[] = {CL_BACKEND_SCALAR, CL_BACKEND_NEON_DOTPROD, CL_BACKEND_NEON_I8MM,
                           CL_BACKEND_KLEIDIAI};
    int nout = 0;
    const double flops = 2.0 * gemm_m * gemm_n * gemm_k;

    for (int bi = 0; bi < 4 && nout < max_rows; bi++) {
        cl_backend b = order[bi];
        if (!cl_backend_available(b))
            continue;
        cl_bench_row *row = &rows[nout++];
        memset(row, 0, sizeof(*row));
        row->name = cl_backend_name(b);
        row->available = 1;
        row->model_bytes = (double)cl_model_nbytes();

        if (b == CL_BACKEND_KLEIDIAI) {
            double pack_us = 0, compute_us = 0;
            /* Warmup + timed compute. Packing is one-time for weights. */
            cl_kleidiai_gemm_f32(gemm_m, gemm_n, gemm_k, Af, Bf, Cf, &pack_us, &compute_us);
            double t0 = cl_now_us();
            int reps = 8;
            for (int r = 0; r < reps; r++)
                cl_kleidiai_gemm_f32(gemm_m, gemm_n, gemm_k, Af, Bf, Cf, &pack_us, &compute_us);
            double t1 = cl_now_us();
            row->gemm_us = (t1 - t0) / reps;
            row->gemm_gflops = (flops / (row->gemm_us * 1e3));
        } else {
            g_backend = b;
            cl_gemm_fn gemm = cl_active_gemm();
            gemm(gemm_m, gemm_n, gemm_k, A, B, as, bs, bias, C);
            double t0 = cl_now_us();
            int reps = (b == CL_BACKEND_SCALAR) ? 2 : 8;
            for (int r = 0; r < reps; r++)
                gemm(gemm_m, gemm_n, gemm_k, A, B, as, bs, bias, C);
            double t1 = cl_now_us();
            row->gemm_us = (t1 - t0) / reps;
            row->gemm_gflops = flops / (row->gemm_us * 1e3);
        }

        g_backend = (b == CL_BACKEND_KLEIDIAI) ? CL_BACKEND_NEON_I8MM : b;
        if (!cl_backend_available(g_backend))
            g_backend = CL_BACKEND_NEON_DOTPROD;
        cl_perception p = cl_perceive(frame, 96, 64);
        double t0 = cl_now_us();
        double pre = 0, e2e = 0;
        for (int i = 0; i < e2e_iters; i++) {
            p = cl_perceive(frame, 96, 64);
            pre += p.preprocess_us;
            e2e += p.total_us;
        }
        (void)t0;
        row->preprocess_us = pre / e2e_iters;
        row->e2e_us = e2e / e2e_iters;
    }

    g_backend = saved;
    cl_xfree(A);
    cl_xfree(B);
    cl_xfree(C);
    cl_xfree(as);
    cl_xfree(bs);
    cl_xfree(bias);
    cl_xfree(frame);
    cl_xfree(Af);
    cl_xfree(Bf);
    cl_xfree(Cf);
    return nout;
}

int cl_bench_json(int gemm_m, int gemm_n, int gemm_k, int e2e_iters, char *out, int out_cap) {
    cl_bench_row rows[4];
    int n = cl_run_bench(gemm_m, gemm_n, gemm_k, e2e_iters, rows, 4);
    int pos = 0;
    pos += snprintf(out + pos, out_cap - pos,
                    "{\"version\":\"%s\",\"cpu\":\"%s\",\"model_bytes\":%.0f,\"gemm\":{\"m\":%d,"
                    "\"n\":%d,\"k\":%d},\"rows\":[",
                    cl_version(), cl_cpu_features_string(), n ? rows[0].model_bytes : 0, gemm_m,
                    gemm_n, gemm_k);
    for (int i = 0; i < n && pos < out_cap - 2; i++) {
        pos += snprintf(out + pos, out_cap - pos,
                        "%s{\"name\":\"%s\",\"gemm_us\":%.3f,\"gemm_gflops\":%.3f,"
                        "\"preprocess_us\":%.3f,\"e2e_us\":%.3f}",
                        i ? "," : "", rows[i].name, rows[i].gemm_us, rows[i].gemm_gflops,
                        rows[i].preprocess_us, rows[i].e2e_us);
    }
    if (pos < out_cap - 3)
        pos += snprintf(out + pos, out_cap - pos, "]}");
    return pos;
}

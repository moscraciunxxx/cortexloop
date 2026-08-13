#ifndef CORTEXLOOP_H
#define CORTEXLOOP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CL_N_CLASSES 8
#define CL_IN_W 32
#define CL_IN_H 32
#define CL_IN_C 3

typedef enum {
    CL_BACKEND_SCALAR = 0,
    CL_BACKEND_NEON_DOTPROD = 1,
    CL_BACKEND_NEON_I8MM = 2,
    CL_BACKEND_KLEIDIAI = 3
} cl_backend;

typedef enum {
    CL_FEAT_NEON = 1u << 0,
    CL_FEAT_DOTPROD = 1u << 1,
    CL_FEAT_I8MM = 1u << 2,
    CL_FEAT_BF16 = 1u << 3,
    CL_FEAT_SME = 1u << 4,
    CL_FEAT_SME2 = 1u << 5
} cl_cpu_feature;

typedef struct {
    int class_id;
    float confidence;
    float logits[CL_N_CLASSES];
    double preprocess_us;
    double infer_us;
    double total_us;
} cl_perception;

typedef struct {
    const char *name;
    int available;
    double gemm_us;
    double gemm_gflops;
    double preprocess_us;
    double e2e_us;
    double model_bytes;
} cl_bench_row;

int cl_init(const char *model_path);
void cl_shutdown(void);
int cl_ready(void);

int cl_set_backend(cl_backend backend);
cl_backend cl_get_backend(void);
const char *cl_backend_name(cl_backend backend);
int cl_backend_available(cl_backend backend);

uint32_t cl_cpu_features(void);
const char *cl_cpu_features_string(void);
const char *cl_class_name(int class_id);

cl_perception cl_perceive(const uint8_t *rgb, int width, int height);

int cl_run_bench(int gemm_m, int gemm_n, int gemm_k, int e2e_iters,
                 cl_bench_row *rows, int max_rows);
int cl_bench_json(int gemm_m, int gemm_n, int gemm_k, int e2e_iters,
                  char *out, int out_cap);

size_t cl_model_nbytes(void);
const char *cl_version(void);

#ifdef __cplusplus
}
#endif

#endif

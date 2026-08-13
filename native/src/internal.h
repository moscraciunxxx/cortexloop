#ifndef CORTEXLOOP_INTERNAL_H
#define CORTEXLOOP_INTERNAL_H

#include "cortexloop.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CL_CONV1_OUT 8
#define CL_CONV2_OUT 16
#define CL_HIDDEN 16
#define CL_ALIGN_K 32

typedef struct {
    int n;
    int c;
    int kh;
    int kw;
    int stride;
    int pad;
    int8_t *w;     /* n * c * kh * kw */
    float *scale;  /* n */
    float *bias;   /* n */
} cl_conv;

typedef struct {
    int out;
    int in;
    int8_t *w;     /* out * in */
    float *scale;  /* out */
    float *bias;   /* out */
} cl_linear;

typedef struct {
    int loaded;
    cl_conv conv1;
    cl_conv conv2;
    cl_linear fc;
    uint8_t *scratch;
    size_t scratch_cap;
} cl_model;

extern cl_model g_model;
extern cl_backend g_backend;

uint32_t cl_detect_features(void);
void *cl_xmalloc(size_t n);
void cl_xfree(void *p);
void *cl_scratch(size_t n);
double cl_now_us(void);

void cl_resize_bilinear_u8_scalar(const uint8_t *src, int sw, int sh,
                                  uint8_t *dst, int dw, int dh);
void cl_resize_bilinear_u8_neon(const uint8_t *src, int sw, int sh,
                                uint8_t *dst, int dw, int dh);

void cl_gemm_i8_scalar(int m, int n, int k, const int8_t *a, const int8_t *b,
                       const float *a_scale, const float *b_scale,
                       const float *bias, float *c);
void cl_gemm_i8_dotprod(int m, int n, int k, const int8_t *a, const int8_t *b,
                        const float *a_scale, const float *b_scale,
                        const float *bias, float *c);
void cl_gemm_i8_i8mm(int m, int n, int k, const int8_t *a, const int8_t *b,
                     const float *a_scale, const float *b_scale,
                     const float *bias, float *c);

void cl_im2col_u8(const uint8_t *src, int h, int w, int c, int kh, int kw,
                  int stride, int pad, int8_t *col, int *out_h, int *out_w);

void cl_quant_rows_f32(const float *src, int rows, int cols, int8_t *dst,
                       float *scales);

void cl_conv_forward(const cl_conv *layer, const uint8_t *src, int h, int w,
                     int c, float *dst, int *out_h, int *out_w, int use_u8_src,
                     const float *src_f32);
void cl_relu(float *x, int n);
void cl_gap(const float *src, int h, int w, int c, float *dst);
void cl_softmax(const float *logits, int n, float *probs);

int cl_load_model(const char *path);
void cl_free_model(void);

cl_perception cl_cnn_forward(const uint8_t *rgb32);

int cl_kleidiai_available(void);
int cl_kleidiai_gemm_f32(int m, int n, int k, const float *a, const float *b,
                         float *c, double *pack_us, double *compute_us);

typedef void (*cl_gemm_fn)(int m, int n, int k, const int8_t *a, const int8_t *b,
                           const float *a_scale, const float *b_scale,
                           const float *bias, float *c);

cl_gemm_fn cl_active_gemm(void);

#ifdef __cplusplus
}
#endif

#endif

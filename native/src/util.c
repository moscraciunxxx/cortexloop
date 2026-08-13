#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

cl_model g_model;
cl_backend g_backend = CL_BACKEND_NEON_DOTPROD;

void *cl_xmalloc(size_t n) {
    void *p = calloc(1, n ? n : 1);
    if (!p) {
        fprintf(stderr, "cortexloop: out of memory (%zu bytes)\n", n);
        abort();
    }
    return p;
}

void cl_xfree(void *p) { free(p); }

void *cl_scratch(size_t n) {
    if (n > g_model.scratch_cap) {
        free(g_model.scratch);
        g_model.scratch_cap = n + (n >> 1) + 4096;
        g_model.scratch = (uint8_t *)cl_xmalloc(g_model.scratch_cap);
    }
    return g_model.scratch;
}

double cl_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

uint32_t cl_detect_features(void) {
    uint32_t f = 0;
#if defined(__aarch64__) || defined(__arm64__)
    f |= CL_FEAT_NEON;
#endif
#if defined(__APPLE__)
    int val = 0;
    size_t len = sizeof(val);
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &val, &len, NULL, 0) == 0 && val)
        f |= CL_FEAT_DOTPROD;
    len = sizeof(val);
    val = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &val, &len, NULL, 0) == 0 && val)
        f |= CL_FEAT_I8MM;
    len = sizeof(val);
    val = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_BF16", &val, &len, NULL, 0) == 0 && val)
        f |= CL_FEAT_BF16;
    len = sizeof(val);
    val = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &val, &len, NULL, 0) == 0 && val)
        f |= CL_FEAT_SME;
    len = sizeof(val);
    val = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &val, &len, NULL, 0) == 0 && val)
        f |= CL_FEAT_SME2;
#else
#if defined(__ARM_FEATURE_DOTPROD)
    f |= CL_FEAT_DOTPROD;
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    f |= CL_FEAT_I8MM;
#endif
#if defined(__ARM_FEATURE_BF16)
    f |= CL_FEAT_BF16;
#endif
#if defined(__ARM_FEATURE_SME) || defined(__ARM_FEATURE_SME2)
    f |= CL_FEAT_SME;
#endif
#if defined(__ARM_FEATURE_SME2)
    f |= CL_FEAT_SME2;
#endif
#endif
    return f;
}

static char g_feat_buf[160];

const char *cl_cpu_features_string(void) {
    uint32_t f = cl_detect_features();
    snprintf(g_feat_buf, sizeof(g_feat_buf), "NEON=%d DotProd=%d I8MM=%d BF16=%d SME=%d SME2=%d",
             !!(f & CL_FEAT_NEON), !!(f & CL_FEAT_DOTPROD), !!(f & CL_FEAT_I8MM),
             !!(f & CL_FEAT_BF16), !!(f & CL_FEAT_SME), !!(f & CL_FEAT_SME2));
    return g_feat_buf;
}

uint32_t cl_cpu_features(void) { return cl_detect_features(); }

const char *cl_version(void) { return "1.0.0"; }

const char *cl_class_name(int class_id) {
    static const char *names[CL_N_CLASSES] = {
        "empty", "wall", "box_red", "box_blue", "box_green", "hazard", "person", "dock"};
    if (class_id < 0 || class_id >= CL_N_CLASSES)
        return "unknown";
    return names[class_id];
}

const char *cl_backend_name(cl_backend backend) {
    switch (backend) {
    case CL_BACKEND_SCALAR:
        return "scalar";
    case CL_BACKEND_NEON_DOTPROD:
        return "neon_dotprod";
    case CL_BACKEND_NEON_I8MM:
        return "neon_i8mm";
    case CL_BACKEND_KLEIDIAI:
        return "kleidiai";
    default:
        return "unknown";
    }
}

#include "internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    fprintf(stderr,
            "CortexLoop native tools\n"
            "  %s bench [--model PATH] [--m 256] [--n 256] [--k 256] [--iters 32]\n"
            "  %s perceive --model PATH --ppm FILE\n",
            argv0, argv0);
}

static int argi(char **argv, int i, int argc, int def) {
    if (i + 1 >= argc)
        return def;
    return atoi(argv[i + 1]);
}

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "bench";
    const char *model = "models/warehouse_int8.bin";
    int m = 256, n = 256, k = 256, iters = 32;
    const char *ppm = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            model = argv[++i];
        else if (!strcmp(argv[i], "--m"))
            m = argi(argv, i, argc, m), i++;
        else if (!strcmp(argv[i], "--n"))
            n = argi(argv, i, argc, n), i++;
        else if (!strcmp(argv[i], "--k"))
            k = argi(argv, i, argc, k), i++;
        else if (!strcmp(argv[i], "--iters"))
            iters = argi(argv, i, argc, iters), i++;
        else if (!strcmp(argv[i], "--ppm") && i + 1 < argc)
            ppm = argv[++i];
        else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        }
    }

    int rc = cl_init(model);
    if (rc != 0) {
        fprintf(stderr, "failed to load model '%s' (code %d). Run: python -m cortexloop.train\n",
                model, rc);
        return 1;
    }
    printf("CortexLoop %s\nCPU: %s\nModel: %zu bytes\nDefault backend: %s\n", cl_version(),
           cl_cpu_features_string(), cl_model_nbytes(), cl_backend_name(cl_get_backend()));

    if (!strcmp(cmd, "perceive") && ppm) {
        FILE *f = fopen(ppm, "rb");
        if (!f) {
            perror(ppm);
            return 1;
        }
        char magic[8];
        int w = 0, h = 0, maxv = 0;
        if (fscanf(f, "%7s %d %d %d", magic, &w, &h, &maxv) != 4 || strcmp(magic, "P6") != 0) {
            fprintf(stderr, "need binary PPM P6\n");
            return 1;
        }
        fgetc(f);
        uint8_t *rgb = (uint8_t *)malloc((size_t)w * h * 3);
        if (fread(rgb, 1, (size_t)w * h * 3, f) != (size_t)w * h * 3) {
            fprintf(stderr, "truncated ppm\n");
            return 1;
        }
        fclose(f);
        cl_perception p = cl_perceive(rgb, w, h);
        printf("class=%s id=%d conf=%.3f preprocess=%.1fus infer=%.1fus total=%.1fus\n",
               cl_class_name(p.class_id), p.class_id, p.confidence, p.preprocess_us, p.infer_us,
               p.total_us);
        free(rgb);
        cl_shutdown();
        return 0;
    }

    char json[4096];
    cl_bench_json(m, n, k, iters, json, (int)sizeof(json));
    puts(json);
    cl_shutdown();
    return 0;
}

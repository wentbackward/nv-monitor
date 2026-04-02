/*
 * demo-load - Visual test harness for nv-monitor
 *
 * Spawns one thread per CPU core, each with a sinusoidal load pattern
 * offset in phase so all cores show different utilization levels.
 * Optionally loads the GPU with varying intensity via the CUDA driver API.
 *
 * Build: gcc -O2 -o demo-load demo-load.c -lpthread -ldl -lm
 * Usage: ./demo-load            # CPU only
 *        ./demo-load --gpu      # CPU + GPU load
 *        ./demo-load --gpu-only # GPU only
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <pthread.h>
#include <dlfcn.h>

static volatile int running = 1;
static unsigned int gpu_blocks = 1024;
#define DEFAULT_MAX_SECONDS 300 /* 5 minute failsafe */

static void on_signal(int sig) { (void)sig; running = 0; }

/* Parse "HH:MM" or "HH:MM:SS" into epoch seconds. Returns -1 on error. */
static time_t parse_time(const char *s) {
    int h, m, sec = 0;
    if (sscanf(s, "%d:%d:%d", &h, &m, &sec) < 2) return -1;
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    tm.tm_hour = h;
    tm.tm_min = m;
    tm.tm_sec = sec;
    time_t target = mktime(&tm);
    /* If time is in the past, assume tomorrow */
    if (target <= now) target += 86400;
    return target;
}

/* Parse duration like "30s", "5m", "1h", or bare seconds. Returns seconds, -1 on error. */
static int parse_duration(const char *s) {
    int val = atoi(s);
    if (val <= 0) return -1;
    int len = strlen(s);
    if (len > 0) {
        switch (s[len - 1]) {
        case 'h': case 'H': return val * 3600;
        case 'm': case 'M': return val * 60;
        case 's': case 'S': return val;
        }
    }
    return val; /* bare number = seconds */
}

/* ── CPU load ──────────────────────────────────────────────────────── */

typedef struct {
    int core_id;
    int total_cores;
} CpuThreadArg;

static void *cpu_load_thread(void *arg) {
    CpuThreadArg *a = (CpuThreadArg *)arg;
    double phase = (2.0 * M_PI * a->core_id) / a->total_cores;
    double speed = 0.6 + 0.4 * ((double)a->core_id / a->total_cores);

    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    while (running) {
        double now;
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        now = (ts.tv_sec - ts_start.tv_sec) + (ts.tv_nsec - ts_start.tv_nsec) / 1e9;

        /* Sinusoidal target: 5% to 95% utilization */
        double target = 0.50 + 0.45 * sin(now * speed + phase);

        /* Busy/sleep cycle: 50ms window */
        int busy_us = (int)(target * 50000);
        int idle_us = 50000 - busy_us;

        /* Busy spin */
        struct timespec spin_end;
        clock_gettime(CLOCK_MONOTONIC, &spin_end);
        long end_ns = spin_end.tv_nsec + busy_us * 1000L;
        spin_end.tv_sec += end_ns / 1000000000L;
        spin_end.tv_nsec = end_ns % 1000000000L;

        while (running) {
            clock_gettime(CLOCK_MONOTONIC, &ts);
            if (ts.tv_sec > spin_end.tv_sec ||
                (ts.tv_sec == spin_end.tv_sec && ts.tv_nsec >= spin_end.tv_nsec))
                break;
            /* burn cycles */
            volatile double x = 1.0001;
            for (int j = 0; j < 100; j++) x *= 1.0001;
            (void)x;
        }

        /* Sleep */
        if (idle_us > 0) {
            struct timespec sl = { .tv_sec = 0, .tv_nsec = idle_us * 1000L };
            nanosleep(&sl, NULL);
        }
    }

    free(a);
    return NULL;
}

/* ── GPU load (CUDA driver API via dlopen) ─────────────────────────── */

/*
 * PTX kernel: fixed 100K iterations of heavy FMA work per thread.
 * No parameters — load is controlled by how often we launch, not kernel size.
 */
static const char *ptx_source =
    ".version 7.0\n"
    ".target sm_50\n"
    ".address_size 64\n"
    ".visible .entry busy_kernel() {\n"
    "    .reg .u32 %r<3>;\n"
    "    .reg .f32 %f<9>;\n"
    "    .reg .pred %p;\n"
    "    mov.f32 %f0, 1.0;\n"
    "    mov.f32 %f1, 0.999999;\n"
    "    mov.f32 %f2, 0.000001;\n"
    "    mov.f32 %f3, 1.0;\n"
    "    mov.f32 %f4, 1.0;\n"
    "    mov.f32 %f5, 1.0;\n"
    "    mov.u32 %r0, 100000;\n"
    "    mov.u32 %r1, 0;\n"
    "LOOP:\n"
    "    fma.rn.f32 %f0, %f0, %f1, %f2;\n"
    "    fma.rn.f32 %f3, %f3, %f1, %f2;\n"
    "    fma.rn.f32 %f4, %f4, %f1, %f2;\n"
    "    fma.rn.f32 %f5, %f5, %f1, %f2;\n"
    "    fma.rn.f32 %f0, %f0, %f1, %f2;\n"
    "    fma.rn.f32 %f3, %f3, %f1, %f2;\n"
    "    fma.rn.f32 %f4, %f4, %f1, %f2;\n"
    "    fma.rn.f32 %f5, %f5, %f1, %f2;\n"
    "    add.u32 %r1, %r1, 1;\n"
    "    setp.lt.u32 %p, %r1, %r0;\n"
    "    @%p bra LOOP;\n"
    "    ret;\n"
    "}\n";

/* CUDA driver API types */
typedef int CUresult;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;

typedef struct {
    int gpu_index;
} GpuThreadArg;

static void *gpu_load_thread(void *arg) {
    GpuThreadArg *ga = (GpuThreadArg *)arg;
    int gpu_idx = ga->gpu_index;
    free(ga);

    void *cuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!cuda) {
        fprintf(stderr, "demo-load: cannot open libcuda.so.1: %s\n", dlerror());
        return NULL;
    }

    /* Load function pointers */
    CUresult (*cuInit)(unsigned) = dlsym(cuda, "cuInit");
    CUresult (*cuDeviceGet)(int *, int) = dlsym(cuda, "cuDeviceGet");
    CUresult (*cuCtxCreate)(CUcontext *, unsigned, int) = dlsym(cuda, "cuCtxCreate_v2");
    CUresult (*cuModuleLoadData)(CUmodule *, const void *) = dlsym(cuda, "cuModuleLoadData");
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = dlsym(cuda, "cuModuleGetFunction");
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                                unsigned, unsigned, unsigned,
                                unsigned, void *, void **, void **) = dlsym(cuda, "cuLaunchKernel");
    CUresult (*cuCtxSynchronize)(void) = dlsym(cuda, "cuCtxSynchronize");
    CUresult (*cuCtxDestroy)(CUcontext) = dlsym(cuda, "cuCtxDestroy_v2");
    CUresult (*cuStreamCreate)(CUstream *, unsigned) = dlsym(cuda, "cuStreamCreate");
    CUresult (*cuStreamSynchronize)(CUstream) = dlsym(cuda, "cuStreamSynchronize");
    CUresult (*cuStreamDestroy)(CUstream) = dlsym(cuda, "cuStreamDestroy_v2");
    if (!cuInit || !cuDeviceGet || !cuCtxCreate || !cuModuleLoadData ||
        !cuModuleGetFunction || !cuLaunchKernel || !cuCtxSynchronize || !cuCtxDestroy ||
        !cuStreamCreate || !cuStreamSynchronize || !cuStreamDestroy) {
        fprintf(stderr, "demo-load: missing CUDA driver symbols\n");
        dlclose(cuda);
        return NULL;
    }

    CUresult rc;
    if ((rc = cuInit(0)) != 0) {
        fprintf(stderr, "demo-load: cuInit failed (%d)\n", rc);
        dlclose(cuda);
        return NULL;
    }

    int dev = 0;
    cuDeviceGet(&dev, gpu_idx);

    CUcontext ctx = NULL;
    if ((rc = cuCtxCreate(&ctx, 0, dev)) != 0) {
        fprintf(stderr, "demo-load: cuCtxCreate failed for GPU %d (%d)\n", gpu_idx, rc);
        dlclose(cuda);
        return NULL;
    }

    CUmodule mod = NULL;
    if ((rc = cuModuleLoadData(&mod, ptx_source)) != 0) {
        fprintf(stderr, "demo-load: cuModuleLoadData failed for GPU %d (%d)\n", gpu_idx, rc);
        cuCtxDestroy(ctx);
        dlclose(cuda);
        return NULL;
    }

    CUfunction func = NULL;
    cuModuleGetFunction(&func, mod, "busy_kernel");

    CUstream stream = NULL;
    cuStreamCreate(&stream, 0);

    /* ── Calibrate: measure how long one kernel takes ── */
    unsigned int blocks = gpu_blocks;

    printf("GPU %d: calibrating...", gpu_idx);
    fflush(stdout);

    /* Warm up */
    cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, stream, NULL, NULL);
    cuStreamSynchronize(stream);
    cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, stream, NULL, NULL);
    cuStreamSynchronize(stream);

    /* Measure */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < 10; i++)
        cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, stream, NULL, NULL);
    cuStreamSynchronize(stream);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double kernel_ms = ((t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e6) / 10.0;
    if (kernel_ms < 0.01) kernel_ms = 0.01;

    printf(" done (kernel=%.2fms, blocks=%u)\n", kernel_ms, blocks);

    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    printf("GPU %d: load active\n", gpu_idx);

    /*
     * Duty-cycle in 500ms windows (NVML samples at ~1s).
     * Queue enough kernels to fill the busy portion, then sleep.
     * No sync between individual kernels — GPU pipeline stays full.
     */
    while (running) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double now = (ts.tv_sec - ts_start.tv_sec) + (ts.tv_nsec - ts_start.tv_nsec) / 1e9;

        /* Sinusoidal intensity */
        double intensity = 0.50 + 0.45 * sin(now * 0.8);

        double window_ms = 200.0;
        double busy_ms = window_ms * intensity;
        int num_kernels = (int)(busy_ms / kernel_ms);
        if (num_kernels < 1) num_kernels = 1;

        /* Queue all kernels at once */
        for (int k = 0; k < num_kernels; k++)
            cuLaunchKernel(func, blocks, 1, 1, 256, 1, 1, 0, stream, NULL, NULL);
        cuStreamSynchronize(stream);

        /* Sleep for idle portion */
        double idle_ms = window_ms - busy_ms;
        if (idle_ms > 1.0) {
            struct timespec sl = {
                .tv_sec = (time_t)(idle_ms / 1000.0),
                .tv_nsec = (long)(fmod(idle_ms, 1000.0) * 1e6)
            };
            nanosleep(&sl, NULL);
        }
    }

    cuStreamDestroy(stream);

    cuCtxDestroy(ctx);
    dlclose(cuda);
    return NULL;
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    int do_cpu = 1, do_gpu = 0;
    int duration_sec = 0;       /* 0 = run until Ctrl-C */
    time_t stop_at = 0;         /* 0 = not set */
    int allow_long_run = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0)
            do_gpu = 1;
        else if (strcmp(argv[i], "--gpu-only") == 0) {
            do_gpu = 1;
            do_cpu = 0;
        } else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            gpu_blocks = (unsigned int)atoi(argv[++i]);
            if (gpu_blocks < 1) gpu_blocks = 1;
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            duration_sec = parse_duration(argv[++i]);
            if (duration_sec <= 0) {
                fprintf(stderr, "Invalid duration: %s (use e.g. 30s, 5m, 1h)\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--until") == 0 && i + 1 < argc) {
            stop_at = parse_time(argv[++i]);
            if (stop_at < 0) {
                fprintf(stderr, "Invalid time: %s (use HH:MM or HH:MM:SS)\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--allow-long-run") == 0) {
            allow_long_run = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [OPTIONS]\n\n"
                   "Synthetic load generator for testing nv-monitor.\n\n"
                   "  --gpu              Enable GPU load alongside CPU load\n"
                   "  --gpu-only         GPU load only (no CPU threads)\n"
                   "  --blocks N         CUDA blocks per kernel launch (default: 1024)\n"
                   "  --duration TIME    Run for specified time (e.g. 30s, 5m, 1h)\n"
                   "  --until HH:MM     Run until specified time (24h format)\n"
                   "  --allow-long-run   Allow runs longer than 5 minutes\n\n"
                   "Without --duration or --until, stops after 5 minutes unless\n"
                   "--allow-long-run is set.\n\n"
                   "Copyright (c) 2026 Paul Gresham Advisory LLC\n"
                   "https://github.com/wentbackward/nv-monitor\n",
                   argv[0]);
            return 0;
        }
    }

    /* Determine stop time */
    time_t now = time(NULL);
    if (stop_at > 0) {
        /* --until takes precedence */
    } else if (duration_sec > 0) {
        stop_at = now + duration_sec;
    } else if (!allow_long_run) {
        /* Default failsafe: 5 minutes */
        stop_at = now + DEFAULT_MAX_SECONDS;
    }
    /* else: allow_long_run with no duration = run until Ctrl-C */

    /* Check failsafe */
    if (stop_at > 0 && !allow_long_run && (stop_at - now) > DEFAULT_MAX_SECONDS) {
        fprintf(stderr, "Error: run time exceeds 5 minutes. Use --allow-long-run to override.\n");
        return 1;
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    int ncpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpus < 1) ncpus = 1;

    pthread_t *cpu_threads = NULL;
    #define MAX_GPUS 16
    pthread_t gpu_threads[MAX_GPUS];
    int n_gpus = 0;

    if (do_cpu) {
        printf("Starting CPU load on %d cores (sinusoidal, phased)\n", ncpus);
        cpu_threads = calloc(ncpus, sizeof(pthread_t));
        for (int i = 0; i < ncpus; i++) {
            CpuThreadArg *a = malloc(sizeof(CpuThreadArg));
            a->core_id = i;
            a->total_cores = ncpus;
            pthread_create(&cpu_threads[i], NULL, cpu_load_thread, a);
        }
    }

    if (do_gpu) {
        /* Detect GPU count via CUDA driver API */
        void *cuda = dlopen("libcuda.so.1", RTLD_LAZY);
        if (cuda) {
            CUresult (*cuInit)(unsigned) = dlsym(cuda, "cuInit");
            CUresult (*cuDeviceGetCount)(int *) = dlsym(cuda, "cuDeviceGetCount");
            if (cuInit && cuDeviceGetCount && cuInit(0) == 0) {
                int count = 0;
                cuDeviceGetCount(&count);
                if (count > MAX_GPUS) count = MAX_GPUS;
                n_gpus = count;
            }
            dlclose(cuda);
        }

        if (n_gpus == 0) {
            fprintf(stderr, "demo-load: no GPUs found\n");
        } else {
            printf("Starting GPU load on %d GPU%s (sinusoidal)\n",
                   n_gpus, n_gpus > 1 ? "s" : "");
            for (int g = 0; g < n_gpus; g++) {
                GpuThreadArg *ga = malloc(sizeof(GpuThreadArg));
                ga->gpu_index = g;
                pthread_create(&gpu_threads[g], NULL, gpu_load_thread, ga);
            }
        }
    }

    if (stop_at > 0) {
        int remaining = (int)(stop_at - time(NULL));
        printf("Will stop in %dm %ds (Ctrl-C to stop early)\n",
               remaining / 60, remaining % 60);
    } else {
        printf("Press Ctrl-C to stop\n");
    }

    /* Wait */
    while (running) {
        if (stop_at > 0 && time(NULL) >= stop_at) {
            printf("\nTime limit reached.\n");
            running = 0;
            break;
        }
        sleep(1);
    }

    printf("\nStopping...\n");

    if (cpu_threads) {
        for (int i = 0; i < ncpus; i++)
            pthread_join(cpu_threads[i], NULL);
        free(cpu_threads);
    }
    for (int g = 0; g < n_gpus; g++)
        pthread_join(gpu_threads[g], NULL);

    printf("Done.\n");
    return 0;
}

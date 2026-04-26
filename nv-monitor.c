/*
 * nv-monitor - System monitor for NVIDIA DGX Spark (Grace + GB10)
 *
 * Displays CPU per-core usage, memory, CPU thermals, GPU utilization,
 * GPU temperature/power/clock, and GPU processes in a single TUI.
 *
 * Build: gcc -O2 -o nv-monitor nv-monitor.c -lncurses -ldl -lpthread
 */

#ifndef VERSION
#define VERSION "dev"
#endif

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <dirent.h>
#include <signal.h>
#include <time.h>
#include <pwd.h>
#include <ncurses.h>
#include <locale.h>
#include <sys/sysinfo.h>
#include <sys/stat.h>
#include <getopt.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>

/* ── NVML types (loaded dynamically) ────────────────────────────────── */

typedef void *nvmlDevice_t;
typedef int   nvmlReturn_t;

typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef struct {
    unsigned int pid;
    unsigned long long usedGpuMemory;
    unsigned int gpuInstanceId;
    unsigned int computeInstanceId;
} nvmlProcessInfo_t;

#define NVML_SUCCESS 0
#define NVML_ERROR_NOT_SUPPORTED 3
#define NVML_TEMPERATURE_GPU 0
#define NVML_CLOCK_GRAPHICS 0
#define NVML_CLOCK_MEM 2
#define NVML_CLOCK_SM 1

/* NVML function pointers */
static nvmlReturn_t (*pNvmlInit)(void);
static nvmlReturn_t (*pNvmlShutdown)(void);
static nvmlReturn_t (*pNvmlDeviceGetCount)(unsigned int *);
static nvmlReturn_t (*pNvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t *);
static nvmlReturn_t (*pNvmlDeviceGetName)(nvmlDevice_t, char *, unsigned int);
static nvmlReturn_t (*pNvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t *);
static nvmlReturn_t (*pNvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
static nvmlReturn_t (*pNvmlDeviceGetTemperature)(nvmlDevice_t, int, unsigned int *);
static nvmlReturn_t (*pNvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int *);
static nvmlReturn_t (*pNvmlDeviceGetClockInfo)(nvmlDevice_t, int, unsigned int *);
static nvmlReturn_t (*pNvmlDeviceGetComputeRunningProcesses)(nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);
static nvmlReturn_t (*pNvmlDeviceGetGraphicsRunningProcesses)(nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);
static nvmlReturn_t (*pNvmlDeviceGetFanSpeed)(nvmlDevice_t, unsigned int *);
static nvmlReturn_t (*pNvmlDeviceGetEncoderUtilization)(nvmlDevice_t, unsigned int *, unsigned int *);
static nvmlReturn_t (*pNvmlDeviceGetDecoderUtilization)(nvmlDevice_t, unsigned int *, unsigned int *);

static void *nvml_handle = NULL;
static int   nvml_ok = 0;
static unsigned int gpu_count = 0;      /* number of GPUs detected */
static int   use_tegra_gpu = 0;         /* prefer Tegra sysfs over NVML for GPU metrics */
static char  cpu_model_name[128] = "";  /* from /proc/cpuinfo */
static char  host_name[256] = "";       /* local hostname, shortened before first dot */

/* ── Constants ──────────────────────────────────────────────────────── */

#define MAX_GPU_PROCS 256
#define REFRESH_MS    1000
#define BAR_CHAR_FULL  ACS_BLOCK
#define COLOR_GRAY     8
#define HISTORY_LEN   20

/* ── CPU state (dynamically allocated at startup) ──────────────────── */

typedef struct {
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
} CpuTick;

static int       num_cpus = 0;
static int       max_cpus = 0;       /* allocated size */
static CpuTick  *prev_ticks = NULL;  /* [max_cpus + 1] — index 0 = aggregate */
static CpuTick  *cur_ticks = NULL;   /* [max_cpus + 1] — current frame */
static double   *cpu_pct = NULL;     /* [max_cpus + 1] */
static unsigned int *cpu_part = NULL; /* [max_cpus] */

/* ── GPU process info ───────────────────────────────────────────────── */

typedef struct {
    unsigned int  pid;
    unsigned int  gpu_id;  /* which GPU this process belongs to */
    unsigned long long mem_bytes;
    char          name[256];
    char          user[64];
    char          type;    /* C=compute, G=graphics */
    double        cpu_pct; /* per-process CPU% */
} GpuProc;

/* ── Per-process CPU tracking ───────────────────────────────────────── */

#define MAX_TRACKED_PIDS 512

typedef struct {
    unsigned int  pid;
    unsigned long long ticks; /* utime + stime */
} ProcCpuSnap;

static ProcCpuSnap prev_proc_snaps[MAX_TRACKED_PIDS];
static int         prev_proc_count = 0;
static unsigned long long prev_total_cpu_ticks = 0;

static unsigned long long read_proc_cpu_ticks(unsigned int pid) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%u/stat", pid);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char buf[1024];
    int n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);
    char *cp = strrchr(buf, ')');
    if (!cp) return 0;
    cp += 2;
    unsigned long utime = 0, stime = 0;
    sscanf(cp, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu",
           &utime, &stime);
    return utime + stime;
}

static unsigned long long read_total_cpu_ticks_sum(void) {
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return 0;
    char line[512];
    unsigned long long sum = 0;
    if (fgets(line, sizeof(line), f)) {
        unsigned long long u, n, s, id, io, ir, si, st;
        sscanf(line + 4, "%llu %llu %llu %llu %llu %llu %llu %llu",
               &u, &n, &s, &id, &io, &ir, &si, &st);
        sum = u + n + s + id + io + ir + si + st;
    }
    fclose(f);
    return sum;
}

static double calc_proc_cpu_pct(unsigned int pid) {
    unsigned long long cur_ticks = read_proc_cpu_ticks(pid);
    unsigned long long cur_total = read_total_cpu_ticks_sum();
    unsigned long long total_delta = cur_total - prev_total_cpu_ticks;

    /* Find previous snapshot for this PID */
    for (int i = 0; i < prev_proc_count; i++) {
        if (prev_proc_snaps[i].pid == pid) {
            unsigned long long proc_delta = cur_ticks - prev_proc_snaps[i].ticks;
            if (total_delta > 0)
                return (double)proc_delta / (double)total_delta * 100.0 * num_cpus;
            return 0.0;
        }
    }
    return 0.0; /* no previous sample */
}

static void update_proc_cpu_snapshots(GpuProc *procs, int count) {
    prev_proc_count = 0;
    for (int i = 0; i < count && prev_proc_count < MAX_TRACKED_PIDS; i++) {
        prev_proc_snaps[prev_proc_count].pid = procs[i].pid;
        prev_proc_snaps[prev_proc_count].ticks = read_proc_cpu_ticks(procs[i].pid);
        prev_proc_count++;
    }
    prev_total_cpu_ticks = read_total_cpu_ticks_sum();
}

/* ── History ring buffers ────────────────────────────────────────────── */

static double cpu_history[HISTORY_LEN];
static double gpu_history[HISTORY_LEN];
static int    history_pos = 0;
static int    history_count = 0;

/* ── Globals ────────────────────────────────────────────────────────── */

static volatile sig_atomic_t g_quit = 0;
static int sort_mode = 0; /* 0=by mem, 1=by pid */
static int delay_ms = REFRESH_MS;
static double last_gpu_util = 0; /* average GPU util captured during draw for history */
static int cpu_columns = 0;  /* 0 = auto, 1-4 = user override */
static int cpu_scroll  = 0;  /* first visible core row offset */
static int skip_history_once = 0; /* suppress history chart for one frame after resize */
static int gpu_view = 0;  /* 0=auto, 1=compact, 2=detailed */

/* Command-line options */
static FILE *log_fp = NULL;
static int   log_interval_ms = 1000;
static int   no_ui = 0;
static int   prom_port = 0;  /* Prometheus metrics port (0 = disabled) */
static const char *prom_token = NULL; /* Bearer token for /metrics auth */

/* ── Signal handler ─────────────────────────────────────────────────── */

static void on_signal(int sig) {
    (void)sig;
    g_quit = 1;
}

/* ── NVML loading ───────────────────────────────────────────────────── */

static int load_nvml(void) {
    const char *paths[] = {
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        "/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1",
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        nvml_handle = dlopen(paths[i], RTLD_LAZY);
        if (nvml_handle) break;
    }
    if (!nvml_handle) return -1;

    /* Try versioned symbol first, then base name */
    #define LOAD(ptr, ...) do { \
        const char *_names[] = { __VA_ARGS__, NULL }; \
        for (int _i = 0; _names[_i]; _i++) { \
            *(void **)(&ptr) = dlsym(nvml_handle, _names[_i]); \
            if (ptr) break; \
        } \
    } while(0)

    LOAD(pNvmlInit,                               "nvmlInit_v2", "nvmlInit");
    LOAD(pNvmlShutdown,                           "nvmlShutdown");
    LOAD(pNvmlDeviceGetCount,                     "nvmlDeviceGetCount_v2", "nvmlDeviceGetCount");
    LOAD(pNvmlDeviceGetHandleByIndex,             "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
    LOAD(pNvmlDeviceGetName,                      "nvmlDeviceGetName");
    LOAD(pNvmlDeviceGetUtilizationRates,          "nvmlDeviceGetUtilizationRates");
    LOAD(pNvmlDeviceGetMemoryInfo,                "nvmlDeviceGetMemoryInfo");
    LOAD(pNvmlDeviceGetTemperature,               "nvmlDeviceGetTemperature");
    LOAD(pNvmlDeviceGetPowerUsage,                "nvmlDeviceGetPowerUsage");
    LOAD(pNvmlDeviceGetClockInfo,                 "nvmlDeviceGetClockInfo");
    LOAD(pNvmlDeviceGetComputeRunningProcesses,   "nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses");
    LOAD(pNvmlDeviceGetGraphicsRunningProcesses,  "nvmlDeviceGetGraphicsRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses");
    LOAD(pNvmlDeviceGetFanSpeed,                  "nvmlDeviceGetFanSpeed");
    LOAD(pNvmlDeviceGetEncoderUtilization,        "nvmlDeviceGetEncoderUtilization");
    LOAD(pNvmlDeviceGetDecoderUtilization,        "nvmlDeviceGetDecoderUtilization");
    #undef LOAD

    if (!pNvmlInit) return -1;
    if (pNvmlInit() != NVML_SUCCESS) return -1;

    return 0;
}

/* ── CPU core type identification ───────────────────────────────────── */

static void read_cpu_model_name(void) {
    /* Try device tree model first (works on ARM SBCs, DGX Spark, etc.) */
    FILE *f = fopen("/sys/firmware/devicetree/base/model", "r");
    if (f) {
        if (fgets(cpu_model_name, sizeof(cpu_model_name), f))
            cpu_model_name[strcspn(cpu_model_name, "\n\r")] = '\0';
        fclose(f);
        if (cpu_model_name[0]) return;
    }

    /* DMI product name (works on DGX Spark, most x86 systems) */
    f = fopen("/sys/devices/virtual/dmi/id/product_name", "r");
    if (f) {
        if (fgets(cpu_model_name, sizeof(cpu_model_name), f))
            cpu_model_name[strcspn(cpu_model_name, "\n\r")] = '\0';
        fclose(f);
        /* Clean up underscores for display */
        for (char *p = cpu_model_name; *p; p++)
            if (*p == '_') *p = ' ';
        if (cpu_model_name[0]) return;
    }

    /* x86: "model name" from /proc/cpuinfo */
    f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char *sep = strchr(line, ':');
        if (!sep) continue;
        if (strncmp(line, "model name", 10) == 0) {
            const char *val = sep + 1;
            while (*val == ' ' || *val == '\t') val++;
            snprintf(cpu_model_name, sizeof(cpu_model_name), "%s", val);
            cpu_model_name[strcspn(cpu_model_name, "\n\r")] = '\0';
            break;
        }
    }
    fclose(f);
}

static void read_cpu_part_ids(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[256];
    int cur_cpu = -1;
    while (fgets(line, sizeof(line), f)) {
        int n;
        if (sscanf(line, "processor : %d", &n) == 1) {
            cur_cpu = n;
        } else if (cur_cpu >= 0 && cur_cpu < max_cpus) {
            unsigned int part;
            if (sscanf(line, "CPU part : %x", &part) == 1)
                cpu_part[cur_cpu] = part;
        }
    }
    fclose(f);
}

static const char *cpu_part_label(int cpu_idx) {
    switch (cpu_part[cpu_idx]) {
    /* Grace (DGX Spark / DGX 300) */
    case 0xd85: return "X925";
    case 0xd87: return "X725";
    /* Jetson Orin (Nano / NX / AGX) */
    case 0xd42: return "A78A";  /* Cortex-A78AE */
    /* Other common ARM cores */
    case 0xd44: return "X4";
    case 0xd43: return "A720";
    case 0xd46: return "A725";
    case 0xd41: return "A78";
    case 0xd40: return "V2";
    case 0xd0b: return "A76";
    case 0xd0a: return "A75";
    case 0xd07: return "A57";   /* Jetson TX1/TX2 */
    case 0xd03: return "A53";   /* Jetson Nano (original) */
    default:    return "";
    }
}

/* ── CPU sampling ───────────────────────────────────────────────────── */

static void read_cpu_ticks(CpuTick ticks[], int *n_cpus) {
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return;

    char line[512];
    int idx = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "cpu", 3) != 0) continue;
        CpuTick t = {0};
        if (line[3] == ' ') {
            /* aggregate */
            sscanf(line + 4, "%llu %llu %llu %llu %llu %llu %llu %llu",
                   &t.user, &t.nice, &t.system, &t.idle,
                   &t.iowait, &t.irq, &t.softirq, &t.steal);
            ticks[0] = t;
        } else {
            int cpunum;
            sscanf(line + 3, "%d", &cpunum);
            sscanf(strchr(line + 3, ' ') + 1, "%llu %llu %llu %llu %llu %llu %llu %llu",
                   &t.user, &t.nice, &t.system, &t.idle,
                   &t.iowait, &t.irq, &t.softirq, &t.steal);
            if (cpunum + 1 < max_cpus) {
                ticks[cpunum + 1] = t;
                idx = cpunum + 1;
            }
        }
    }
    *n_cpus = idx;
    fclose(f);
}

static void compute_cpu_usage(void) {
    memset(cur_ticks, 0, (max_cpus + 1) * sizeof(CpuTick));
    int n = 0;
    read_cpu_ticks(cur_ticks, &n);
    if (n > max_cpus) n = max_cpus;
    num_cpus = n;

    for (int i = 0; i <= n; i++) {
        unsigned long long prev_idle  = prev_ticks[i].idle + prev_ticks[i].iowait;
        unsigned long long cur_idle   = cur_ticks[i].idle + cur_ticks[i].iowait;
        unsigned long long prev_total = prev_ticks[i].user + prev_ticks[i].nice +
                                        prev_ticks[i].system + prev_ticks[i].idle +
                                        prev_ticks[i].iowait + prev_ticks[i].irq +
                                        prev_ticks[i].softirq + prev_ticks[i].steal;
        unsigned long long cur_total  = cur_ticks[i].user + cur_ticks[i].nice +
                                        cur_ticks[i].system + cur_ticks[i].idle +
                                        cur_ticks[i].iowait + cur_ticks[i].irq +
                                        cur_ticks[i].softirq + cur_ticks[i].steal;
        unsigned long long totald = cur_total - prev_total;
        unsigned long long idled  = cur_idle - prev_idle;
        if (totald == 0)
            cpu_pct[i] = 0.0;
        else
            cpu_pct[i] = (double)(totald - idled) / (double)totald * 100.0;
    }

    memcpy(prev_ticks, cur_ticks, (max_cpus + 1) * sizeof(CpuTick));
}

/* ── Memory info ────────────────────────────────────────────────────── */

typedef struct {
    /* Raw values from /proc/meminfo */
    unsigned long long total_kb;
    unsigned long long free_kb;
    unsigned long long avail_kb;
    unsigned long long buffers_kb;
    unsigned long long cached_kb;
    unsigned long long swap_total_kb;
    unsigned long long swap_free_kb;
    /* Derived values */
    unsigned long long app_kb;      /* actual application memory */
    unsigned long long bufcache_kb; /* buffers + cached */
    unsigned long long swap_used_kb;
} MemInfo;

/* Compute derived fields from raw /proc/meminfo values */
static void meminfo_calc(MemInfo *m) {
    m->bufcache_kb = m->buffers_kb + m->cached_kb;
    m->app_kb = (m->total_kb > m->free_kb + m->bufcache_kb)
              ? m->total_kb - m->free_kb - m->bufcache_kb : 0;
    m->swap_used_kb = (m->swap_total_kb > m->swap_free_kb)
                    ? m->swap_total_kb - m->swap_free_kb : 0;
}

static void read_meminfo(MemInfo *m) {
    memset(m, 0, sizeof(*m));
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;

    long long huge_total = -1, huge_free = -1, huge_size = -1;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemTotal: %llu kB", &m->total_kb) == 1) continue;
        if (sscanf(line, "MemFree: %llu kB", &m->free_kb) == 1) continue;
        if (sscanf(line, "MemAvailable: %llu kB", &m->avail_kb) == 1) continue;
        if (sscanf(line, "Buffers: %llu kB", &m->buffers_kb) == 1) continue;
        if (sscanf(line, "Cached: %llu kB", &m->cached_kb) == 1) continue;
        if (sscanf(line, "SwapTotal: %llu kB", &m->swap_total_kb) == 1) continue;
        if (sscanf(line, "SwapFree: %llu kB", &m->swap_free_kb) == 1) continue;
        if (sscanf(line, "HugePages_Total: %lld", &huge_total) == 1) continue;
        if (sscanf(line, "HugePages_Free: %lld", &huge_free) == 1) continue;
        if (sscanf(line, "Hugepagesize: %lld kB", &huge_size) == 1) continue;
    }
    fclose(f);

    /* DGX Spark: when HugePages are active, MemAvailable is inaccurate.
     * Use HugePages_Free * Hugepagesize instead, and report swap as 0
     * since hugetlbfs pages are not swappable.
     * See: docs.nvidia.com/dgx/dgx-spark/known-issues.html */
    if (huge_total > 0 && huge_free >= 0 && huge_size > 0) {
        m->avail_kb = (unsigned long long)(huge_free * huge_size);
        m->swap_free_kb = m->swap_total_kb; /* effective 0 swap used */
    }

    meminfo_calc(m);
}

/* ── CPU thermals ───────────────────────────────────────────────────── */

static int read_cpu_temp(void) {
    /* Find highest thermal zone temp */
    int max_temp = 0;
    for (int i = 0; i < 20; i++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", i);
        FILE *f = fopen(path, "r");
        if (!f) break;
        int t = 0;
        if (fscanf(f, "%d", &t) == 1 && t > max_temp)
            max_temp = t;
        fclose(f);
    }
    return max_temp / 1000; /* millidegrees to degrees */
}

/* ── CPU frequency ──────────────────────────────────────────────────── */

static int read_cpu_freq_mhz(void) {
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "r");
    if (!f) return 0;
    int khz = 0;
    (void)!fscanf(f, "%d", &khz);
    fclose(f);
    return khz / 1000;
}

static void read_host_name(void) {
    if (gethostname(host_name, sizeof(host_name) - 1) != 0) {
        host_name[0] = '\0';
        return;
    }
    host_name[sizeof(host_name) - 1] = '\0';
    char *dot = strchr(host_name, '.');
    if (dot) *dot = '\0';
}

/* Forward declarations for process lookup (used by Tegra GPU scanner) */
static void get_proc_cmdline(unsigned int pid, char *buf, int len);
static void get_proc_user(unsigned int pid, char *buf, int len);
static double calc_proc_cpu_pct(unsigned int pid);

/* ── Tegra GPU sysfs fallback (Jetson Orin / Nano / NX / AGX) ──────── */

static int tegra_gpu_available = 0;
static char tegra_gpu_load_path[256] = "";
static int tegra_gpu_therm_zone = -1; /* thermal zone index for GPU-therm */

static void detect_tegra_gpu(void) {
    /* Try known Tegra GPU load paths */
    const char *gpu_paths[] = {
        "/sys/devices/gpu.0/load",
        "/sys/devices/platform/bus@0/17000000.gpu/load",
        "/sys/devices/platform/17000000.gpu/load",
        NULL
    };
    for (int i = 0; gpu_paths[i]; i++) {
        FILE *f = fopen(gpu_paths[i], "r");
        if (f) {
            tegra_gpu_available = 1;
            snprintf(tegra_gpu_load_path, sizeof(tegra_gpu_load_path), "%s", gpu_paths[i]);
            fclose(f);
            break;
        }
    }

    /* Find GPU thermal zone */
    for (int i = 0; i < 20; i++) {
        char path[128], type[64] = "";
        snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/type", i);
        FILE *f = fopen(path, "r");
        if (!f) break;
        if (fgets(type, sizeof(type), f)) {
            type[strcspn(type, "\n\r")] = '\0';
            if (strcasecmp(type, "GPU-therm") == 0 ||
                strcasecmp(type, "gpu-thermal") == 0) {
                tegra_gpu_therm_zone = i;
                fclose(f);
                break;
            }
        }
        fclose(f);
    }
}

static int read_tegra_gpu_util(void) {
    FILE *f = fopen(tegra_gpu_load_path, "r");
    if (!f) return -1;
    int load = 0;
    (void)!fscanf(f, "%d", &load);
    fclose(f);
    return load / 10; /* scale is 0-1000 -> 0-100% */
}

static int read_tegra_gpu_temp(void) {
    if (tegra_gpu_therm_zone < 0) return -1;
    char path[128];
    snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", tegra_gpu_therm_zone);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int t = 0;
    (void)!fscanf(f, "%d", &t);
    fclose(f);
    return t / 1000;
}

/* Scan /proc for processes with open fds to GPU device nodes.
 * Used on Jetson where NVML process listing returns garbage. */
static dev_t tegra_gpu_dev = 0; /* device ID of /dev/nvhost-gpu or /dev/dri/card0 */

static void detect_tegra_gpu_dev(void) {
    const char *dev_paths[] = {
        "/dev/nvhost-gpu",
        "/dev/dri/card0",
        "/dev/dri/renderD128",
        NULL
    };
    struct stat st;
    for (int i = 0; dev_paths[i]; i++) {
        if (stat(dev_paths[i], &st) == 0 && S_ISCHR(st.st_mode)) {
            tegra_gpu_dev = st.st_rdev;
            break;
        }
    }
}

static int scan_tegra_gpu_procs(GpuProc *procs, int max_procs) {
    if (!tegra_gpu_dev && !use_tegra_gpu) return 0;
    int n = 0;
    pid_t my_pid = getpid();

    DIR *proc_dir = opendir("/proc");
    if (!proc_dir) return 0;

    struct dirent *pent;
    while ((pent = readdir(proc_dir)) && n < max_procs) {
        /* Skip non-numeric entries */
        unsigned int pid = 0;
        if (sscanf(pent->d_name, "%u", &pid) != 1 || pid == 0) continue;
        if ((pid_t)pid == my_pid) continue; /* skip ourselves */

        /* Check if this PID already found */
        int dup = 0;
        for (int i = 0; i < n; i++)
            if (procs[i].pid == pid) { dup = 1; break; }
        if (dup) continue;

        char fd_dir[64];
        snprintf(fd_dir, sizeof(fd_dir), "/proc/%u/fd", pid);
        DIR *fds = opendir(fd_dir);
        if (!fds) continue;

        int found = 0;
        struct dirent *fent;
        while ((fent = readdir(fds))) {
            char fd_path[128], link_target[256];
            snprintf(fd_path, sizeof(fd_path), "/proc/%u/fd/%s", pid, fent->d_name);

            /* Check 1: device file matches GPU device ID */
            struct stat fd_stat;
            if (tegra_gpu_dev && stat(fd_path, &fd_stat) == 0 &&
                S_ISCHR(fd_stat.st_mode) &&
                fd_stat.st_rdev == tegra_gpu_dev) {
                found = 1;
                break;
            }

            /* Check 2: symlink target contains nvhost GPU or DRI render node */
            int llen = readlink(fd_path, link_target, sizeof(link_target) - 1);
            if (llen > 0) {
                link_target[llen] = '\0';
                if (strstr(link_target, "nvhost") && strstr(link_target, "gpu")) {
                    found = 1;
                    break;
                }
                if (strstr(link_target, "/dev/dri/render")) {
                    found = 1;
                    break;
                }
            }
        }
        closedir(fds);

        if (found) {
            GpuProc *p = &procs[n];
            p->pid = pid;
            p->mem_bytes = 0; /* not available via fd scan */
            p->type = 'C';
            p->cpu_pct = calc_proc_cpu_pct(pid);
            get_proc_cmdline(pid, p->name, sizeof(p->name));
            get_proc_user(pid, p->user, sizeof(p->user));
            n++;
        }
    }
    closedir(proc_dir);
    return n;
}

/* ── Process name lookup ────────────────────────────────────────────── */

static void get_proc_name(unsigned int pid, char *buf, int len) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%u/comm", pid);
    FILE *f = fopen(path, "r");
    if (f) {
        if (fgets(buf, len, f)) {
            char *nl = strchr(buf, '\n');
            if (nl) *nl = '\0';
        }
        fclose(f);
    } else {
        snprintf(buf, len, "[pid %u]", pid);
    }
}

static void get_proc_cmdline(unsigned int pid, char *buf, int len) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%u/cmdline", pid);
    FILE *f = fopen(path, "r");
    if (f) {
        int n = fread(buf, 1, len - 1, f);
        fclose(f);
        if (n > 0) {
            buf[n] = '\0';
            /* Replace nulls with spaces */
            for (int i = 0; i < n - 1; i++)
                if (buf[i] == '\0') buf[i] = ' ';
            /* Shorten the first arg (command) to its basename, keep the rest */
            char *space = strchr(buf, ' ');
            char *slash = NULL;
            if (space)
                slash = memrchr(buf, '/', space - buf);
            else
                slash = strrchr(buf, '/');
            if (slash)
                memmove(buf, slash + 1, strlen(slash + 1) + 1);
            return;
        }
    }
    get_proc_name(pid, buf, len);
}

static void get_proc_user(unsigned int pid, char *buf, int len) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%u/status", pid);
    FILE *f = fopen(path, "r");
    if (!f) { snprintf(buf, len, "?"); return; }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        unsigned int uid;
        if (sscanf(line, "Uid:\t%u", &uid) == 1) {
            struct passwd *pw = getpwuid(uid);
            if (pw)
                snprintf(buf, len, "%s", pw->pw_name);
            else
                snprintf(buf, len, "%u", uid);
            fclose(f);
            return;
        }
    }
    fclose(f);
    snprintf(buf, len, "?");
}

/* ── Drawing helpers ────────────────────────────────────────────────── */

static void draw_bar(int y, int x, int width, double pct, int color_pair) {
    int filled = (int)(pct / 100.0 * width + 0.5);
    if (filled > width) filled = width;

    move(y, x);
    attron(COLOR_PAIR(color_pair));
    for (int i = 0; i < filled; i++)
        addch(ACS_BLOCK);
    attroff(COLOR_PAIR(color_pair));

    attron(COLOR_PAIR(8)); /* dim */
    for (int i = filled; i < width; i++)
        addch(ACS_BULLET);
    attroff(COLOR_PAIR(8));
}

static void draw_bar_segmented(int y, int x, int width,
                               double pct_used, double pct_bufcache,
                               int color_used, int color_cache) {
    int filled_used = (int)(pct_used / 100.0 * width + 0.5);
    int filled_cache = (int)(pct_bufcache / 100.0 * width + 0.5);
    if (filled_used + filled_cache > width) filled_cache = width - filled_used;

    move(y, x);
    attron(COLOR_PAIR(color_used));
    for (int i = 0; i < filled_used; i++) addch(ACS_BLOCK);
    attroff(COLOR_PAIR(color_used));

    attron(COLOR_PAIR(color_cache));
    for (int i = 0; i < filled_cache; i++) addch(ACS_BLOCK);
    attroff(COLOR_PAIR(color_cache));

    attron(COLOR_PAIR(8));
    for (int i = filled_used + filled_cache; i < width; i++) addch(ACS_BULLET);
    attroff(COLOR_PAIR(8));
}

static const char *fmt_bytes(unsigned long long bytes, char *buf, int len) {
    if (bytes >= (1ULL << 30))
        snprintf(buf, len, "%.1fG", (double)bytes / (1ULL << 30));
    else if (bytes >= (1ULL << 20))
        snprintf(buf, len, "%.1fM", (double)bytes / (1ULL << 20));
    else if (bytes >= (1ULL << 10))
        snprintf(buf, len, "%.1fK", (double)bytes / (1ULL << 10));
    else
        snprintf(buf, len, "%lluB", bytes);
    return buf;
}

static const char *fmt_rate(double bytes_per_sec, char *buf, int len) {
    if (bytes_per_sec >= (1ULL << 30))
        snprintf(buf, len, "%.1fG/s", bytes_per_sec / (double)(1ULL << 30));
    else if (bytes_per_sec >= (1ULL << 20))
        snprintf(buf, len, "%.1fM/s", bytes_per_sec / (double)(1ULL << 20));
    else if (bytes_per_sec >= (1ULL << 10))
        snprintf(buf, len, "%.1fK/s", bytes_per_sec / (double)(1ULL << 10));
    else
        snprintf(buf, len, "%.0fB/s", bytes_per_sec);
    return buf;
}

/* ── Uptime ─────────────────────────────────────────────────────────── */

static void fmt_uptime(char *buf, int len) {
    struct sysinfo si;
    if (sysinfo(&si) != 0) { snprintf(buf, len, "?"); return; }
    long s = si.uptime;
    int days = s / 86400; s %= 86400;
    int hrs  = s / 3600;  s %= 3600;
    int mins = s / 60;
    if (days > 0)
        snprintf(buf, len, "%dd %dh %dm", days, hrs, mins);
    else
        snprintf(buf, len, "%dh %dm", hrs, mins);
}

/* ── Load average ───────────────────────────────────────────────────── */

static void get_loadavg(double *l1, double *l5, double *l15) {
    FILE *f = fopen("/proc/loadavg", "r");
    if (f) { (void)!fscanf(f, "%lf %lf %lf", l1, l5, l15); fclose(f); }
}

/* ── History chart ──────────────────────────────────────────────────── */

/* Unicode block elements: ▁▂▃▄▅▆▇█ (U+2581..U+2588) */
static const char *block_chars[] = {
    " ", "\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83",
    "\xe2\x96\x84", "\xe2\x96\x85", "\xe2\x96\x86",
    "\xe2\x96\x87", "\xe2\x96\x88"
};

static void draw_history_chart(int top_y, int total_w, int chart_h) {
    int n = history_count < HISTORY_LEN ? history_count : HISTORY_LEN;
    if (n == 0) return;

    int margin = 2;
    int label_w = 5; /* "100% " */
    int left_x = margin + label_w;
    int right_x = total_w - margin;
    int avail_w = right_x - left_x;
    if (avail_w < 10) return;

    /* Title centered above chart */
    int title_x = left_x;
    attron(A_BOLD | COLOR_PAIR(7));
    mvprintw(top_y - 1, title_x, "CPU");
    attroff(A_BOLD | COLOR_PAIR(7));
    attron(COLOR_PAIR(8));
    printw("/");
    attroff(COLOR_PAIR(8));
    attron(A_BOLD | COLOR_PAIR(6));
    printw("GPU");
    attroff(A_BOLD | COLOR_PAIR(6));
    attron(COLOR_PAIR(8));
    printw(" history");
    attroff(COLOR_PAIR(8));

    /* Fixed column width based on max samples — prevents rescaling as history fills.
     * Each sample: cpu_w + gpu_w + 1 gap char */
    int col_w = avail_w / HISTORY_LEN;
    if (col_w < 3) col_w = 3;
    int gap = 1;
    int bar_w = col_w - gap;
    int cpu_w = bar_w / 2;
    int gpu_w = bar_w - cpu_w;

    /* Clamp visible samples to what fits in avail_w. On very narrow terminals
     * this shows fewer samples rather than overflowing into other lines. */
    int max_visible = avail_w / col_w;
    if (max_visible < 1) return;
    int visible = n;
    if (visible > max_visible) visible = max_visible;

    /* Right-align: new samples appear on the right */
    int chart_total = visible * col_w;
    int x_start = right_x - chart_total;

    for (int s = 0; s < visible; s++) {
        int idx = (history_pos - visible + s + HISTORY_LEN) % HISTORY_LEN;
        double cpu_val = cpu_history[idx];
        double gpu_val = gpu_history[idx];

        int cpu_blocks = (int)(cpu_val / 100.0 * chart_h * 8 + 0.5);
        int gpu_blocks = (int)(gpu_val / 100.0 * chart_h * 8 + 0.5);

        int x = x_start + s * col_w;

        for (int row = 0; row < chart_h; row++) {
            int ry = top_y + chart_h - 1 - row;
            int row_base = row * 8;

            int cpu_fill = cpu_blocks - row_base;
            if (cpu_fill < 0) cpu_fill = 0;
            if (cpu_fill > 8) cpu_fill = 8;

            int gpu_fill = gpu_blocks - row_base;
            if (gpu_fill < 0) gpu_fill = 0;
            if (gpu_fill > 8) gpu_fill = 8;

            move(ry, x);
            attron(COLOR_PAIR(2)); /* green = CPU */
            for (int c = 0; c < cpu_w; c++)
                printw("%s", block_chars[cpu_fill]);
            attroff(COLOR_PAIR(2));
            attron(COLOR_PAIR(6)); /* cyan = GPU */
            for (int c = 0; c < gpu_w; c++)
                printw("%s", block_chars[gpu_fill]);
            attroff(COLOR_PAIR(6));
            /* gap between samples */
            printw(" ");
        }
    }

    /* Y-axis labels */
    attron(COLOR_PAIR(8));
    mvprintw(top_y, margin, "100%%");
    mvprintw(top_y + chart_h - 1, margin, "  0%%");

    /* X-axis: t-N labels (seconds ago), right-aligned with 0 on the right */
    int x_row = top_y + chart_h;
    for (int t = 0; t < visible; t += 5) {
        int s = visible - 1 - t; /* sample index from right */
        int x = x_start + s * col_w;
        if (x >= left_x && x < right_x - 2)
            mvprintw(x_row, x, "%-3d", t);
    }
    /* label */
    mvprintw(x_row, margin, "  t=");
    attroff(COLOR_PAIR(8));
}

static void record_history(double cpu, double gpu) {
    cpu_history[history_pos] = cpu;
    gpu_history[history_pos] = gpu;
    history_pos = (history_pos + 1) % HISTORY_LEN;
    if (history_count < HISTORY_LEN) history_count++;
}

/* ── Aggregate network throughput ───────────────────────────────────── */

typedef struct {
    unsigned long long rx_bytes;
    unsigned long long tx_bytes;
    double rx_bytes_sec;
    double tx_bytes_sec;
    int valid;
} NetTotals;

static NetTotals net_totals = {0};
static unsigned long long net_prev_rx = 0;
static unsigned long long net_prev_tx = 0;
static struct timespec    net_prev_time;
static int                net_prev_valid = 0;
static double             net_scale_bytes_sec = 1024.0 * 1024.0; /* auto-scaling floor: 1 MiB/s */

static void read_net_totals(void) {
    FILE *f = fopen("/proc/net/dev", "r");
    if (!f) {
        net_totals.valid = 0;
        return;
    }

    unsigned long long rx_total = 0;
    unsigned long long tx_total = 0;
    char line[512];
    int line_no = 0;
    while (fgets(line, sizeof(line), f)) {
        line_no++;
        if (line_no <= 2) continue;

        char *colon = strchr(line, ':');
        if (!colon) continue;
        *colon = '\0';

        char ifname[64];
        snprintf(ifname, sizeof(ifname), "%s", line);
        char *name = ifname;
        while (*name == ' ') name++;
        if (strcmp(name, "lo") == 0) continue;

        unsigned long long rx_bytes = 0, tx_bytes = 0;
        unsigned long long discard[14];
        if (sscanf(colon + 1,
                   " %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                   &rx_bytes,
                   &discard[0], &discard[1], &discard[2], &discard[3], &discard[4], &discard[5], &discard[6],
                   &tx_bytes,
                   &discard[7], &discard[8], &discard[9], &discard[10], &discard[11], &discard[12], &discard[13]) == 16) {
            rx_total += rx_bytes;
            tx_total += tx_bytes;
        }
    }
    fclose(f);

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double dt = 0.0;
    if (net_prev_valid) {
        dt = (now.tv_sec - net_prev_time.tv_sec) +
             (now.tv_nsec - net_prev_time.tv_nsec) / 1e9;
        if (dt <= 0) dt = 1.0;
    }

    net_totals.rx_bytes = rx_total;
    net_totals.tx_bytes = tx_total;
    if (net_prev_valid) {
        if (rx_total >= net_prev_rx)
            net_totals.rx_bytes_sec = (double)(rx_total - net_prev_rx) / dt;
        else
            net_totals.rx_bytes_sec = 0;  /* counter wrapped, skip frame */
        if (tx_total >= net_prev_tx)
            net_totals.tx_bytes_sec = (double)(tx_total - net_prev_tx) / dt;
        else
            net_totals.tx_bytes_sec = 0;  /* counter wrapped, skip frame */
    } else {
        net_totals.rx_bytes_sec = 0;
        net_totals.tx_bytes_sec = 0;
    }
    net_totals.valid = 1;
    double total_bytes_sec = net_totals.rx_bytes_sec + net_totals.tx_bytes_sec;
    double decayed_scale = net_scale_bytes_sec * 0.95;
    if (decayed_scale < 1024.0 * 1024.0) decayed_scale = 1024.0 * 1024.0;
    net_scale_bytes_sec = total_bytes_sec > decayed_scale ? total_bytes_sec : decayed_scale;
    net_prev_rx = rx_total;
    net_prev_tx = tx_total;
    net_prev_time = now;
    net_prev_valid = 1;
}

/* ── RDMA types (used by CSV logging and Prometheus) ───────────────── */

#define MAX_RDMA_PORTS 16

typedef struct {
    char device[64];
    int  port;
    char state[32];
    char rate[32];
    unsigned long long xmit_bytes;
    unsigned long long recv_bytes;
    unsigned long long xmit_pkts;
    unsigned long long recv_pkts;
    unsigned long long errors;
    double xmit_bytes_sec;
    double recv_bytes_sec;
} RdmaPort;

static RdmaPort rdma_ports[MAX_RDMA_PORTS];
static int       rdma_count = 0;
static int       rdma_available = 0;

/* ── CSV logging ────────────────────────────────────────────────────── */

static void log_csv_header(FILE *f) {
    fprintf(f, "timestamp,cpu_avg_pct");
    for (int i = 1; i <= num_cpus; i++)
        fprintf(f, ",cpu%d_pct", i - 1);
    fprintf(f, ",cpu_temp_c,cpu_freq_mhz");
    fprintf(f, ",mem_used_kb,mem_total_kb,mem_bufcache_kb");
    fprintf(f, ",swap_used_kb,swap_total_kb");
    fprintf(f, ",net_rx_Bps,net_tx_Bps");
    for (unsigned int g = 0; g < gpu_count; g++)
        fprintf(f, ",gpu%u_util_pct,gpu%u_temp_c,gpu%u_power_mw,gpu%u_clock_mhz", g, g, g, g);
    for (int i = 0; i < rdma_count; i++)
        fprintf(f, ",rdma_%s_p%d_xmit_Bps,rdma_%s_p%d_recv_Bps",
                rdma_ports[i].device, rdma_ports[i].port,
                rdma_ports[i].device, rdma_ports[i].port);
    fprintf(f, "\n");
    fflush(f);
}

static void log_csv_row(FILE *f) {
    /* Timestamp */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    fprintf(f, "%04d-%02d-%02dT%02d:%02d:%02d.%03ld",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000);

    /* CPU */
    fprintf(f, ",%.1f", cpu_pct[0]);
    for (int i = 1; i <= num_cpus; i++)
        fprintf(f, ",%.1f", cpu_pct[i]);

    fprintf(f, ",%d,%d", read_cpu_temp(), read_cpu_freq_mhz());

    /* Memory */
    MemInfo mi;
    read_meminfo(&mi);
    fprintf(f, ",%llu,%llu,%llu", mi.app_kb, mi.total_kb, mi.bufcache_kb);
    fprintf(f, ",%llu,%llu", mi.swap_used_kb, mi.swap_total_kb);
    fprintf(f, ",%.0f,%.0f", net_totals.rx_bytes_sec, net_totals.tx_bytes_sec);

    /* GPU */
    for (unsigned int g = 0; g < gpu_count; g++) {
        nvmlDevice_t dev;
        if (pNvmlDeviceGetHandleByIndex(g, &dev) == NVML_SUCCESS) {
            nvmlUtilization_t util = {0};
            if (!use_tegra_gpu && pNvmlDeviceGetUtilizationRates)
                pNvmlDeviceGetUtilizationRates(dev, &util);
            if (use_tegra_gpu) {
                int tutil = read_tegra_gpu_util();
                if (tutil >= 0) util.gpu = (unsigned int)tutil;
            }

            unsigned int temp = 0;
            if (!use_tegra_gpu && pNvmlDeviceGetTemperature)
                pNvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp);
            if (use_tegra_gpu && tegra_gpu_therm_zone >= 0) {
                int ttemp = read_tegra_gpu_temp();
                if (ttemp > 0) temp = (unsigned int)ttemp;
            }

            unsigned int power_mw = 0;
            if (pNvmlDeviceGetPowerUsage)
                pNvmlDeviceGetPowerUsage(dev, &power_mw);

            unsigned int clk = 0;
            if (pNvmlDeviceGetClockInfo)
                pNvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &clk);

            fprintf(f, ",%u,%u,%u,%u", util.gpu, temp, power_mw, clk);
        } else {
            fprintf(f, ",,,,");
        }
    }
    if (gpu_count == 0) fprintf(f, ",,,,");

    for (int i = 0; i < rdma_count; i++)
        fprintf(f, ",%.0f,%.0f", rdma_ports[i].xmit_bytes_sec, rdma_ports[i].recv_bytes_sec);

    fprintf(f, "\n");
    fflush(f);
}

/* ── RDMA / InfiniBand monitoring ───────────────────────────────────── */

static unsigned long long rdma_prev_xmit[MAX_RDMA_PORTS];
static unsigned long long rdma_prev_recv[MAX_RDMA_PORTS];
static struct timespec    rdma_prev_time;
static int                rdma_prev_valid = 0;

static unsigned long long read_sysfs_ull(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    unsigned long long val = 0;
    (void)!fscanf(f, "%llu", &val);
    fclose(f);
    return val;
}

static void read_sysfs_str(const char *path, char *buf, int len) {
    FILE *f = fopen(path, "r");
    if (!f) { buf[0] = '\0'; return; }
    if (!fgets(buf, len, f)) buf[0] = '\0';
    fclose(f);
    buf[strcspn(buf, "\n\r")] = '\0';
}

static void read_rdma_ports(void) {
    DIR *ib_dir = opendir("/sys/class/infiniband");
    if (!ib_dir) { rdma_available = 0; rdma_count = 0; return; }

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double dt = 0;
    if (rdma_prev_valid) {
        dt = (now.tv_sec - rdma_prev_time.tv_sec) +
             (now.tv_nsec - rdma_prev_time.tv_nsec) / 1e9;
        if (dt <= 0) dt = 1;
    }

    rdma_available = 1;
    int idx = 0;
    struct dirent *dev_ent;
    while ((dev_ent = readdir(ib_dir)) && idx < MAX_RDMA_PORTS) {
        if (dev_ent->d_name[0] == '.') continue;

        /* Scan ports (typically 1-2) */
        for (int p = 1; p <= 2 && idx < MAX_RDMA_PORTS; p++) {
            char path[256];
            snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/state", dev_ent->d_name, p);
            FILE *test = fopen(path, "r");
            if (!test) continue;
            fclose(test);

            RdmaPort *r = &rdma_ports[idx];
            snprintf(r->device, sizeof(r->device), "%s", dev_ent->d_name);
            r->port = p;

            read_sysfs_str(path, r->state, sizeof(r->state));
            /* Strip numeric prefix like "4: ACTIVE" -> "ACTIVE" */
            char *colon = strchr(r->state, ':');
            if (colon) {
                const char *s = colon + 1;
                while (*s == ' ') s++;
                memmove(r->state, s, strlen(s) + 1);
            }

            snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/rate", dev_ent->d_name, p);
            read_sysfs_str(path, r->rate, sizeof(r->rate));

            /* Counters — RDMA counters are in units of 4 bytes (32-bit words) for data */
            snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/counters/port_xmit_data", dev_ent->d_name, p);
            r->xmit_bytes = read_sysfs_ull(path) * 4;
            snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/counters/port_rcv_data", dev_ent->d_name, p);
            r->recv_bytes = read_sysfs_ull(path) * 4;
            snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/counters/port_xmit_packets", dev_ent->d_name, p);
            r->xmit_pkts = read_sysfs_ull(path);
            snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/counters/port_rcv_packets", dev_ent->d_name, p);
            r->recv_pkts = read_sysfs_ull(path);

            /* Sum error counters */
            r->errors = 0;
            const char *err_counters[] = {
                "symbol_error_counter", "port_rcv_errors",
                "port_rcv_constraint_errors", "port_xmit_constraint_errors",
                "link_error_recovery_counter", "link_downed_counter",
                NULL
            };
            for (int e = 0; err_counters[e]; e++) {
                snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/counters/%s",
                         dev_ent->d_name, p, err_counters[e]);
                r->errors += read_sysfs_ull(path);
            }

            /* Rate calculation */
            if (rdma_prev_valid && idx < rdma_count) {
                if (r->xmit_bytes >= rdma_prev_xmit[idx])
                    r->xmit_bytes_sec = (double)(r->xmit_bytes - rdma_prev_xmit[idx]) / dt;
                else
                    r->xmit_bytes_sec = 0;  /* counter wrapped, skip frame */
                if (r->recv_bytes >= rdma_prev_recv[idx])
                    r->recv_bytes_sec = (double)(r->recv_bytes - rdma_prev_recv[idx]) / dt;
                else
                    r->recv_bytes_sec = 0;  /* counter wrapped, skip frame */
            } else {
                r->xmit_bytes_sec = 0;
                r->recv_bytes_sec = 0;
            }

            rdma_prev_xmit[idx] = r->xmit_bytes;
            rdma_prev_recv[idx] = r->recv_bytes;
            idx++;
        }
    }
    closedir(ib_dir);
    rdma_count = idx;
    rdma_prev_time = now;
    rdma_prev_valid = 1;
}

/* ── Prometheus metrics exporter ────────────────────────────────────── */

static int   prom_sock = -1;
static pthread_t prom_thread;

#define PROM_BYTES_PER_GPU 512  /* estimated Prometheus output per GPU */
#define PROM_BASE_SIZE 8192     /* base buffer for CPU/memory/system metrics */

typedef struct {
    int      valid;
    char     name[96];
    unsigned int util_gpu;
    unsigned int temp;
    unsigned int power_mw;
    int      has_power;
    unsigned int clk_gfx, clk_mem;
    unsigned long long mem_total, mem_used;
    int      has_mem;
    unsigned int fan;
    int      has_fan;
    unsigned int enc, dec;
    int      has_enc, has_dec;
} PromGpu;

static int      prom_buf_size = 0;
static char    *prom_body = NULL;
static PromGpu *prom_gpus = NULL;

/* Format all metrics into buf. Returns bytes written. */
static int format_metrics(char *buf, int buflen) {
    int off = 0;

    #define PM(...) do { \
        int _n = snprintf(buf + off, (size_t)(buflen - off), __VA_ARGS__); \
        if (_n > 0) { \
            if (_n >= buflen - off) { off = buflen - 1; goto pm_done; } \
            off += _n; \
        } \
    } while(0)

    /* Build info */
    PM("# HELP nv_build_info nv-monitor version\n"
       "# TYPE nv_build_info gauge\n"
       "nv_build_info{version=\"%s\"} 1\n", VERSION);

    /* Uptime */
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        PM("# HELP nv_uptime_seconds System uptime\n"
           "# TYPE nv_uptime_seconds gauge\n"
           "nv_uptime_seconds %ld\n", si.uptime);
    }

    /* Load average */
    double l1 = 0, l5 = 0, l15 = 0;
    get_loadavg(&l1, &l5, &l15);
    PM("# HELP nv_load_average System load average\n"
       "# TYPE nv_load_average gauge\n"
       "nv_load_average{interval=\"1m\"} %.2f\n"
       "nv_load_average{interval=\"5m\"} %.2f\n"
       "nv_load_average{interval=\"15m\"} %.2f\n", l1, l5, l15);

    /* CPU usage */
    PM("# HELP nv_cpu_usage_percent CPU utilization\n"
       "# TYPE nv_cpu_usage_percent gauge\n"
       "nv_cpu_usage_percent{cpu=\"overall\"} %.1f\n", cpu_pct[0]);
    for (int i = 1; i <= num_cpus; i++) {
        const char *lbl = cpu_part_label(i - 1);
        if (lbl[0])
            PM("nv_cpu_usage_percent{cpu=\"%d\",type=\"%s\"} %.1f\n",
               i - 1, lbl, cpu_pct[i]);
        else
            PM("nv_cpu_usage_percent{cpu=\"%d\"} %.1f\n",
               i - 1, cpu_pct[i]);
    }

    /* CPU temperature */
    PM("# HELP nv_cpu_temperature_celsius CPU temperature\n"
       "# TYPE nv_cpu_temperature_celsius gauge\n"
       "nv_cpu_temperature_celsius %d\n", read_cpu_temp());

    /* CPU frequency */
    PM("# HELP nv_cpu_frequency_mhz CPU frequency\n"
       "# TYPE nv_cpu_frequency_mhz gauge\n"
       "nv_cpu_frequency_mhz %d\n", read_cpu_freq_mhz());

    /* Memory */
    MemInfo mi;
    read_meminfo(&mi);
    PM("# HELP nv_memory_total_bytes Total system memory\n"
       "# TYPE nv_memory_total_bytes gauge\n"
       "nv_memory_total_bytes %llu\n"
       "# HELP nv_memory_used_bytes Application memory used\n"
       "# TYPE nv_memory_used_bytes gauge\n"
       "nv_memory_used_bytes %llu\n"
       "# HELP nv_memory_bufcache_bytes Buffer and cache memory\n"
       "# TYPE nv_memory_bufcache_bytes gauge\n"
       "nv_memory_bufcache_bytes %llu\n",
       mi.total_kb * 1024ULL, mi.app_kb * 1024ULL, mi.bufcache_kb * 1024ULL);

    if (mi.swap_total_kb > 0) {
        PM("# HELP nv_swap_total_bytes Total swap\n"
           "# TYPE nv_swap_total_bytes gauge\n"
           "nv_swap_total_bytes %llu\n"
           "# HELP nv_swap_used_bytes Swap used\n"
           "# TYPE nv_swap_used_bytes gauge\n"
           "nv_swap_used_bytes %llu\n",
           mi.swap_total_kb * 1024ULL, mi.swap_used_kb * 1024ULL);
    }

    if (net_totals.valid) {
        PM("# HELP nv_network_receive_bytes_total Total bytes received across all non-loopback interfaces\n"
           "# TYPE nv_network_receive_bytes_total counter\n"
           "nv_network_receive_bytes_total %llu\n"
           "# HELP nv_network_transmit_bytes_total Total bytes transmitted across all non-loopback interfaces\n"
           "# TYPE nv_network_transmit_bytes_total counter\n"
           "nv_network_transmit_bytes_total %llu\n"
           "# HELP nv_network_receive_bytes_per_second Aggregate receive throughput across all non-loopback interfaces\n"
           "# TYPE nv_network_receive_bytes_per_second gauge\n"
           "nv_network_receive_bytes_per_second %.0f\n"
           "# HELP nv_network_transmit_bytes_per_second Aggregate transmit throughput across all non-loopback interfaces\n"
           "# TYPE nv_network_transmit_bytes_per_second gauge\n"
           "nv_network_transmit_bytes_per_second %.0f\n",
           net_totals.rx_bytes, net_totals.tx_bytes,
           net_totals.rx_bytes_sec, net_totals.tx_bytes_sec);
    }

    /* GPU — collect data first, then format grouped by metric family */
    PromGpu *gpus = prom_gpus;
    int n_gpus = 0;
    if (gpus && gpu_count > 0)
        memset(gpus, 0, gpu_count * sizeof(PromGpu));

    if (nvml_ok) {
        unsigned int dev_count = 0;
        pNvmlDeviceGetCount(&dev_count);

        for (unsigned int d = 0; d < dev_count && d < gpu_count; d++) {
            PromGpu *g = &gpus[n_gpus];
            memset(g, 0, sizeof(*g));
            nvmlDevice_t dev;
            if (pNvmlDeviceGetHandleByIndex(d, &dev) != NVML_SUCCESS) continue;
            g->valid = 1;
            pNvmlDeviceGetName(dev, g->name, sizeof(g->name));

            nvmlUtilization_t util = {0};
            if (!use_tegra_gpu && pNvmlDeviceGetUtilizationRates)
                pNvmlDeviceGetUtilizationRates(dev, &util);
            if (use_tegra_gpu) {
                int tutil = read_tegra_gpu_util();
                if (tutil >= 0) util.gpu = (unsigned int)tutil;
            }
            g->util_gpu = util.gpu;

            if (!use_tegra_gpu && pNvmlDeviceGetTemperature)
                pNvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &g->temp);
            if (use_tegra_gpu && tegra_gpu_therm_zone >= 0) {
                int ttemp = read_tegra_gpu_temp();
                if (ttemp > 0) g->temp = (unsigned int)ttemp;
            }

            g->has_power = (pNvmlDeviceGetPowerUsage &&
                            pNvmlDeviceGetPowerUsage(dev, &g->power_mw) == NVML_SUCCESS);

            if (pNvmlDeviceGetClockInfo) {
                pNvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &g->clk_gfx);
                pNvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &g->clk_mem);
            }

            nvmlMemory_t mem = {0};
            g->has_mem = (pNvmlDeviceGetMemoryInfo &&
                          pNvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS &&
                          mem.total > 0);
            if (g->has_mem) { g->mem_total = mem.total; g->mem_used = mem.used; }

            unsigned int period;
            g->has_fan = (pNvmlDeviceGetFanSpeed &&
                          pNvmlDeviceGetFanSpeed(dev, &g->fan) == NVML_SUCCESS);
            g->has_enc = (pNvmlDeviceGetEncoderUtilization &&
                          pNvmlDeviceGetEncoderUtilization(dev, &g->enc, &period) == NVML_SUCCESS);
            g->has_dec = (pNvmlDeviceGetDecoderUtilization &&
                          pNvmlDeviceGetDecoderUtilization(dev, &g->dec, &period) == NVML_SUCCESS);
            n_gpus++;
        }
    }

    if (n_gpus > 0) {
        PM("# HELP nv_gpu_info GPU device information\n"
           "# TYPE nv_gpu_info gauge\n");
        for (int d = 0; d < n_gpus; d++)
            PM("nv_gpu_info{gpu=\"%d\",name=\"%s\"} 1\n", d, gpus[d].name);

        PM("# HELP nv_gpu_utilization_percent GPU compute utilization\n"
           "# TYPE nv_gpu_utilization_percent gauge\n");
        for (int d = 0; d < n_gpus; d++)
            PM("nv_gpu_utilization_percent{gpu=\"%d\"} %u\n", d, gpus[d].util_gpu);

        PM("# HELP nv_gpu_temperature_celsius GPU temperature\n"
           "# TYPE nv_gpu_temperature_celsius gauge\n");
        for (int d = 0; d < n_gpus; d++)
            PM("nv_gpu_temperature_celsius{gpu=\"%d\"} %u\n", d, gpus[d].temp);

        PM("# HELP nv_gpu_power_watts GPU power draw\n"
           "# TYPE nv_gpu_power_watts gauge\n");
        for (int d = 0; d < n_gpus; d++)
            if (gpus[d].has_power)
                PM("nv_gpu_power_watts{gpu=\"%d\"} %.1f\n", d, gpus[d].power_mw / 1000.0);

        PM("# HELP nv_gpu_clock_mhz GPU clock speed\n"
           "# TYPE nv_gpu_clock_mhz gauge\n");
        for (int d = 0; d < n_gpus; d++) {
            if (gpus[d].clk_gfx)
                PM("nv_gpu_clock_mhz{gpu=\"%d\",type=\"graphics\"} %u\n", d, gpus[d].clk_gfx);
            if (gpus[d].clk_mem)
                PM("nv_gpu_clock_mhz{gpu=\"%d\",type=\"memory\"} %u\n", d, gpus[d].clk_mem);
        }

        PM("# HELP nv_gpu_memory_total_bytes GPU memory total\n"
           "# TYPE nv_gpu_memory_total_bytes gauge\n");
        for (int d = 0; d < n_gpus; d++)
            if (gpus[d].has_mem)
                PM("nv_gpu_memory_total_bytes{gpu=\"%d\"} %llu\n", d, gpus[d].mem_total);

        PM("# HELP nv_gpu_memory_used_bytes GPU memory used\n"
           "# TYPE nv_gpu_memory_used_bytes gauge\n");
        for (int d = 0; d < n_gpus; d++)
            if (gpus[d].has_mem)
                PM("nv_gpu_memory_used_bytes{gpu=\"%d\"} %llu\n", d, gpus[d].mem_used);

        PM("# HELP nv_gpu_fan_speed_percent GPU fan speed\n"
           "# TYPE nv_gpu_fan_speed_percent gauge\n");
        for (int d = 0; d < n_gpus; d++)
            if (gpus[d].has_fan)
                PM("nv_gpu_fan_speed_percent{gpu=\"%d\"} %u\n", d, gpus[d].fan);

        PM("# HELP nv_gpu_encoder_utilization_percent GPU encoder utilization\n"
           "# TYPE nv_gpu_encoder_utilization_percent gauge\n");
        for (int d = 0; d < n_gpus; d++)
            if (gpus[d].has_enc)
                PM("nv_gpu_encoder_utilization_percent{gpu=\"%d\"} %u\n", d, gpus[d].enc);

        PM("# HELP nv_gpu_decoder_utilization_percent GPU decoder utilization\n"
           "# TYPE nv_gpu_decoder_utilization_percent gauge\n");
        for (int d = 0; d < n_gpus; d++)
            if (gpus[d].has_dec)
                PM("nv_gpu_decoder_utilization_percent{gpu=\"%d\"} %u\n", d, gpus[d].dec);
    }

    /* RDMA / InfiniBand */
    if (rdma_available && rdma_count > 0) {
        PM("# HELP nv_rdma_info RDMA port information\n"
           "# TYPE nv_rdma_info gauge\n");
        for (int i = 0; i < rdma_count; i++)
            PM("nv_rdma_info{device=\"%s\",port=\"%d\",state=\"%s\",rate=\"%s\"} 1\n",
               rdma_ports[i].device, rdma_ports[i].port,
               rdma_ports[i].state, rdma_ports[i].rate);

        PM("# HELP nv_rdma_xmit_bytes_total Total bytes transmitted\n"
           "# TYPE nv_rdma_xmit_bytes_total counter\n");
        for (int i = 0; i < rdma_count; i++)
            PM("nv_rdma_xmit_bytes_total{device=\"%s\",port=\"%d\"} %llu\n",
               rdma_ports[i].device, rdma_ports[i].port, rdma_ports[i].xmit_bytes);

        PM("# HELP nv_rdma_recv_bytes_total Total bytes received\n"
           "# TYPE nv_rdma_recv_bytes_total counter\n");
        for (int i = 0; i < rdma_count; i++)
            PM("nv_rdma_recv_bytes_total{device=\"%s\",port=\"%d\"} %llu\n",
               rdma_ports[i].device, rdma_ports[i].port, rdma_ports[i].recv_bytes);

        PM("# HELP nv_rdma_xmit_packets_total Total packets transmitted\n"
           "# TYPE nv_rdma_xmit_packets_total counter\n");
        for (int i = 0; i < rdma_count; i++)
            PM("nv_rdma_xmit_packets_total{device=\"%s\",port=\"%d\"} %llu\n",
               rdma_ports[i].device, rdma_ports[i].port, rdma_ports[i].xmit_pkts);

        PM("# HELP nv_rdma_recv_packets_total Total packets received\n"
           "# TYPE nv_rdma_recv_packets_total counter\n");
        for (int i = 0; i < rdma_count; i++)
            PM("nv_rdma_recv_packets_total{device=\"%s\",port=\"%d\"} %llu\n",
               rdma_ports[i].device, rdma_ports[i].port, rdma_ports[i].recv_pkts);

        PM("# HELP nv_rdma_errors_total Total RDMA errors\n"
           "# TYPE nv_rdma_errors_total counter\n");
        for (int i = 0; i < rdma_count; i++)
            PM("nv_rdma_errors_total{device=\"%s\",port=\"%d\"} %llu\n",
               rdma_ports[i].device, rdma_ports[i].port, rdma_ports[i].errors);

        PM("# HELP nv_rdma_xmit_bytes_per_second Transmit throughput\n"
           "# TYPE nv_rdma_xmit_bytes_per_second gauge\n");
        for (int i = 0; i < rdma_count; i++)
            PM("nv_rdma_xmit_bytes_per_second{device=\"%s\",port=\"%d\"} %.0f\n",
               rdma_ports[i].device, rdma_ports[i].port, rdma_ports[i].xmit_bytes_sec);

        PM("# HELP nv_rdma_recv_bytes_per_second Receive throughput\n"
           "# TYPE nv_rdma_recv_bytes_per_second gauge\n");
        for (int i = 0; i < rdma_count; i++)
            PM("nv_rdma_recv_bytes_per_second{device=\"%s\",port=\"%d\"} %.0f\n",
               rdma_ports[i].device, rdma_ports[i].port, rdma_ports[i].recv_bytes_sec);
    }

pm_done:
    #undef PM
    return off;
}

/* Minimal HTTP handler for a single connection */
static void prom_handle(int fd) {
    /* Set timeouts to prevent stalled clients from blocking the server */
    struct timeval tv = { .tv_sec = 2, .tv_usec = 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    char req[512];
    int n = (int)recv(fd, req, sizeof(req) - 1, 0);
    if (n <= 0) return;
    req[n] = '\0';

    /* Bearer token auth if configured */
    if (prom_token) {
        char expected[512];
        snprintf(expected, sizeof(expected), "Authorization: Bearer %s", prom_token);
        if (!strstr(req, expected)) {
            static const char resp_401[] =
                "HTTP/1.1 401 Unauthorized\r\n"
                "Content-Type: text/plain\r\n"
                "Connection: close\r\n\r\n"
                "Unauthorized\n";
            send(fd, resp_401, sizeof(resp_401) - 1, MSG_NOSIGNAL);
            return;
        }
    }

    if (strstr(req, "GET /metrics")) {
        if (!prom_body) return;
        int bodylen = format_metrics(prom_body, prom_buf_size);

        char hdr[128];
        int hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
            "Content-Length: %d\r\n"
            "Connection: close\r\n\r\n", bodylen);

        send(fd, hdr, hlen, MSG_NOSIGNAL);
        send(fd, prom_body, bodylen, MSG_NOSIGNAL);
    } else {
        /* Landing page with link to /metrics */
        static const char resp[] =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "Connection: close\r\n\r\n"
            "<html><body><h1>nv-monitor</h1>"
            "<p><a href=\"/metrics\">Metrics</a></p>"
            "</body></html>\n";
        send(fd, resp, sizeof(resp) - 1, MSG_NOSIGNAL);
    }
}

/* Server thread — blocks on poll() with 1s timeout for clean shutdown */
static void *prom_server(void *arg) {
    (void)arg;
    /* Pre-allocate buffers once for the lifetime of the thread */
    prom_buf_size = PROM_BASE_SIZE + (gpu_count * PROM_BYTES_PER_GPU) +
                    (num_cpus * 80);
    prom_body = malloc(prom_buf_size);
    prom_gpus = calloc(gpu_count > 0 ? gpu_count : 1, sizeof(PromGpu));

    while (!g_quit) {
        struct pollfd pfd = { .fd = prom_sock, .events = POLLIN };
        if (poll(&pfd, 1, 1000) <= 0) continue;

        int fd = accept(prom_sock, NULL, NULL);
        if (fd < 0) continue;
        prom_handle(fd);
        close(fd);
    }
    free(prom_body);  prom_body = NULL;
    free(prom_gpus);  prom_gpus = NULL;
    return NULL;
}

static int prom_start(void) {
    if (!prom_port) return 0;

    prom_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (prom_sock < 0) {
        perror("prometheus: socket");
        return -1;
    }

    int opt = 1;
    setsockopt(prom_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port   = htons((uint16_t)prom_port),
        .sin_addr.s_addr = INADDR_ANY,
    };

    if (bind(prom_sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("prometheus: bind");
        close(prom_sock);
        prom_sock = -1;
        return -1;
    }

    if (listen(prom_sock, 4) < 0) {
        perror("prometheus: listen");
        close(prom_sock);
        prom_sock = -1;
        return -1;
    }

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 524288); /* 512 KB — room for large GPU arrays */

    if (pthread_create(&prom_thread, &attr, prom_server, NULL) != 0) {
        perror("prometheus: pthread_create");
        close(prom_sock);
        prom_sock = -1;
        pthread_attr_destroy(&attr);
        return -1;
    }

    pthread_attr_destroy(&attr);
    fprintf(stderr, "Prometheus metrics at http://0.0.0.0:%d/metrics\n", prom_port);
    return 0;
}

static void prom_stop(void) {
    if (prom_sock >= 0) {
        pthread_join(prom_thread, NULL);
        close(prom_sock);
        prom_sock = -1;
    }
}

/* ── Usage ──────────────────────────────────────────────────────────── */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  -c COLS   CPU display columns (1-4, default: auto)\n"
        "  -g MODE   GPU view mode (auto, compact, detailed)\n"
        "  -l FILE   Log statistics to CSV file\n"
        "  -i MS     Log interval in milliseconds (default: 1000)\n"
        "  -n        No UI (headless mode, requires -l or -p)\n"
        "  -p PORT   Expose Prometheus metrics on PORT\n"
        "  -t TOKEN  Require Bearer token for /metrics (or NV_MONITOR_TOKEN env)\n"
        "  -r MS     UI refresh interval in milliseconds (default: 1000)\n"
        "  -v        Show version\n"
        "  -h        Show this help\n"
        "\n"
        "Examples:\n"
        "  %s -l stats.csv                  TUI + logging every 1s\n"
        "  %s -l stats.csv -i 5000          TUI + logging every 5s\n"
        "  %s -n -l stats.csv -i 500        Headless, log every 500ms\n"
        "  %s -r 2000                       TUI refreshing every 2s\n"
        "\n"
        "Copyright (c) 2026 Paul Gresham Advisory LLC\n"
        "https://github.com/wentbackward/nv-monitor\n",
        prog, prog, prog, prog, prog);
}

/* ── Main draw ──────────────────────────────────────────────────────── */

static void draw_screen(void) {
    int rows, cols;
    getmaxyx(stdscr, rows, cols);

    erase();

    int y = 0;

    /* ── Header ─────────────────────────────────────────────────────── */
    char upbuf[64];
    fmt_uptime(upbuf, sizeof(upbuf));
    double l1, l5, l15;
    get_loadavg(&l1, &l5, &l15);
    int right_x = cols;
    {
        char info[128];
        int len = snprintf(info, sizeof(info), "up %s  load %.2f %.2f %.2f", upbuf, l1, l5, l15);
        if (len > 0 && len < cols) {
            right_x = cols - len - 1;
            mvprintw(y, right_x, "%s", info);
        }
    }

    int x = 0;
    const char *title = " nv-monitor";
    int title_len = (int)strlen(title);
    attron(A_BOLD | COLOR_PAIR(6));
    mvprintw(y, x, "%s", title);
    attroff(A_BOLD | COLOR_PAIR(6));
    x += title_len;

    if (host_name[0] && x < right_x) {
        char host_label[300];
        int remaining = right_x - x;
        int host_len = snprintf(host_label, sizeof(host_label), "  @%s", host_name);
        if (host_len > remaining) host_len = remaining;
        if (host_len > 0) {
            attron(A_BOLD | COLOR_PAIR(3));
            mvprintw(y, x, "%.*s", host_len, host_label);
            attroff(A_BOLD | COLOR_PAIR(3));
            x += host_len;
        }
    }

    if (x < right_x) {
        const char *cpu = cpu_model_name[0] ? cpu_model_name : "Unknown CPU";
        int remaining = right_x - x;
        if (remaining > 2) {
            attron(COLOR_PAIR(7));
            mvprintw(y, x, "  %.*s", remaining - 2, cpu);
            attroff(COLOR_PAIR(7));
        }
    }
    y += 1;

    attron(COLOR_PAIR(8));
    mvhline(y, 0, ACS_HLINE, cols);
    attroff(COLOR_PAIR(8));
    y += 1;

    /* ── CPU section ────────────────────────────────────────────────── */
    int cpu_temp = read_cpu_temp();
    int cpu_freq = read_cpu_freq_mhz();

    /* Auto-calculate column count: ~36 chars per column minimum */
    int ncols;
    if (cpu_columns > 0) {
        ncols = cpu_columns;
    } else {
        ncols = cols / 36;
        if (ncols < 1) ncols = 1;
        if (ncols > 4) ncols = 4;
    }
    /* Don't have more columns than cores */
    int total_core_rows = (num_cpus + ncols - 1) / ncols;
    if (total_core_rows < 1 && num_cpus > 0) total_core_rows = 1;

    /* Narrow terminal: put "Overall:" on its own line to avoid collision */
    int narrow_header = (cols < 90);

    /* CPU vertical budget: total rows minus fixed sections.
     * Per-GPU estimate: header(1) + util(1) + vram(1) + enc/dec(1) + blank(1)
     * + proc header(1) + ~2 procs(2) + other row(1) = ~10 rows.
     * Cap at 3 GPUs visible — beyond that, scrolling the GPU section is a
     * separate problem. */
    int gpu_rows = (gpu_count == 0 ? 0 :
                    (gpu_count > 3 ? 30 : (int)gpu_count * 10));
    int header_rows_extra = narrow_header ? 1 : 0;
    /* base: title(2) + CPU header(1) + blank(1) + mem(3) + net(2) + blank(1)
     * + separator(1) + blank(1) + footer(1) = 13 */
    int base_fixed = 13 + gpu_rows + header_rows_extra;
    int history_rows = 7;
    int fixed_rows = base_fixed + history_rows;
    int cpu_max_rows = rows - fixed_rows;
    /* If too cramped, drop history to give cores more room */
    int show_history = 1;
    if (cpu_max_rows < 5) {
        show_history = 0;
        fixed_rows = base_fixed;
        cpu_max_rows = rows - fixed_rows;
    }
    if (cpu_max_rows < 3) cpu_max_rows = 3;
    if (cpu_max_rows > total_core_rows) cpu_max_rows = total_core_rows;

    /* Clamp scroll */
    int max_scroll = total_core_rows - cpu_max_rows;
    if (max_scroll < 0) max_scroll = 0;
    if (cpu_scroll > max_scroll) cpu_scroll = max_scroll;
    if (cpu_scroll < 0) cpu_scroll = 0;

    int scrollable = (total_core_rows > cpu_max_rows);

    /* CPU header */
    attron(A_BOLD | COLOR_PAIR(3));
    mvprintw(y, 1, "CPU");
    attroff(A_BOLD | COLOR_PAIR(3));
    printw("  %d cores", num_cpus);
    if (scrollable) {
        int first_core = cpu_scroll * ncols;
        int last_core = (cpu_scroll + cpu_max_rows) * ncols - 1;
        if (last_core >= num_cpus) last_core = num_cpus - 1;
        const char *up   = (cpu_scroll > 0) ? "\xe2\x86\x91" : " ";        /* ↑ */
        const char *down = (cpu_scroll < max_scroll) ? "\xe2\x86\x93" : " "; /* ↓ */
        attron(COLOR_PAIR(8));
        printw(" [%d-%d] ", first_core, last_core);
        attroff(COLOR_PAIR(8));
        attron(A_BOLD | COLOR_PAIR(6));
        printw("%s%s", up, down);
        attroff(A_BOLD | COLOR_PAIR(6));
    }
    if (cpu_freq > 0) printw("  %d MHz", cpu_freq);
    if (cpu_temp > 0) printw("  %d C", cpu_temp);

    if (narrow_header) {
        /* Overall bar on its own line (full width) */
        y += 1;
        attron(A_BOLD);
        mvprintw(y, 1, "Overall:");
        attroff(A_BOLD);
        int bw = cols - 19;
        if (bw < 10) bw = 10;
        int color = cpu_pct[0] > 90 ? 1 : (cpu_pct[0] > 60 ? 3 : 2);
        draw_bar(y, 11, bw, cpu_pct[0], color);
        mvprintw(y, 11 + bw, " %4.1f%%", cpu_pct[0]);
    } else {
        attron(A_BOLD);
        mvprintw(y, cols / 2 + 1, "Overall: ");
        attroff(A_BOLD);
        int bw = cols / 2 - 17;
        if (bw < 10) bw = 10;
        int color = cpu_pct[0] > 90 ? 1 : (cpu_pct[0] > 60 ? 3 : 2);
        draw_bar(y, cols / 2 + 10, bw, cpu_pct[0], color);
        mvprintw(y, cols / 2 + 10 + bw, " %4.1f%%", cpu_pct[0]);
    }
    y += 1;

    /* Per-core bars - N columns, scrollable */
    int lbl_w = 9; /* "XX YYYY " — core number + type label */
    int col_w = cols / ncols;
    int bar_w = col_w - lbl_w - 8; /* label + suffix " xxx.x%" + margin */
    if (bar_w < 5) bar_w = 5;

    int visible = cpu_max_rows;
    for (int r = 0; r < visible; r++) {
        for (int c = 0; c < ncols; c++) {
            int core_idx = (cpu_scroll + r) * ncols + c;
            if (core_idx >= num_cpus) break;
            int x = c * col_w + 1;

            int color = cpu_pct[core_idx + 1] > 90 ? 1 :
                       (cpu_pct[core_idx + 1] > 60 ? 3 : 2);
            const char *lbl = cpu_part_label(core_idx);
            mvprintw(y, x, "%2d ", core_idx);
            attron(COLOR_PAIR(8));
            printw("%-4s ", lbl);
            attroff(COLOR_PAIR(8));
            draw_bar(y, x + lbl_w, bar_w, cpu_pct[core_idx + 1], color);
            mvprintw(y, x + lbl_w + bar_w, " %4.1f%%", cpu_pct[core_idx + 1]);
        }
        y++;
    }

    y += 1;

    /* ── Memory section ─────────────────────────────────────────────── */
    MemInfo mi;
    read_meminfo(&mi);
    double pct_app = mi.total_kb ? (double)mi.app_kb / mi.total_kb * 100.0 : 0;
    double pct_bufcache = mi.total_kb ? (double)mi.bufcache_kb / mi.total_kb * 100.0 : 0;

    attron(A_BOLD | COLOR_PAIR(4));
    mvprintw(y, 1, "MEM");
    attroff(A_BOLD | COLOR_PAIR(4));

    char tb[16], ab[16], bb[16];
    fmt_bytes(mi.total_kb * 1024ULL, tb, sizeof(tb));
    fmt_bytes(mi.app_kb * 1024ULL, ab, sizeof(ab));
    fmt_bytes(mi.bufcache_kb * 1024ULL, bb, sizeof(bb));
    printw("  ");
    attron(COLOR_PAIR(2));
    printw("%s used", ab);
    attroff(COLOR_PAIR(2));
    printw(" + ");
    attron(COLOR_PAIR(4));
    printw("%s buf/cache", bb);
    attroff(COLOR_PAIR(4));
    printw(" / %s", tb);
    y++;

    {
        int bw = cols - 13;
        if (bw < 10) bw = 10;
        draw_bar_segmented(y, 4, bw, pct_app, pct_bufcache, 2, 4);
        mvprintw(y, 4 + bw, " %.1f%%", pct_app + pct_bufcache);
    }
    y++;

    /* Swap */
    if (mi.swap_total_kb > 0) {
        double swap_pct = (double)mi.swap_used_kb / mi.swap_total_kb * 100.0;
        attron(A_BOLD | COLOR_PAIR(4));
        mvprintw(y, 1, "SWP");
        attroff(A_BOLD | COLOR_PAIR(4));
        char stb[16], sub[16];
        fmt_bytes(mi.swap_used_kb * 1024ULL, sub, sizeof(sub));
        fmt_bytes(mi.swap_total_kb * 1024ULL, stb, sizeof(stb));
        printw("  %s / %s", sub, stb);
        y++;
        {
            int bw = cols - 13;
            if (bw < 10) bw = 10;
            int color = swap_pct > 80 ? 1 : (swap_pct > 40 ? 3 : 5);
            draw_bar(y, 4, bw, swap_pct, color);
            mvprintw(y, 4 + bw, " %.1f%%", swap_pct);
        }
        y++;
    }

    if (net_totals.valid) {
        char rx[16], tx[16], scale[16];
        attron(A_BOLD | COLOR_PAIR(6));
        mvprintw(y, 1, "NET");
        attroff(A_BOLD | COLOR_PAIR(6));
        attron(COLOR_PAIR(2));
        printw("  ");
        printw("%s", fmt_rate(net_totals.rx_bytes_sec, rx, sizeof(rx)));
        attroff(COLOR_PAIR(2));
        printw(" down");
        attron(COLOR_PAIR(3));
        printw("  %s", fmt_rate(net_totals.tx_bytes_sec, tx, sizeof(tx)));
        attroff(COLOR_PAIR(3));
        printw(" up");
        y++;

        mvprintw(y, 1, "  I/O ");
        int bw = cols - 24;
        if (bw < 10) bw = 10;
        double rx_pct = net_scale_bytes_sec > 0 ? net_totals.rx_bytes_sec / net_scale_bytes_sec * 100.0 : 0;
        double tx_pct = net_scale_bytes_sec > 0 ? net_totals.tx_bytes_sec / net_scale_bytes_sec * 100.0 : 0;
        if (rx_pct < 0) rx_pct = 0;
        if (tx_pct < 0) tx_pct = 0;
        if (rx_pct + tx_pct > 100.0) {
            double scale_down = 100.0 / (rx_pct + tx_pct);
            rx_pct *= scale_down;
            tx_pct *= scale_down;
        }
        draw_bar_segmented(y, 7, bw, rx_pct, tx_pct, 2, 3);
        attron(COLOR_PAIR(8));
        mvprintw(y, 7 + bw + 1, "%s peak", fmt_rate(net_scale_bytes_sec, scale, sizeof(scale)));
        attroff(COLOR_PAIR(8));
        y++;
    }

    y += 1;
    attron(COLOR_PAIR(8));
    mvhline(y, 0, ACS_HLINE, cols);
    attroff(COLOR_PAIR(8));
    y += 1;

    /* ── GPU section ────────────────────────────────────────────────── */
    if (!nvml_ok && !use_tegra_gpu) {
        attron(COLOR_PAIR(1));
        mvprintw(y, 1, "GPU: NVML not available");
        attroff(COLOR_PAIR(1));
        y += 2;
    } else {
        double gpu_util_sum = 0;
        unsigned int gpu_util_n = 0;

        /* Determine view mode */
        int use_compact;
        if (gpu_view == 1) use_compact = 1;
        else if (gpu_view == 2) use_compact = 0;
        else {
            /* Auto: compact if detailed would overflow */
            int est_detail_rows = (int)gpu_count * 10;
            int available = rows - y - 8;
            use_compact = (est_detail_rows > available && gpu_count > 2);
        }

        /* ── Collect GPU data for all GPUs ────────────────────────── */
        typedef struct {
            char name[96];
            unsigned int util_gpu, temp, power_mw, clk_gfx;
            int has_power, has_mem, has_fan, has_enc, has_dec;
            unsigned int fan, enc, dec;
            unsigned long long mem_total, mem_used;
        } GpuSnapshot;

        /* Use stack array — bounded by gpu_count which was validated at startup */
        GpuSnapshot gpu_snaps[256]; /* stack, not heap — draw_screen is on main thread */
        if (gpu_count > 256) gpu_count = 256; /* safety */

        GpuProc all_procs_combined[MAX_GPU_PROCS * 2];
        int n_all_combined = 0;

        for (unsigned int d = 0; d < gpu_count; d++) {
            GpuSnapshot *gs = &gpu_snaps[d];
            memset(gs, 0, sizeof(*gs));
            snprintf(gs->name, sizeof(gs->name), "Unknown");

            nvmlDevice_t dev;
            if (pNvmlDeviceGetHandleByIndex &&
                pNvmlDeviceGetHandleByIndex(d, &dev) != NVML_SUCCESS) continue;

            pNvmlDeviceGetName(dev, gs->name, sizeof(gs->name));

            nvmlUtilization_t util = {0};
            if (!use_tegra_gpu && pNvmlDeviceGetUtilizationRates)
                pNvmlDeviceGetUtilizationRates(dev, &util);
            if (use_tegra_gpu) {
                int tutil = read_tegra_gpu_util();
                if (tutil >= 0) util.gpu = (unsigned int)tutil;
            }
            gs->util_gpu = util.gpu;
            gpu_util_sum += (double)util.gpu;
            gpu_util_n++;

            if (!use_tegra_gpu && pNvmlDeviceGetTemperature)
                pNvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &gs->temp);
            if (use_tegra_gpu && tegra_gpu_therm_zone >= 0) {
                int ttemp = read_tegra_gpu_temp();
                if (ttemp > 0) gs->temp = (unsigned int)ttemp;
            }

            gs->has_power = (pNvmlDeviceGetPowerUsage &&
                             pNvmlDeviceGetPowerUsage(dev, &gs->power_mw) == NVML_SUCCESS);
            if (pNvmlDeviceGetClockInfo)
                pNvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &gs->clk_gfx);
            gs->has_fan = (pNvmlDeviceGetFanSpeed &&
                           pNvmlDeviceGetFanSpeed(dev, &gs->fan) == NVML_SUCCESS);

            nvmlMemory_t mem = {0};
            gs->has_mem = (pNvmlDeviceGetMemoryInfo &&
                           pNvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS &&
                           mem.total > 0);
            if (gs->has_mem) { gs->mem_total = mem.total; gs->mem_used = mem.used; }

            unsigned int enc_period = 0, dec_period = 0;
            gs->has_enc = (pNvmlDeviceGetEncoderUtilization &&
                           pNvmlDeviceGetEncoderUtilization(dev, &gs->enc, &enc_period) == NVML_SUCCESS);
            gs->has_dec = (pNvmlDeviceGetDecoderUtilization &&
                           pNvmlDeviceGetDecoderUtilization(dev, &gs->dec, &dec_period) == NVML_SUCCESS);

            /* Collect processes for this GPU */
            nvmlProcessInfo_t comp_procs[MAX_GPU_PROCS];
            nvmlProcessInfo_t gfx_procs[MAX_GPU_PROCS];
            unsigned int n_comp = MAX_GPU_PROCS, n_gfx = MAX_GPU_PROCS;

            if (pNvmlDeviceGetComputeRunningProcesses) {
                int rc = pNvmlDeviceGetComputeRunningProcesses(dev, &n_comp, comp_procs);
                if (rc != NVML_SUCCESS) n_comp = 0;
            } else n_comp = 0;

            if (pNvmlDeviceGetGraphicsRunningProcesses) {
                int rc = pNvmlDeviceGetGraphicsRunningProcesses(dev, &n_gfx, gfx_procs);
                if (rc != NVML_SUCCESS) n_gfx = 0;
            } else n_gfx = 0;

            for (unsigned int i = 0; i < n_comp && n_all_combined < MAX_GPU_PROCS * 2; i++) {
                if (comp_procs[i].pid == 0 || comp_procs[i].pid > 4194304) continue;
                GpuProc *p = &all_procs_combined[n_all_combined++];
                p->pid = comp_procs[i].pid;
                p->gpu_id = d;
                unsigned long long pmem = comp_procs[i].usedGpuMemory;
                p->mem_bytes = (pmem == 0xFFFFFFFFFFFFFFFFULL) ? 0 : pmem;
                p->type = 'C';
                p->cpu_pct = calc_proc_cpu_pct(p->pid);
                get_proc_cmdline(p->pid, p->name, sizeof(p->name));
                get_proc_user(p->pid, p->user, sizeof(p->user));
            }
            for (unsigned int i = 0; i < n_gfx && n_all_combined < MAX_GPU_PROCS * 2; i++) {
                if (gfx_procs[i].pid == 0 || gfx_procs[i].pid > 4194304) continue;
                int dup = 0;
                for (int j = 0; j < n_all_combined; j++)
                    if (all_procs_combined[j].pid == gfx_procs[i].pid) { dup = 1; break; }
                if (dup) continue;
                GpuProc *p = &all_procs_combined[n_all_combined++];
                p->pid = gfx_procs[i].pid;
                p->gpu_id = d;
                unsigned long long pmem = gfx_procs[i].usedGpuMemory;
                p->mem_bytes = (pmem == 0xFFFFFFFFFFFFFFFFULL) ? 0 : pmem;
                p->type = 'G';
                p->cpu_pct = calc_proc_cpu_pct(p->pid);
                get_proc_cmdline(p->pid, p->name, sizeof(p->name));
                get_proc_user(p->pid, p->user, sizeof(p->user));
            }
        }

        /* Tegra fallback for process listing */
        if (n_all_combined == 0 && use_tegra_gpu)
            n_all_combined = scan_tegra_gpu_procs(all_procs_combined, MAX_GPU_PROCS * 2);

        update_proc_cpu_snapshots(all_procs_combined, n_all_combined);

        /* Sort processes: by GPU index first, then by sort_mode within each GPU */
        for (int i = 0; i < n_all_combined - 1; i++)
            for (int j = i + 1; j < n_all_combined; j++) {
                int sw = 0;
                if (all_procs_combined[j].gpu_id < all_procs_combined[i].gpu_id)
                    sw = 1;
                else if (all_procs_combined[j].gpu_id == all_procs_combined[i].gpu_id) {
                    if (sort_mode == 0)
                        sw = all_procs_combined[j].mem_bytes > all_procs_combined[i].mem_bytes;
                    else
                        sw = all_procs_combined[j].pid < all_procs_combined[i].pid;
                }
                if (sw) {
                    GpuProc tmp = all_procs_combined[i];
                    all_procs_combined[i] = all_procs_combined[j];
                    all_procs_combined[j] = tmp;
                }
            }

        /* ── Render: compact or detailed ──────────────────────────── */
        if (use_compact) {
            /* Compact: one row per GPU with mini-bars */
            int mini_bar_w = 10;
            attron(A_BOLD | COLOR_PAIR(6));
            mvprintw(y, 1, " GPU  NAME");
            attroff(A_BOLD | COLOR_PAIR(6));
            attron(COLOR_PAIR(8));
            {
                int hdr_x = cols > 80 ? 30 : 18;
                mvprintw(y, hdr_x, "UTIL");
                mvprintw(y, hdr_x + mini_bar_w + 7, "TEMP  POWER  CLOCK");
                if (cols > 100) {
                    int vram_x = hdr_x + mini_bar_w + 32;
                    mvprintw(y, vram_x, "VRAM");
                }
            }
            attroff(COLOR_PAIR(8));
            y++;

            for (unsigned int d = 0; d < gpu_count && y < rows - 4; d++) {
                GpuSnapshot *gs = &gpu_snaps[d];

                /* GPU index */
                attron(A_BOLD | COLOR_PAIR(6));
                mvprintw(y, 1, " %3u", d);
                attroff(A_BOLD | COLOR_PAIR(6));

                /* Name (truncated) */
                char short_name[32];
                /* Strip "NVIDIA " prefix for compactness */
                const char *nm = gs->name;
                if (strncmp(nm, "NVIDIA ", 7) == 0) nm += 7;
                int nm_max = cols > 80 ? 22 : 12;
                snprintf(short_name, sizeof(short_name), "%-.*s", nm_max, nm);
                printw("  %s", short_name);

                /* Util mini-bar */
                int bar_x = cols > 80 ? 30 : 18;
                int util_color = gs->util_gpu > 90 ? 1 : (gs->util_gpu > 60 ? 3 : 6);
                draw_bar(y, bar_x, mini_bar_w, (double)gs->util_gpu, util_color);
                mvprintw(y, bar_x + mini_bar_w + 1, "%3u%%", gs->util_gpu);

                /* Temp, power, clock */
                int info_x = bar_x + mini_bar_w + 7;
                mvprintw(y, info_x, "%3uC", gs->temp);
                if (gs->has_power)
                    mvprintw(y, info_x + 6, "%5.0fW", gs->power_mw / 1000.0);
                else
                    mvprintw(y, info_x + 6, "     ");
                if (gs->clk_gfx)
                    mvprintw(y, info_x + 13, "%4uMHz", gs->clk_gfx);

                /* VRAM mini-bar (if room and available) */
                if (cols > 100) {
                    int vram_x = info_x + 22;
                    if (gs->has_mem) {
                        double mem_pct = (double)gs->mem_used / gs->mem_total * 100.0;
                        int mc = mem_pct > 90 ? 1 : (mem_pct > 60 ? 3 : 5);
                        draw_bar(y, vram_x, mini_bar_w, mem_pct, mc);
                        char ub[16], tb2[16];
                        fmt_bytes(gs->mem_used, ub, sizeof(ub));
                        fmt_bytes(gs->mem_total, tb2, sizeof(tb2));
                        mvprintw(y, vram_x + mini_bar_w + 1, "%s/%s", ub, tb2);
                    } else {
                        attron(COLOR_PAIR(7));
                        mvprintw(y, vram_x, "unified");
                        attroff(COLOR_PAIR(7));
                    }
                }
                y++;
            }

            y++;

            /* Combined process list with GPU column */
            if (n_all_combined > 0 && y < rows - 3) {
                attron(A_BOLD | COLOR_PAIR(7));
                mvprintw(y, 1, " GPU  %-8s %-10s %-4s %9s %-10s %s",
                         "PID", "USER", "TYPE", "CPU%", "GPU-MEM", "COMMAND");
                attroff(A_BOLD | COLOR_PAIR(7));
                y++;

                for (int i = 0; i < n_all_combined && y < rows - 2; i++) {
                    GpuProc *p = &all_procs_combined[i];
                    char mb[16];
                    if (p->mem_bytes > 0) fmt_bytes(p->mem_bytes, mb, sizeof(mb));
                    else snprintf(mb, sizeof(mb), "N/A");

                    int name_max = cols - 58;
                    if (name_max < 0) name_max = 0;
                    char truncname[256];
                    snprintf(truncname, sizeof(truncname), "%-.*s", name_max, p->name);

                    int pc = (p->type == 'C') ? 5 : 7;
                    attron(COLOR_PAIR(6));
                    mvprintw(y, 1, " %3u", p->gpu_id);
                    attroff(COLOR_PAIR(6));
                    printw("  %-8u %-10s ", p->pid, p->user);
                    attron(COLOR_PAIR(pc));
                    printw("%-4c", p->type);
                    attroff(COLOR_PAIR(pc));
                    printw(" %8.1f%% %-10s %s", p->cpu_pct, mb, truncname);
                    y++;
                }

                /* Other processes summary */
                double gpu_proc_cpu = 0;
                for (int i = 0; i < n_all_combined; i++)
                    gpu_proc_cpu += all_procs_combined[i].cpu_pct;
                double total_cpu = cpu_pct[0] * num_cpus;
                double other_cpu = total_cpu - gpu_proc_cpu;
                if (other_cpu < 0) other_cpu = 0;
                attron(COLOR_PAIR(8));
                mvprintw(y, 1, " %3s  %-8s %-10s %-4s %8.1f%%",
                         "", "", "", "", other_cpu);
                printw(" %-10s %s", "", "(other processes)");
                attroff(COLOR_PAIR(8));
                y++;
            }

        } else {
            /* ── Detailed view (existing layout) ──────────────────── */
            for (unsigned int d = 0; d < gpu_count && y < rows - 4; d++) {
                GpuSnapshot *gs = &gpu_snaps[d];

                /* GPU header line */
                attron(A_BOLD | COLOR_PAIR(6));
                mvprintw(y, 1, "GPU %u", d);
                attroff(A_BOLD | COLOR_PAIR(6));
                printw("  %s  %u C", gs->name, gs->temp);
                if (gs->has_power) printw("  %.1fW", gs->power_mw / 1000.0);
                if (gs->clk_gfx) printw("  %u MHz", gs->clk_gfx);
                if (gs->has_fan) printw("  Fan %u%%", gs->fan);
                y++;

                /* GPU utilization bar */
                mvprintw(y, 1, "  GPU ");
                {
                    int bx = 7;
                    int bw = cols - bx - 7;
                    if (bw < 10) bw = 10;
                    int color = gs->util_gpu > 90 ? 1 : (gs->util_gpu > 60 ? 3 : 6);
                    draw_bar(y, bx, bw, (double)gs->util_gpu, color);
                    mvprintw(y, bx + bw + 1, "%3u%%", gs->util_gpu);
                }
                y++;

                /* Memory usage */
                if (gs->has_mem) {
                    mvprintw(y, 1, "  VRAM");
                    int bx = 7;
                    int bw = cols - bx - 18;
                    if (bw < 10) bw = 10;
                    double mem_pct = (double)gs->mem_used / gs->mem_total * 100.0;
                    int color = mem_pct > 90 ? 1 : (mem_pct > 60 ? 3 : 5);
                    draw_bar(y, bx, bw, mem_pct, color);
                    char ub2[16], tb2[16];
                    fmt_bytes(gs->mem_used, ub2, sizeof(ub2));
                    fmt_bytes(gs->mem_total, tb2, sizeof(tb2));
                    mvprintw(y, bx + bw + 1, "%s/%s", ub2, tb2);
                } else {
                    mvprintw(y, 1, "  VRAM");
                    attron(COLOR_PAIR(7));
                    printw("  unified memory (shared with CPU)");
                    attroff(COLOR_PAIR(7));
                }
                y++;

                /* Encoder/Decoder */
                if (gs->has_enc || gs->has_dec) {
                    mvprintw(y, 1, "  ");
                    if (gs->has_enc) printw("ENC %u%%  ", gs->enc);
                    if (gs->has_dec) printw("DEC %u%%", gs->dec);
                    y++;
                }

                y++;

                /* Per-GPU process list */
                int n_this_gpu = 0;
                for (int i = 0; i < n_all_combined; i++)
                    if (all_procs_combined[i].gpu_id == d) n_this_gpu++;

                if (n_this_gpu > 0) {
                    attron(A_BOLD | COLOR_PAIR(7));
                    mvprintw(y, 1, "  %-8s %-12s %-4s %9s %-12s %s",
                             "PID", "USER", "TYPE", "CPU%", "GPU-MEM", "COMMAND");
                    attroff(A_BOLD | COLOR_PAIR(7));
                    y++;

                    for (int i = 0; i < n_all_combined && y < rows - 2; i++) {
                        GpuProc *p = &all_procs_combined[i];
                        if (p->gpu_id != d) continue;
                        char mb[16];
                        if (p->mem_bytes > 0) fmt_bytes(p->mem_bytes, mb, sizeof(mb));
                        else snprintf(mb, sizeof(mb), "N/A");

                        int name_max = cols - 54;
                        if (name_max < 0) name_max = 0;
                        char truncname[256];
                        snprintf(truncname, sizeof(truncname), "%-.*s", name_max, p->name);

                        int pc = (p->type == 'C') ? 5 : 7;
                        mvprintw(y, 1, "  %-8u %-12s ", p->pid, p->user);
                        attron(COLOR_PAIR(pc));
                        printw("%-4c", p->type);
                        attroff(COLOR_PAIR(pc));
                        printw(" %8.1f%% %-12s %s", p->cpu_pct, mb, truncname);
                        y++;
                    }
                }

                /* Other processes (only on last GPU in detailed view) */
                if (d == gpu_count - 1) {
                    double gpu_proc_cpu = 0;
                    for (int i = 0; i < n_all_combined; i++)
                        gpu_proc_cpu += all_procs_combined[i].cpu_pct;
                    double total_cpu = cpu_pct[0] * num_cpus;
                    double other_cpu = total_cpu - gpu_proc_cpu;
                    if (other_cpu < 0) other_cpu = 0;
                    attron(COLOR_PAIR(8));
                    mvprintw(y, 1, "  %-8s %-12s %-4s %8.1f%%",
                             "", "", "", other_cpu);
                    printw(" %-12s %s", "", "(other processes)");
                    attroff(COLOR_PAIR(8));
                    y++;
                }
            }
        }

        last_gpu_util = gpu_util_n > 0 ? gpu_util_sum / gpu_util_n : 0;
    }

    /* ── History chart (full width) ───────────────────────────────── */
    record_history(cpu_pct[0], last_gpu_util);
    if (show_history && !skip_history_once) {
        int chart_h = 5;
        int chart_top = rows - 3 - chart_h; /* -3: footer + x-axis + gap */
        if (chart_top > y + 1 && cols > 20) {
            draw_history_chart(chart_top, cols, chart_h);
        }
    }
    skip_history_once = 0;

    /* ── Footer ─────────────────────────────────────────────────────── */
    attron(COLOR_PAIR(8));
    mvhline(rows - 1, 0, ACS_HLINE, cols);
    attroff(COLOR_PAIR(8));
    move(rows - 1, 1);
    attron(A_BOLD | COLOR_PAIR(7));
    printw(" q");
    attroff(A_BOLD | COLOR_PAIR(7));
    printw(":quit ");
    attron(A_BOLD | COLOR_PAIR(7));
    printw("s");
    attroff(A_BOLD | COLOR_PAIR(7));
    printw(":sort ");
    attron(A_BOLD | COLOR_PAIR(7));
    printw("c");
    attroff(A_BOLD | COLOR_PAIR(7));
    printw(":cols ");
    attron(A_BOLD | COLOR_PAIR(7));
    printw("g");
    attroff(A_BOLD | COLOR_PAIR(7));
    printw(":gpu ");
    attron(A_BOLD | COLOR_PAIR(7));
    printw("j/k");
    attroff(A_BOLD | COLOR_PAIR(7));
    printw(":scroll ");
    attron(A_BOLD | COLOR_PAIR(7));
    printw("+/-");
    attroff(A_BOLD | COLOR_PAIR(7));
    printw(":speed  ");
    attron(COLOR_PAIR(8));
    printw("%.1fs", delay_ms / 1000.0);
    attroff(COLOR_PAIR(8));

    /* Version, right-aligned */
    attron(COLOR_PAIR(8));
    mvprintw(rows - 1, cols - (int)strlen(VERSION) - 2, "%s ", VERSION);
    attroff(COLOR_PAIR(8));

    refresh();
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    setlocale(LC_ALL, "");
    setlocale(LC_NUMERIC, "C"); /* Force decimal point for Prometheus exposition format */
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    const char *log_path = NULL;
    int opt;
    while ((opt = getopt(argc, argv, "c:g:l:i:np:t:r:vh")) != -1) {
        switch (opt) {
        case 'c': cpu_columns = atoi(optarg); if (cpu_columns < 0 || cpu_columns > 4) cpu_columns = 0; break;
        case 'g':
            if (strcmp(optarg, "compact") == 0) gpu_view = 1;
            else if (strcmp(optarg, "detailed") == 0) gpu_view = 2;
            else gpu_view = 0;
            break;
        case 'l': log_path = optarg; break;
        case 'i': log_interval_ms = atoi(optarg); break;
        case 'n': no_ui = 1; break;
        case 'p': prom_port = atoi(optarg); break;
        case 't': prom_token = optarg; break;
        case 'r': delay_ms = atoi(optarg); break;
        case 'v': printf("nv-monitor %s\n", VERSION); return 0;
        case 'h': print_usage(argv[0]); return 0;
        default:  print_usage(argv[0]); return 1;
        }
    }

    /* Token: CLI flag takes precedence, then env var */
    if (!prom_token)
        prom_token = getenv("NV_MONITOR_TOKEN");

    if (no_ui && !log_path && !prom_port) {
        fprintf(stderr, "Error: -n (no UI) requires -l <file> or -p <port>\n");
        return 1;
    }
    if (log_interval_ms < 100) log_interval_ms = 100;
    if (delay_ms < 250) delay_ms = 250;

    /* Open log file */
    if (log_path) {
        log_fp = fopen(log_path, "w");
        if (!log_fp) {
            perror(log_path);
            return 1;
        }
    }

    /* Load NVML */
    nvml_ok = (load_nvml() == 0);
    if (nvml_ok && pNvmlDeviceGetCount)
        pNvmlDeviceGetCount(&gpu_count);

    /* Read CPU info */
    read_host_name();
    read_cpu_model_name();
    read_cpu_part_ids();

    /* Detect Tegra GPU sysfs (Jetson fallback) */
    detect_tegra_gpu();
    detect_tegra_gpu_dev();
    /* On Tegra/Jetson, NVML returns SUCCESS but zeros for util/temp — prefer sysfs */
    if (tegra_gpu_available)
        use_tegra_gpu = 1;

    /* Detect CPU count and allocate arrays */
    max_cpus = (int)sysconf(_SC_NPROCESSORS_CONF);
    if (max_cpus < 1) max_cpus = 1;
    max_cpus += 16; /* headroom for hotplug */
    prev_ticks = calloc(max_cpus + 1, sizeof(CpuTick));
    cur_ticks  = calloc(max_cpus + 1, sizeof(CpuTick));
    cpu_pct    = calloc(max_cpus + 1, sizeof(double));
    cpu_part   = calloc(max_cpus, sizeof(unsigned int));
    if (!prev_ticks || !cur_ticks || !cpu_pct || !cpu_part) {
        fprintf(stderr, "Failed to allocate CPU arrays for %d cores\n", max_cpus);
        return 1;
    }

    /* Initial CPU tick read */
    read_cpu_ticks(prev_ticks, &num_cpus);
    usleep(100000); /* brief pause for first delta */
    compute_cpu_usage();

    /* Detect RDMA/InfiniBand ports */
    read_net_totals();
    read_rdma_ports();

    /* Write CSV header after first CPU sample (so we know num_cpus) */
    if (log_fp)
        log_csv_header(log_fp);

    /* Start Prometheus exporter if requested */
    if (prom_port && prom_start() != 0)
        return 1;

    if (no_ui) {
        /* ── Headless mode ──────────────────────────────────────────── */
        int headless_interval = log_fp ? log_interval_ms : delay_ms;
        if (log_fp)
            fprintf(stderr, "Logging to %s every %dms (Ctrl+C to stop)\n",
                    log_path, headless_interval);
        else
            fprintf(stderr, "Running headless (Ctrl+C to stop)\n");
        while (!g_quit) {
            compute_cpu_usage();
            read_net_totals();
            read_rdma_ports();
            if (log_fp) log_csv_row(log_fp);
            usleep(headless_interval * 1000);
        }
        fprintf(stderr, "\nStopped.\n");
    } else {
        /* ── TUI mode ───────────────────────────────────────────────── */
        initscr();
        cbreak();
        noecho();
        curs_set(0);
        nodelay(stdscr, TRUE);
        keypad(stdscr, TRUE);

        if (has_colors()) {
            start_color();
            use_default_colors();
            init_pair(1, COLOR_RED,     -1); /* high/critical */
            init_pair(2, COLOR_GREEN,   -1); /* normal/good */
            init_pair(3, COLOR_YELLOW,  -1); /* medium */
            init_pair(4, COLOR_BLUE,    -1); /* buf/cache */
            init_pair(5, COLOR_MAGENTA, -1); /* compute */
            init_pair(6, COLOR_CYAN,    -1); /* headers/gpu */
            init_pair(7, COLOR_WHITE,   -1); /* bold text */
            init_pair(8, 244,           -1); /* dim/gray (256-color) */
        }

        int log_elapsed = 0;

        while (!g_quit) {
            compute_cpu_usage();
            read_net_totals();
            read_rdma_ports();
            /* Force full repaint to recover from terminal corruption */
            clearok(stdscr, TRUE);
            draw_screen();

            /* Log at log_interval_ms if logging enabled */
            if (log_fp) {
                log_elapsed += delay_ms;
                if (log_elapsed >= log_interval_ms) {
                    log_csv_row(log_fp);
                    log_elapsed = 0;
                }
            }

            /* Input handling - poll within the refresh interval */
            int elapsed = 0;
            while (elapsed < delay_ms && !g_quit) {
                int ch = getch();
                if (ch == 'q' || ch == 'Q' || ch == 27) {
                    g_quit = 1;
                    break;
                } else if (ch == 's' || ch == 'S') {
                    sort_mode = (sort_mode + 1) % 2;
                    break;
                } else if (ch == '+' || ch == '=') {
                    if (delay_ms > 250) delay_ms -= 250;
                } else if (ch == '-' || ch == '_') {
                    if (delay_ms < 5000) delay_ms += 250;
                } else if (ch == 'c' || ch == 'C') {
                    cpu_columns = (cpu_columns + 1) % 5; /* 0=auto,1,2,3,4 */
                    cpu_scroll = 0;
                    skip_history_once = 1;
                    break;
                } else if (ch == 'g' || ch == 'G') {
                    gpu_view = (gpu_view + 1) % 3; /* 0=auto,1=compact,2=detailed */
                    skip_history_once = 1;
                    break;
                } else if (ch == 'j' || ch == KEY_DOWN) {
                    cpu_scroll++;
                    skip_history_once = 1;
                    break;
                } else if (ch == 'k' || ch == KEY_UP) {
                    if (cpu_scroll > 0) cpu_scroll--;
                    skip_history_once = 1;
                    break;
                } else if (ch == KEY_RESIZE) {
                    cpu_scroll = 0;
                    skip_history_once = 1;
                    break; /* redraw immediately */
                }
                usleep(50000);
                elapsed += 50;
            }
        }

        endwin();
    }

    prom_stop();
    if (log_fp) fclose(log_fp);
    if (nvml_ok && pNvmlShutdown) pNvmlShutdown();
    if (nvml_handle) dlclose(nvml_handle);

    /* Free startup allocations */
    free(prev_ticks); prev_ticks = NULL;
    free(cur_ticks);  cur_ticks = NULL;
    free(cpu_pct);    cpu_pct = NULL;
    free(cpu_part);   cpu_part = NULL;

    return 0;
}

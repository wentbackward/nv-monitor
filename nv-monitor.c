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
static char  cpu_model_name[128] = "";  /* from /proc/cpuinfo */

/* ── Constants ──────────────────────────────────────────────────────── */

#define MAX_CPUS      128
#define MAX_GPU_PROCS 64
#define REFRESH_MS    1000
#define BAR_CHAR_FULL  ACS_BLOCK
#define COLOR_GRAY     8
#define HISTORY_LEN   20

/* ── CPU state ──────────────────────────────────────────────────────── */

typedef struct {
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
} CpuTick;

static int       num_cpus = 0;
static CpuTick   prev_ticks[MAX_CPUS + 1]; /* index 0 = aggregate */
static double     cpu_pct[MAX_CPUS + 1];
static unsigned int cpu_part[MAX_CPUS];     /* ARM CPU part IDs */

/* ── GPU process info ───────────────────────────────────────────────── */

typedef struct {
    unsigned int  pid;
    unsigned long long mem_bytes;
    char          name[256];
    char          user[64];
    char          type;    /* C=compute, G=graphics */
    double        cpu_pct; /* per-process CPU% */
} GpuProc;

/* ── Per-process CPU tracking ───────────────────────────────────────── */

#define MAX_TRACKED_PIDS 128

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
static double last_gpu_util = 0; /* captured during draw for history */

/* Command-line options */
static FILE *log_fp = NULL;
static int   log_interval_ms = 1000;
static int   no_ui = 0;
static int   prom_port = 0;  /* Prometheus metrics port (0 = disabled) */

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
        } else if (cur_cpu >= 0 && cur_cpu < MAX_CPUS) {
            unsigned int part;
            if (sscanf(line, "CPU part : %x", &part) == 1)
                cpu_part[cur_cpu] = part;
        }
    }
    fclose(f);
}

static const char *cpu_part_label(int cpu_idx) {
    switch (cpu_part[cpu_idx]) {
    case 0xd85: return "X925";
    case 0xd87: return "X725";
    case 0xd44: return "X4";
    case 0xd43: return "A720";
    case 0xd46: return "A725";
    case 0xd41: return "A78";
    case 0xd40: return "V2";
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
            if (cpunum + 1 < MAX_CPUS)
                ticks[cpunum + 1] = t;
            idx = cpunum + 1;
        }
    }
    *n_cpus = idx;
    fclose(f);
}

static void compute_cpu_usage(void) {
    CpuTick cur[MAX_CPUS + 1];
    int n = 0;
    read_cpu_ticks(cur, &n);
    num_cpus = n;

    for (int i = 0; i <= n; i++) {
        unsigned long long prev_idle  = prev_ticks[i].idle + prev_ticks[i].iowait;
        unsigned long long cur_idle   = cur[i].idle + cur[i].iowait;
        unsigned long long prev_total = prev_ticks[i].user + prev_ticks[i].nice +
                                        prev_ticks[i].system + prev_ticks[i].idle +
                                        prev_ticks[i].iowait + prev_ticks[i].irq +
                                        prev_ticks[i].softirq + prev_ticks[i].steal;
        unsigned long long cur_total  = cur[i].user + cur[i].nice +
                                        cur[i].system + cur[i].idle +
                                        cur[i].iowait + cur[i].irq +
                                        cur[i].softirq + cur[i].steal;
        unsigned long long totald = cur_total - prev_total;
        unsigned long long idled  = cur_idle - prev_idle;
        if (totald == 0)
            cpu_pct[i] = 0.0;
        else
            cpu_pct[i] = (double)(totald - idled) / (double)totald * 100.0;
    }

    memcpy(prev_ticks, cur, sizeof(cur));
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

    /* Each sample gets 2 sub-columns (CPU + GPU), scaled to fill width */
    int col_w = avail_w / n; /* chars per sample */
    if (col_w < 2) col_w = 2;
    int cpu_w = col_w / 2;
    int gpu_w = col_w - cpu_w;

    /* Recalculate how many samples fit */
    int visible = avail_w / col_w;
    if (visible > n) visible = n;

    /* Center the chart within available space */
    int chart_total = visible * col_w;
    int x_start = left_x + (avail_w - chart_total) / 2;

    for (int s = 0; s < visible; s++) {
        int idx = (history_pos - visible + s + HISTORY_LEN) % HISTORY_LEN;
        double cpu_val = cpu_history[idx];
        double gpu_val = gpu_history[idx];

        int cpu_blocks = (int)(cpu_val / 100.0 * chart_h * 8 + 0.5);
        int gpu_blocks = (int)(gpu_val / 100.0 * chart_h * 8 + 0.5);

        int x = x_start + s * col_w;

        int cpu_color = cpu_val > 90 ? 1 : (cpu_val > 60 ? 3 : 2);
        int gpu_color = gpu_val > 90 ? 1 : (gpu_val > 60 ? 3 : 6);

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
            attron(COLOR_PAIR(cpu_color));
            for (int c = 0; c < cpu_w; c++)
                printw("%s", block_chars[cpu_fill]);
            attroff(COLOR_PAIR(cpu_color));
            attron(COLOR_PAIR(gpu_color));
            for (int c = 0; c < gpu_w; c++)
                printw("%s", block_chars[gpu_fill]);
            attroff(COLOR_PAIR(gpu_color));
        }
    }

    /* Y-axis labels */
    attron(COLOR_PAIR(8));
    mvprintw(top_y, margin, "100%%");
    mvprintw(top_y + chart_h - 1, margin, "  0%%");
    attroff(COLOR_PAIR(8));
}

static void record_history(double cpu, double gpu) {
    cpu_history[history_pos] = cpu;
    gpu_history[history_pos] = gpu;
    history_pos = (history_pos + 1) % HISTORY_LEN;
    if (history_count < HISTORY_LEN) history_count++;
}

/* ── CSV logging ────────────────────────────────────────────────────── */

static void log_csv_header(FILE *f) {
    fprintf(f, "timestamp,cpu_avg_pct");
    for (int i = 1; i <= num_cpus; i++)
        fprintf(f, ",cpu%d_pct", i - 1);
    fprintf(f, ",cpu_temp_c,cpu_freq_mhz");
    fprintf(f, ",mem_used_kb,mem_total_kb,mem_bufcache_kb");
    fprintf(f, ",swap_used_kb,swap_total_kb");
    fprintf(f, ",gpu_util_pct,gpu_temp_c,gpu_power_mw,gpu_clock_mhz");
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

    /* GPU */
    if (nvml_ok) {
        nvmlDevice_t dev;
        if (pNvmlDeviceGetHandleByIndex(0, &dev) == NVML_SUCCESS) {
            nvmlUtilization_t util = {0};
            pNvmlDeviceGetUtilizationRates(dev, &util);

            unsigned int temp = 0;
            pNvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp);

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
    } else {
        fprintf(f, ",,,,");
    }

    fprintf(f, "\n");
    fflush(f);
}

/* ── Prometheus metrics exporter ────────────────────────────────────── */

static int   prom_sock = -1;
static pthread_t prom_thread;

#define PROM_BUF_SIZE 8192
#define PROM_MAX_GPUS 8

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

    /* GPU — collect data first, then format grouped by metric family */
    PromGpu gpus[PROM_MAX_GPUS];
    int n_gpus = 0;

    if (nvml_ok) {
        unsigned int dev_count = 0;
        pNvmlDeviceGetCount(&dev_count);

        for (unsigned int d = 0; d < dev_count && (int)d < PROM_MAX_GPUS; d++) {
            PromGpu *g = &gpus[n_gpus];
            memset(g, 0, sizeof(*g));
            nvmlDevice_t dev;
            if (pNvmlDeviceGetHandleByIndex(d, &dev) != NVML_SUCCESS) continue;
            g->valid = 1;
            pNvmlDeviceGetName(dev, g->name, sizeof(g->name));

            nvmlUtilization_t util = {0};
            pNvmlDeviceGetUtilizationRates(dev, &util);
            g->util_gpu = util.gpu;

            pNvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &g->temp);

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

    if (strstr(req, "GET /metrics")) {
        char body[PROM_BUF_SIZE];
        int bodylen = format_metrics(body, sizeof(body));

        char hdr[128];
        int hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
            "Content-Length: %d\r\n"
            "Connection: close\r\n\r\n", bodylen);

        send(fd, hdr, hlen, MSG_NOSIGNAL);
        send(fd, body, bodylen, MSG_NOSIGNAL);
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
    while (!g_quit) {
        struct pollfd pfd = { .fd = prom_sock, .events = POLLIN };
        if (poll(&pfd, 1, 1000) <= 0) continue;

        int fd = accept(prom_sock, NULL, NULL);
        if (fd < 0) continue;
        prom_handle(fd);
        close(fd);
    }
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
    pthread_attr_setstacksize(&attr, 131072); /* 128 KB — minimal for aarch64 */

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
        "  -l FILE   Log statistics to CSV file\n"
        "  -i MS     Log interval in milliseconds (default: 1000)\n"
        "  -n        No UI (headless mode, requires -l or -p)\n"
        "  -p PORT   Expose Prometheus metrics on PORT\n"
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
    attron(A_BOLD | COLOR_PAIR(6));
    mvprintw(y, 0, " nv-monitor");
    attroff(A_BOLD | COLOR_PAIR(6));
    attron(COLOR_PAIR(7));
    printw("  %s", cpu_model_name[0] ? cpu_model_name : "Unknown CPU");
    attroff(COLOR_PAIR(7));

    char upbuf[64];
    fmt_uptime(upbuf, sizeof(upbuf));
    double l1, l5, l15;
    get_loadavg(&l1, &l5, &l15);
    {
        char info[128];
        int len = snprintf(info, sizeof(info), "up %s  load %.2f %.2f %.2f", upbuf, l1, l5, l15);
        mvprintw(y, cols - len - 1, "%s", info);
    }
    y += 1;

    attron(COLOR_PAIR(8));
    mvhline(y, 0, ACS_HLINE, cols);
    attroff(COLOR_PAIR(8));
    y += 1;

    /* ── CPU section ────────────────────────────────────────────────── */
    int cpu_temp = read_cpu_temp();
    int cpu_freq = read_cpu_freq_mhz();

    attron(A_BOLD | COLOR_PAIR(3));
    mvprintw(y, 1, "CPU");
    attroff(A_BOLD | COLOR_PAIR(3));
    printw("  %d cores  %d MHz  %d C", num_cpus, cpu_freq, cpu_temp);

    attron(A_BOLD);
    mvprintw(y, cols / 2 + 1, "Overall: ");
    attroff(A_BOLD);
    {
        int bw = cols / 2 - 17; /* 10 label + 7 suffix " xxx.x%" */
        if (bw < 10) bw = 10;
        int color = cpu_pct[0] > 90 ? 1 : (cpu_pct[0] > 60 ? 3 : 2);
        draw_bar(y, cols / 2 + 10, bw, cpu_pct[0], color);
        mvprintw(y, cols / 2 + 10 + bw, " %4.1f%%", cpu_pct[0]);
    }
    y += 1;

    /* Per-core bars - two columns */
    int half = (num_cpus + 1) / 2;
    int lbl_w = 9; /* "XX YYYY " — core number + type label */
    int bar_w = cols / 2 - lbl_w - 7; /* label + suffix " xxx.x%" */
    if (bar_w < 5) bar_w = 5;

    for (int i = 0; i < half; i++) {
        int cpu_l = i;
        int cpu_r = i + half;

        /* Left column */
        int color = cpu_pct[cpu_l + 1] > 90 ? 1 : (cpu_pct[cpu_l + 1] > 60 ? 3 : 2);
        const char *lbl_l = cpu_part_label(cpu_l);
        mvprintw(y, 1, "%2d ", cpu_l);
        attron(COLOR_PAIR(8));
        printw("%-4s ", lbl_l);
        attroff(COLOR_PAIR(8));
        draw_bar(y, 1 + lbl_w, bar_w, cpu_pct[cpu_l + 1], color);
        mvprintw(y, 1 + lbl_w + bar_w, " %4.1f%%", cpu_pct[cpu_l + 1]);

        /* Right column */
        if (cpu_r < num_cpus) {
            int rx = cols / 2 + 1;
            color = cpu_pct[cpu_r + 1] > 90 ? 1 : (cpu_pct[cpu_r + 1] > 60 ? 3 : 2);
            const char *lbl_r = cpu_part_label(cpu_r);
            mvprintw(y, rx, "%2d ", cpu_r);
            attron(COLOR_PAIR(8));
            printw("%-4s ", lbl_r);
            attroff(COLOR_PAIR(8));
            draw_bar(y, rx + lbl_w - 1, bar_w, cpu_pct[cpu_r + 1], color);
            mvprintw(y, rx + lbl_w - 1 + bar_w, " %4.1f%%", cpu_pct[cpu_r + 1]);
        }
        y++;
        if (y >= rows - 2) break;
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

    y += 1;
    attron(COLOR_PAIR(8));
    mvhline(y, 0, ACS_HLINE, cols);
    attroff(COLOR_PAIR(8));
    y += 1;

    /* ── GPU section ────────────────────────────────────────────────── */
    if (!nvml_ok) {
        attron(COLOR_PAIR(1));
        mvprintw(y, 1, "GPU: NVML not available");
        attroff(COLOR_PAIR(1));
        y += 2;
    } else {
        unsigned int dev_count = 0;
        pNvmlDeviceGetCount(&dev_count);

        for (unsigned int d = 0; d < dev_count && y < rows - 4; d++) {
            nvmlDevice_t dev;
            if (pNvmlDeviceGetHandleByIndex(d, &dev) != NVML_SUCCESS) continue;

            char name[96] = "Unknown";
            pNvmlDeviceGetName(dev, name, sizeof(name));

            nvmlUtilization_t util = {0};
            pNvmlDeviceGetUtilizationRates(dev, &util);
            last_gpu_util = (double)util.gpu;

            unsigned int temp = 0;
            pNvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp);

            unsigned int power_mw = 0;
            int has_power = (pNvmlDeviceGetPowerUsage && pNvmlDeviceGetPowerUsage(dev, &power_mw) == NVML_SUCCESS);

            unsigned int clk_gfx = 0, clk_mem = 0;
            if (pNvmlDeviceGetClockInfo) {
                pNvmlDeviceGetClockInfo(dev, NVML_CLOCK_GRAPHICS, &clk_gfx);
                pNvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &clk_mem);
            }

            unsigned int fan = 0;
            int has_fan = (pNvmlDeviceGetFanSpeed && pNvmlDeviceGetFanSpeed(dev, &fan) == NVML_SUCCESS);

            /* GPU header line */
            attron(A_BOLD | COLOR_PAIR(6));
            mvprintw(y, 1, "GPU %u", d);
            attroff(A_BOLD | COLOR_PAIR(6));
            printw("  %s  %u C", name, temp);
            if (has_power) printw("  %.1fW", power_mw / 1000.0);
            if (clk_gfx) printw("  %u MHz", clk_gfx);
            if (has_fan) printw("  Fan %u%%", fan);
            y++;

            /* GPU utilization bar */
            mvprintw(y, 1, "  GPU ");
            {
                int bx = 7;
                int bw = cols - bx - 7; /* leave room for " 100%" */
                if (bw < 10) bw = 10;
                int color = util.gpu > 90 ? 1 : (util.gpu > 60 ? 3 : 6);
                draw_bar(y, bx, bw, (double)util.gpu, color);
                mvprintw(y, bx + bw + 1, "%3u%%", util.gpu);
            }
            y++;

            /* Memory usage */
            nvmlMemory_t mem = {0};
            int has_mem = (pNvmlDeviceGetMemoryInfo &&
                           pNvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS &&
                           mem.total > 0);

            if (has_mem) {
                mvprintw(y, 1, "  VRAM");
                int bx = 7;
                int bw = cols - bx - 18; /* room for " 12.3G/34.5G" */
                if (bw < 10) bw = 10;
                double mem_pct = (double)mem.used / mem.total * 100.0;
                int color = mem_pct > 90 ? 1 : (mem_pct > 60 ? 3 : 5);
                draw_bar(y, bx, bw, mem_pct, color);
                char ub2[16], tb2[16];
                fmt_bytes(mem.used, ub2, sizeof(ub2));
                fmt_bytes(mem.total, tb2, sizeof(tb2));
                mvprintw(y, bx + bw + 1, "%s/%s", ub2, tb2);
            } else {
                mvprintw(y, 1, "  VRAM");
                attron(COLOR_PAIR(7));
                printw("  unified memory (shared with CPU)");
                attroff(COLOR_PAIR(7));
            }
            y++;

            /* Encoder/Decoder utilization if available */
            unsigned int enc_util = 0, dec_util = 0, enc_period = 0, dec_period = 0;
            int has_enc = (pNvmlDeviceGetEncoderUtilization &&
                          pNvmlDeviceGetEncoderUtilization(dev, &enc_util, &enc_period) == NVML_SUCCESS);
            int has_dec = (pNvmlDeviceGetDecoderUtilization &&
                          pNvmlDeviceGetDecoderUtilization(dev, &dec_util, &dec_period) == NVML_SUCCESS);
            if (has_enc || has_dec) {
                mvprintw(y, 1, "  ");
                if (has_enc) printw("ENC %u%%  ", enc_util);
                if (has_dec) printw("DEC %u%%", dec_util);
                y++;
            }

            y++;

            /* GPU processes */
            nvmlProcessInfo_t comp_procs[MAX_GPU_PROCS];
            nvmlProcessInfo_t gfx_procs[MAX_GPU_PROCS];
            unsigned int n_comp = MAX_GPU_PROCS, n_gfx = MAX_GPU_PROCS;
            GpuProc all_procs[MAX_GPU_PROCS * 2];
            int n_all = 0;

            if (pNvmlDeviceGetComputeRunningProcesses)
                pNvmlDeviceGetComputeRunningProcesses(dev, &n_comp, comp_procs);
            else
                n_comp = 0;

            if (pNvmlDeviceGetGraphicsRunningProcesses)
                pNvmlDeviceGetGraphicsRunningProcesses(dev, &n_gfx, gfx_procs);
            else
                n_gfx = 0;

            for (unsigned int i = 0; i < n_comp && n_all < MAX_GPU_PROCS * 2; i++) {
                GpuProc *p = &all_procs[n_all++];
                p->pid = comp_procs[i].pid;
                p->mem_bytes = comp_procs[i].usedGpuMemory;
                p->type = 'C';
                p->cpu_pct = calc_proc_cpu_pct(p->pid);
                get_proc_cmdline(p->pid, p->name, sizeof(p->name));
                get_proc_user(p->pid, p->user, sizeof(p->user));
            }
            for (unsigned int i = 0; i < n_gfx && n_all < MAX_GPU_PROCS * 2; i++) {
                /* Skip duplicates */
                int dup = 0;
                for (int j = 0; j < n_all; j++)
                    if (all_procs[j].pid == gfx_procs[i].pid) { dup = 1; break; }
                if (dup) continue;
                GpuProc *p = &all_procs[n_all++];
                p->pid = gfx_procs[i].pid;
                p->mem_bytes = gfx_procs[i].usedGpuMemory;
                p->type = 'G';
                p->cpu_pct = calc_proc_cpu_pct(p->pid);
                get_proc_cmdline(p->pid, p->name, sizeof(p->name));
                get_proc_user(p->pid, p->user, sizeof(p->user));
            }

            /* Save snapshots for next frame's delta calculation */
            update_proc_cpu_snapshots(all_procs, n_all);

            /* Sort by memory descending */
            for (int i = 0; i < n_all - 1; i++)
                for (int j = i + 1; j < n_all; j++) {
                    int swap = 0;
                    if (sort_mode == 0)
                        swap = all_procs[j].mem_bytes > all_procs[i].mem_bytes;
                    else
                        swap = all_procs[j].pid < all_procs[i].pid;
                    if (swap) {
                        GpuProc tmp = all_procs[i];
                        all_procs[i] = all_procs[j];
                        all_procs[j] = tmp;
                    }
                }

            if (n_all > 0) {
                attron(A_BOLD | COLOR_PAIR(7));
                mvprintw(y, 1, "  %-8s %-12s %-4s %6s %-12s %s",
                         "PID", "USER", "TYPE", "CPU%", "GPU MEM", "COMMAND");
                attroff(A_BOLD | COLOR_PAIR(7));
                y++;

                for (int i = 0; i < n_all && y < rows - 2; i++) {
                    GpuProc *p = &all_procs[i];
                    char mb[16];
                    fmt_bytes(p->mem_bytes, mb, sizeof(mb));

                    int name_max = cols - 53;
                    if (name_max < 10) name_max = 10;
                    char truncname[256];
                    snprintf(truncname, sizeof(truncname), "%-.*s", name_max, p->name);

                    int pc = (p->type == 'C') ? 5 : 7;
                    mvprintw(y, 1, "  %-8u %-12s ", p->pid, p->user);
                    attron(COLOR_PAIR(pc));
                    printw("%-4c", p->type);
                    attroff(COLOR_PAIR(pc));
                    printw(" %5.1f%% %-12s %s", p->cpu_pct, mb, truncname);
                    y++;
                }

                /* Non-GPU processes summary */
                double gpu_proc_cpu = 0;
                for (int i = 0; i < n_all; i++)
                    gpu_proc_cpu += all_procs[i].cpu_pct;
                /* Scale overall CPU to per-core basis to match process CPU% */
                double total_cpu = cpu_pct[0] * num_cpus;
                double other_cpu = total_cpu - gpu_proc_cpu;
                if (other_cpu < 0) other_cpu = 0;
                attron(COLOR_PAIR(8));
                mvprintw(y, 1, "  %-8s %-12s %-4s %5.1f%%",
                         "", "", "", other_cpu);
                printw(" %-12s %s", "", "(other processes)");
                attroff(COLOR_PAIR(8));
                y++;
            }
        }
    }

    /* ── History chart (full width) ───────────────────────────────── */
    record_history(cpu_pct[0], last_gpu_util);
    {
        int chart_h = 5;
        int chart_top = rows - 2 - chart_h;
        if (chart_top > y + 1 && cols > 20) {
            draw_history_chart(chart_top, cols, chart_h);
        }
    }

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
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    const char *log_path = NULL;
    int opt;
    while ((opt = getopt(argc, argv, "l:i:np:r:vh")) != -1) {
        switch (opt) {
        case 'l': log_path = optarg; break;
        case 'i': log_interval_ms = atoi(optarg); break;
        case 'n': no_ui = 1; break;
        case 'p': prom_port = atoi(optarg); break;
        case 'r': delay_ms = atoi(optarg); break;
        case 'v': printf("nv-monitor %s\n", VERSION); return 0;
        case 'h': print_usage(argv[0]); return 0;
        default:  print_usage(argv[0]); return 1;
        }
    }

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

    /* Read CPU info */
    read_cpu_model_name();
    read_cpu_part_ids();

    /* Initial CPU tick read */
    read_cpu_ticks(prev_ticks, &num_cpus);
    usleep(100000); /* brief pause for first delta */
    compute_cpu_usage();

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
                } else if (ch == KEY_RESIZE) {
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

    return 0;
}

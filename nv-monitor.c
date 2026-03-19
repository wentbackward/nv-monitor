/*
 * nv-monitor - System monitor for NVIDIA DGX Spark (Grace + GB10)
 *
 * Displays CPU per-core usage, memory, CPU thermals, GPU utilization,
 * GPU temperature/power/clock, and GPU processes in a single TUI.
 *
 * Build: gcc -O2 -o nv-monitor nv-monitor.c -lncurses -ldl -lpthread
 */

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

/* ── Constants ──────────────────────────────────────────────────────── */

#define MAX_CPUS      128
#define MAX_GPU_PROCS 64
#define REFRESH_MS    1000
#define BAR_CHAR_FULL  ACS_BLOCK
#define COLOR_GRAY     8

/* ── CPU state ──────────────────────────────────────────────────────── */

typedef struct {
    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
} CpuTick;

static int       num_cpus = 0;
static CpuTick   prev_ticks[MAX_CPUS + 1]; /* index 0 = aggregate */
static double     cpu_pct[MAX_CPUS + 1];

/* ── GPU process info ───────────────────────────────────────────────── */

typedef struct {
    unsigned int  pid;
    unsigned long long mem_bytes;
    char          name[256];
    char          user[64];
    char          type; /* C=compute, G=graphics */
} GpuProc;

/* ── Globals ────────────────────────────────────────────────────────── */

static volatile sig_atomic_t g_quit = 0;
static int sort_mode = 0; /* 0=by mem, 1=by pid */

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
    unsigned long long total_kb;
    unsigned long long avail_kb;
    unsigned long long buffers_kb;
    unsigned long long cached_kb;
    unsigned long long swap_total_kb;
    unsigned long long swap_free_kb;
} MemInfo;

static void read_meminfo(MemInfo *m) {
    memset(m, 0, sizeof(*m));
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemTotal: %llu kB", &m->total_kb) == 1) continue;
        if (sscanf(line, "MemAvailable: %llu kB", &m->avail_kb) == 1) continue;
        if (sscanf(line, "Buffers: %llu kB", &m->buffers_kb) == 1) continue;
        if (sscanf(line, "Cached: %llu kB", &m->cached_kb) == 1) continue;
        if (sscanf(line, "SwapTotal: %llu kB", &m->swap_total_kb) == 1) continue;
        if (sscanf(line, "SwapFree: %llu kB", &m->swap_free_kb) == 1) continue;
    }
    fclose(f);
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
            /* Trim to just the command name */
            char *slash = strrchr(buf, '/');
            if (slash && slash < buf + n) {
                char *space = strchr(slash, ' ');
                if (space) *space = '\0';
                memmove(buf, slash + 1, strlen(slash + 1) + 1);
            } else {
                char *space = strchr(buf, ' ');
                if (space) *space = '\0';
            }
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
    printw("  DGX Spark (Grace + GB10)");
    attroff(COLOR_PAIR(7));

    char upbuf[64];
    fmt_uptime(upbuf, sizeof(upbuf));
    double l1, l5, l15;
    get_loadavg(&l1, &l5, &l15);
    mvprintw(y, cols - 50, "up %s  load %.2f %.2f %.2f", upbuf, l1, l5, l15);
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
        int bw = cols / 2 - 14;
        if (bw < 10) bw = 10;
        int color = cpu_pct[0] > 90 ? 1 : (cpu_pct[0] > 60 ? 3 : 2);
        draw_bar(y, cols / 2 + 10, bw, cpu_pct[0], color);
        printw(" %4.1f%%", cpu_pct[0]);
    }
    y += 1;

    /* Per-core bars - two columns */
    int half = (num_cpus + 1) / 2;
    int bar_w = cols / 2 - 10;
    if (bar_w < 5) bar_w = 5;

    for (int i = 0; i < half; i++) {
        int cpu_l = i + 1;
        int cpu_r = i + half + 1;

        /* Left column */
        int color = cpu_pct[cpu_l] > 90 ? 1 : (cpu_pct[cpu_l] > 60 ? 3 : 2);
        mvprintw(y, 1, "%2d ", i);
        draw_bar(y, 4, bar_w, cpu_pct[cpu_l], color);
        printw(" %4.1f%%", cpu_pct[cpu_l]);

        /* Right column */
        if (cpu_r <= num_cpus) {
            int rx = cols / 2 + 1;
            color = cpu_pct[cpu_r] > 90 ? 1 : (cpu_pct[cpu_r] > 60 ? 3 : 2);
            mvprintw(y, rx, "%2d ", i + half);
            draw_bar(y, rx + 3, bar_w, cpu_pct[cpu_r], color);
            printw(" %4.1f%%", cpu_pct[cpu_r]);
        }
        y++;
        if (y >= rows - 2) break;
    }

    y += 1;

    /* ── Memory section ─────────────────────────────────────────────── */
    MemInfo mi;
    read_meminfo(&mi);
    unsigned long long used_kb = mi.total_kb - mi.avail_kb;
    double pct_used = mi.total_kb ? (double)used_kb / mi.total_kb * 100.0 : 0;
    double pct_bufcache = mi.total_kb ? (double)(mi.buffers_kb + mi.cached_kb) / mi.total_kb * 100.0 : 0;
    /* bufcache is part of used, show the non-bufcache used portion */
    double pct_app = pct_used - pct_bufcache;
    if (pct_app < 0) pct_app = 0;

    attron(A_BOLD | COLOR_PAIR(4));
    mvprintw(y, 1, "MEM");
    attroff(A_BOLD | COLOR_PAIR(4));

    char tb[16], ub[16], bb[16];
    fmt_bytes(mi.total_kb * 1024ULL, tb, sizeof(tb));
    fmt_bytes(used_kb * 1024ULL, ub, sizeof(ub));
    fmt_bytes((mi.buffers_kb + mi.cached_kb) * 1024ULL, bb, sizeof(bb));
    printw("  %s used / %s total  (buf/cache %s)", ub, tb, bb);
    y++;

    {
        int bw = cols - 6;
        if (bw < 10) bw = 10;
        draw_bar_segmented(y, 4, bw, pct_app, pct_bufcache, 2, 4);
        char pb[16];
        snprintf(pb, sizeof(pb), " %.1f%%", pct_used);
        printw("%s", pb);
    }
    y++;

    /* Swap */
    if (mi.swap_total_kb > 0) {
        unsigned long long swap_used = mi.swap_total_kb - mi.swap_free_kb;
        double swap_pct = (double)swap_used / mi.swap_total_kb * 100.0;
        attron(A_BOLD | COLOR_PAIR(4));
        mvprintw(y, 1, "SWP");
        attroff(A_BOLD | COLOR_PAIR(4));
        char stb[16], sub[16];
        fmt_bytes(swap_used * 1024ULL, sub, sizeof(sub));
        fmt_bytes(mi.swap_total_kb * 1024ULL, stb, sizeof(stb));
        printw("  %s / %s", sub, stb);
        y++;
        {
            int bw = cols - 6;
            if (bw < 10) bw = 10;
            int color = swap_pct > 80 ? 1 : (swap_pct > 40 ? 3 : 5);
            draw_bar(y, 4, bw, swap_pct, color);
            printw(" %.1f%%", swap_pct);
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
                get_proc_cmdline(p->pid, p->name, sizeof(p->name));
                get_proc_user(p->pid, p->user, sizeof(p->user));
            }

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
                mvprintw(y, 1, "  %-8s %-12s %-4s %-12s %s", "PID", "USER", "TYPE", "GPU MEM", "COMMAND");
                attroff(A_BOLD | COLOR_PAIR(7));
                y++;

                for (int i = 0; i < n_all && y < rows - 2; i++) {
                    GpuProc *p = &all_procs[i];
                    char mb[16];
                    fmt_bytes(p->mem_bytes, mb, sizeof(mb));

                    int name_max = cols - 44;
                    if (name_max < 10) name_max = 10;
                    char truncname[256];
                    snprintf(truncname, sizeof(truncname), "%-.*s", name_max, p->name);

                    int pc = (p->type == 'C') ? 5 : 7;
                    mvprintw(y, 1, "  %-8u %-12s ", p->pid, p->user);
                    attron(COLOR_PAIR(pc));
                    printw("%-4c", p->type);
                    attroff(COLOR_PAIR(pc));
                    printw(" %-12s %s", mb, truncname);
                    y++;
                }
            }
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
    printw(":speed ");

    refresh();
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void) {
    setlocale(LC_ALL, "");
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    /* Load NVML */
    nvml_ok = (load_nvml() == 0);

    /* Initial CPU tick read */
    read_cpu_ticks(prev_ticks, &num_cpus);
    usleep(100000); /* brief pause for first delta */
    compute_cpu_usage();

    /* Init ncurses */
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

    int delay_ms = REFRESH_MS;

    while (!g_quit) {
        compute_cpu_usage();
        draw_screen();

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

    if (nvml_ok && pNvmlShutdown) pNvmlShutdown();
    if (nvml_handle) dlclose(nvml_handle);

    return 0;
}

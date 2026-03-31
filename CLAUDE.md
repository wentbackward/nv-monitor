# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

nv-monitor is a single-file C terminal system monitor for the NVIDIA DGX Spark (Grace ARM CPU + GB10 Blackwell GPU). It combines CPU, memory, and GPU monitoring in one ncurses TUI.

## Build

```bash
make          # builds nv-monitor binary (-O3 -march=native)
make portable # builds without -march=native (for CI/distribution)
make test     # builds and runs unit tests
make install-user # installs to ~/.local/bin by default
make clean    # removes binaries
```

Direct compilation: `cc -O2 -Wall -Wextra -std=gnu11 -o nv-monitor nv-monitor.c -lncursesw -ldl -lpthread`

Dependencies: a C compiler (`cc`, `gcc`, or `clang`) and `libncurses-dev`

Works on both **aarch64** (DGX Spark) and **x86_64**. On x86, ARM core type labels are omitted; everything else works identically.

## Architecture

Everything is in `nv-monitor.c` (~1640 lines). Key sections:

- **NVML dynamic loading** (line ~115): Loads `libnvidia-ml.so.1` via `dlopen`/`dlsym` at runtime. Uses a variadic LOAD macro to try versioned symbols first (e.g. `nvmlInit_v2` before `nvmlInit`). All NVML function pointers are prefixed with `p` (e.g. `pNvmlInit`).
- **CPU sampling**: Reads `/proc/stat` delta between frames to compute per-core usage percentages.
- **Memory**: Parses `/proc/meminfo` for used/available/buffers/cached/swap.
- **CPU thermals/freq**: Reads from `/sys/class/thermal/` and `/sys/devices/system/cpu/`.
- **GPU process info**: Queries both compute and graphics process lists via NVML, resolves PID to command name and user via `/proc/<pid>/`.
- **TUI rendering**: ncursesw (wide character support) with color pairs (1=red/critical, 2=green/normal, 3=yellow/medium, 6=cyan/headers). `draw_screen()` is the main render function.
- **History chart**: Ring buffer of last 20 CPU/GPU samples, rendered as full-width vertical bar chart using Unicode block elements (▁▂▃▄▅▆▇█).
- **CSV logging**: Opt-in via `-l FILE`, writes timestamped rows with all CPU/memory/GPU metrics. Shares derived calculations with the TUI via `meminfo_calc()`.
- **Prometheus exporter**: Opt-in via `-p PORT`. Runs a minimal HTTP server on a dedicated pthread, serving OpenMetrics-formatted metrics at `/metrics`. Uses POSIX sockets with `poll()` for clean shutdown. Zero overhead when not enabled. Enables multi-machine monitoring via Prometheus/Grafana.

## DGX Spark specifics

- The GB10 GPU uses **unified memory** shared with the Grace CPU. `nvmlDeviceGetMemoryInfo` returns NOT_SUPPORTED — the code detects this and shows "unified memory" instead of a VRAM bar.
- Target arch is **aarch64**. NVML library paths include both aarch64 and x86_64 fallbacks.
- No fan speed sensor on GB10 (handled gracefully via return code check).
- **Grace CPU is big.LITTLE**: 10x Cortex-X925 (performance, 3.9 GHz) + 10x Cortex-X725 (efficiency, 2.8 GHz). Core types are identified via ARM CPU part IDs in `/proc/cpuinfo` (`0xd85` = X925, `0xd87` = X725) and shown per-core in the TUI.
- DGX Spark ships with `performance` cpufreq governor (cores pinned to max). Per-core frequency display is not shown since it's static, but if future hardware (e.g. DGX 300) uses dynamic governors, per-core freq could be read from `/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`.
- When HugePages are active, `MemAvailable` is inaccurate — the code uses `HugePages_Free * Hugepagesize` instead (per NVIDIA known-issues docs).

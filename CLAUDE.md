# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

nv-monitor is a single-file C terminal system monitor for the NVIDIA DGX Spark (Grace ARM CPU + GB10 Blackwell GPU). It combines CPU, memory, and GPU monitoring in one ncurses TUI.

## Build

```bash
make          # builds nv-monitor binary
make clean    # removes binary
```

Direct compilation: `gcc -O2 -Wall -Wextra -std=gnu11 -o nv-monitor nv-monitor.c -lncursesw -ldl -lpthread`

Dependencies: `build-essential`, `libncurses-dev`

## Architecture

Everything is in `nv-monitor.c` (~940 lines). Key sections:

- **NVML dynamic loading** (line ~115): Loads `libnvidia-ml.so.1` via `dlopen`/`dlsym` at runtime. Uses a variadic LOAD macro to try versioned symbols first (e.g. `nvmlInit_v2` before `nvmlInit`). All NVML function pointers are prefixed with `p` (e.g. `pNvmlInit`).
- **CPU sampling**: Reads `/proc/stat` delta between frames to compute per-core usage percentages.
- **Memory**: Parses `/proc/meminfo` for used/available/buffers/cached/swap.
- **CPU thermals/freq**: Reads from `/sys/class/thermal/` and `/sys/devices/system/cpu/`.
- **GPU process info**: Queries both compute and graphics process lists via NVML, resolves PID to command name and user via `/proc/<pid>/`.
- **TUI rendering**: ncursesw (wide character support) with color pairs (1=red/critical, 2=green/normal, 3=yellow/medium, 6=cyan/headers). `draw_screen()` is the main render function.
- **History chart**: Ring buffer of last 20 CPU/GPU samples, rendered as vertical bar chart using Unicode block elements (▁▂▃▄▅▆▇█) in the bottom-right corner.

## DGX Spark specifics

- The GB10 GPU uses **unified memory** shared with the Grace CPU. `nvmlDeviceGetMemoryInfo` returns NOT_SUPPORTED — the code detects this and shows "unified memory" instead of a VRAM bar.
- Target arch is **aarch64**. NVML library paths include both aarch64 and x86_64 fallbacks.
- No fan speed sensor on GB10 (handled gracefully via return code check).

# Performance Report

nv-monitor is designed for minimal resource impact, particularly on edge devices where every MB and CPU cycle matters. This report compares nv-monitor against standard Linux monitoring tools.

## Summary

| Metric | nv-monitor | top | htop |
|--------|-----------|-----|------|
| **Monitors GPU** | Yes | No | No |
| **Prometheus export** | Yes | No | No |
| **CSV logging** | Yes | No | No |
| **RDMA monitoring** | Yes | No | No |

nv-monitor provides significantly more functionality while using less CPU than either tool.

## DGX Spark (Grace + GB10, 128 GB, 20 cores)

```
Binary sizes:
  nv-monitor: 76K
  top:        136K
  htop:       391K

Measurements (10s sample):
  nv-monitor (headless)            RSS: 20908 KB  Private: 15088 KB  CPU: 0.10%
  nv-monitor (headless+prometheus) RSS: 20920 KB  Private: 15100 KB  CPU: 0.10%
  top (batch mode)                 RSS:  5508 KB  Private:  2080 KB  CPU: 0.70%

NVML library: 2.2 MB (shared with other GPU processes)
```

- **7x less CPU than top** — while monitoring CPU, memory, GPU, and RDMA
- NVML shared library code (2.2 MB) is mapped once in physical RAM and shared across all GPU processes (nvidia-smi, VLLM, etc.)
- Private memory cost of nv-monitor itself: ~15 MB (mostly NVML's internal heap allocations)

## Jetson Orin Nano Super (8 GB, 6 cores)

```
Binary sizes:
  nv-monitor: 64K
  top:        118K
  htop:       265K

Measurements (10s sample):
  nv-monitor (headless)            RSS: 15312 KB  Private: 12932 KB  CPU: 0.20%
  nv-monitor (headless+prometheus) RSS: 15328 KB  Private: 12940 KB  CPU: 0.20%
  top (batch mode)                 RSS:  3424 KB  Private:   788 KB  CPU: 2.10%

NVML + Tegra libraries: ~1.7 MB (split across multiple Tegra shared objects)
```

- **10x less CPU than top** — on a 6-core edge device where CPU budget is tight
- Adding Prometheus export adds zero measurable CPU overhead
- Tegra NVML is split across multiple smaller shared objects (libnvrm_gpu.so, libnvidia-ml.so, etc.), totalling ~1.7 MB of shared library code

## Memory breakdown

RSS (Resident Set Size) is misleading for GPU-aware tools because it includes shared library pages. The actual unique memory cost:

| Component | DGX Spark | Orin Nano |
|-----------|-----------|-----------|
| nv-monitor code + stack | ~3 MB | ~3 MB |
| NVML private heap | ~12 MB | ~10 MB |
| NVML shared library code | 2.2 MB (shared) | 1.7 MB (shared) |
| Prometheus thread stack | +128 KB | +128 KB |
| **Total unique to nv-monitor** | **~15 MB** | **~13 MB** |

The NVML shared library code is mapped into physical memory once by the kernel and shared across all processes that use it. On a system already running GPU workloads, this memory is already allocated.

## Prometheus overhead

| Mode | Threads | Additional CPU | Additional memory |
|------|---------|---------------|-------------------|
| Headless only | 2 | baseline | baseline |
| Headless + Prometheus | 3 | +0.00% | +128 KB (thread stack) |

The Prometheus server thread uses `poll()` with a 1-second timeout and only wakes on incoming scrape requests. Between scrapes, it consumes zero CPU.

## Methodology

Results collected using `bench.sh` included in this repository. Run it on any target device:

```bash
./bench.sh
```

The script measures:
- Binary sizes
- RSS, private (unique), and shared memory via `/proc/<pid>/smaps_rollup`
- CPU usage over a 10-second sample via `/proc/<pid>/stat` tick deltas
- NVML shared library memory map regions

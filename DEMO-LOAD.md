# demo-load — Synthetic CPU/GPU Load Generator

A lightweight test harness for validating nv-monitor across single machines or multi-node clusters. Generates controllable, sinusoidal CPU and GPU loads with zero dependencies beyond a C compiler and NVIDIA drivers.

Useful for:
- **End-to-end testing** of nv-monitor TUI, CSV logging, and Prometheus pipelines
- **Visual health-checks** — verify bars, colors, and history charts are rendering correctly
- **Multi-node validation** — run on every box in a cluster without installing bulky benchmarking tools

## Building

```bash
make demo-load
```

Or directly:

```bash
gcc -O2 -o demo-load demo-load.c -lpthread -ldl -lm
```

No CUDA toolkit required — the GPU load uses the CUDA driver API via `dlopen`, so only the NVIDIA driver needs to be installed.

## Usage

```bash
./demo-load                        # CPU load only (all cores)
./demo-load --gpu                  # CPU + GPU load
./demo-load --gpu-only             # GPU load only
./demo-load --gpu --blocks 4096    # GPU with more blocks (for bigger GPUs)
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--gpu` | Enable GPU load alongside CPU | off |
| `--gpu-only` | GPU load only, no CPU threads | off |
| `--blocks N` | CUDA blocks per kernel launch | 1024 |
| `-h` | Show help | |

### Tuning GPU load with `--blocks`

The `--blocks` flag controls how many CUDA thread blocks are launched per kernel. More blocks = more GPU hardware utilised per launch. The right value depends on your GPU:

| GPU | Suggested `--blocks` |
|-----|---------------------|
| GB10 (DGX Spark) | 256 - 512 |
| RTX 3050/3060 | 1024 (default) |
| RTX 4090 | 2048 - 4096 |
| A100 / H100 | 4096 - 8192 |

The tool auto-calibrates kernel timing at startup, so the sinusoidal pattern adapts to any GPU — `--blocks` just controls the peak utilisation ceiling.

## How It Works

### CPU

Spawns one thread per CPU core. Each thread runs a busy-wait/sleep duty cycle following a sine wave, with:
- **Phase offset** per core — so all cores show different utilisation levels at any given moment
- **Slightly different frequencies** per core — creating a rolling wave pattern across cores

### GPU

Loads a small PTX kernel (FMA-heavy compute loop) via the CUDA driver API at runtime. At startup, it calibrates by measuring kernel execution time, then uses a duty-cycle approach:
1. Each 200ms window is divided into busy time and idle time based on a sine wave
2. During the busy portion, kernels are queued back-to-back on a CUDA stream with no synchronisation between launches — keeping the GPU pipeline fully saturated
3. A single sync at the end, then sleep for the idle portion

This matches how NVML measures utilisation (fraction of time the GPU was active), producing smooth, realistic-looking load curves.

## Example: Multi-Node Production Test

Run demo-load on each node while nv-monitor exports Prometheus metrics:

```bash
# On each node:
./demo-load --gpu &
./nv-monitor -n -p 9101
```

Then scrape all nodes from Prometheus and verify metrics flow end-to-end through to Grafana — without deploying real workloads.

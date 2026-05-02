# nv-monitor

Local monitoring TUI, CSV logger, and Prometheus/OpenMetrics exporter for NVIDIA GPU systems — all in a single <80KB binary with zero runtime dependencies. Built for the **DGX Spark** (Grace + GB10), works on any Linux system with an NVIDIA GPU.

Accurately monitor a single machine or an entire cluster with minimal overhead. Reports metrics to NVIDIA specifications via NVML, with correct handling of unified memory, HugePages, and ARM big.LITTLE core topology. Includes `demo-load`, a zero-dependency synthetic CPU/GPU load generator for validating your monitoring pipeline end-to-end.

![C](https://img.shields.io/badge/lang-C-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Arch](https://img.shields.io/badge/arch-aarch64%20%7C%20x86__64-orange) ![Build](https://github.com/wentbackward/nv-monitor/actions/workflows/build.yml/badge.svg)

## Display

### CPU Section
- **Overall** aggregate usage bar across all cores
- **Per-core** usage bars in dual-column layout with ARM core type labels (**X925** = performance cores at 3.9 GHz, **X725** = efficiency cores at 2.8 GHz on the Grace big.LITTLE architecture)
- CPU temperature (highest thermal zone) and frequency

### Memory Section
- **Used** (green) — actual application memory (total - free - buffers - cached)
- **Buf/cache** (blue) — kernel buffers and page cache (reclaimable)
- Swap usage bar
- Correctly handles **HugePages** on DGX Spark where `MemAvailable` is inaccurate

### GPU Section
- **GPU utilization** bar with temperature, power draw (watts), and clock speed
- **VRAM** bar, or "unified memory" label on DGX Spark where CPU/GPU share memory
- **ENC/DEC** — hardware video encoder (NVENC) and decoder (NVDEC) utilization percentage

### GPU Processes
- **PID** — process ID
- **USER** — process owner
- **TYPE** — **C** (Compute: CUDA/inference workloads) or **G** (Graphics: rendering, e.g. Xorg)
- **CPU%** — per-process CPU usage (delta-based, per-core scale)
- **GPU MEM** — GPU memory allocated by the process
- **COMMAND** — binary name with arguments
- **(other processes)** — summary row showing CPU usage from non-GPU processes

### History Chart
- Full-width rolling graph of CPU (green) and GPU (cyan) utilization over the last 20 samples using Unicode block elements (▁▂▃▄▅▆▇█)

### General
- Color-coded bars: green (normal), yellow (>60%), red (>90%)
- **CSV Logging** — log all stats to file with configurable interval
- **Headless Mode** — run without TUI for unattended data collection
- 1s default refresh, adjustable at runtime or via CLI
- NVML loaded dynamically at runtime — no hard dependency on NVIDIA drivers

<table>
<tr>
<td><strong>aarch64</strong> (DGX Spark — Grace + GB10)</td>
<td><strong>x86_64</strong> (Laptop — Ryzen 7 + RTX 3050)</td>
<td><strong>arm64</strong> (Jetson Orin Nano)</td>
</tr>
<tr>
<td><img src="screenshots/nv-monitor-arm.png" alt="nv-monitor on ARM"></td>
<td><img src="screenshots/nv-monitor-x86.png" alt="nv-monitor on x86"></td>
<td><img src="screenshots/nv-monitor-jetson.png" alt="nv-monitor on Jetson"></td>
</tr>
</table>
<table>
<tr>
<td><strong>PowerEdge XE9680</strong> (H100 Datacenter Grade)</td>
<td><strong>GB200</strong> (Datacenter Grade)</td>
<td><strong>GB200</strong> (Datacenter Grade)</td>
</tr>
<tr>
<td><img src="screenshots/poweredge-xe9680-H100.png" alt="nv-monitor on PowerEdge XE9680 with H100"></td>
<td><img src="screenshots/GB200-2.png" alt="nv-monitor on GB200"></td>
<td><img src="screenshots/GB200.png" alt="nv-monitor on GB200"></td>
</tr>
</table>
<table>
<tr>
<td><strong>PowerEdge XE9680</strong> (H100 Datacenter Grade) - Compact View</td>
</tr>
<tr>
<td><img src="screenshots/poweredge-xe9680-H100.png" alt="nv-monitor on ARM"></td>
</tr>
</table>

## Download

There's a [binary release](https://github.com/wentbackward/nv-monitor/releases) built on every release via GitHub CI/CD pipelines.

### Arch Linux

On Arch-based Linux distributions, you can install nv-monitor directly with pacman when the AUR is enabled:

```bash
pacman -S nv-monitor
```

## Building

Requires `gcc` and `libncurses-dev`:

```bash
sudo apt install build-essential libncurses-dev
make
```

## Usage

```bash
./nv-monitor                           # TUI only
./nv-monitor -l stats.csv              # TUI + log every 1s
./nv-monitor -l stats.csv -i 5000      # TUI + log every 5s
./nv-monitor -n -l stats.csv -i 500    # Headless, log every 500ms
./nv-monitor -r 2000                   # TUI refreshing every 2s
./nv-monitor -p 9101                   # TUI + Prometheus metrics on :9101
./nv-monitor -n -p 9101                # Headless Prometheus exporter
```

Or install system-wide:

```bash
sudo make install
```

### Command-line options

| Flag      | Description                                | Default |
|-----------|--------------------------------------------|---------|
| `-c COLS` | CPU display columns (1-4, 0=auto)          | auto    |
| `-l FILE` | Log statistics to CSV file                 | off     |
| `-i MS`   | Log interval in milliseconds               | 1000    |
| `-n`      | Headless mode (no TUI, requires `-l`/`-p`) | off     |
| `-p PORT` | Expose Prometheus metrics on PORT          | off     |
| `-t TOKEN`| Require Bearer token for `/metrics`        | off     |
| `-r MS`   | UI refresh interval in milliseconds        | 1000    |
| `-v`      | Show version                               |         |
| `-h`      | Show help                                  |         |

### Interactive controls

| Key         | Action                                       |
|-------------|----------------------------------------------|
| `q`/Esc     | Quit                                         |
| `s`         | Toggle sort (GPU memory / PID)               |
| `c`         | Cycle CPU column count (auto → 1 → 2 → 3 → 4)|
| `j`/`k` or ↑/↓ | Scroll CPU cores when they exceed the viewport |
| `+`/`-`     | Adjust refresh rate (250ms steps)            |

The CPU section auto-selects the column count based on terminal width and adapts on resize. When cores exceed the available vertical space, a scroll indicator `[first-last] ↑↓` appears on the CPU header line. Use `-c N` to override the auto column count at startup.

## Prometheus Metrics

Pass `-p PORT` to expose a Prometheus-compatible metrics endpoint:

```bash
./nv-monitor -p 9101              # TUI + metrics at http://localhost:9101/metrics
./nv-monitor -n -p 9101           # Pure headless exporter
curl -s localhost:9101/metrics     # Check it works
```

### Available metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `nv_build_info` | gauge | `version` | nv-monitor version |
| `nv_uptime_seconds` | gauge | | System uptime |
| `nv_load_average` | gauge | `interval` | Load average (1m, 5m, 15m) |
| `nv_cpu_usage_percent` | gauge | `cpu`, `type` | Per-core CPU utilization (type = ARM core: X925, X725, etc.) |
| `nv_cpu_temperature_celsius` | gauge | | CPU temperature |
| `nv_cpu_frequency_mhz` | gauge | | CPU frequency |
| `nv_memory_total_bytes` | gauge | | Total system memory |
| `nv_memory_used_bytes` | gauge | | Application memory used |
| `nv_memory_bufcache_bytes` | gauge | | Buffer and cache memory |
| `nv_swap_total_bytes` | gauge | | Total swap |
| `nv_swap_used_bytes` | gauge | | Swap used |
| `nv_gpu_info` | gauge | `gpu`, `name` | GPU device name |
| `nv_gpu_utilization_percent` | gauge | `gpu` | GPU compute utilization |
| `nv_gpu_temperature_celsius` | gauge | `gpu` | GPU temperature |
| `nv_gpu_power_watts` | gauge | `gpu` | GPU power draw |
| `nv_gpu_clock_mhz` | gauge | `gpu`, `type` | GPU clock speed (graphics, memory) |
| `nv_gpu_memory_total_bytes` | gauge | `gpu` | GPU memory total |
| `nv_gpu_memory_used_bytes` | gauge | `gpu` | GPU memory used |
| `nv_gpu_fan_speed_percent` | gauge | `gpu` | Fan speed |
| `nv_gpu_encoder_utilization_percent` | gauge | `gpu` | Hardware encoder utilization |
| `nv_gpu_decoder_utilization_percent` | gauge | `gpu` | Hardware decoder utilization |

### Prometheus scrape config

```yaml
scrape_configs:
  - job_name: 'nv-monitor'
    authorization:
      credentials: 'my-secret-token'
    static_configs:
      - targets: ['dgx-spark:9101']
```

No new dependencies are required — the exporter uses POSIX sockets and adds ~128 KB of memory overhead.

### Security

The exporter supports optional Bearer token authentication:

```bash
./nv-monitor -p 9101 -t my-secret-token           # token via CLI flag
NV_MONITOR_TOKEN=my-secret-token ./nv-monitor -p 9101  # token via env var (preferred)
```

The env var is preferred over `-t` since CLI arguments are visible in `ps` output. Without `-t` or `NV_MONITOR_TOKEN`, no auth is required (backwards compatible).

**Design rationale:** nv-monitor is a lightweight, single-purpose endpoint — it intentionally does not implement TLS. For transport security, layer it with the tools you already have:

- **Tailscale** — zero-config encrypted mesh, just run nv-monitor on the tailnet
- **SSH tunnel** — `ssh -L 9101:localhost:9101 dgx-spark`
- **Reverse proxy** — nginx/caddy with TLS termination
- **Service mesh** — Istio, Linkerd, etc.

This keeps the binary small, dependency-free, and composable with existing infrastructure.

## Synthetic Load Testing

A companion tool `demo-load` generates sinusoidal CPU and GPU loads for visual testing and multi-node validation — no bulky benchmarking tools required. See [DEMO-LOAD.md](DEMO-LOAD.md) for details.

```bash
make demo-load
./demo-load --gpu          # CPU + GPU sinusoidal load
```

## Requirements

- Linux (reads from `/proc` and `/sys`)
- ncurses (TUI mode)
- NVIDIA drivers with NVML (for GPU monitoring — CPU/memory work without it)

### Platform support

| Platform | Status |
|----------|--------|
| DGX Spark (aarch64, Grace + GB10) | Primary target — full support including unified memory, HugePages, big.LITTLE core labels |
| GB200 NVL (aarch64, up to 208 GPUs) | Supported — dynamic allocation scales to any CPU/GPU count, scrollable TUI |
| Dell PowerEdge XE9680 (x86_64, 8x H100) | Tested — 112 cores, 8 GPUs with VRAM, multi-GPU CSV/Prometheus export |
| Jetson Orin (Nano / NX / AGX) | GPU via Tegra sysfs, A78AE core labels, legacy glibc binary available |
| Any Linux + NVIDIA GPU (x86_64) | Fully supported — CPU, memory, GPU, processes, Prometheus exporter |
| Linux without NVIDIA GPU | CPU and memory monitoring only, GPU section shows "NVML not available" |
| RDMA / InfiniBand | Community-verified on real hardware — auto-detected via `/sys/class/infiniband/`, feedback welcome |

### A note on RDMA and cross-node bandwidth

nv-monitor captures per-port TX/RX throughput over QSFP/InfiniBand links — this is cross-node traffic by definition. In a Prometheus/Grafana setup you can visualise each node's fabric utilisation side by side and infer traffic flows.

For fabric-level visibility (topology, per-peer bandwidth, congestion maps, hop-by-hop latency), use your subnet manager or NVIDIA UFM alongside nv-monitor. Networking is a complex domain — nv-monitor focuses on endpoint metrics and leaves fabric management to dedicated tools.

## Contributors

- Prometheus metrics exporter by [Tim Messerschmidt (@SeraphimSerapis)](https://github.com/SeraphimSerapis)

## License

MIT

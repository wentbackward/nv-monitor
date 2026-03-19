# nv-monitor

A lightweight terminal system monitor built for the **NVIDIA DGX Spark** (Grace CPU + GB10 GPU). Think htop + nvtop in a single 73KB binary.

![C](https://img.shields.io/badge/lang-C-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Arch](https://img.shields.io/badge/arch-aarch64-orange)

## Features

- **CPU**: Per-core usage bars (dual-column layout), frequency, temperature, overall aggregate
- **Memory**: Used/total with buf/cache segmented bar, swap usage
- **GPU**: Utilization bar, temperature, power draw, clock speed, encoder/decoder utilization
- **GPU Processes**: PID, user, type (Compute/Graphics), GPU memory, command name
- **Unified Memory**: Gracefully handles the DGX Spark's shared CPU/GPU memory architecture
- Color-coded bars (green/yellow/red) based on utilization thresholds
- 1s default refresh, adjustable at runtime
- NVML loaded dynamically at runtime — no hard dependency on NVIDIA drivers

## Building

Requires `gcc` and `libncurses-dev`:

```bash
sudo apt install build-essential libncurses-dev
make
```

## Usage

```bash
./nv-monitor
```

Or install system-wide:

```bash
sudo make install
```

### Controls

| Key     | Action                              |
|---------|-------------------------------------|
| `q`/Esc | Quit                                |
| `s`     | Toggle sort (GPU memory / PID)      |
| `+`/`-` | Adjust refresh rate (250ms steps)   |

## Requirements

- Linux (reads from `/proc` and `/sys`)
- ncurses
- NVIDIA drivers with NVML (for GPU monitoring — CPU/memory work without it)

Tested on DGX Spark (aarch64, CUDA 13.0, driver 580.x) but should work on any Linux system with an NVIDIA GPU.

## License

MIT

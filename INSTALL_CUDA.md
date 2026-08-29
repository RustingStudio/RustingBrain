# CUDA Setup

RustingBrain builds and trains on CPU by default. CUDA is only needed for the
GPU matrix benchmark and later GPU backend work.

The CUDA feature uses dynamic loading, so Rust can compile the feature without
calling `nvcc` during the build. Running the benchmark still requires an NVIDIA
driver and cuBLAS runtime libraries.

## Basic Commands

Compile CUDA support:

```bash
cargo check --features cuda
```

Run the benchmark:

```bash
cargo run --release --example cuda_benchmark --features cuda
```

The `cuda` feature currently points to CUDA 13.1 bindings. Other binding
versions are available:

```bash
cargo check --features cuda-11-8
cargo check --features cuda-12-0
cargo check --features cuda-12-4
cargo check --features cuda-12-6
cargo check --features cuda-12-8
cargo check --features cuda-13-0
cargo check --features cuda-13-1
```

Use the closest version at or below the CUDA runtime installed on the machine.

## CachyOS / Arch

Install CUDA and NVIDIA utilities:

```bash
sudo pacman -Syu
sudo pacman -S cuda nvidia-utils nvidia-settings
```

If you use the default CachyOS kernel and need the NVIDIA kernel module:

```bash
sudo pacman -S linux-cachyos-nvidia-open
sudo reboot
```

Check the driver:

```bash
nvidia-smi
```

Check CUDA and cuBLAS:

```bash
/opt/cuda/bin/nvcc --version
find /opt/cuda/lib64 -name 'libcublas.so*'
```

Expose CUDA tools and libraries in the current shell:

```bash
export PATH=/opt/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH
```

Add those two exports to your shell profile if you want them to persist.

## Ubuntu / Debian

Install the NVIDIA driver first, then use NVIDIA's CUDA download page for the
toolkit package that matches your distribution:

<https://developer.nvidia.com/cuda-downloads>

After installation:

```bash
nvidia-smi
nvcc --version
ldconfig -p | grep libcublas
```

If the libraries are installed but not visible at runtime:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Troubleshooting

If the benchmark says it cannot load `cublas`, the Rust code compiled but the
runtime library is not visible. Install the CUDA toolkit or add the CUDA library
directory to `LD_LIBRARY_PATH`.

Useful checks:

```bash
nvidia-smi
ldconfig -p | grep libcublas
find /opt/cuda /usr/local/cuda -name 'libcublas.so*' 2>/dev/null
```

Official NVIDIA docs:

- <https://developer.nvidia.com/cuda-downloads>
- <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>
- <https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html>

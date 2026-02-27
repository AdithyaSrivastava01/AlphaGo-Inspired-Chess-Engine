# CUDA Acceleration: Matrix Multiplication and 2D Convolution

High‑performance GPU computing project that implements and benchmarks matrix multiplication and 2D image convolution on CPU and CUDA. The repo includes multiple GPU strategies (naive and shared‑memory tiled), a cuBLAS baseline, a Python interface via ctypes, and a reproducible benchmarking/visualization workflow with ready‑to‑use figures and CSVs.

**Highlights**
- CPU baseline and three GPU paths: naive CUDA kernel, shared‑memory tiled kernel, and cuBLAS `sgemm`.
- 2D convolution on GPU and CPU with a Python test harness for analysis and visualization.
- Reproducible benchmarks across sizes (256 → 4096), CSV outputs, and publication‑ready plots.
- Practical CUDA engineering: shared memory tiling, memory coalescing, 2D grid/block configuration, and CUDA event timing.
- Figures and a short write‑up included: `Analysis_Report_GPU_Acceleration_with_CUDA.pdf`.

**Tech & Keywords**
- CUDA C/C++, cuBLAS, GPU kernels, shared memory, tiling, memory coalescing
- Performance benchmarking, CUDA events, numerical computing, HPC
- 2D convolution, image processing (Sobel, Gaussian blur, sharpen)
- Python interop via `ctypes`, NumPy, Matplotlib, Pandas
- Bash automation, reproducible experiments, GCP GPU runs

---

## Repo Structure
- `Matrix_Multiplication/`
  - `matrix_cpu.c` — CPU triple‑nested loop baseline.
  - `matrix_gpu.cu` — Naive CUDA kernel (global memory loads per multiply‑accumulate), 2D grid of 16×16 threads.
  - `matrix_tiled.cu` — Shared‑memory tiled CUDA kernel (`TILE_WIDTH=16`) for reduced global memory traffic.
  - `matrix_cublas.cu` — cuBLAS `cublasSgemm` baseline for peak GEMM performance.
  - `run_all_comparison.sh` — Runs binaries across sizes and writes summary CSV‑style text.
  - `plot_cpu_results.py`, `compare_cpu_gpu.py`, `part4_visualization.py` — Creates plots and summary tables.
  - Figures: `cpu_performance.png`, `cpu_gpu_comparison.png`, `part4_complete_analysis.png`.
- `Convolution/`
  - `matrix_lib.cu` — CUDA shared library exposing: `gpu_matrix_multiply`, `gpu_convolution`, and CPU reference `cpu_convolution`.
  - `test_library.py` — Comprehensive Python benchmark and visualization suite (saves figures and CSV).
  - `convolution_cpu_test.c` — CPU‑only convolution perf test (multiple sizes).
  - `build_library.sh` — Builds `libmatrix.so` using NVCC.
  - Figures: `convolution_filters_demo.png`, `part8_convolution_complete_analysis.png` (+ GCP results in `gcp_results/`).
- `Analysis_Report_GPU_Acceleration_with_CUDA.pdf` — Short write‑up summarizing the work.

---

## Quick Start

Prerequisites
- NVIDIA GPU + drivers, CUDA Toolkit (nvcc), and cuBLAS (bundled with CUDA).
- Python 3.9+ with `numpy`, `pandas`, `matplotlib`, `pillow`.
  - Install: `python3 -m pip install -U numpy pandas matplotlib pillow`

Build (Matrix Multiplication)
- From repo root, compile CPU/GPU binaries:
  - `gcc -O3 -o Matrix_Multiplication/matrix_cpu Matrix_Multiplication/matrix_cpu.c`
  - `nvcc -O3 -o Matrix_Multiplication/matrix_gpu Matrix_Multiplication/matrix_gpu.cu`
  - `nvcc -O3 -o Matrix_Multiplication/matrix_tiled Matrix_Multiplication/matrix_tiled.cu`
  - `nvcc -O3 -lcublas -o Matrix_Multiplication/matrix_cublas Matrix_Multiplication/matrix_cublas.cu`

Run (Matrix Multiplication)
- Single run (example N=1024):
  - `./Matrix_Multiplication/matrix_cpu 1024`
  - `./Matrix_Multiplication/matrix_gpu 1024`
  - `./Matrix_Multiplication/matrix_tiled 1024`
  - `./Matrix_Multiplication/matrix_cublas 1024`
- End‑to‑end comparison across sizes:
  - `bash Matrix_Multiplication/run_all_comparison.sh`
  - Generates `cublas_comparison.txt` and prints per‑size timings for Naive, Optimized, and cuBLAS.
- Plotting from saved results (`cpu_results.txt`, `gpu_results.txt`):
  - `python3 Matrix_Multiplication/plot_cpu_results.py` → `cpu_performance.png`
  - `python3 Matrix_Multiplication/compare_cpu_gpu.py` → `cpu_gpu_comparison.png`

Build & Run (Convolution + Python)
- Build shared library:
  - `cd Convolution && bash build_library.sh`
  - Produces `Convolution/libmatrix.so`.
- Run the comprehensive suite (benchmarks + figures + CSV):
  - `python3 Convolution/test_library.py`
  - Outputs:
    - `Convolution/part8_convolution_complete_analysis.png` (3‑panel figure: CPU vs GPU times, filter comparison, speedup)
    - `Convolution/convolution_filters_demo.png` (visual demo on synthetic images)
    - `Convolution/convolution_performance_results.csv` (benchmark dataset)

---

## What This Shows (Engineering Notes)
- GPU kernel design: 2D thread blocks (16×16), 2D grids, bounds checks for edge tiles.
- Shared memory tiling: co‑loads `A` and `B` tiles into `__shared__` to reduce global memory transactions and improve reuse.
- Timing & correctness: CUDA event timing around kernels; lightweight checksum verification to avoid dead‑code elimination.
- Baseline vs optimized: naive kernel → tiled kernel → cuBLAS peak; scripts generate comparison tables/plots for each N.
- Python interop: clean `ctypes` signatures and contiguous NumPy arrays enable direct calls to CUDA kernels from Python.
- Image filters: Sobel (X/Y), blur, Gaussian blur, sharpen; zero‑padding at boundaries; GPU/CPU parity validated with MSE.
- Reproducibility: scripts and CSVs checked into repo; figures are regenerated by running the suite.

---

## Example Commands (Cheat Sheet)
- Build all binaries (from repo root):
  - `gcc -O3 -o Matrix_Multiplication/matrix_cpu Matrix_Multiplication/matrix_cpu.c`
  - `nvcc -O3 -o Matrix_Multiplication/matrix_gpu Matrix_Multiplication/matrix_gpu.cu`
  - `nvcc -O3 -o Matrix_Multiplication/matrix_tiled Matrix_Multiplication/matrix_tiled.cu`
  - `nvcc -O3 -lcublas -o Matrix_Multiplication/matrix_cublas Matrix_Multiplication/matrix_cublas.cu`
- Convolution shared library:
  - `cd Convolution && bash build_library.sh && cd -`
- Full Python analysis:
  - `python3 Convolution/test_library.py`
- Matrix comparison across sizes:
  - `bash Matrix_Multiplication/run_all_comparison.sh`

---

## Results & Artifacts
- Matrix multiplication figures:
  - `Matrix_Multiplication/cpu_performance.png`
  - `Matrix_Multiplication/cpu_gpu_comparison.png`
  - `Matrix_Multiplication/part4_complete_analysis.png`
- Convolution figures and data:
  - `Convolution/part8_convolution_complete_analysis.png`
  - `Convolution/convolution_filters_demo.png`
  - `Convolution/convolution_performance_results.csv`
- Write‑up:
  - `Analysis_Report_GPU_Acceleration_with_CUDA.pdf`


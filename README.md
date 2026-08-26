# LB-Cube: A Modern C++23 / CUDA Lattice Boltzmann Solver

LB-Cube is a high-performance Lattice Boltzmann Method (LBM) solver for academic research on rectangular and cubic periodic domains. The codebase implements compile-time lattice models for D2Q9, D3Q19, and D3Q27, a strict Structure of Arrays (SoA) memory layout, and a double-population ping-pong update scheme for fused collision and streaming.

The current solver targets weakly compressible isothermal Navier-Stokes simulations using the BGK collision operator. Its design emphasizes readable high-performance C++ while avoiding runtime polymorphism in performance-critical paths.

## Key Features

- Compile-time lattice traits for D2Q9, D3Q19, and D3Q27.
- Zero-overhead templated physics kernels constrained with C++23 concepts.
- Host-side SoA memory layout using `std::mdspan` views.
- Ping-pong population buffers for race-free fused collision-streaming.
- CPU reference backend with periodic pull streaming.
- CUDA backend using raw device population buffers and configurable launch blocks.
- Precision templating with `float` or `double`.
- Stateless `__host__ __device__` mathematical kernels shared by CPU and GPU paths.
- Integral diagnostics and legacy ASCII VTK output for ParaView visualization.
- GoogleTest validation, including mass conservation and Taylor-Green vortex decay.

## Prerequisites

Recommended toolchain:

- CMake 3.24 or newer.
- C++23-compatible host compiler, such as GCC 13+ or a recent Clang.
- CUDA Toolkit 12.2 or newer for GPU builds.
- CUDA-capable NVIDIA GPU for GPU execution.
- Eigen3.
- GoogleTest for validation tests.

The project uses CMake targets for Eigen, GoogleTest, and, when enabled, the CUDA runtime. On HPC systems, these dependencies are typically provided through environment modules.

## Compilation & Installation

Configure and build out of source:

```bash
git clone <repository-url> LB-Cube
cd LB-Cube

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### CPU-Only Build Without `nvcc`

If the machine does not provide the NVIDIA CUDA compiler (`nvcc`) or CUDA Toolkit, disable the CUDA backend at configure time:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DLB_CUBE_ENABLE_CUDA=OFF

cmake --build build-cpu -j
```

This builds the CPU executable and validation tests without including `cuda_runtime.h`, compiling `.cu` files, or requiring `nvcc`.

Run the CPU-only executable with `-cpu`:

```bash
./build-cpu/lbm_sim -nx 64 -ny 64 -nz 64 -steps 1000 -out_freq 100 -cpu
```

In a CPU-only build, the `-gpu` runtime option is unavailable and will report an error.

Run the validation tests:

```bash
ctest --test-dir build --output-on-failure
```

If CUDA or dependencies are installed in non-standard locations, provide CMake hints:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT=/path/to/cuda \
  -DEigen3_DIR=/path/to/eigen/cmake \
  -DGTest_DIR=/path/to/gtest/cmake
```

The default CUDA architecture setting is `native`, which asks CMake to target the GPU architecture available on the build node. For cross-compilation or login-node builds, specify the architecture explicitly:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80
```

## Running a Simulation

The main executable is `lbm_sim`. It currently runs 3D D3Q19 simulations.

Command-line options:

- `-nx <N>`: number of grid points in the x direction. Default: `64`.
- `-ny <N>`: number of grid points in the y direction. Default: `64`.
- `-nz <N>`: number of grid points in the z direction. Default: `64`.
- `-steps <N>`: total number of time steps. Default: `1000`.
- `-out_freq <N>`: output frequency in time steps. Default: `100`.
- `-init <name>`: initial condition. Options are `tgv` and `rest`. Default: `tgv`.
- `-cpu`: use the CPU backend.
- `-gpu`: use the CUDA backend. This is the default.

Example GPU run:

```bash
./build/lbm_sim -nx 128 -ny 128 -nz 128 -steps 5000 -out_freq 250 -gpu
```

Example CPU run:

```bash
./build/lbm_sim -nx 64 -ny 64 -nz 64 -steps 1000 -out_freq 100 -cpu
```

The default `tgv` initial condition is a small-amplitude Taylor-Green-style vortex, so the diagnostic kinetic energy and velocity field are non-zero. Use `-init rest` only for a uniform equilibrium sanity check; in that case `total_kinetic_energy` and `max_velocity_magnitude` will remain zero and ParaView will show a uniform field.

At startup, the executable prints the grid size, selected backend, relaxation parameter, and requested number of steps. At shutdown, it reports elapsed wall time and throughput in MLUPS, meaning million lattice updates per second.

## HPC Cluster Execution

On SGE-based clusters, build the code in a persistent project or software directory, then run simulations from node-local scratch storage when possible. Large VTK output should be written to scratch during execution and copied back to archive storage at the end of the job.

Example `qsub` script:

```bash
#!/bin/bash
#$ -N lb_cube_d3q19
#$ -cwd
#$ -pe smp 8
#$ -l h_rt=04:00:00
#$ -l gpu=1
#$ -j y
#$ -o lb_cube.$JOB_ID.log

set -euo pipefail

module purge
module load gcc/13
module load cuda/12.2
module load cmake/3.24

PROJECT_DIR="$HOME/LB-Cube"
ARCHIVE_DIR="$PROJECT_DIR/runs/$JOB_ID"
SCRATCH_DIR="${TMPDIR:-/tmp}/lb_cube_$JOB_ID"

mkdir -p "$SCRATCH_DIR" "$ARCHIVE_DIR"
cd "$SCRATCH_DIR"

"$PROJECT_DIR/build/lbm_sim" \
  -nx 128 -ny 128 -nz 128 \
  -steps 5000 \
  -out_freq 250 \
  -gpu

cp diagnostics.csv lbm_*.vtk "$ARCHIVE_DIR"/
```

Adjust resource requests, module names, and CUDA architecture settings to match the local cluster configuration.

## Outputs & Visualization

LB-Cube writes two primary output types:

- `diagnostics.csv`: integral quantities over time, including total mass, total kinetic energy, and maximum velocity magnitude.
- `lbm_XXXXXX.vtk`: legacy ASCII VTK files containing density, velocity magnitude, and velocity vector fields.

The VTK files use the `STRUCTURED_POINTS` dataset format and can be opened directly in ParaView:

```bash
paraview lbm_000000.vtk
```

For time-series visualization, open the generated `lbm_*.vtk` sequence in ParaView and enable file-series loading.

If the field appears blank, check which array ParaView is coloring by. Uniform rest-state density has no visible contrast. For vortex runs, color by `velocity_magnitude` or add glyphs/streamlines using the `velocity` vector field.

## Repository Structure

```text
include/
  lattice_traits.hpp    Compile-time lattice definitions
  lattice_memory.hpp    SoA host memory and macroscopic state
  lattice_physics.hpp   Stateless LBM math kernels
  lattice_core.hpp      CPU fused collision-streaming loop
  lattice_cuda.cuh      CUDA kernel and launcher declarations
  lattice_io.hpp        Diagnostics and VTK output

src/
  lattice_cuda.cu       CUDA backend implementation
  main.cpp              Simulation executable

tests/
  test_validation.cpp   GoogleTest validation suite
```

## Citation / Acknowledgments

If you use LB-Cube in academic work, please cite the software repository and the associated publication once available.

Suggested placeholder citation:

```text
LB-Cube: A Modern C++23/CUDA Lattice Boltzmann Solver.
CRECK Modeling Lab, Politecnico di Milano.
Repository: <repository-url>
```

This software originated within the CRECK Modeling Lab at Politecnico di Milano as a research-oriented platform for high-performance LBM development.

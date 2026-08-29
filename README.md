# LB-Cube

![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)
![CUDA](https://img.shields.io/badge/CUDA-optional-green)
![CMake](https://img.shields.io/badge/CMake-3.24%2B-064F8C)
![GoogleTest](https://img.shields.io/badge/tests-GoogleTest-informational)

LB-Cube is a modern C++23/CUDA Lattice Boltzmann Method (LBM) research solver
for direct numerical simulation of incompressible and weakly compressible flows,
with an emphasis on turbulent reactive mixing. The codebase uses compile-time
lattice traits, strict Structure-of-Arrays storage, ping-pong population buffers,
and stateless physics kernels shared by the CPU and CUDA backends.

The solver currently supports fluid dynamics with D2Q9 and D3Q19 lattices,
passive and reactive scalar transport with D2Q5 and D3Q7 lattices, and
BGK, TRT, and MRT collision operators. The design avoids virtual dispatch and
runtime polymorphism in hot paths, keeping the numerical kernels suitable for
high-performance CPU and GPU execution.

## Features

- **Fluid lattices:** D2Q9, D3Q19, and D3Q27 compile-time lattice traits.
- **Scalar lattices:** D2Q5 and D3Q7 for advection-diffusion-reaction transport.
- **Collision operators:** BGK, TRT, and MRT, selected at compile time.
- **Reactive mixing:** fused two-species `A + B -> C` scalar reaction step.
- **Modern C++:** C++23 concepts, templates, `std::mdspan`, and type-safe traits.
- **CUDA acceleration:** optional GPU backend with raw device buffers and fused kernels.
- **Precision templating:** `float` and `double` execution paths through `Real`.
- **SoA memory layout:** population-major storage with contiguous spatial domains for coalesced access.
- **Ping-pong updates:** two population buffers per field for race-free streaming.
- **On-the-fly statistics:** scalar means, variances, covariance, segregation intensity, and reaction rates.
- **Visualization output:** legacy ASCII VTK files readable directly by ParaView.
- **Validation suite:** GoogleTest coverage for transformations, physics decay, convergence, and ADR kinetics.

## Repository Layout

```text
include/
  lattice_traits.hpp    Compile-time lattice definitions and opposite maps
  lattice_memory.hpp    SoA host memory and macroscopic state types
  lattice_physics.hpp   Stateless BGK/TRT/scalar physics functions
  lattice_mrt.hpp       D2Q9 and D3Q19 MRT transformations and collision
  lattice_core.hpp      CPU fused collision-streaming and ADR loops
  lattice_cuda.cuh      CUDA kernel declarations and launchers
  lattice_io.hpp        Diagnostics, statistics, and VTK output

src/
  main.cpp              3D fluid simulation executable
  simulate_reaction.cpp 2D reactive mixing executable
  compare_operators.cpp BGK/TRT/MRT reactive-mixing comparison executable
  lattice_cuda.cu       CUDA backend implementation

tests/
  test_validation.cpp   GoogleTest validation and convergence suite
```

## Prerequisites

Required for all builds:

- CMake 3.24 or newer.
- A C++23-capable compiler, such as GCC 13+, Clang 16+, or a recent AppleClang.
- Eigen3.
- GoogleTest.

Required only for CUDA builds:

- NVIDIA CUDA Toolkit, recommended 12.2 or newer.
- `nvcc` available on `PATH`, or `CUDAToolkit_ROOT` provided to CMake.
- A CUDA-capable NVIDIA GPU for GPU execution.

On HPC systems, these dependencies are typically provided through environment
modules. The default CUDA architecture setting is `native`, which asks CMake to
compile for the GPU available on the build node.

## Build Instructions

### CUDA Build

Use this path on a workstation or cluster node with a working CUDA compiler:

```bash
git clone <repository-url> LB-Cube
cd LB-Cube

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLB_CUBE_ENABLE_CUDA=ON

cmake --build build -j
```

If CUDA or dependencies are installed in non-standard locations:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLB_CUBE_ENABLE_CUDA=ON \
  -DCUDAToolkit_ROOT=/path/to/cuda \
  -DEigen3_DIR=/path/to/eigen/cmake \
  -DGTest_DIR=/path/to/gtest/cmake
```

For login-node builds or cross-compilation, set the CUDA architecture explicitly:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLB_CUBE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
```

### CPU-Only Build Without `nvcc`

If the machine does not have the NVIDIA CUDA compiler, disable the CUDA backend:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DLB_CUBE_ENABLE_CUDA=OFF

cmake --build build-cpu -j
```

This builds the CPU executables and tests without compiling `.cu` files or
including `cuda_runtime.h`.

## Running Tests

Run the full GoogleTest validation suite through CTest:

```bash
ctest --test-dir build-cpu --output-on-failure
```

Or run the test executable directly:

```bash
./build-cpu/lbm_validation_tests
```

For CUDA-enabled builds, replace `build-cpu` with `build`:

```bash
ctest --test-dir build --output-on-failure
```

The validation suite includes:

- MRT transformation identity tests for D2Q9 and D3Q19.
- 2D shear-wave and Taylor-Green vortex decay tests.
- 3D angled shear-wave tests for BGK, TRT, and MRT.
- Spatial and temporal convergence checks.
- Passive scalar advection-diffusion validation.
- Batch reaction kinetics for `A + B -> C`.

## Running a Fluid Simulation

The main executable is `lbm_sim`, which currently runs D3Q19 fluid simulations.

```bash
./build/lbm_sim -nx 128 -ny 128 -nz 128 -steps 5000 -out_freq 250 -gpu
```

CPU-only example:

```bash
./build-cpu/lbm_sim -nx 64 -ny 64 -nz 64 -steps 1000 -out_freq 100 -cpu
```

Command-line options:

- `-nx <N>`: number of grid points in x. Default: `64`.
- `-ny <N>`: number of grid points in y. Default: `64`.
- `-nz <N>`: number of grid points in z. Default: `64`.
- `-steps <N>`: number of time steps. Default: `1000`.
- `-out_freq <N>`: VTK and diagnostics output frequency. Default: `100`.
- `-init <name>`: initial condition, currently `tgv` or `rest`.
- `-cpu`: use the CPU backend.
- `-gpu`: use the CUDA backend when available.

In a CPU-only build, requesting `-gpu` will fail with a clear runtime error.

## Running Reactive Mixing

The `lbm_reaction_sim` executable runs a 2D segregated reactive-mixing case with
a Taylor-Green vortex, species A initialized in the left half of the domain, and
species B initialized in the right half.

```bash
./build-cpu/lbm_reaction_sim -cpu
```

Useful options include:

- `-nx <N>` and `-ny <N>`: 2D domain size.
- `-steps <N>`: number of time steps.
- `-out_freq <N>`: output frequency.
- `-u0 <value>`: Taylor-Green velocity amplitude.
- `-tau <value>`: fluid relaxation time.
- `-tau_c <value>`: scalar relaxation time.
- `-k_react <value>`: second-order reaction-rate constant.
- `-cpu` or `-gpu`: backend selection, depending on build configuration.

## Running the Operator Benchmark

The `lbm_compare_operators` executable runs the same 256x256 reactive-mixing
case three times, once with each fluid collision operator:

- BGK
- TRT
- MRT

The scalar transport and reaction model are kept fixed, so differences in the
outputs reflect the fluid operator used to generate the mixing field.

Run it from the project root:

```bash
./build-cpu/lbm_compare_operators
```

or, for a CUDA-enabled build tree:

```bash
./build/lbm_compare_operators
```

The benchmark uses fixed settings:

- Grid: `256 x 256`
- Steps per operator: `5000`
- Reactive statistics frequency: every `10` steps
- VTK output frequency: every `500` steps
- Fluid lattice: D2Q9
- Scalar lattices: D2Q5 for species A and B
- Initial fluid field: Taylor-Green vortex with `U0 = 0.04`, `tau = 0.6`
- Reaction rate: `k_react = 0.05`

Outputs are written to separate directories:

```text
output_BGK/
  stats_BGK.csv
  comparison_BGK_000000.vtk
  comparison_BGK_000500.vtk
  ...

output_TRT/
  stats_TRT.csv
  comparison_TRT_000000.vtk
  comparison_TRT_000500.vtk
  ...

output_MRT/
  stats_MRT.csv
  comparison_MRT_000000.vtk
  comparison_MRT_000500.vtk
  ...
```

Each CSV file contains:

- `mean_A`, `mean_B`
- `var_A`, `var_B`
- `covariance`
- `segregation_intensity`
- `true_reaction_rate`
- `mixed_reaction_rate`

Each VTK snapshot contains:

- fluid `velocity`
- species concentration `C_A`
- species concentration `C_B`
- local `reaction_rate`

## Outputs and Visualization

LB-Cube writes lightweight CSV diagnostics for quantitative analysis and legacy
ASCII VTK files for visualization. The VTK files use the `STRUCTURED_POINTS`
dataset format and can be opened directly in ParaView:

```bash
paraview output_MRT/comparison_MRT_000500.vtk
```

For time-series visualization, open a numbered VTK sequence in ParaView and
enable file-series loading. For rest-state or weak-flow cases, color by velocity
magnitude, concentration, or reaction rate rather than uniform density.

## HPC Cluster Execution

For SGE-based clusters, build in a persistent project directory and run large
jobs from node-local scratch storage. Copy only the final CSV files and selected
VTK snapshots back to archive storage to avoid unnecessary parallel-filesystem
traffic.

Example `qsub` script:

```bash
#!/bin/bash
#$ -N lb_cube_compare
#$ -cwd
#$ -pe smp 16
#$ -l h_rt=08:00:00
#$ -j y
#$ -o lb_cube_compare.$JOB_ID.log

set -euo pipefail

module purge
module load gcc/13
module load cmake/3.24

PROJECT_DIR="$HOME/LB-Cube"
ARCHIVE_DIR="$PROJECT_DIR/runs/$JOB_ID"
SCRATCH_DIR="${TMPDIR:-/tmp}/lb_cube_compare_$JOB_ID"

mkdir -p "$SCRATCH_DIR" "$ARCHIVE_DIR"
cd "$SCRATCH_DIR"

"$PROJECT_DIR/build-cpu/lbm_compare_operators"

cp -r output_BGK output_TRT output_MRT "$ARCHIVE_DIR"/
```

For GPU fluid runs, load the CUDA module and build with
`-DLB_CUBE_ENABLE_CUDA=ON`.

## Development Notes

LB-Cube follows a strict separation of concerns:

- `lattice_traits.hpp` defines lattice constants and topology.
- `lattice_memory.hpp` owns population storage and multidimensional views.
- `lattice_physics.hpp` and `lattice_mrt.hpp` contain stateless mathematical kernels.
- `lattice_core.hpp` and `lattice_cuda.cuh` implement execution backends.
- `lattice_io.hpp` handles diagnostics and visualization output.

New kernels should preserve compile-time dispatch, avoid dynamic allocation in
inner loops, and keep host/device-compatible mathematical functions stateless.

## Citation and Acknowledgments

If you use LB-Cube in academic work, please cite the software repository and the
associated publication once available.

Suggested placeholder citation:

```text
LB-Cube: A Modern C++23/CUDA Lattice Boltzmann Solver for Reactive Mixing.
CRECK Modeling Lab, Politecnico di Milano.
Repository: <repository-url>
```

This software originated within the CRECK Modeling Lab at Politecnico di Milano
as a research-oriented platform for high-performance LBM development.

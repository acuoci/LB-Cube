# LB-Cube: A C++23 Lattice Boltzmann Framework for Transport Phenomena

![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)
![CMake](https://img.shields.io/badge/CMake-3.24%2B-064F8C)
![CUDA](https://img.shields.io/badge/CUDA-optional-green)
![GoogleTest](https://img.shields.io/badge/tests-GoogleTest-informational)

LB-Cube is a high-performance, dimension-agnostic Lattice Boltzmann framework
for transport phenomena, specializing in turbulent hydrodynamics and
high-Schmidt-number reactive scalar transport. The codebase targets
liquid-phase reactive mixing of the form `A + B -> C`, with explicit support for
2D and 3D hydrodynamic lattices, scalar advection-diffusion-reaction transport,
and production-oriented diagnostics for DNS and implicit-LES studies.

The framework is built around compile-time lattice traits, strict
Structure-of-Arrays population storage, ping-pong time integration, and
stateless single-cell physics kernels. This keeps hot loops free from virtual
dispatch while preserving a clean separation between lattice definitions,
memory layout, collision physics, core time-stepping, validation, and output.

## Core Features & Architecture

- **Modern C++23:** LB-Cube is built strictly around native `<mdspan>` for
  zero-cost multidimensional array views over flat population buffers. This
  exposes the same logical `[Q, Y, X]` or `[Q, Z, Y, X]` layout to CPU kernels,
  diagnostics, and future GPU-oriented memory abstractions.
- **Hydrodynamics:** Fluid models include `D2Q9`, `D3Q19`, and `D3Q27`.
  Collision operators include BGK, TRT, MRT, and Regularized LBM (RLBM), with
  compile-time dispatch through `CollisionType`. The RLBM path is used for
  robust high-Reynolds-number turbulent flows where implicit-LES-like stability
  is required.
- **Reactive Scalar Transport:** Scalar advection-diffusion uses compact
  `D2Q5` and `D3Q7` lattices coupled to the local fluid velocity reconstructed
  from the hydrodynamic populations. Multiple species are represented as
  independent `LatticeMemory` fields, preserving simple SoA access and avoiding
  hidden interleaving.
- **High-Schmidt Stability:** Reactive scalar transport includes a decoupled
  max-dissipation TRT-style scalar collision. It discards even non-equilibrium
  scalar modes and relaxes only the odd component with the physical scalar rate,
  suppressing high-frequency checkerboard/Gibbs oscillations at `Sc > 1`.
- **Robust Chemistry:** The `A + B -> C` reaction source is integrated with an
  explicit positivity-preserving exact batch analytical update at each lattice
  node. This avoids stiff ODE solvers while retaining a fused, allocation-free
  reaction step suitable for production mixing simulations.
- **Diagnostics and Output:** The production executables provide real-time
  telemetry, CSV statistics, scalar means and variances, true versus mixed
  reaction rates, min/max bounds, and legacy VTK output for ParaView.

## Compiler Requirements

LB-Cube strictly requires **C++23**.

This is not a cosmetic requirement: the framework relies on the final standard
`<mdspan>` header. When using GCC, use **GCC 16 or newer**. Equivalent Clang,
AppleClang, or MSVC versions are acceptable only if they provide conforming
C++23 `<mdspan>` support.

Required dependencies:

- CMake 3.24 or newer
- C++23 compiler with native `<mdspan>`
- Eigen3
- GoogleTest

Optional CUDA backend requirements:

- NVIDIA CUDA Toolkit
- `nvcc` available on `PATH`
- CUDA-capable NVIDIA GPU

CPU-only builds are fully supported and do not require `nvcc`.

## Build Instructions

### CPU Build

Use this configuration on machines without CUDA or when validating the CPU
reference backend:

```bash
mkdir build-cpu
cd build-cpu
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=23 -DLB_CUBE_ENABLE_CUDA=OFF
cmake --build . -j 4
```

### CUDA Build

Use this configuration on a node with a working CUDA compiler:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=23 -DLB_CUBE_ENABLE_CUDA=ON
cmake --build . -j 4
```

By default, CMake uses `CMAKE_CUDA_ARCHITECTURES=native`, which targets the GPU
available on the build node. On clusters where compilation happens away from the
compute GPU, set the architecture explicitly:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DLB_CUBE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
```

## Flagship Executable

The flagship production executable is:

```bash
./build-cpu/lbm_turbulent_reactive_shear_3d
```

It runs a 3D liquid-phase turbulent reactive shear layer using:

- fluid lattice: `D3Q27`
- fluid collision: `CollisionType::RLBM`
- scalar lattices: two `D3Q7` reactant fields
- chemistry: fused `A + B -> C` exact local batch update
- scalar stabilization: max-dissipation decoupled scalar collision

Example:

```bash
./build-cpu/lbm_turbulent_reactive_shear_3d \
  --Nx 128 --Ny 128 --Nz 128 \
  --tau_f 0.505 \
  --tau_s 0.5005 \
  --U0 0.05 \
  --k_react 0.1 \
  --steps 10000 \
  --stat_freq 10 \
  --screen_freq 100 \
  --vtk_freq 1000 \
  --vtk_burst_length 1000 \
  --vtk_burst_freq 100
```

Supported command-line parameters include:

- `--Nx`, `--Ny`, `--Nz`: grid dimensions
- `--tau_f`: fluid relaxation time
- `--tau_s`: scalar relaxation time
- `--U0`: shear velocity amplitude
- `--k_react`: second-order reaction-rate constant
- `--steps`: total simulation steps
- `--stat_freq`: CSV statistics frequency
- `--screen_freq`: console telemetry frequency
- `--vtk_freq`: regular binary VTK output frequency
- `--vtk_burst_length`: duration of high-frequency burst output
- `--vtk_burst_freq`: VTK interval during burst mode

At startup, the executable prints a full parameter recap including:

- kinematic viscosity: `nu = (tau_f - 0.5) / 3`
- scalar diffusivity for `D3Q7`: `D = (tau_s - 0.5) / 4`
- Schmidt number: `Sc = nu / D`
- Damkohler number: `Da = k_react * Ny / U0`

### Flagship Outputs

The executable writes high-frequency scalar statistics to:

```text
statistics_shear_3d.csv
```

The CSV includes:

```text
step,time,u_max,E_k,dissipation_rate,
mean_Ca,var_Ca,min_Ca,max_Ca,
mean_Cb,var_Cb,mean_Cc,var_Cc,
rate_true,rate_mixed
```

`rate_true = <k C_A C_B>` captures the actual segregated reaction rate, while
`rate_mixed = k <C_A><C_B>` gives the perfectly mixed reference rate. Their
separation is intended for intensity-of-segregation analysis.

Binary legacy VTK files are written to:

```text
vtk_shear_3d/
```

VTK output contains:

- velocity vector
- `C_A`
- `C_B`
- reconstructed product `C_C = 0.5 * (1 - C_A - C_B)`

The VTK system supports dual-frequency output:

- regular output every `--vtk_freq` steps
- burst-mode output every `--vtk_burst_freq` steps after the kinetic energy
  drops below 95% of its initial value

## Other Executables

LB-Cube also builds several focused tools:

- `lbm_sim`: baseline fluid simulation driver
- `lbm_reaction_sim`: 2D reactive mixing executable
- `lbm_compare_operators`: BGK/TRT/MRT/RLBM comparison on a shared 2D reactive case
- `lbm_turbulent_shear_layer`: 2D high-Re doubly periodic shear-layer gauntlet
- `lbm_turbulent_tgv_3d`: 3D turbulent Taylor-Green vortex stability benchmark

## Testing

Build the validation target:

```bash
cmake --build build-cpu --target lbm_validation_tests -j 4
```

Run the full GoogleTest suite through CTest:

```bash
ctest --test-dir build-cpu --output-on-failure
```

Or run the test executable directly:

```bash
./build-cpu/lbm_validation_tests
```

The `lbm_validation_tests` target covers:

- mass conservation
- MRT transformation identity checks
- 2D shear-wave decay
- 2D Taylor-Green vortex decay
- 3D Taylor-Green and angled shear-wave validation
- spatial and temporal convergence studies
- passive scalar advection-diffusion
- 2D and 3D reactive mixing with `D2Q5` and `D3Q7`
- exact `A + B -> C` batch reaction kinetics

## Notes for HPC Runs

For cluster execution, prefer out-of-source builds and run simulations from
scratch storage. Keep large VTK output on scratch during the job and archive only
selected snapshots, CSV statistics, logs, and configuration files after
completion. The production executable flushes console telemetry so schedulers
capture progress promptly.

## Citation and Acknowledgments

Citation metadata will be added before publication. Please cite the repository
and associated paper/preprint when available.

LB-Cube originates from research software development for transport phenomena
and reactive mixing studies within the CRECK Modeling Lab at Politecnico di
Milano.

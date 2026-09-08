# LB-Cube: A C++23 Lattice Boltzmann Framework for Transport Phenomena

![C++23](https://img.shields.io/badge/C%2B%2B-23-blue)
![CMake](https://img.shields.io/badge/CMake-3.24%2B-064F8C)
![OpenMP](https://img.shields.io/badge/OpenMP-optional-green)
![CUDA](https://img.shields.io/badge/CUDA-optional-green)
![FFTW](https://img.shields.io/badge/FFTW-optional-informational)
![GoogleTest](https://img.shields.io/badge/tests-GoogleTest-informational)

LB-Cube is a high-performance, dimension-agnostic Lattice Boltzmann framework for turbulent hydrodynamics and liquid-phase reactive scalar transport. It targets direct numerical simulation and implicit-LES-style studies of transport phenomena, with special attention to high-Schmidt-number mixing and the irreversible reaction

```text
A + B -> C
```

The code is built around compile-time lattice traits, structure-of-arrays population storage, out-of-place streaming, and stateless single-cell collision kernels. This keeps the hot loops free of virtual dispatch while preserving a clean separation between lattice definitions, memory layout, physics, time integration, validation, and production diagnostics.

LB-Cube provides general 2D/3D LBM infrastructure, but its current main scientific application is `lbm_turbulent_reactive_double_shear_3d`, a production driver for triply periodic turbulent reactive double-shear-layer simulations.

## Current Capabilities

**Hydrodynamics**

- Fluid lattices: `D2Q9`, `D3Q19`, `D3Q27`
- Collision operators: BGK, TRT, MRT, RLBM
- CPU stepping through compile-time `CollisionType` dispatch
- Optional CUDA backend for selected lower-level executables and kernels
- OpenMP acceleration for CPU physics, statistics, initialization, and production diagnostics

**Scalar Transport and Chemistry**

- Scalar lattices: `D2Q5`, `D3Q7`
- Multi-species transport using independent scalar `LatticeMemory` fields
- Velocity-coupled scalar equilibria for advective transport
- High-Schmidt scalar stabilization through a max-dissipation decoupled TRT-style scalar collision
- Explicit, local, positivity-preserving analytical update for `A + B -> C`
- Product `C` reconstructed diagnostically from stoichiometry rather than transported as a third scalar field

**Production Diagnostics**

- High-frequency global hydrodynamic, scalar, reaction, conservation, and resolution diagnostics
- Plane-averaged y-profile output
- Full-domain PDFs, joint PDFs, and conditional scalar statistics
- FFTW-based x-z plane scalar and velocity spectra, directional spectra, radial spectra, and A-B covariance co-spectra
- Multi-width a-priori box-filter diagnostics using the optimized separable periodic 3D box filter
- Filtered SGS scalar moments, filtered conditional means, SGS PDFs, and LES reaction metrics
- Binary legacy VTK output for ParaView
- JSON metadata for reproducibility

## Requirements

LB-Cube requires C++23.

The project uses an mdspan compatibility layer:

- Native C++23 `<mdspan>` is used by default.
- Kokkos standalone mdspan can be selected with `-DUSE_KOKKOS_MDSPAN=ON`.

For native `<mdspan>`, use a compiler that implements the final C++23 mdspan standard. With GCC, this generally means GCC 16 or newer. On clusters constrained to older host compilers, such as GCC 12.5 with `nvcc` 12.6, use the Kokkos standalone mdspan path.

Required build dependencies for the current CMake configuration:

- CMake 3.24 or newer
- C++23 compiler
- Eigen3
- GoogleTest

Optional dependencies:

- OpenMP for CPU parallelism
- FFTW3 double precision for spectral output
- CUDA Toolkit for the optional CUDA backend
- Kokkos standalone mdspan for non-native mdspan compilers

## Build

### CPU Release Build

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DLB_CUBE_ENABLE_CUDA=OFF

cmake --build build-cpu -j 4
```

### CPU Build With FFTW Spectra

Spectral diagnostics in `lbm_turbulent_reactive_double_shear_3d` require FFTW at configure time:

```bash
cmake -S . -B build-fftw \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DLB_CUBE_ENABLE_CUDA=OFF \
  -DLB_CUBE_ENABLE_FFTW=ON

cmake --build build-fftw -j 4
```

If FFTW is installed in a nonstandard location, provide either a prefix:

```bash
cmake -S . -B build-fftw \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DLB_CUBE_ENABLE_CUDA=OFF \
  -DLB_CUBE_ENABLE_FFTW=ON \
  -DLB_CUBE_FFTW_ROOT=/path/to/fftw
```

or explicit include/library paths:

```bash
cmake -S . -B build-fftw \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DLB_CUBE_ENABLE_CUDA=OFF \
  -DLB_CUBE_ENABLE_FFTW=ON \
  -DLB_CUBE_FFTW_INCLUDE_DIR=/path/to/fftw/include \
  -DLB_CUBE_FFTW_LIBRARY=/path/to/fftw/lib/libfftw3.so
```

Use `libfftw3.a` instead of `libfftw3.so` if your cluster provides only static libraries.

### OpenMP

OpenMP is enabled by default when CMake finds it:

```bash
cmake -S . -B build-omp \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DLB_CUBE_ENABLE_OPENMP=ON \
  -DLB_CUBE_ENABLE_CUDA=OFF
```

CMake prints whether OpenMP was found and activated. If auto-detection fails, manual settings can be supplied:

```bash
cmake -S . -B build-omp \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DLB_CUBE_ENABLE_OPENMP=ON \
  -DLB_CUBE_OPENMP_CXX_FLAGS="-fopenmp" \
  -DLB_CUBE_OPENMP_LINK_FLAGS="-fopenmp" \
  -DLB_CUBE_OPENMP_LIBRARIES="/path/to/libgomp.so"
```

Recommended runtime settings on a CPU node:

```bash
export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

For NUMA-heavy runs, launch through the site scheduler or `numactl` according to local policy.

### Kokkos mdspan Compatibility

Use this when the compiler does not provide native C++23 `<mdspan>`:

```bash
cmake -S . -B build-kokkos-mdspan \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DLB_CUBE_ENABLE_CUDA=OFF \
  -DUSE_KOKKOS_MDSPAN=ON
```

CMake uses `find_package(mdspan REQUIRED)` and links the `std::mdspan` target provided by the standalone package.

### CUDA Backend

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DLB_CUBE_ENABLE_CUDA=ON

cmake --build build-cuda -j 4
```

By default, CMake uses `CMAKE_CUDA_ARCHITECTURES=native`. On clusters where compilation happens away from the target GPU, set the architecture explicitly:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=23 \
  -DLB_CUBE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
```

## Testing

Run all registered tests:

```bash
ctest --test-dir build-cpu --output-on-failure
```

Important test targets include:

- `lbm_validation_tests`
- `lbm_double_shear_parameter_tests`
- `lbm_box_filter_tests`
- `lbm_fftw_test` when `LB_CUBE_ENABLE_FFTW=ON`

Direct examples:

```bash
./build-cpu/lbm_validation_tests
./build-cpu/lbm_double_shear_parameter_tests
./build-cpu/lbm_box_filter_tests
```

The validation suite covers fluid decay tests, MRT transformation identities, 3D angled shear waves, scalar advection-diffusion, reactive mixing, double-shear parameter conversion, and periodic box-filter correctness.

## Executables

The build currently defines:

| Target | Purpose |
| --- | --- |
| `lbm_sim` | Baseline fluid simulation driver |
| `lbm_reaction_sim` | 2D reactive mixing driver |
| `lbm_compare_operators` | BGK/TRT/MRT/RLBM comparison on a shared 2D reactive case |
| `lbm_turbulent_shear_layer` | 2D high-Reynolds doubly periodic shear-layer gauntlet |
| `lbm_turbulent_tgv_3d` | 3D turbulent Taylor-Green vortex stability benchmark |
| `lbm_turbulent_reactive_shear_3d` | Earlier 3D reactive shear-layer driver |
| `lbm_turbulent_reactive_double_shear_3d` | Current production 3D turbulent reactive double-shear-layer executable |

## Main Solver: turbulent_reactive_double_shear_3d

The main production executable is:

```bash
./build-cpu/lbm_turbulent_reactive_double_shear_3d
```

It simulates a triply periodic 3D turbulent reactive double shear layer using:

- hydrodynamic lattice: `D3Q27`
- hydrodynamic collision: `CollisionType::RLBM`
- scalar lattice: two `D3Q7` memories for `A` and `B`
- chemistry: local analytical `A + B -> C` update
- scalar stabilization: max-dissipation scalar collision
- diagnostics: CSV, VTK, profiles, PDFs, spectra, filtering, and metadata

### Physical Problem

The solver initializes a canonical 3D double shear layer in a triply periodic box. The central stream and outer stream move in opposite streamwise directions, with two smooth interfaces at `Ny/4` and `3*Ny/4`. Two miscible reactants are initialized on complementary sides of the same smooth interface structure and react through `A + B -> C`.

The velocity perturbation is applied only at initialization. After that, the flow, scalar transport, and reaction evolve freely.

### Governing Parameters

The principal physical parameters are:

```text
Nx, Ny, Nz      grid dimensions
U0              half velocity difference scale
C0              reference reactant concentration
delta_ratio     initial shear thickness divided by Ny
Re_delta        shear-layer Reynolds number
Sc              Schmidt number
Da_delta        shear-layer Damkohler number
```

Derived quantities printed at startup include:

```text
delta0    = delta_ratio * Ny
DeltaU    = 2 * U0
nu        = (tau_f - 0.5) / 3
D         = (tau_s - 0.5) / 4
tau_delta = delta0 / DeltaU
tau_chem  = 1 / (k_react * C0), when k_react > 0
```

### Initial Condition

The two shear layers are located at:

```text
y1 = Ny / 4
y2 = 3 Ny / 4
delta0 = delta_ratio * Ny
```

The base profile is:

```text
S(y) = tanh((y - y1) / delta0) - tanh((y - y2) / delta0) - 1
```

Velocity:

```text
u_x = U0 * S(y) + u'_x
u_y = u'_y
u_z = u'_z
rho = 1
```

The perturbation `u'` is optional and deterministic. It is generated from the curl of a localized random vector potential, uses normalized periodic coordinates, low integer Fourier modes, a fixed seed, and an RMS normalization relative to:

```text
DeltaU = 2 * U0
```

Scalars are not perturbed:

```text
C_A = 0.5 * C0 * (1 + S(y))
C_B = 0.5 * C0 * (1 - S(y))
C_C = 0.5 * (C0 - C_A - C_B)
Z   = 0.5 * (1 + (C_A - C_B) / C0)
```

### Reaction Model

The reaction source is treated locally inside the fused scalar reaction step. The code applies the exact analytical batch update for the equal-stoichiometry reaction `A + B -> C`, with positivity safeguards for the reactant concentrations used to compute the source term.

The global diagnostic reaction rates are:

```text
rate_true  = < k_react * C_A * C_B >
rate_mixed = k_react * <C_A> * <C_B>
```

The ratio `reaction_efficiency = rate_true / rate_mixed` measures the loss of reaction rate due to segregation when the denominator is positive.

### Perturbation

The optional perturbation is deterministic, smooth, low-wavenumber, localized near the two shear layers, and divergence-free by construction. It is generated from the curl of a localized random vector potential:

```text
u' = curl(A)
```

The Fourier modes use normalized periodic coordinates and integer mode numbers, so the same seed and mode band represent the same nondimensional field across grid refinements. The final perturbation is mean-subtracted and normalized with:

```text
sqrt(<u'_x^2 + u'_y^2 + u'_z^2>) = perturb_amplitude * DeltaU
```

The scalar fields are not perturbed.

### Parameterization Modes

The executable supports two mutually exclusive modes.

**Direct LBM mode**

Use `--tau_f`, `--tau_s`, and `--k_react`. This is the default when no physical nondimensional group is explicitly provided.

```text
nu = (tau_f - 0.5) / 3
D  = (tau_s - 0.5) / 4
```

**Physical mode**

Use all three of `--Re_delta`, `--Sc`, and `--Da_delta`. The executable then computes the LBM parameters:

```text
delta0  = delta_ratio * Ny
DeltaU  = 2 * U0
nu      = DeltaU * delta0 / Re_delta
D       = nu / Sc
k_react = Da_delta * DeltaU / (C0 * delta0)
tau_f   = 0.5 + 3 * nu
tau_s   = 0.5 + 4 * D
```

The reported nondimensional groups are:

```text
Re_delta = DeltaU * delta0 / nu
Sc       = nu / D
Da_delta = k_react * C0 * delta0 / DeltaU
```

The mode selection is strict:

- Supplying none of `--Re_delta`, `--Sc`, `--Da_delta` selects direct mode.
- Supplying any one of them requires all three.
- Explicitly mixing physical mode with explicit `--tau_f`, `--tau_s`, or `--k_react` is rejected.

### Main CLI Options

Use `--help` to print the executable's command-line reference.

| Option | Default | Meaning |
| --- | ---: | --- |
| `--Nx` | `128` | Grid nodes in x |
| `--Ny` | `128` | Grid nodes in y |
| `--Nz` | `128` | Grid nodes in z |
| `--tau_f` | `0.505` | Direct-mode fluid relaxation time |
| `--tau_s` | `0.5005` | Direct-mode scalar relaxation time |
| `--k_react` | `0.1` | Direct-mode reaction-rate constant |
| `--Re_delta` | none | Physical-mode shear-layer Reynolds number |
| `--Sc` | none | Physical-mode Schmidt number |
| `--Da_delta` | none | Physical-mode Damkohler number |
| `--U0` | `0.05` | Shear velocity amplitude |
| `--C0` | `1.0` | Reference reactant concentration |
| `--delta_ratio` | `0.0625` | Initial shear thickness divided by `Ny` |
| `--steps` | `10000` | Total LBM steps |
| `--stat_freq` | `10` | Global statistics CSV interval |
| `--screen_freq` | `100` | Console telemetry interval |
| `--vtk_freq` | `1000` | VTK interval; `0` disables VTK |
| `--profile_freq` | `0` | y-profile interval; `0` disables profiles |
| `--pdf_freq` | `0` | PDF and conditional-statistics interval; `0` disables PDFs |
| `--spectrum_freq` | `0` | Spectral-output interval; `0` disables spectra |
| `--filter_freq` | `0` | A-priori filter interval; `0` disables filtering |
| `--checkpoint_freq` | `0` | Checkpoint interval in completed steps; `0` disables step-based checkpointing |
| `--checkpoint_walltime` | `0` | Approximate wall-clock checkpoint interval in hours; `0` disables wall-clock checkpointing |
| `--checkpoint_keep` | `2` | Number of completed checkpoint files retained |
| `--checkpoint_assumed_bandwidth` | `250` | Initial checkpoint write-bandwidth estimate in MiB/s for graceful walltime planning |
| `--max_walltime` | `0` | Gracefully checkpoint and exit before this elapsed wall time; `0` disables |
| `--walltime_safety_margin` | `0.25` | Reserved time, in hours, before `--max_walltime` for final checkpoint writing |
| `--restart_from` | none | Binary checkpoint file to restart from |

### Perturbation Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--perturb_amplitude` | `0.02` | Target perturbation RMS divided by `DeltaU` |
| `--perturb_seed` | `12345` | Deterministic random seed |
| `--perturb_kmin` | `1` | Minimum integer Fourier mode norm |
| `--perturb_kmax` | `4` | Maximum integer Fourier mode norm |
| `--perturb_width` | `2.0` | Localization width divided by `delta0` |

Set `--perturb_amplitude 0` to recover the unperturbed y-only double-shear initialization.

### PDF Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--pdf_bins` | `128` | Marginal PDF bin count |
| `--joint_pdf_bins` | `128` | Joint PDF bin count per dimension |
| `--pdf_log_chi_min` | `-12` | Minimum `log10(chi*)` bin edge |
| `--pdf_log_chi_max` | `2` | Maximum `log10(chi*)` bin edge |
| `--pdf_log_R_min` | `-12` | Minimum `log10(R*)` bin edge |
| `--pdf_log_R_max` | `2` | Maximum `log10(R*)` bin edge |

Dimensionless variables:

```text
a        = C_A / C0
b        = C_B / C0
chi_star = chi_Z * delta0 / DeltaU
R_star   = k_react * C_A * C_B * delta0 / (C0 * DeltaU)
```

### Filtering Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--filter_widths` | disabled | Comma-separated positive odd widths, for example `1,3,5,9` |
| `--filter_width` | disabled | Backward-compatible single-width alias |
| `--filter_freq` | `0` | Filter interval |
| `--filter_conditional_bins` | `64` | Filtered conditional bin count |
| `--filter_log_tauZZ_min` | `-12` | Minimum `log10(tau_ZZ)` conditional edge |
| `--filter_log_tauZZ_max` | `0` | Maximum `log10(tau_ZZ)` conditional edge |
| `--filter_pdf_bins` | `128` | SGS marginal PDF bin count |
| `--filter_joint_pdf_bins` | `128` | SGS joint PDF bin count per dimension |
| `--filter_tauAB_min` | `-0.5` | Minimum `tau_AB` PDF bin edge |
| `--filter_tauAB_max` | `0.5` | Maximum `tau_AB` PDF bin edge |

Filtering is enabled only when a non-empty width list and `--filter_freq > 0` are both provided. Supplying only one side is a configuration error. Widths must be positive odd integers, are sorted internally, and are processed sequentially so peak memory does not scale with the number of requested widths.

The production path uses the optimized separable periodic 3D box filter:

```text
box_filter_3d_separable
```

The naive filter remains in the codebase as a reference implementation for tests.

### Running the Solver

The executable writes outputs in the current working directory. For production runs, launch from a dedicated run directory so CSV, metadata, VTK, profile, PDF, spectrum, and filtering outputs remain grouped with the job.

Checkpointing is optional. `--steps` always means the absolute final simulation step, including after restart. For example, restarting from `checkpoint_00012000.bin` with `--steps 28800` continues from completed step 12000 through step 28800.

#### Direct Mode Example

```bash
./build-cpu/lbm_turbulent_reactive_double_shear_3d \
  --Nx 128 --Ny 128 --Nz 128 \
  --U0 0.05 \
  --C0 1.0 \
  --delta_ratio 0.0625 \
  --tau_f 0.506 \
  --tau_s 0.502 \
  --k_react 0.01 \
  --steps 100 \
  --stat_freq 10 \
  --screen_freq 10 \
  --vtk_freq 100
```

#### Physical Mode Example

```bash
./build-cpu/lbm_turbulent_reactive_double_shear_3d \
  --Nx 256 --Ny 256 --Nz 256 \
  --U0 0.05 \
  --C0 1.0 \
  --delta_ratio 0.0625 \
  --Re_delta 400 \
  --Sc 10 \
  --Da_delta 10 \
  --steps 12000 \
  --stat_freq 10 \
  --screen_freq 100 \
  --vtk_freq 3000
```

For this case:

```text
delta0  = 16
DeltaU  = 0.1
nu      = 0.004
D       = 0.0004
tau_f   = 0.512
tau_s   = 0.5016
k_react = 0.0625
```

#### Full Diagnostics Example

Build with FFTW before enabling `--spectrum_freq`.

```bash
./build-fftw/lbm_turbulent_reactive_double_shear_3d \
  --Nx 128 --Ny 128 --Nz 128 \
  --U0 0.05 \
  --C0 1.0 \
  --delta_ratio 0.0625 \
  --Re_delta 400 \
  --Sc 4 \
  --Da_delta 1 \
  --perturb_amplitude 0.02 \
  --perturb_seed 12345 \
  --steps 1000 \
  --stat_freq 10 \
  --screen_freq 100 \
  --vtk_freq 500 \
  --profile_freq 100 \
  --pdf_freq 250 \
  --spectrum_freq 250 \
  --filter_widths 1,3,5,9,17 \
  --filter_freq 250
```

## Output and Diagnostics

### Global Statistics

File:

```text
statistics_double_shear_3d.csv
```

This high-frequency CSV preserves the full global diagnostic stream.

Hydrodynamics:

- time, maximum velocity, mean kinetic energy, and kinetic-energy decay rate
- transverse kinetic energy, fluctuation kinetic energy, component RMS values, and enstrophy
- physical viscous dissipation `epsilon = 2 * nu * <S_ij S_ij>`
- momentum thicknesses `theta_1`, `theta_2`, `theta_avg`
- vorticity thicknesses `delta_omega_1`, `delta_omega_2`, `delta_omega_avg`
- `Re_theta = DeltaU * theta_avg / nu`

Scalars and mixing:

- scalar means, variances, min/max bounds, and product statistics
- mixture fraction `Z`, scalar dissipation `chi_Z`, scalar mixing times, and scalar-budget diagnostics
- `chi_Z = 2 * D * |grad Z|^2`
- `tau_mix = var_Z / mean_chi_Z`
- online scalar budget diagnostics, including `scalar_variance_decay_rate`, `scalar_budget_ratio`, and `tau_eff`

Reaction and segregation:

- true and perfectly mixed reaction rates
- reaction efficiency, covariance, segregation index, and reactant correlation
- `Da_mix`, `Da_eta`, reaction localization metrics, and reaction-scalar-dissipation correlation

Resolution and numerical quality:

- Kolmogorov and Batchelor scales, `dx/eta_K`, `dx/eta_B`
- density, Mach number, mixture-fraction drift, and mass drift
- `mean_Z_drift` tracks drift in the conserved mixture-fraction-like scalar
- `relative_mass_drift` tracks hydrodynamic density conservation

### Metadata

File:

```text
metadata_double_shear_3d.json
```

The metadata records the parameterization mode, numerical parameters, physical nondimensional groups, perturbation definition, output frequencies, PDF/filter/spectrum settings, and perturbation diagnostics.

For restarted runs it also records `restart_from`, `restart_step`, and the checkpoint format version.

### Checkpoints

Directory:

```text
checkpoints_double_shear_3d/
```

Checkpoint files use names such as:

```text
checkpoint_00006400.bin
```

Each checkpoint is written after a fully completed time step. Current checkpoints use format version 3 and store only the active population buffer for:

- `D3Q27` fluid populations
- `D3Q7` species `A` populations
- `D3Q7` species `B` populations

The inactive ping-pong buffers are scratch storage and are completely overwritten by the next out-of-place LBM step. They are therefore reconstructed as ordinary allocated write buffers on restart rather than serialized. The checkpoint also stores the active ping-pong buffer indices, current completed step, initial diagnostic reference means, previous statistics state, previous kinetic-energy state, scalar limiter interval/cumulative diagnostic state, and compatibility metadata. Product `C` is not stored because it is reconstructed diagnostically from `A` and `B`. New checkpoints are always written in the compact version-3 format; older version-1/version-2 checkpoints do not contain the limiter diagnostic restart state and are rejected with a clear compatibility error.

Checkpoint writes are atomic-style:

```text
checkpoint_XXXXXXXX.bin.tmp -> checkpoint_XXXXXXXX.bin
```

The executable writes the temporary file, flushes/closes it, renames it to the final path, and only then applies retention. Retention keeps the most recent `--checkpoint_keep` completed `.bin` files and ignores temporary files.

On restart, the executable validates grid dimensions, precision, lattice identifiers, `tau_f`, `tau_s`, `k_react`, `U0`, `C0`, `delta_ratio`, `Re_delta`, `Sc`, `Da_delta`, `delta0`, `DeltaU`, `nu`, and `D`. Output frequencies may be changed after restart. Main and filter CSV histories are appended, and step-0 output is not regenerated.

For HPC batch queues, `--max_walltime` enables a portable graceful exit that does not rely on scheduler signals. At safe points between completed steps the executable checks:

```text
elapsed_walltime + estimated_checkpoint_write_time + walltime_safety_margin >= max_walltime
```

Before the first checkpoint, `estimated_checkpoint_write_time` is initialized from the compact version-3 checkpoint size and `--checkpoint_assumed_bandwidth`, with a minimum floor of 1 second. After successful checkpoint writes, the estimate is refined conservatively as the larger of the initial estimate and `1.25 * max_measured_checkpoint_write_time`. When the condition is met before the requested final step, the code writes a normal atomic checkpoint for the completed step, applies the usual `--checkpoint_keep` retention policy, reports the checkpoint filename, and exits with return code `0`. This forced checkpoint is independent of `--checkpoint_freq` and `--checkpoint_walltime`; normal completion at `--steps` does not force an extra checkpoint.

### VTK

Directory:

```text
vtk_double_shear_3d/
```

When `--vtk_freq > 0`, binary legacy VTK files are written at:

```text
0, vtk_freq, 2*vtk_freq, ...
```

When `--vtk_freq 0`, no VTK files are written.

Fields:

- velocity
- `C_A`
- `C_B`
- reconstructed `C_C`

### Plane Profiles

Directory:

```text
profiles_double_shear_3d/
```

Files:

```text
profile_00000000.csv
profile_00001000.csv
...
```

Each file contains one row per y index with plane means, fluctuation RMS values, scalar means, `Z`, `chi_Z`, reaction rate, and local reactant covariance.

### PDFs and Conditional Statistics

Directory:

```text
pdfs_double_shear_3d/step_XXXXXXXX/
```

Files include:

- `pdf_Z.csv`
- `pdf_log_chi.csv`
- `pdf_log_R.csv`
- `joint_pdf_Ca_Cb.csv`
- `joint_pdf_Z_log_chi.csv`
- `joint_pdf_log_R_log_chi.csv`
- `conditional_chi_given_Z.csv`
- `conditional_R_given_Z.csv`
- `conditional_R_given_log_chi.csv`
- `pdf_metadata.json`

Probabilities are stored as probability mass normalized by the full domain cell count, not as bin-width-normalized densities.

The marginal PDFs use `Z`, `log10(chi_star)`, and `log10(R_star)`. Joint PDFs cover `P(C_A/C0, C_B/C0)`, `P(Z, log10(chi_star))`, and `P(log10(R_star), log10(chi_star))`. Conditional means include `<chi_Z | Z>`, `<R | Z>`, and `<R | log10(chi_star)>`.

### Spectra

Directory:

```text
spectra_double_shear_3d/step_XXXXXXXX/
```

Spectra require `LB_CUBE_ENABLE_FFTW=ON`. Runtime requests for spectra fail clearly if FFTW support was not compiled in.

The spectral module performs 2D real-to-complex FFTs in homogeneous x-z planes only, subtracts the local x-z plane mean, applies the same mixing-layer y-weight:

```text
w(y) = 4 * Zbar(y) * (1 - Zbar(y))
```

and writes:

- scalar spectra: `spectrum_Z_2D.csv`, `spectrum_Z_kx.csv`, `spectrum_Z_kz.csv`, `spectrum_Z_kh.csv`
- velocity fluctuation spectra: `spectrum_U_2D.csv`, `spectrum_U_kx.csv`, `spectrum_U_kz.csv`, `spectrum_U_kh.csv`
- reactant covariance co-spectra: `cospectrum_AB_2D.csv`, `cospectrum_AB_kx.csv`, `cospectrum_AB_kz.csv`, `cospectrum_AB_kh.csv`
- cumulative covariance resolution: `cumulative_cospectrum_AB_kh.csv`
- validation metadata: `spectrum_metadata.json`

The 2D spectra use signed `kx` indices, nonnegative FFTW real-to-complex `kz` indices, Hermitian multiplicity, and explicit FFTW normalization. Parseval and reduction-conservation checks are written to metadata.

### A-Priori Filtering

Main file:

```text
filter_statistics_double_shear_3d.csv
```

Additional directories:

```text
filter_conditionals_double_shear_3d/step_XXXXXXXX/
filter_pdfs_double_shear_3d/step_XXXXXXXX/
```

For each configured width, the filter module computes:

```text
Abar, Bbar, ABbar
tau_AB = ABbar - Abar * Bbar
tau_AA = overline(A^2) - Abar^2
tau_BB = overline(B^2) - Bbar^2
tau_ZZ = overline(Z^2) - Zbar^2
rho_AB_SGS = tau_AB / sqrt(tau_AA * tau_BB)
```

with:

```text
Zbar = 0.5 * (1 + (Abar - Bbar) / C0)
```

The filtered reaction identity is:

```text
R_exact_filtered = k_react * ABbar
R_LES_naive      = k_react * Abar * Bbar
R_SGS            = k_react * tau_AB
R_exact_filtered = R_LES_naive + R_SGS
```

The module also writes exact identity and conservation diagnostics, filtered conditional means, and SGS PDFs. The identity width `1` is supported and should recover zero SGS moments to roundoff.

### Filtered Conditionals and SGS PDFs

Filtered conditional outputs are written to:

```text
filter_conditionals_double_shear_3d/step_XXXXXXXX/
```

For every configured filter width, the files report:

```text
<tau_AB | Zbar>
<rho_AB_SGS | Zbar>
<tau_AB | log10(tau_ZZ)>
```

SGS PDF outputs are written to:

```text
filter_pdfs_double_shear_3d/step_XXXXXXXX/
```

They include:

```text
P(tau_AB)
P(log10(tau_ZZ))
P(tau_AB, log10(tau_ZZ))
```

`tau_AB` uses linear bins. Positive `tau_ZZ` uses logarithmic bins, with zero, underflow, overflow, and included probabilities recorded in metadata. Probabilities are normalized by the full domain cell count.

## Performance and Memory Notes

- Build in `Release`.
- Use explicit compiler and dependency paths in the CMake command rather than relying on login-node defaults.
- Set OpenMP placement and binding explicitly.
- Keep VTK, PDF, spectrum, and filter frequencies coarse enough for the parallel file system.
- Use `--vtk_freq 0`, `--profile_freq 0`, `--pdf_freq 0`, `--spectrum_freq 0`, and disabled filtering for pure performance runs.
- Enable spectra only in FFTW builds.
- Use physical mode for grid-refinement studies so `tau_f`, `tau_s`, and `k_react` are derived consistently from `Re_delta`, `Sc`, `Da_delta`, `U0`, `C0`, and `delta_ratio`.
- Large 3D cases such as `512^3` require substantial memory for fluid populations, scalar populations, and optional diagnostic temporaries.
- Spectra add FFT work at output steps only.
- Multi-width filtering is diagnostic post-processing during the run. Widths are processed sequentially, so peak memory is independent of the number of requested widths.

Example production-style command:

```bash
export OMP_NUM_THREADS=128
export OMP_PLACES=cores
export OMP_PROC_BIND=close

./build-fftw/lbm_turbulent_reactive_double_shear_3d \
  --Nx 384 --Ny 384 --Nz 384 \
  --U0 0.05 \
  --C0 1.0 \
  --delta_ratio 0.0625 \
  --Re_delta 400 \
  --Sc 10 \
  --Da_delta 10 \
  --perturb_amplitude 0.02 \
  --perturb_seed 12345 \
  --perturb_kmin 1 \
  --perturb_kmax 4 \
  --perturb_width 2.0 \
  --steps 12000 \
  --stat_freq 10 \
  --screen_freq 100 \
  --vtk_freq 3000 \
  --profile_freq 1000 \
  --pdf_freq 3000 \
  --spectrum_freq 3000 \
  --filter_widths 1,3,5,9,17,33 \
  --filter_freq 3000
```

## Repository Layout

```text
include/
  lattice_traits.hpp       Lattice definitions and compile-time traits
  lattice_memory.hpp       Population storage and mdspan views
  lattice_physics.hpp      Equilibria, collision kernels, scalar physics, chemistry
  lattice_mrt.hpp          MRT moment transforms and collision utilities
  lattice_core.hpp         CPU stepping, scalar stepping, fused reaction stepping
  lattice_cuda.cuh         CUDA kernel declarations and device-side stepping
  lattice_filter.hpp       Naive and optimized periodic 3D box filters
  mdspan_compat.hpp        Native/Kokkos mdspan compatibility aliases

src/
  turbulent_reactive_double_shear_3d.cpp
  turbulent_tgv_3d.cpp
  turbulent_shear_layer.cpp
  compare_operators.cpp

tests/
  test_validation.cpp
  test_double_shear_parameters.cpp
  test_box_filter.cpp
  test_fftw.cpp
```

## Development Principles

- Keep collision dispatch compile-time and lattice-generic where possible.
- Preserve exact mathematical identities in diagnostics and tests.
- Prefer deterministic, normalized, reproducible initial conditions for parameter studies.
- Avoid dense matrix operations in hot collision paths.
- Keep production diagnostics opt-in when they are expensive.
- Keep reference implementations, such as the naive box filter, available for validation even when optimized paths are used in production.

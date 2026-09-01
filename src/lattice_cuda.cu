/**
 * @file lattice_cuda.cu
 * @brief CUDA implementation of fused pull-streaming fluid time steps.
 *
 * This translation unit contains device-only indexing helpers, the templated
 * CUDA kernel, host launch wrappers, and explicit template instantiations for the
 * supported lattice/precision combinations.
 */

#include "lattice_cuda.cuh"

#include <array>
#include <type_traits>

namespace lbm {

namespace detail {

/**
 * @brief Dependent false value used to reject unsupported compile-time branches.
 *
 * The assertion is instantiated only when an unsupported MRT lattice is selected
 * for a concrete kernel specialization.
 *
 * @tparam Lattice Lattice traits type used by the selected solver branch.
 */
template <typename Lattice>
inline constexpr bool always_false_v = false;

/**
 * @brief Compute the periodic upstream coordinate for one CUDA thread.
 *
 * The kernel uses pull streaming, so the source coordinate for population `i` is
 * the destination coordinate minus the discrete velocity component. Adding the
 * extent before modulo implements periodic wrapping for one-cell negative
 * displacements.
 *
 * @param coordinate Destination coordinate along one axis.
 * @param velocity Discrete velocity component along the same axis.
 * @param extent Domain size along the axis.
 * @return Wrapped source coordinate.
 */
__device__ inline std::size_t periodic_pull_index(
    std::size_t coordinate,
    int velocity,
    std::size_t extent) {
    const auto signed_coordinate = static_cast<std::ptrdiff_t>(coordinate);
    const auto signed_extent = static_cast<std::ptrdiff_t>(extent);
    const auto signed_index = signed_coordinate - static_cast<std::ptrdiff_t>(velocity) + signed_extent;
    return static_cast<std::size_t>(signed_index % signed_extent);
}

/**
 * @brief Convert `[q, z, y, x]` coordinates into the flat SoA device index.
 *
 * The formula mirrors `std::layout_right` host views with `X` contiguous. Keeping
 * this mapping identical to `LatticeMemory` is essential for host/device
 * round-trips and CPU/GPU result comparisons.
 *
 * @param i Discrete population index.
 * @param x Cell x coordinate.
 * @param y Cell y coordinate.
 * @param z Cell z coordinate.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z.
 * @return Flat scalar offset in the population buffer.
 */
__host__ __device__ inline std::size_t population_index(
    int i,
    std::size_t x,
    std::size_t y,
    std::size_t z,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent) {
    return (((static_cast<std::size_t>(i) * z_extent + z) * y_extent + y) * x_extent + x);
}

/**
 * @brief Apply the compile-time selected collision operator inside a CUDA thread.
 *
 * @tparam CT Compile-time collision operator selection.
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param local_pops Populations gathered by pull streaming and overwritten with
 * post-collision values.
 * @param omega Even/BGK relaxation frequency.
 */
template <CollisionType CT, IsLatticeModel Lattice, std::floating_point Real>
__device__ inline void collide_cell(
    std::array<Real, static_cast<std::size_t>(Lattice::Q)>& local_pops,
    Real omega) {
    if constexpr (CT == CollisionType::BGK) {
        collide_bgk<Lattice, Real>(local_pops, omega);
    } else if constexpr (CT == CollisionType::TRT) {
        const MacroState<Lattice, Real> macro =
            compute_macro_state<Lattice, Real>(local_pops);
        const Real omega_minus = compute_omega_minus<Real>(omega);
        collide_trt<Lattice, Real>(local_pops, macro, omega, omega_minus);
    } else if constexpr (CT == CollisionType::MRT) {
        const MacroState<Lattice, Real> macro =
            compute_macro_state<Lattice, Real>(local_pops);
        if constexpr (std::is_same_v<Lattice, D2Q9>) {
            mrt::MrtRelaxationRates_D2Q9<Real> relaxation_rates{};
            relaxation_rates.s_nu = omega;
            mrt::collide_mrt_d2q9<Real>(local_pops, macro, relaxation_rates);
        } else if constexpr (std::is_same_v<Lattice, D3Q19>) {
            mrt::MrtRelaxationRates_D3Q19<Real> relaxation_rates{};
            relaxation_rates.s_nu = omega;
            mrt::collide_mrt_d3q19<Real>(local_pops, macro, relaxation_rates);
        } else if constexpr (std::is_same_v<Lattice, D3Q27>) {
            mrt::MrtRelaxationRates_D3Q27<Real> relaxation_rates{};
            relaxation_rates.s_nu = omega;
            mrt::collide_mrt_d3q27<Real>(local_pops, macro, relaxation_rates);
        } else {
            static_assert(
                always_false_v<Lattice>,
                "MRT is currently only supported for D2Q9, D3Q19, and D3Q27.");
        }
    } else if constexpr (CT == CollisionType::RLBM) {
        const MacroState<Lattice, Real> macro =
            compute_macro_state<Lattice, Real>(local_pops);
        collide_regularized<Lattice, Real>(local_pops, macro, omega);
    }
}

} // namespace detail

/**
 * @brief CUDA kernel for one fused collision-streaming update.
 *
 * Each thread owns one destination cell. It gathers populations from neighboring
 * cells using periodic pull streaming, applies the compile-time selected
 * collision operator, and writes the updated populations into the next buffer.
 * No inter-thread synchronization is required because the scheme only reads from
 * the current buffer and writes to disjoint locations in the next buffer.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @tparam CT Compile-time collision operator.
 * @param current_populations Device pointer to the current SoA buffer.
 * @param next_populations Device pointer to the next SoA buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z.
 * @param omega BGK relaxation frequency.
 */
template <
    IsLatticeModel Lattice,
    std::floating_point Real,
    CollisionType CT>
__global__ void kernel_step(
    const Real* current_populations,
    Real* next_populations,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega) {
    const std::size_t x = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const std::size_t y = static_cast<std::size_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const std::size_t z = static_cast<std::size_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= x_extent || y >= y_extent || z >= z_extent) {
        return;
    }

    std::array<Real, static_cast<std::size_t>(Lattice::Q)> local_pops{};

    if constexpr (Lattice::D == 2) {
        for (int i = 0; i < Lattice::Q; ++i) {
            const auto direction_offset = static_cast<std::size_t>(i * Lattice::D);
            const int cx = Lattice::directions[direction_offset];
            const int cy = Lattice::directions[direction_offset + 1];
            const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
            const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);

            local_pops[static_cast<std::size_t>(i)] =
                current_populations[detail::population_index(
                    i, nx, ny, 0, x_extent, y_extent, z_extent)];
        }

        detail::collide_cell<CT, Lattice, Real>(local_pops, omega);

        for (int i = 0; i < Lattice::Q; ++i) {
            next_populations[detail::population_index(
                i, x, y, 0, x_extent, y_extent, z_extent)] =
                local_pops[static_cast<std::size_t>(i)];
        }
    } else {
        for (int i = 0; i < Lattice::Q; ++i) {
            const auto direction_offset = static_cast<std::size_t>(i * Lattice::D);
            const int cx = Lattice::directions[direction_offset];
            const int cy = Lattice::directions[direction_offset + 1];
            const int cz = Lattice::directions[direction_offset + 2];
            const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
            const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);
            const std::size_t nz = detail::periodic_pull_index(z, cz, z_extent);

            local_pops[static_cast<std::size_t>(i)] =
                current_populations[detail::population_index(
                    i, nx, ny, nz, x_extent, y_extent, z_extent)];
        }

        detail::collide_cell<CT, Lattice, Real>(local_pops, omega);

        for (int i = 0; i < Lattice::Q; ++i) {
            next_populations[detail::population_index(
                i, x, y, z, x_extent, y_extent, z_extent)] =
                local_pops[static_cast<std::size_t>(i)];
        }
    }
}

/**
 * @brief CUDA kernel for one passive-scalar advection-diffusion update.
 *
 * The fluid populations are sampled at the destination cell to reconstruct the
 * advecting velocity, while scalar populations are pulled from upstream scalar
 * neighbors. The scalar collision is local and includes an optional source term.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_scalar Device pointer to the scalar current SoA buffer.
 * @param next_scalar Device pointer to the scalar next SoA buffer.
 * @param current_fluid Device pointer to the fluid current SoA buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param source_term Scalar source contribution applied locally.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
__global__ void kernel_scalar_step(
    const Real* current_scalar,
    Real* next_scalar,
    const Real* current_fluid,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega_c,
    Real source_term) {
    static_assert(FluidLattice::D == ScalarLattice::D);

    const std::size_t x = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const std::size_t y = static_cast<std::size_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const std::size_t z = static_cast<std::size_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= x_extent || y >= y_extent || z >= z_extent) {
        return;
    }

    std::array<Real, static_cast<std::size_t>(FluidLattice::Q)> fluid_pops{};
    std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> scalar_pops{};

    if constexpr (ScalarLattice::D == 2) {
        for (int i = 0; i < FluidLattice::Q; ++i) {
            fluid_pops[static_cast<std::size_t>(i)] =
                current_fluid[detail::population_index(
                    i, x, y, 0, x_extent, y_extent, z_extent)];
        }
        const MacroState<FluidLattice, Real> fluid_macro =
            compute_macro_state<FluidLattice, Real>(fluid_pops);

        for (int i = 0; i < ScalarLattice::Q; ++i) {
            const auto direction_offset = static_cast<std::size_t>(i * ScalarLattice::D);
            const int cx = ScalarLattice::c[direction_offset];
            const int cy = ScalarLattice::c[direction_offset + 1];
            const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
            const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);

            scalar_pops[static_cast<std::size_t>(i)] =
                current_scalar[detail::population_index(
                    i, nx, ny, 0, x_extent, y_extent, z_extent)];
        }

        collide_scalar_bgk<ScalarLattice, Real>(
            scalar_pops,
            fluid_macro.velocity,
            omega_c,
            source_term);

        for (int i = 0; i < ScalarLattice::Q; ++i) {
            next_scalar[detail::population_index(
                i, x, y, 0, x_extent, y_extent, z_extent)] =
                scalar_pops[static_cast<std::size_t>(i)];
        }
    } else {
        for (int i = 0; i < FluidLattice::Q; ++i) {
            fluid_pops[static_cast<std::size_t>(i)] =
                current_fluid[detail::population_index(
                    i, x, y, z, x_extent, y_extent, z_extent)];
        }
        const MacroState<FluidLattice, Real> fluid_macro =
            compute_macro_state<FluidLattice, Real>(fluid_pops);

        for (int i = 0; i < ScalarLattice::Q; ++i) {
            const auto direction_offset = static_cast<std::size_t>(i * ScalarLattice::D);
            const int cx = ScalarLattice::c[direction_offset];
            const int cy = ScalarLattice::c[direction_offset + 1];
            const int cz = ScalarLattice::c[direction_offset + 2];
            const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
            const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);
            const std::size_t nz = detail::periodic_pull_index(z, cz, z_extent);

            scalar_pops[static_cast<std::size_t>(i)] =
                current_scalar[detail::population_index(
                    i, nx, ny, nz, x_extent, y_extent, z_extent)];
        }

        collide_scalar_bgk<ScalarLattice, Real>(
            scalar_pops,
            fluid_macro.velocity,
            omega_c,
            source_term);

        for (int i = 0; i < ScalarLattice::Q; ++i) {
            next_scalar[detail::population_index(
                i, x, y, z, x_extent, y_extent, z_extent)] =
                scalar_pops[static_cast<std::size_t>(i)];
        }
    }
}

/**
 * @brief CUDA kernel for fused `A + B -> C` scalar reaction transport.
 *
 * Both reactants are advanced in a single thread-local operation so the reaction
 * source is computed once and no source arrays are allocated.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_a Device pointer to species A current SoA buffer.
 * @param next_a Device pointer to species A next SoA buffer.
 * @param current_b Device pointer to species B current SoA buffer.
 * @param next_b Device pointer to species B next SoA buffer.
 * @param current_fluid Device pointer to fluid current SoA buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param k_react Second-order reaction-rate constant.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
__global__ void kernel_reaction_AB(
    const Real* current_a,
    Real* next_a,
    const Real* current_b,
    Real* next_b,
    const Real* current_fluid,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega_c,
    Real k_react) {
    static_assert(FluidLattice::D == ScalarLattice::D);

    const std::size_t x = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const std::size_t y = static_cast<std::size_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const std::size_t z = static_cast<std::size_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= x_extent || y >= y_extent || z >= z_extent) {
        return;
    }

    std::array<Real, static_cast<std::size_t>(FluidLattice::Q)> fluid_pops{};
    std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> a_pops{};
    std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> b_pops{};

    if constexpr (ScalarLattice::D == 2) {
        for (int i = 0; i < FluidLattice::Q; ++i) {
            fluid_pops[static_cast<std::size_t>(i)] =
                current_fluid[detail::population_index(
                    i, x, y, 0, x_extent, y_extent, z_extent)];
        }
        const MacroState<FluidLattice, Real> fluid_macro =
            compute_macro_state<FluidLattice, Real>(fluid_pops);

        for (int i = 0; i < ScalarLattice::Q; ++i) {
            const auto direction_offset = static_cast<std::size_t>(i * ScalarLattice::D);
            const int cx = ScalarLattice::c[direction_offset];
            const int cy = ScalarLattice::c[direction_offset + 1];
            const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
            const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);
            const auto q = static_cast<std::size_t>(i);

            a_pops[q] = current_a[detail::population_index(
                i, nx, ny, 0, x_extent, y_extent, z_extent)];
            b_pops[q] = current_b[detail::population_index(
                i, nx, ny, 0, x_extent, y_extent, z_extent)];
        }

        const Real concentration_a = compute_concentration<ScalarLattice, Real>(a_pops);
        const Real concentration_b = compute_concentration<ScalarLattice, Real>(b_pops);
        const Real reaction_source =
            compute_reaction_ab_source<Real>(concentration_a, concentration_b, k_react);

        collide_scalar_bgk<ScalarLattice, Real>(
            a_pops, fluid_macro.velocity, omega_c, reaction_source);
        collide_scalar_bgk<ScalarLattice, Real>(
            b_pops, fluid_macro.velocity, omega_c, reaction_source);

        for (int i = 0; i < ScalarLattice::Q; ++i) {
            const auto q = static_cast<std::size_t>(i);
            next_a[detail::population_index(i, x, y, 0, x_extent, y_extent, z_extent)] = a_pops[q];
            next_b[detail::population_index(i, x, y, 0, x_extent, y_extent, z_extent)] = b_pops[q];
        }
    } else {
        for (int i = 0; i < FluidLattice::Q; ++i) {
            fluid_pops[static_cast<std::size_t>(i)] =
                current_fluid[detail::population_index(
                    i, x, y, z, x_extent, y_extent, z_extent)];
        }
        const MacroState<FluidLattice, Real> fluid_macro =
            compute_macro_state<FluidLattice, Real>(fluid_pops);

        for (int i = 0; i < ScalarLattice::Q; ++i) {
            const auto direction_offset = static_cast<std::size_t>(i * ScalarLattice::D);
            const int cx = ScalarLattice::c[direction_offset];
            const int cy = ScalarLattice::c[direction_offset + 1];
            const int cz = ScalarLattice::c[direction_offset + 2];
            const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
            const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);
            const std::size_t nz = detail::periodic_pull_index(z, cz, z_extent);
            const auto q = static_cast<std::size_t>(i);

            a_pops[q] = current_a[detail::population_index(
                i, nx, ny, nz, x_extent, y_extent, z_extent)];
            b_pops[q] = current_b[detail::population_index(
                i, nx, ny, nz, x_extent, y_extent, z_extent)];
        }

        const Real concentration_a = compute_concentration<ScalarLattice, Real>(a_pops);
        const Real concentration_b = compute_concentration<ScalarLattice, Real>(b_pops);
        const Real reaction_source =
            compute_reaction_ab_source<Real>(concentration_a, concentration_b, k_react);

        collide_scalar_bgk<ScalarLattice, Real>(
            a_pops, fluid_macro.velocity, omega_c, reaction_source);
        collide_scalar_bgk<ScalarLattice, Real>(
            b_pops, fluid_macro.velocity, omega_c, reaction_source);

        for (int i = 0; i < ScalarLattice::Q; ++i) {
            const auto q = static_cast<std::size_t>(i);
            next_a[detail::population_index(i, x, y, z, x_extent, y_extent, z_extent)] = a_pops[q];
            next_b[detail::population_index(i, x, y, z, x_extent, y_extent, z_extent)] = b_pops[q];
        }
    }
}

/**
 * @brief Validate launch inputs and enqueue the CUDA step kernel.
 *
 * The wrapper leaves synchronization to the caller so timing code can measure
 * kernels in batches or include/exclude host-device transfers explicitly.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_populations Device pointer to the current SoA buffer.
 * @param next_populations Device pointer to the next SoA buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z, overwritten to 1 for 2D lattices.
 * @param omega BGK relaxation frequency, or TRT even relaxation frequency.
 * @param block CUDA block dimensions.
 * @return CUDA validation or launch status.
 */
template <
    IsLatticeModel Lattice,
    std::floating_point Real,
    CollisionType CT>
cudaError_t launch_step_gpu(
    const Real* current_populations,
    Real* next_populations,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega,
    dim3 block) {
    if (current_populations == nullptr || next_populations == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    if (block.x == 0 || block.y == 0 || block.z == 0) {
        return cudaErrorInvalidConfiguration;
    }

    if constexpr (Lattice::D == 2) {
        z_extent = 1;
        block.z = 1;
    }

    if (x_extent == 0 || y_extent == 0 || z_extent == 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid{
        static_cast<unsigned int>((x_extent + block.x - 1) / block.x),
        static_cast<unsigned int>((y_extent + block.y - 1) / block.y),
        static_cast<unsigned int>((z_extent + block.z - 1) / block.z)
    };

    kernel_step<Lattice, Real, CT><<<grid, block>>>(
        current_populations,
        next_populations,
        x_extent,
        y_extent,
        z_extent,
        omega);

    return cudaGetLastError();
}

/**
 * @brief Validate inputs and enqueue the CUDA passive-scalar step kernel.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_scalar Device pointer to the scalar current SoA buffer.
 * @param next_scalar Device pointer to the scalar next SoA buffer.
 * @param current_fluid Device pointer to the fluid current SoA buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z, overwritten to 1 for 2D lattices.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param source_term Scalar source contribution.
 * @param block CUDA block dimensions.
 * @return CUDA validation or launch status.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
cudaError_t launch_scalar_step_gpu(
    const Real* current_scalar,
    Real* next_scalar,
    const Real* current_fluid,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega_c,
    Real source_term,
    dim3 block) {
    static_assert(FluidLattice::D == ScalarLattice::D);

    if (current_scalar == nullptr || next_scalar == nullptr || current_fluid == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    if (block.x == 0 || block.y == 0 || block.z == 0) {
        return cudaErrorInvalidConfiguration;
    }

    if constexpr (ScalarLattice::D == 2) {
        z_extent = 1;
        block.z = 1;
    }

    if (x_extent == 0 || y_extent == 0 || z_extent == 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid{
        static_cast<unsigned int>((x_extent + block.x - 1) / block.x),
        static_cast<unsigned int>((y_extent + block.y - 1) / block.y),
        static_cast<unsigned int>((z_extent + block.z - 1) / block.z)
    };

    kernel_scalar_step<FluidLattice, ScalarLattice, Real><<<grid, block>>>(
        current_scalar,
        next_scalar,
        current_fluid,
        x_extent,
        y_extent,
        z_extent,
        omega_c,
        source_term);

    return cudaGetLastError();
}

/**
 * @brief Validate inputs and enqueue the CUDA fused reaction kernel.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_a Device pointer to species A current SoA buffer.
 * @param next_a Device pointer to species A next SoA buffer.
 * @param current_b Device pointer to species B current SoA buffer.
 * @param next_b Device pointer to species B next SoA buffer.
 * @param current_fluid Device pointer to fluid current SoA buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z, overwritten to 1 for 2D lattices.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param k_react Second-order reaction-rate constant.
 * @param block CUDA block dimensions.
 * @return CUDA validation or launch status.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
cudaError_t launch_reaction_AB_gpu(
    const Real* current_a,
    Real* next_a,
    const Real* current_b,
    Real* next_b,
    const Real* current_fluid,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega_c,
    Real k_react,
    dim3 block) {
    static_assert(FluidLattice::D == ScalarLattice::D);

    if (current_a == nullptr || next_a == nullptr ||
        current_b == nullptr || next_b == nullptr ||
        current_fluid == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    if (block.x == 0 || block.y == 0 || block.z == 0) {
        return cudaErrorInvalidConfiguration;
    }

    if constexpr (ScalarLattice::D == 2) {
        z_extent = 1;
        block.z = 1;
    }

    if (x_extent == 0 || y_extent == 0 || z_extent == 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid{
        static_cast<unsigned int>((x_extent + block.x - 1) / block.x),
        static_cast<unsigned int>((y_extent + block.y - 1) / block.y),
        static_cast<unsigned int>((z_extent + block.z - 1) / block.z)
    };

    kernel_reaction_AB<FluidLattice, ScalarLattice, Real><<<grid, block>>>(
        current_a,
        next_a,
        current_b,
        next_b,
        current_fluid,
        x_extent,
        y_extent,
        z_extent,
        omega_c,
        k_react);

    return cudaGetLastError();
}

/**
 * @brief Convenience launcher that reads domain extents from host memory metadata.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param mem Host-side memory object used only for dimensions.
 * @param current_populations Device pointer to the current SoA buffer.
 * @param next_populations Device pointer to the next SoA buffer.
 * @param omega BGK relaxation frequency.
 * @param block CUDA block dimensions.
 * @return CUDA validation or launch status.
 */
template <
    IsLatticeModel Lattice,
    std::floating_point Real,
    CollisionType CT>
cudaError_t launch_step_gpu(
    const LatticeMemory<Lattice, Real>& mem,
    const Real* current_populations,
    Real* next_populations,
    Real omega,
    dim3 block) {
    return launch_step_gpu<Lattice, Real, CT>(
        current_populations,
        next_populations,
        mem.x_extent(),
        mem.y_extent(),
        mem.z_extent(),
        omega,
        block);
}

/**
 * @brief Convenience scalar launcher that reads extents from host memory metadata.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param scalar_mem Host scalar memory object used for dimensions.
 * @param fluid_mem Host fluid memory object used for dimension consistency.
 * @param current_scalar Device pointer to the scalar current SoA buffer.
 * @param next_scalar Device pointer to the scalar next SoA buffer.
 * @param current_fluid Device pointer to the fluid current SoA buffer.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param source_term Scalar source contribution.
 * @param block CUDA block dimensions.
 * @return CUDA validation or launch status.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
cudaError_t launch_scalar_step_gpu(
    const LatticeMemory<ScalarLattice, Real>& scalar_mem,
    const LatticeMemory<FluidLattice, Real>& fluid_mem,
    const Real* current_scalar,
    Real* next_scalar,
    const Real* current_fluid,
    Real omega_c,
    Real source_term,
    dim3 block) {
    static_assert(FluidLattice::D == ScalarLattice::D);

    if (scalar_mem.x_extent() != fluid_mem.x_extent() ||
        scalar_mem.y_extent() != fluid_mem.y_extent() ||
        scalar_mem.z_extent() != fluid_mem.z_extent()) {
        return cudaErrorInvalidValue;
    }

    return launch_scalar_step_gpu<FluidLattice, ScalarLattice, Real>(
        current_scalar,
        next_scalar,
        current_fluid,
        scalar_mem.x_extent(),
        scalar_mem.y_extent(),
        scalar_mem.z_extent(),
        omega_c,
        source_term,
        block);
}

/**
 * @brief Explicit template instantiations for supported lattice/precision pairs.
 *
 * Keeping instantiations in this CUDA translation unit prevents ordinary host
 * targets from compiling device code while still exposing concrete symbols for
 * the application launcher.
 */
template __global__ void kernel_step<D2Q9, float, CollisionType::BGK>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D2Q9, double, CollisionType::BGK>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);
template __global__ void kernel_step<D3Q19, float, CollisionType::BGK>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D3Q19, double, CollisionType::BGK>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);
template __global__ void kernel_step<D3Q27, float, CollisionType::BGK>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D3Q27, double, CollisionType::BGK>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);

template __global__ void kernel_step<D2Q9, float, CollisionType::TRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D2Q9, double, CollisionType::TRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);
template __global__ void kernel_step<D3Q19, float, CollisionType::TRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D3Q19, double, CollisionType::TRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);
template __global__ void kernel_step<D3Q27, float, CollisionType::TRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D3Q27, double, CollisionType::TRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);

template __global__ void kernel_step<D2Q9, float, CollisionType::MRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D2Q9, double, CollisionType::MRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);
template __global__ void kernel_step<D3Q19, float, CollisionType::MRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D3Q19, double, CollisionType::MRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);
template __global__ void kernel_step<D3Q27, float, CollisionType::MRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D3Q27, double, CollisionType::MRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);

template __global__ void kernel_step<D2Q9, float, CollisionType::RLBM>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D2Q9, double, CollisionType::RLBM>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);
template __global__ void kernel_step<D3Q19, float, CollisionType::RLBM>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D3Q19, double, CollisionType::RLBM>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);
template __global__ void kernel_step<D3Q27, float, CollisionType::RLBM>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D3Q27, double, CollisionType::RLBM>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);

template cudaError_t launch_step_gpu<D2Q9, float, CollisionType::BGK>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D2Q9, double, CollisionType::BGK>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);
template cudaError_t launch_step_gpu<D3Q19, float, CollisionType::BGK>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D3Q19, double, CollisionType::BGK>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);
template cudaError_t launch_step_gpu<D3Q27, float, CollisionType::BGK>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D3Q27, double, CollisionType::BGK>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);

template cudaError_t launch_step_gpu<D2Q9, float, CollisionType::TRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D2Q9, double, CollisionType::TRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);
template cudaError_t launch_step_gpu<D3Q19, float, CollisionType::TRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D3Q19, double, CollisionType::TRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);
template cudaError_t launch_step_gpu<D3Q27, float, CollisionType::TRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D3Q27, double, CollisionType::TRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);

template cudaError_t launch_step_gpu<D2Q9, float, CollisionType::MRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D2Q9, double, CollisionType::MRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);
template cudaError_t launch_step_gpu<D3Q19, float, CollisionType::MRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D3Q19, double, CollisionType::MRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);
template cudaError_t launch_step_gpu<D3Q27, float, CollisionType::MRT>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D3Q27, double, CollisionType::MRT>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);

template cudaError_t launch_step_gpu<D2Q9, float, CollisionType::RLBM>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D2Q9, double, CollisionType::RLBM>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);
template cudaError_t launch_step_gpu<D3Q19, float, CollisionType::RLBM>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D3Q19, double, CollisionType::RLBM>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);
template cudaError_t launch_step_gpu<D3Q27, float, CollisionType::RLBM>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D3Q27, double, CollisionType::RLBM>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);

template cudaError_t launch_step_gpu<D2Q9, float, CollisionType::BGK>(
    const LatticeMemory<D2Q9, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D2Q9, double, CollisionType::BGK>(
    const LatticeMemory<D2Q9, double>&, const double*, double*, double, dim3);
template cudaError_t launch_step_gpu<D3Q19, float, CollisionType::BGK>(
    const LatticeMemory<D3Q19, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D3Q19, double, CollisionType::BGK>(
    const LatticeMemory<D3Q19, double>&, const double*, double*, double, dim3);
template cudaError_t launch_step_gpu<D3Q27, float, CollisionType::BGK>(
    const LatticeMemory<D3Q27, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D3Q27, double, CollisionType::BGK>(
    const LatticeMemory<D3Q27, double>&, const double*, double*, double, dim3);

template cudaError_t launch_step_gpu<D2Q9, float, CollisionType::TRT>(
    const LatticeMemory<D2Q9, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D2Q9, double, CollisionType::TRT>(
    const LatticeMemory<D2Q9, double>&, const double*, double*, double, dim3);
template cudaError_t launch_step_gpu<D3Q19, float, CollisionType::TRT>(
    const LatticeMemory<D3Q19, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D3Q19, double, CollisionType::TRT>(
    const LatticeMemory<D3Q19, double>&, const double*, double*, double, dim3);
template cudaError_t launch_step_gpu<D3Q27, float, CollisionType::TRT>(
    const LatticeMemory<D3Q27, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D3Q27, double, CollisionType::TRT>(
    const LatticeMemory<D3Q27, double>&, const double*, double*, double, dim3);

template cudaError_t launch_step_gpu<D2Q9, float, CollisionType::MRT>(
    const LatticeMemory<D2Q9, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D2Q9, double, CollisionType::MRT>(
    const LatticeMemory<D2Q9, double>&, const double*, double*, double, dim3);
template cudaError_t launch_step_gpu<D3Q19, float, CollisionType::MRT>(
    const LatticeMemory<D3Q19, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D3Q19, double, CollisionType::MRT>(
    const LatticeMemory<D3Q19, double>&, const double*, double*, double, dim3);
template cudaError_t launch_step_gpu<D3Q27, float, CollisionType::MRT>(
    const LatticeMemory<D3Q27, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D3Q27, double, CollisionType::MRT>(
    const LatticeMemory<D3Q27, double>&, const double*, double*, double, dim3);

template cudaError_t launch_step_gpu<D2Q9, float, CollisionType::RLBM>(
    const LatticeMemory<D2Q9, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D2Q9, double, CollisionType::RLBM>(
    const LatticeMemory<D2Q9, double>&, const double*, double*, double, dim3);
template cudaError_t launch_step_gpu<D3Q19, float, CollisionType::RLBM>(
    const LatticeMemory<D3Q19, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D3Q19, double, CollisionType::RLBM>(
    const LatticeMemory<D3Q19, double>&, const double*, double*, double, dim3);
template cudaError_t launch_step_gpu<D3Q27, float, CollisionType::RLBM>(
    const LatticeMemory<D3Q27, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D3Q27, double, CollisionType::RLBM>(
    const LatticeMemory<D3Q27, double>&, const double*, double*, double, dim3);

template __global__ void kernel_scalar_step<D2Q9, D2Q5, float>(
    const float*, float*, const float*, std::size_t, std::size_t, std::size_t, float, float);
template __global__ void kernel_scalar_step<D2Q9, D2Q5, double>(
    const double*, double*, const double*, std::size_t, std::size_t, std::size_t, double, double);
template __global__ void kernel_scalar_step<D3Q19, D3Q7, float>(
    const float*, float*, const float*, std::size_t, std::size_t, std::size_t, float, float);
template __global__ void kernel_scalar_step<D3Q19, D3Q7, double>(
    const double*, double*, const double*, std::size_t, std::size_t, std::size_t, double, double);
template __global__ void kernel_scalar_step<D3Q27, D3Q7, float>(
    const float*, float*, const float*, std::size_t, std::size_t, std::size_t, float, float);
template __global__ void kernel_scalar_step<D3Q27, D3Q7, double>(
    const double*, double*, const double*, std::size_t, std::size_t, std::size_t, double, double);

template cudaError_t launch_scalar_step_gpu<D2Q9, D2Q5, float>(
    const float*, float*, const float*, std::size_t, std::size_t, std::size_t, float, float, dim3);
template cudaError_t launch_scalar_step_gpu<D2Q9, D2Q5, double>(
    const double*, double*, const double*, std::size_t, std::size_t, std::size_t, double, double, dim3);
template cudaError_t launch_scalar_step_gpu<D3Q19, D3Q7, float>(
    const float*, float*, const float*, std::size_t, std::size_t, std::size_t, float, float, dim3);
template cudaError_t launch_scalar_step_gpu<D3Q19, D3Q7, double>(
    const double*, double*, const double*, std::size_t, std::size_t, std::size_t, double, double, dim3);
template cudaError_t launch_scalar_step_gpu<D3Q27, D3Q7, float>(
    const float*, float*, const float*, std::size_t, std::size_t, std::size_t, float, float, dim3);
template cudaError_t launch_scalar_step_gpu<D3Q27, D3Q7, double>(
    const double*, double*, const double*, std::size_t, std::size_t, std::size_t, double, double, dim3);

template cudaError_t launch_scalar_step_gpu<D2Q9, D2Q5, float>(
    const LatticeMemory<D2Q5, float>&,
    const LatticeMemory<D2Q9, float>&,
    const float*,
    float*,
    const float*,
    float,
    float,
    dim3);
template cudaError_t launch_scalar_step_gpu<D2Q9, D2Q5, double>(
    const LatticeMemory<D2Q5, double>&,
    const LatticeMemory<D2Q9, double>&,
    const double*,
    double*,
    const double*,
    double,
    double,
    dim3);
template cudaError_t launch_scalar_step_gpu<D3Q19, D3Q7, float>(
    const LatticeMemory<D3Q7, float>&,
    const LatticeMemory<D3Q19, float>&,
    const float*,
    float*,
    const float*,
    float,
    float,
    dim3);
template cudaError_t launch_scalar_step_gpu<D3Q19, D3Q7, double>(
    const LatticeMemory<D3Q7, double>&,
    const LatticeMemory<D3Q19, double>&,
    const double*,
    double*,
    const double*,
    double,
    double,
    dim3);
template cudaError_t launch_scalar_step_gpu<D3Q27, D3Q7, float>(
    const LatticeMemory<D3Q7, float>&,
    const LatticeMemory<D3Q27, float>&,
    const float*,
    float*,
    const float*,
    float,
    float,
    dim3);
template cudaError_t launch_scalar_step_gpu<D3Q27, D3Q7, double>(
    const LatticeMemory<D3Q7, double>&,
    const LatticeMemory<D3Q27, double>&,
    const double*,
    double*,
    const double*,
    double,
    double,
    dim3);

template __global__ void kernel_reaction_AB<D2Q9, D2Q5, float>(
    const float*, float*, const float*, float*, const float*,
    std::size_t, std::size_t, std::size_t, float, float);
template __global__ void kernel_reaction_AB<D2Q9, D2Q5, double>(
    const double*, double*, const double*, double*, const double*,
    std::size_t, std::size_t, std::size_t, double, double);
template __global__ void kernel_reaction_AB<D3Q19, D3Q7, float>(
    const float*, float*, const float*, float*, const float*,
    std::size_t, std::size_t, std::size_t, float, float);
template __global__ void kernel_reaction_AB<D3Q19, D3Q7, double>(
    const double*, double*, const double*, double*, const double*,
    std::size_t, std::size_t, std::size_t, double, double);
template __global__ void kernel_reaction_AB<D3Q27, D3Q7, float>(
    const float*, float*, const float*, float*, const float*,
    std::size_t, std::size_t, std::size_t, float, float);
template __global__ void kernel_reaction_AB<D3Q27, D3Q7, double>(
    const double*, double*, const double*, double*, const double*,
    std::size_t, std::size_t, std::size_t, double, double);

template cudaError_t launch_reaction_AB_gpu<D2Q9, D2Q5, float>(
    const float*, float*, const float*, float*, const float*,
    std::size_t, std::size_t, std::size_t, float, float, dim3);
template cudaError_t launch_reaction_AB_gpu<D2Q9, D2Q5, double>(
    const double*, double*, const double*, double*, const double*,
    std::size_t, std::size_t, std::size_t, double, double, dim3);
template cudaError_t launch_reaction_AB_gpu<D3Q19, D3Q7, float>(
    const float*, float*, const float*, float*, const float*,
    std::size_t, std::size_t, std::size_t, float, float, dim3);
template cudaError_t launch_reaction_AB_gpu<D3Q19, D3Q7, double>(
    const double*, double*, const double*, double*, const double*,
    std::size_t, std::size_t, std::size_t, double, double, dim3);
template cudaError_t launch_reaction_AB_gpu<D3Q27, D3Q7, float>(
    const float*, float*, const float*, float*, const float*,
    std::size_t, std::size_t, std::size_t, float, float, dim3);
template cudaError_t launch_reaction_AB_gpu<D3Q27, D3Q7, double>(
    const double*, double*, const double*, double*, const double*,
    std::size_t, std::size_t, std::size_t, double, double, dim3);

} // namespace lbm

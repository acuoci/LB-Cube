#pragma once
/**
 * @file lattice_core.hpp
 * @brief CPU fused collision-streaming loop for periodic LBM domains.
 *
 * This header ties the SoA memory views and stateless physics functions together
 * for host execution. The update uses a pull-streaming scheme: each destination
 * cell gathers incoming populations from periodic neighbors, performs the
 * compile-time selected collision, and writes the result to the next ping-pong
 * buffer.
 */

#include <array>
#include <cstddef>
#include <concepts>
#include <type_traits>

#include "lattice_memory.hpp"
#include "lattice_mrt.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

namespace lbm {

#ifndef LB_CUBE_COLLISION_TYPE_DEFINED
#define LB_CUBE_COLLISION_TYPE_DEFINED
/**
 * @brief Compile-time collision operator selection for fused LBM steps.
 *
 * The enum is used as a non-type template parameter so BGK and TRT dispatch is
 * resolved during compilation, leaving no runtime branch in the inner loop.
 */
enum class CollisionType {
    /** @brief Single-relaxation-time BGK collision. */
    BGK,
    /** @brief Two-relaxation-time symmetric/anti-symmetric collision. */
    TRT,
    /** @brief Multiple-relaxation-time moment-space collision. */
    MRT
};
#endif

namespace detail {

/**
 * @brief Dependent false value used to reject unsupported compile-time branches.
 *
 * Keeping the expression dependent on the lattice type ensures the assertion is
 * evaluated only when an unsupported MRT instantiation is actually requested.
 *
 * @tparam Lattice Lattice traits type used by the selected solver branch.
 */
template <typename Lattice>
inline constexpr bool always_false_v = false;

/**
 * @brief Compute a wrapped upstream coordinate for pull streaming.
 *
 * For a destination coordinate `x` and lattice velocity component `c`, the
 * source coordinate is `x - c`. Adding the extent before the modulo operation
 * handles negative one-cell displacements and implements periodic boundaries.
 *
 * @param coordinate Destination coordinate along one axis.
 * @param velocity Discrete velocity component along the same axis.
 * @param extent Domain size along the axis.
 * @return Periodically wrapped source coordinate.
 */
[[nodiscard]] constexpr std::size_t periodic_pull_index(
    std::size_t coordinate,
    int velocity,
    std::size_t extent) noexcept {
    const auto signed_coordinate = static_cast<std::ptrdiff_t>(coordinate);
    const auto signed_extent = static_cast<std::ptrdiff_t>(extent);
    const auto signed_index = signed_coordinate - static_cast<std::ptrdiff_t>(velocity) + signed_extent;
    return static_cast<std::size_t>(signed_index % signed_extent);
}

/**
 * @brief Apply the selected collision model to a gathered cell population array.
 *
 * @tparam CT Compile-time collision operator selection.
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param local_pops Populations gathered by pull streaming and overwritten with
 * post-collision values.
 * @param omega Even/BGK relaxation frequency.
 */
template <CollisionType CT, IsLatticeModel Lattice, std::floating_point Real>
inline void collide_cell(
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
        } else {
            static_assert(
                always_false_v<Lattice>,
                "MRT is currently only supported for D2Q9 and D3Q19.");
        }
    }
}

} // namespace detail

/**
 * @brief Advance the host population buffers by one fused LBM time step.
 *
 * The function reads from `mem.get_current_view()`, writes to
 * `mem.get_next_view()`, and calls `mem.swap_buffers()` when the entire domain
 * has been updated. The 2D and 3D code paths and the collision operator are
 * selected with `if constexpr`, so each compiled instantiation contains only the
 * relevant loop nest, indexing rank, and collision formula.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision used by the population buffers.
 * @tparam CT Compile-time collision operator. Defaults to BGK to preserve the
 * existing `step_cpu<Lattice, Real>(...)` call style.
 * @param mem Host SoA population storage participating in the ping-pong scheme.
 * @param omega BGK relaxation frequency, or TRT even relaxation frequency.
 */
template <
    IsLatticeModel Lattice,
    std::floating_point Real,
    CollisionType CT = CollisionType::BGK>
inline void step_cpu(LatticeMemory<Lattice, Real>& mem, Real omega) {
    auto current_view = mem.get_current_view();
    auto next_view = mem.get_next_view();

    if constexpr (Lattice::D == 2) {
        const std::size_t y_extent = current_view.extent(1);
        const std::size_t x_extent = current_view.extent(2);

        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                std::array<Real, static_cast<std::size_t>(Lattice::Q)> local_pops{};

                for (int i = 0; i < Lattice::Q; ++i) {
                    const auto direction_offset = static_cast<std::size_t>(i * Lattice::D);
                    const int cx = Lattice::directions[direction_offset];
                    const int cy = Lattice::directions[direction_offset + 1];
                    const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
                    const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);

                    local_pops[static_cast<std::size_t>(i)] =
                        current_view[static_cast<std::size_t>(i), ny, nx];
                }

                detail::collide_cell<CT, Lattice, Real>(local_pops, omega);

                for (int i = 0; i < Lattice::Q; ++i) {
                    next_view[static_cast<std::size_t>(i), y, x] =
                        local_pops[static_cast<std::size_t>(i)];
                }
            }
        }
    } else {
        const std::size_t z_extent = current_view.extent(1);
        const std::size_t y_extent = current_view.extent(2);
        const std::size_t x_extent = current_view.extent(3);

        for (std::size_t z = 0; z < z_extent; ++z) {
            for (std::size_t y = 0; y < y_extent; ++y) {
                for (std::size_t x = 0; x < x_extent; ++x) {
                    std::array<Real, static_cast<std::size_t>(Lattice::Q)> local_pops{};

                    for (int i = 0; i < Lattice::Q; ++i) {
                        const auto direction_offset = static_cast<std::size_t>(i * Lattice::D);
                        const int cx = Lattice::directions[direction_offset];
                        const int cy = Lattice::directions[direction_offset + 1];
                        const int cz = Lattice::directions[direction_offset + 2];
                        const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
                        const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);
                        const std::size_t nz = detail::periodic_pull_index(z, cz, z_extent);

                        local_pops[static_cast<std::size_t>(i)] =
                            current_view[static_cast<std::size_t>(i), nz, ny, nx];
                    }

                    detail::collide_cell<CT, Lattice, Real>(local_pops, omega);

                    for (int i = 0; i < Lattice::Q; ++i) {
                        next_view[static_cast<std::size_t>(i), z, y, x] =
                            local_pops[static_cast<std::size_t>(i)];
                    }
                }
            }
        }
    }

    mem.swap_buffers();
}

/**
 * @brief Advance a passive scalar field by one CPU LBM advection-diffusion step.
 *
 * The scalar field has its own lattice and ping-pong buffers, while the fluid
 * field supplies the advecting velocity at the destination cell. The scalar
 * populations are streamed with a pull scheme using `ScalarLattice::c`, relaxed
 * toward the first-order scalar equilibrium, and optionally receive a local
 * source contribution.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision shared by fluid and scalar populations.
 * @param scalar_mem Scalar SoA population storage participating in its own
 * ping-pong scheme.
 * @param fluid_mem Read-only fluid SoA population storage used to reconstruct
 * the advecting velocity.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param source_term Uniform scalar source contribution, defaulting to zero for
 * pure advection-diffusion.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
inline void step_scalar_cpu(
    LatticeMemory<ScalarLattice, Real>& scalar_mem,
    const LatticeMemory<FluidLattice, Real>& fluid_mem,
    Real omega_c,
    Real source_term = Real{}) {
    static_assert(FluidLattice::D == ScalarLattice::D);

    auto scalar_current = scalar_mem.get_current_view();
    auto scalar_next = scalar_mem.get_next_view();
    auto fluid_current = fluid_mem.get_current_view();

    if constexpr (ScalarLattice::D == 2) {
        const std::size_t y_extent = scalar_current.extent(1);
        const std::size_t x_extent = scalar_current.extent(2);

        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                std::array<Real, static_cast<std::size_t>(FluidLattice::Q)> fluid_pops{};
                for (int i = 0; i < FluidLattice::Q; ++i) {
                    fluid_pops[static_cast<std::size_t>(i)] =
                        fluid_current[static_cast<std::size_t>(i), y, x];
                }
                const MacroState<FluidLattice, Real> fluid_macro =
                    compute_macro_state<FluidLattice, Real>(fluid_pops);

                std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> scalar_pops{};
                for (int i = 0; i < ScalarLattice::Q; ++i) {
                    const auto direction_offset = static_cast<std::size_t>(i * ScalarLattice::D);
                    const int cx = ScalarLattice::c[direction_offset];
                    const int cy = ScalarLattice::c[direction_offset + 1];
                    const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
                    const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);

                    scalar_pops[static_cast<std::size_t>(i)] =
                        scalar_current[static_cast<std::size_t>(i), ny, nx];
                }

                collide_scalar_bgk<ScalarLattice, Real>(
                    scalar_pops,
                    fluid_macro.velocity,
                    omega_c,
                    source_term);

                for (int i = 0; i < ScalarLattice::Q; ++i) {
                    scalar_next[static_cast<std::size_t>(i), y, x] =
                        scalar_pops[static_cast<std::size_t>(i)];
                }
            }
        }
    } else {
        const std::size_t z_extent = scalar_current.extent(1);
        const std::size_t y_extent = scalar_current.extent(2);
        const std::size_t x_extent = scalar_current.extent(3);

        for (std::size_t z = 0; z < z_extent; ++z) {
            for (std::size_t y = 0; y < y_extent; ++y) {
                for (std::size_t x = 0; x < x_extent; ++x) {
                    std::array<Real, static_cast<std::size_t>(FluidLattice::Q)> fluid_pops{};
                    for (int i = 0; i < FluidLattice::Q; ++i) {
                        fluid_pops[static_cast<std::size_t>(i)] =
                            fluid_current[static_cast<std::size_t>(i), z, y, x];
                    }
                    const MacroState<FluidLattice, Real> fluid_macro =
                        compute_macro_state<FluidLattice, Real>(fluid_pops);

                    std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> scalar_pops{};
                    for (int i = 0; i < ScalarLattice::Q; ++i) {
                        const auto direction_offset = static_cast<std::size_t>(i * ScalarLattice::D);
                        const int cx = ScalarLattice::c[direction_offset];
                        const int cy = ScalarLattice::c[direction_offset + 1];
                        const int cz = ScalarLattice::c[direction_offset + 2];
                        const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
                        const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);
                        const std::size_t nz = detail::periodic_pull_index(z, cz, z_extent);

                        scalar_pops[static_cast<std::size_t>(i)] =
                            scalar_current[static_cast<std::size_t>(i), nz, ny, nx];
                    }

                    collide_scalar_bgk<ScalarLattice, Real>(
                        scalar_pops,
                        fluid_macro.velocity,
                        omega_c,
                        source_term);

                    for (int i = 0; i < ScalarLattice::Q; ++i) {
                        scalar_next[static_cast<std::size_t>(i), z, y, x] =
                            scalar_pops[static_cast<std::size_t>(i)];
                    }
                }
            }
        }
    }

    scalar_mem.swap_buffers();
}

/**
 * @brief Advance two reacting scalar species with fused advection-diffusion-reaction.
 *
 * The reaction model is the isothermal second-order process `A + B -> C`. The
 * local source increment is evaluated from the closed-form one-step batch
 * solution, then applied inside the scalar BGK collision without allocating
 * source-term fields. Both scalar memories are swapped after the full domain
 * update.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision shared by fluid and species populations.
 * @param fluid_mem Read-only fluid populations used to reconstruct advecting velocity.
 * @param species_a_mem Scalar population storage for reactant A.
 * @param species_b_mem Scalar population storage for reactant B.
 * @param omega_c Scalar BGK relaxation frequency for both reactants.
 * @param k_react Second-order reaction-rate constant.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
inline void step_reaction_AB(
    const LatticeMemory<FluidLattice, Real>& fluid_mem,
    LatticeMemory<ScalarLattice, Real>& species_a_mem,
    LatticeMemory<ScalarLattice, Real>& species_b_mem,
    Real omega_c,
    Real k_react) {
    static_assert(FluidLattice::D == ScalarLattice::D);

    auto fluid_current = fluid_mem.get_current_view();
    auto a_current = species_a_mem.get_current_view();
    auto a_next = species_a_mem.get_next_view();
    auto b_current = species_b_mem.get_current_view();
    auto b_next = species_b_mem.get_next_view();

    if constexpr (ScalarLattice::D == 2) {
        const std::size_t y_extent = a_current.extent(1);
        const std::size_t x_extent = a_current.extent(2);

        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                std::array<Real, static_cast<std::size_t>(FluidLattice::Q)> fluid_pops{};
                for (int i = 0; i < FluidLattice::Q; ++i) {
                    fluid_pops[static_cast<std::size_t>(i)] =
                        fluid_current[static_cast<std::size_t>(i), y, x];
                }
                const MacroState<FluidLattice, Real> fluid_macro =
                    compute_macro_state<FluidLattice, Real>(fluid_pops);

                std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> a_pops{};
                std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> b_pops{};
                for (int i = 0; i < ScalarLattice::Q; ++i) {
                    const auto direction_offset = static_cast<std::size_t>(i * ScalarLattice::D);
                    const int cx = ScalarLattice::c[direction_offset];
                    const int cy = ScalarLattice::c[direction_offset + 1];
                    const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
                    const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);
                    const auto q = static_cast<std::size_t>(i);

                    a_pops[q] = a_current[q, ny, nx];
                    b_pops[q] = b_current[q, ny, nx];
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
                    a_next[q, y, x] = a_pops[q];
                    b_next[q, y, x] = b_pops[q];
                }
            }
        }
    } else {
        const std::size_t z_extent = a_current.extent(1);
        const std::size_t y_extent = a_current.extent(2);
        const std::size_t x_extent = a_current.extent(3);

        for (std::size_t z = 0; z < z_extent; ++z) {
            for (std::size_t y = 0; y < y_extent; ++y) {
                for (std::size_t x = 0; x < x_extent; ++x) {
                    std::array<Real, static_cast<std::size_t>(FluidLattice::Q)> fluid_pops{};
                    for (int i = 0; i < FluidLattice::Q; ++i) {
                        fluid_pops[static_cast<std::size_t>(i)] =
                            fluid_current[static_cast<std::size_t>(i), z, y, x];
                    }
                    const MacroState<FluidLattice, Real> fluid_macro =
                        compute_macro_state<FluidLattice, Real>(fluid_pops);

                    std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> a_pops{};
                    std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> b_pops{};
                    for (int i = 0; i < ScalarLattice::Q; ++i) {
                        const auto direction_offset = static_cast<std::size_t>(i * ScalarLattice::D);
                        const int cx = ScalarLattice::c[direction_offset];
                        const int cy = ScalarLattice::c[direction_offset + 1];
                        const int cz = ScalarLattice::c[direction_offset + 2];
                        const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
                        const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);
                        const std::size_t nz = detail::periodic_pull_index(z, cz, z_extent);
                        const auto q = static_cast<std::size_t>(i);

                        a_pops[q] = a_current[q, nz, ny, nx];
                        b_pops[q] = b_current[q, nz, ny, nx];
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
                        a_next[q, z, y, x] = a_pops[q];
                        b_next[q, z, y, x] = b_pops[q];
                    }
                }
            }
        }
    }

    species_a_mem.swap_buffers();
    species_b_mem.swap_buffers();
}

} // namespace lbm

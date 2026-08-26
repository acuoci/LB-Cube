#pragma once
/**
 * @file lattice_physics.hpp
 * @brief Stateless single-cell LBM physics functions shared by CPU and CUDA code.
 *
 * The functions in this header operate only on local population arrays and
 * macroscopic value types. They are annotated for host and device compilation so
 * the CPU reference implementation and CUDA kernels use the same BGK collision
 * and equilibrium formulas.
 */

#include <Eigen/Dense>

#include <array>
#include <concepts>
#include <cstddef>

#include "lattice_memory.hpp"
#include "lattice_traits.hpp"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace lbm {

/**
 * @brief Reconstruct density and velocity from the populations of one cell.
 *
 * The macroscopic density is `rho = sum_i f_i`. Momentum is reconstructed as
 * `sum_i f_i c_i`, where `c_i` is read from the compile-time flattened velocity
 * table, and velocity is `u = momentum / rho`.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision used for local populations and returned
 * macroscopic fields.
 * @param local_pops Population values for one lattice node, ordered according to
 * the lattice traits.
 * @return Macroscopic density and velocity for the node.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
__host__ __device__ inline MacroState<Lattice, Real> compute_macro_state(
    const std::array<Real, static_cast<std::size_t>(Lattice::Q)>& local_pops) {
    using Velocity = typename MacroState<Lattice, Real>::Velocity;

    Real density{};
    Velocity momentum = Velocity::Zero();

    for (int i = 0; i < Lattice::Q; ++i) {
        const Real population = local_pops[static_cast<std::size_t>(i)];
        density += population;

        for (int d = 0; d < Lattice::D; ++d) {
            const auto direction_index = static_cast<std::size_t>(i * Lattice::D + d);
            momentum[d] += population * static_cast<Real>(Lattice::directions[direction_index]);
        }
    }

    MacroState<Lattice, Real> macro{};
    macro.density = density;
    macro.velocity = momentum / density;
    return macro;
}

/**
 * @brief Compute the second-order isothermal equilibrium population.
 *
 * This implements the standard weakly compressible LBM equilibrium
 * `f_i^eq = w_i rho [1 + 3(c_i.u) + 4.5(c_i.u)^2 - 1.5(u.u)]`, corresponding to
 * `c_s^2 = 1/3`. All constants are explicitly cast to `Real` so the same
 * expression can be instantiated for `float` or `double`.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision used for the computation.
 * @param i Discrete velocity index whose equilibrium population is requested.
 * @param macro Density and velocity at the lattice node.
 * @return Equilibrium value for population `i`.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
__host__ __device__ inline Real compute_equilibrium(
    int i,
    const MacroState<Lattice, Real>& macro) {
    using Velocity = typename MacroState<Lattice, Real>::Velocity;

    Velocity direction = Velocity::Zero();
    for (int d = 0; d < Lattice::D; ++d) {
        const auto direction_index = static_cast<std::size_t>(i * Lattice::D + d);
        direction[d] = static_cast<Real>(Lattice::directions[direction_index]);
    }

    const Real weight = static_cast<Real>(Lattice::weights[static_cast<std::size_t>(i)]);
    const Real c_dot_u = direction.dot(macro.velocity);
    const Real u_dot_u = macro.velocity.dot(macro.velocity);

    return weight * macro.density *
           (static_cast<Real>(1.0) +
            static_cast<Real>(3.0) * c_dot_u +
            static_cast<Real>(4.5) * c_dot_u * c_dot_u -
           static_cast<Real>(1.5) * u_dot_u);
}

/**
 * @brief Apply in-place BGK relaxation to the populations of one cell.
 *
 * The BGK collision updates each population as
 * `f_i <- f_i + omega * (f_i^eq - f_i)`, where `omega = 1 / tau`. The function is
 * deliberately local and stateless, making it usable in both the pull-streaming
 * CPU loop and the CUDA kernel after populations have been gathered from
 * neighboring cells.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision used for populations and relaxation.
 * @param local_pops Population array for one lattice node. Values are overwritten
 * with post-collision populations.
 * @param omega Relaxation frequency, equal to inverse relaxation time.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
__host__ __device__ inline void collide_bgk(
    std::array<Real, static_cast<std::size_t>(Lattice::Q)>& local_pops,
    Real omega) {
    const MacroState<Lattice, Real> macro = compute_macro_state<Lattice, Real>(local_pops);

    for (int i = 0; i < Lattice::Q; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const Real equilibrium = compute_equilibrium<Lattice, Real>(i, macro);
        local_pops[index] += omega * (equilibrium - local_pops[index]);
    }
}

} // namespace lbm

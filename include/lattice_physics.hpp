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
 * `f_i^eq = w_i rho [1 + (c_i.u)/c_s^2 + 0.5(c_i.u)^2/c_s^4
 * - 0.5(u.u)/c_s^2]`. For the fluid lattices currently provided,
 * `c_s^2 = 1/3`, which recovers the familiar coefficients 3, 4.5, and 1.5.
 * All constants are explicitly cast to `Real` so the same expression can be
 * instantiated for `float` or `double`.
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
    const Real cs2 = static_cast<Real>(Lattice::cs2);
    const Real inv_cs2 = static_cast<Real>(1.0) / cs2;
    const Real c_dot_u = direction.dot(macro.velocity);
    const Real u_dot_u = macro.velocity.dot(macro.velocity);

    return weight * macro.density *
           (static_cast<Real>(1.0) +
            inv_cs2 * c_dot_u +
            static_cast<Real>(0.5) * inv_cs2 * inv_cs2 * c_dot_u * c_dot_u -
            static_cast<Real>(0.5) * inv_cs2 * u_dot_u);
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

/**
 * @brief Reconstruct passive-scalar concentration from local scalar populations.
 *
 * For the pure LBM scalar model, concentration is the zeroth moment
 * `C = sum_i g_i`.
 *
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision used for scalar populations.
 * @param scalar_pops Scalar populations at one cell.
 * @return Local scalar concentration.
 */
template <IsLatticeModel ScalarLattice, std::floating_point Real>
__host__ __device__ inline Real compute_concentration(
    const std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)>& scalar_pops) {
    Real concentration{};
    for (int i = 0; i < ScalarLattice::Q; ++i) {
        concentration += scalar_pops[static_cast<std::size_t>(i)];
    }

    return concentration;
}

/**
 * @brief Compute the advection-diffusion scalar equilibrium population.
 *
 * The equilibrium is first order in the advecting fluid velocity:
 * `g_i^eq = w_i C (1 + (c_i.u) / c_s^2)`. This is the standard passive-scalar
 * LBM equilibrium for advection by an externally supplied velocity field.
 *
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision used for concentration and velocity.
 * @param i Scalar lattice direction index.
 * @param concentration Local scalar concentration.
 * @param fluid_velocity Fluid velocity reconstructed from the fluid populations.
 * @return Equilibrium scalar population for direction `i`.
 */
template <IsLatticeModel ScalarLattice, std::floating_point Real>
__host__ __device__ inline Real compute_scalar_equilibrium(
    int i,
    Real concentration,
    const Eigen::Matrix<Real, ScalarLattice::D, 1>& fluid_velocity) {
    Real c_dot_u{};
    for (int d = 0; d < ScalarLattice::D; ++d) {
        const auto direction_index = static_cast<std::size_t>(i * ScalarLattice::D + d);
        c_dot_u += static_cast<Real>(ScalarLattice::c[direction_index]) * fluid_velocity[d];
    }

    const Real weight = static_cast<Real>(ScalarLattice::weights[static_cast<std::size_t>(i)]);
    const Real cs2 = static_cast<Real>(ScalarLattice::cs2);

    return weight * concentration *
           (static_cast<Real>(1.0) + c_dot_u / cs2);
}

/**
 * @brief Apply BGK collision and source forcing to local scalar populations.
 *
 * The scalar populations relax toward `compute_scalar_equilibrium` using
 * `omega_c`, then receive an isotropically distributed local source term
 * `w_i * source_term`. Setting `source_term = 0` gives pure passive
 * advection-diffusion.
 *
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision used for scalar populations.
 * @param scalar_pops Scalar populations gathered at one destination cell. Values
 * are overwritten with post-collision populations.
 * @param fluid_velocity Advecting fluid velocity at the destination cell.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param source_term Local scalar source added over one time step.
 */
template <IsLatticeModel ScalarLattice, std::floating_point Real>
__host__ __device__ inline void collide_scalar_bgk(
    std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)>& scalar_pops,
    const Eigen::Matrix<Real, ScalarLattice::D, 1>& fluid_velocity,
    Real omega_c,
    Real source_term) {
    const Real concentration = compute_concentration<ScalarLattice, Real>(scalar_pops);

    for (int i = 0; i < ScalarLattice::Q; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const Real weight = static_cast<Real>(ScalarLattice::weights[index]);
        const Real equilibrium =
            compute_scalar_equilibrium<ScalarLattice, Real>(i, concentration, fluid_velocity);

        scalar_pops[index] += omega_c * (equilibrium - scalar_pops[index]) +
                              weight * source_term;
    }
}

} // namespace lbm

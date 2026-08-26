#pragma once

#include <Eigen/Dense>

#include <array>
#include <concepts>

#include "lattice_memory.hpp"
#include "lattice_traits.hpp"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace lbm {

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

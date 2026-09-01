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
#include <cmath>
#include <limits>
#include <type_traits>

#include "lattice_memory.hpp"
#include "lattice_traits.hpp"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace lbm {

namespace detail {

/**
 * @brief Dependent false value used for unsupported compile-time physics branches.
 *
 * @tparam Lattice Lattice traits type selected by a collision operator.
 */
template <typename Lattice>
inline constexpr bool unsupported_regularized_lattice_v = false;

} // namespace detail

/**
 * @brief Hermite projection prefactor for lattices with `c_s^2 = 1/3`.
 *
 * Regularized reconstruction uses `1 / (2 c_s^4)`. For D2Q9, D3Q19, and D3Q27,
 * `c_s^2 = 1/3`, so the prefactor is exactly `9 / 2`.
 *
 * @tparam Real Floating-point precision used for the reconstruction.
 */
template <std::floating_point Real>
inline constexpr Real hermite_projection_prefactor = Real{9} / Real{2};

/**
 * @brief Independent components of a symmetric second-order tensor in 2D.
 *
 * The regularized collision reconstructs the non-equilibrium stress tensor from
 * population moments. Naming the tensor components avoids fragile positional
 * indexing in later RLBM code while keeping the type trivially copyable for
 * host/device use.
 *
 * @tparam Real Floating-point precision used for tensor accumulation.
 */
template <typename Real>
struct SymmetricTensor2D {
    /** @brief `xx` normal component. */
    Real xx{0};
    /** @brief `yy` normal component. */
    Real yy{0};
    /** @brief `xy` shear component, equal to `yx` by symmetry. */
    Real xy{0};
};

/**
 * @brief Independent components of a symmetric second-order tensor in 3D.
 *
 * The six stored values represent the full tensor because
 * `Pi_xy = Pi_yx`, `Pi_xz = Pi_zx`, and `Pi_yz = Pi_zy`. This compact form is
 * enough for regularized reconstruction without carrying redundant off-diagonal
 * entries through hot kernels.
 *
 * @tparam Real Floating-point precision used for tensor accumulation.
 */
template <typename Real>
struct SymmetricTensor3D {
    /** @brief `xx` normal component. */
    Real xx{0};
    /** @brief `yy` normal component. */
    Real yy{0};
    /** @brief `zz` normal component. */
    Real zz{0};
    /** @brief `xy` shear component. */
    Real xy{0};
    /** @brief `xz` shear component. */
    Real xz{0};
    /** @brief `yz` shear component. */
    Real yz{0};
};

/**
 * @brief Compute the D2Q9 non-equilibrium momentum-flux tensor.
 *
 * The tensor is the second-order moment of the non-equilibrium populations,
 * `Pi_neq = sum_i (f_i - f_i^eq) c_i c_i`. This is the central quantity used by
 * regularized LBM operators to reconstruct the non-equilibrium part from
 * hydrodynamic stress rather than relaxing all higher-order artifacts directly.
 *
 * @tparam Real Floating-point precision used for populations and accumulation.
 * @param pops Local D2Q9 populations before regularized reconstruction.
 * @param feq Local D2Q9 equilibrium populations computed from the same macro state.
 * @return Independent components of `Pi_neq` in 2D.
 */
template <std::floating_point Real>
__host__ __device__ inline SymmetricTensor2D<Real> compute_pi_neq_d2q9(
    const std::array<Real, 9>& pops,
    const std::array<Real, 9>& feq) {
    SymmetricTensor2D<Real> pi_neq{};

    for (int i = 0; i < D2Q9::Q; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const auto direction_offset = static_cast<std::size_t>(i * D2Q9::D);
        const Real f_neq = pops[index] - feq[index];
        const Real cx = static_cast<Real>(D2Q9::c[direction_offset]);
        const Real cy = static_cast<Real>(D2Q9::c[direction_offset + 1]);

        pi_neq.xx += f_neq * cx * cx;
        pi_neq.yy += f_neq * cy * cy;
        pi_neq.xy += f_neq * cx * cy;
    }

    return pi_neq;
}

/**
 * @brief Compute the D3Q19 non-equilibrium momentum-flux tensor.
 *
 * The tensor is formed by projecting the non-equilibrium population vector onto
 * the symmetric second-order Hermite basis, `sum_i f_i^neq c_i_alpha c_i_beta`.
 * Storing only the six independent components keeps the result compact and
 * directly usable by the upcoming D3Q19 regularized collision operator.
 *
 * @tparam Real Floating-point precision used for populations and accumulation.
 * @param pops Local D3Q19 populations before regularized reconstruction.
 * @param feq Local D3Q19 equilibrium populations computed from the same macro state.
 * @return Independent components of `Pi_neq` in 3D.
 */
template <std::floating_point Real>
__host__ __device__ inline SymmetricTensor3D<Real> compute_pi_neq_d3q19(
    const std::array<Real, 19>& pops,
    const std::array<Real, 19>& feq) {
    SymmetricTensor3D<Real> pi_neq{};

    for (int i = 0; i < D3Q19::Q; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const auto direction_offset = static_cast<std::size_t>(i * D3Q19::D);
        const Real f_neq = pops[index] - feq[index];
        const Real cx = static_cast<Real>(D3Q19::c[direction_offset]);
        const Real cy = static_cast<Real>(D3Q19::c[direction_offset + 1]);
        const Real cz = static_cast<Real>(D3Q19::c[direction_offset + 2]);

        pi_neq.xx += f_neq * cx * cx;
        pi_neq.yy += f_neq * cy * cy;
        pi_neq.zz += f_neq * cz * cz;
        pi_neq.xy += f_neq * cx * cy;
        pi_neq.xz += f_neq * cx * cz;
        pi_neq.yz += f_neq * cy * cz;
    }

    return pi_neq;
}

/**
 * @brief Compute the D3Q27 non-equilibrium momentum-flux tensor.
 *
 * D3Q27 carries the full tensor-product velocity set, so the second-order
 * non-equilibrium stress is accumulated directly from all 27 population
 * residuals. The returned symmetric tensor feeds the D3Q27 recursive
 * regularized collision branch.
 *
 * @tparam Real Floating-point precision used for populations and accumulation.
 * @param pops Local D3Q27 populations before regularized reconstruction.
 * @param feq Local D3Q27 equilibrium populations computed from the same macro state.
 * @return Independent components of `Pi_neq` in 3D.
 */
template <std::floating_point Real>
__host__ __device__ inline SymmetricTensor3D<Real> compute_pi_neq_d3q27(
    const std::array<Real, 27>& pops,
    const std::array<Real, 27>& feq) {
    SymmetricTensor3D<Real> pi_neq{};

    for (int i = 0; i < D3Q27::Q; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const auto direction_offset = static_cast<std::size_t>(i * D3Q27::D);
        const Real f_neq = pops[index] - feq[index];
        const Real cx = static_cast<Real>(D3Q27::c[direction_offset]);
        const Real cy = static_cast<Real>(D3Q27::c[direction_offset + 1]);
        const Real cz = static_cast<Real>(D3Q27::c[direction_offset + 2]);

        pi_neq.xx += f_neq * cx * cx;
        pi_neq.yy += f_neq * cy * cy;
        pi_neq.zz += f_neq * cz * cz;
        pi_neq.xy += f_neq * cx * cy;
        pi_neq.xz += f_neq * cx * cz;
        pi_neq.yz += f_neq * cy * cz;
    }

    return pi_neq;
}

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
 * @brief Compute the TRT odd relaxation frequency from the even one.
 *
 * The Two-Relaxation-Time model relaxes symmetric and anti-symmetric population
 * components independently. Given `omega_plus = 1 / tau_plus`, this helper
 * applies the standard magic-parameter relation
 * `Lambda = (tau_plus - 0.5) * (tau_minus - 0.5)` and returns
 * `omega_minus = 1 / tau_minus`. The default `Lambda = 1/4` is a common
 * stability-oriented choice for periodic and simple boundary-condition studies.
 *
 * @tparam Real Floating-point precision used for relaxation parameters.
 * @param omega_plus Relaxation frequency for the even, viscosity-controlling
 * population component.
 * @param magic_param TRT magic parameter `Lambda`.
 * @return Odd relaxation frequency `omega_minus`.
 */
template <std::floating_point Real>
__host__ __device__ inline Real compute_omega_minus(
    Real omega_plus,
    Real magic_param = Real{1} / Real{4}) {
    const Real tau_plus = Real{1} / omega_plus;
    const Real tau_minus =
        magic_param / (tau_plus - Real{1} / Real{2}) + Real{1} / Real{2};
    return Real{1} / tau_minus;
}

/**
 * @brief Apply an in-place Two-Relaxation-Time collision to one cell.
 *
 * TRT decomposes each population pair `(i, opposite[i])` into even and odd
 * components. The even component controls the physical viscosity through
 * `omega_plus`; the odd component is relaxed with `omega_minus`, usually chosen
 * from a magic-parameter relation to improve numerical stability and boundary
 * behavior. Equilibria are computed first for every direction so the symmetric
 * and anti-symmetric equilibrium parts are paired consistently.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`, including an
 * `opposite` table where `c[opposite[i]] = -c[i]`.
 * @tparam Real Floating-point precision used for populations and relaxation.
 * @param pops Local population array for one lattice node. Values are
 * overwritten with post-TRT populations.
 * @param macro Density and velocity at the node, typically reconstructed from
 * `pops` before calling this function.
 * @param omega_plus Relaxation frequency for the even population component.
 * @param omega_minus Relaxation frequency for the odd population component.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
__host__ __device__ inline void collide_trt(
    std::array<Real, static_cast<std::size_t>(Lattice::Q)>& pops,
    const MacroState<Lattice, Real>& macro,
    Real omega_plus,
    Real omega_minus) {
    std::array<Real, static_cast<std::size_t>(Lattice::Q)> equilibrium{};
    std::array<Real, static_cast<std::size_t>(Lattice::Q)> post_collision{};

    for (int i = 0; i < Lattice::Q; ++i) {
        equilibrium[static_cast<std::size_t>(i)] =
            compute_equilibrium<Lattice, Real>(i, macro);
    }

    for (int i = 0; i < Lattice::Q; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const auto inverse = static_cast<std::size_t>(Lattice::opposite[index]);

        const Real even = Real{0.5} * (pops[index] + pops[inverse]);
        const Real equilibrium_even =
            Real{0.5} * (equilibrium[index] + equilibrium[inverse]);
        const Real odd = Real{0.5} * (pops[index] - pops[inverse]);
        const Real equilibrium_odd =
            Real{0.5} * (equilibrium[index] - equilibrium[inverse]);

        post_collision[index] =
            pops[index] -
            omega_plus * (even - equilibrium_even) -
            omega_minus * (odd - equilibrium_odd);
    }

    pops = post_collision;
}

/**
 * @brief Apply the regularized LBM collision operator to one cell.
 *
 * The regularized collision first computes the non-equilibrium stress tensor
 * `Pi_neq = sum_i (f_i - f_i^eq) c_i c_i`, then reconstructs the
 * non-equilibrium distribution with the second-order Hermite projection plus
 * the recursive third-order contribution
 * `a_abc^(3,neq) = u_a Pi_bc + u_b Pi_ac + u_c Pi_ab`. Including the recursive
 * third-order term preserves the hydrodynamic information needed for
 * second-order Navier-Stokes recovery while filtering non-hydrodynamic
 * artifacts before relaxation. The D3Q19 branch uses the D3Q19-compatible
 * paired third-order projection because the 19-velocity stencil does not carry
 * independent `aaa` or `xyz` third-order Hermite tensors. The D3Q27 branch uses
 * the full ten-component third-order tensor-product Hermite basis.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`. Currently
 * supported for `D2Q9`, `D3Q19`, and `D3Q27`.
 * @tparam Real Floating-point precision used for populations and reconstruction.
 * @param pops Local population array. Values are overwritten with post-RLBM populations.
 * @param macro Density and velocity reconstructed from the pre-collision populations.
 * @param omega Relaxation frequency controlling the non-equilibrium stress decay.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
__host__ __device__ inline void collide_regularized(
    std::array<Real, static_cast<std::size_t>(Lattice::Q)>& pops,
    const MacroState<Lattice, Real>& macro,
    Real omega) {
    std::array<Real, static_cast<std::size_t>(Lattice::Q)> feq{};
    for (int i = 0; i < Lattice::Q; ++i) {
        feq[static_cast<std::size_t>(i)] =
            compute_equilibrium<Lattice, Real>(i, macro);
    }

    if constexpr (std::is_same_v<Lattice, D2Q9>) {
        const SymmetricTensor2D<Real> pi_neq = compute_pi_neq_d2q9<Real>(pops, feq);
        const Real cs2 = static_cast<Real>(D2Q9::cs2);
        const Real ux = macro.velocity[0];
        const Real uy = macro.velocity[1];
        const Real a_xxx = Real{3} * ux * pi_neq.xx;
        const Real a_yyy = Real{3} * uy * pi_neq.yy;
        const Real a_xxy = Real{2} * ux * pi_neq.xy + uy * pi_neq.xx;
        const Real a_xyy = Real{2} * uy * pi_neq.xy + ux * pi_neq.yy;

        for (int i = 0; i < D2Q9::Q; ++i) {
            const auto index = static_cast<std::size_t>(i);
            const auto direction_offset = static_cast<std::size_t>(i * D2Q9::D);
            const Real cx = static_cast<Real>(D2Q9::c[direction_offset]);
            const Real cy = static_cast<Real>(D2Q9::c[direction_offset + 1]);
            const Real q_xx = cx * cx - cs2;
            const Real q_yy = cy * cy - cs2;
            const Real q_xy = cx * cy;
            const Real second_order_contraction =
                q_xx * pi_neq.xx +
                q_yy * pi_neq.yy +
                Real{2} * q_xy * pi_neq.xy;
            const Real h_xxx = cx * cx * cx - Real{3} * cs2 * cx;
            const Real h_yyy = cy * cy * cy - Real{3} * cs2 * cy;
            const Real h_xxy = cx * cx * cy - cs2 * cy;
            const Real h_xyy = cx * cy * cy - cs2 * cx;
            const Real third_order_contraction =
                a_xxx * h_xxx +
                a_yyy * h_yyy +
                Real{3} * (a_xxy * h_xxy + a_xyy * h_xyy);
            const Real reconstructed_neq =
                static_cast<Real>(D2Q9::weights[index]) *
                hermite_projection_prefactor<Real> *
                (second_order_contraction + third_order_contraction);

            pops[index] = feq[index] + (Real{1} - omega) * reconstructed_neq;
        }
    } else if constexpr (std::is_same_v<Lattice, D3Q19>) {
        const SymmetricTensor3D<Real> pi_neq = compute_pi_neq_d3q19<Real>(pops, feq);
        const Real cs2 = static_cast<Real>(D3Q19::cs2);
        const Real ux = macro.velocity[0];
        const Real uy = macro.velocity[1];
        const Real uz = macro.velocity[2];
        const Real a_xxx = Real{3} * ux * pi_neq.xx;
        const Real a_yyy = Real{3} * uy * pi_neq.yy;
        const Real a_zzz = Real{3} * uz * pi_neq.zz;
        const Real a_xxy = Real{2} * ux * pi_neq.xy + uy * pi_neq.xx;
        const Real a_xxz = Real{2} * ux * pi_neq.xz + uz * pi_neq.xx;
        const Real a_xyy = Real{2} * uy * pi_neq.xy + ux * pi_neq.yy;
        const Real a_yyz = Real{2} * uy * pi_neq.yz + uz * pi_neq.yy;
        const Real a_xzz = Real{2} * uz * pi_neq.xz + ux * pi_neq.zz;
        const Real a_yzz = Real{2} * uz * pi_neq.yz + uy * pi_neq.zz;
        const Real a_xyz = ux * pi_neq.yz + uy * pi_neq.xz + uz * pi_neq.xy;

        for (int i = 0; i < D3Q19::Q; ++i) {
            const auto index = static_cast<std::size_t>(i);
            const auto direction_offset = static_cast<std::size_t>(i * D3Q19::D);
            const Real cx = static_cast<Real>(D3Q19::c[direction_offset]);
            const Real cy = static_cast<Real>(D3Q19::c[direction_offset + 1]);
            const Real cz = static_cast<Real>(D3Q19::c[direction_offset + 2]);
            const Real q_xx = cx * cx - cs2;
            const Real q_yy = cy * cy - cs2;
            const Real q_zz = cz * cz - cs2;
            const Real q_xy = cx * cy;
            const Real q_xz = cx * cz;
            const Real q_yz = cy * cz;
            const Real second_order_contraction =
                q_xx * pi_neq.xx +
                q_yy * pi_neq.yy +
                q_zz * pi_neq.zz +
                Real{2} * (q_xy * pi_neq.xy + q_xz * pi_neq.xz + q_yz * pi_neq.yz);
            const Real h_xxx = cx * cx * cx - Real{3} * cs2 * cx;
            const Real h_yyy = cy * cy * cy - Real{3} * cs2 * cy;
            const Real h_zzz = cz * cz * cz - Real{3} * cs2 * cz;
            const Real h_xxy = cx * cx * cy - cs2 * cy;
            const Real h_xxz = cx * cx * cz - cs2 * cz;
            const Real h_xyy = cx * cy * cy - cs2 * cx;
            const Real h_yyz = cy * cy * cz - cs2 * cz;
            const Real h_xzz = cx * cz * cz - cs2 * cx;
            const Real h_yzz = cy * cz * cz - cs2 * cy;
            const Real h_xyz = cx * cy * cz;
            static_cast<void>(a_xxx);
            static_cast<void>(a_yyy);
            static_cast<void>(a_zzz);
            static_cast<void>(a_xyz);
            static_cast<void>(h_xxx);
            static_cast<void>(h_yyy);
            static_cast<void>(h_zzz);
            static_cast<void>(h_xyz);

            const Real third_order_contraction =
                Real{3} * (h_xxy + h_yzz) * (a_xxy + a_yzz) +
                (h_xxy - h_yzz) * (a_xxy - a_yzz) +
                Real{3} * (h_xzz + h_xyy) * (a_xzz + a_xyy) +
                (h_xzz - h_xyy) * (a_xzz - a_xyy) +
                Real{3} * (h_yyz + h_xxz) * (a_yyz + a_xxz) +
                (h_yyz - h_xxz) * (a_yyz - a_xxz);
            const Real reconstructed_neq =
                static_cast<Real>(D3Q19::weights[index]) *
                hermite_projection_prefactor<Real> *
                (second_order_contraction + third_order_contraction);

            pops[index] = feq[index] + (Real{1} - omega) * reconstructed_neq;
        }
    } else if constexpr (std::is_same_v<Lattice, D3Q27>) {
        const SymmetricTensor3D<Real> pi_neq = compute_pi_neq_d3q27<Real>(pops, feq);
        const Real cs2 = static_cast<Real>(D3Q27::cs2);
        const Real ux = macro.velocity[0];
        const Real uy = macro.velocity[1];
        const Real uz = macro.velocity[2];
        const Real a_xxx = Real{3} * ux * pi_neq.xx;
        const Real a_yyy = Real{3} * uy * pi_neq.yy;
        const Real a_zzz = Real{3} * uz * pi_neq.zz;
        const Real a_xxy = Real{2} * ux * pi_neq.xy + uy * pi_neq.xx;
        const Real a_xxz = Real{2} * ux * pi_neq.xz + uz * pi_neq.xx;
        const Real a_xyy = ux * pi_neq.yy + Real{2} * uy * pi_neq.xy;
        const Real a_yyz = Real{2} * uy * pi_neq.yz + uz * pi_neq.yy;
        const Real a_xzz = ux * pi_neq.zz + Real{2} * uz * pi_neq.xz;
        const Real a_yzz = uy * pi_neq.zz + Real{2} * uz * pi_neq.yz;
        const Real a_xyz = ux * pi_neq.yz + uy * pi_neq.xz + uz * pi_neq.xy;

        for (int i = 0; i < D3Q27::Q; ++i) {
            const auto index = static_cast<std::size_t>(i);
            const auto direction_offset = static_cast<std::size_t>(i * D3Q27::D);
            const Real cx = static_cast<Real>(D3Q27::c[direction_offset]);
            const Real cy = static_cast<Real>(D3Q27::c[direction_offset + 1]);
            const Real cz = static_cast<Real>(D3Q27::c[direction_offset + 2]);
            const Real q_xx = cx * cx - cs2;
            const Real q_yy = cy * cy - cs2;
            const Real q_zz = cz * cz - cs2;
            const Real q_xy = cx * cy;
            const Real q_xz = cx * cz;
            const Real q_yz = cy * cz;
            const Real second_order_contraction =
                q_xx * pi_neq.xx +
                q_yy * pi_neq.yy +
                q_zz * pi_neq.zz +
                Real{2} * (q_xy * pi_neq.xy + q_xz * pi_neq.xz + q_yz * pi_neq.yz);
            const Real h_xxx = cx * cx * cx - Real{3} * cs2 * cx;
            const Real h_yyy = cy * cy * cy - Real{3} * cs2 * cy;
            const Real h_zzz = cz * cz * cz - Real{3} * cs2 * cz;
            const Real h_xxy = cx * cx * cy - cs2 * cy;
            const Real h_xxz = cx * cx * cz - cs2 * cz;
            const Real h_xyy = cx * cy * cy - cs2 * cx;
            const Real h_yyz = cy * cy * cz - cs2 * cz;
            const Real h_xzz = cx * cz * cz - cs2 * cx;
            const Real h_yzz = cy * cz * cz - cs2 * cy;
            const Real h_xyz = cx * cy * cz;
            const Real third_order_contraction =
                a_xxx * h_xxx +
                a_yyy * h_yyy +
                a_zzz * h_zzz +
                Real{3} * (
                    a_xxy * h_xxy +
                    a_xxz * h_xxz +
                    a_xyy * h_xyy +
                    a_yyz * h_yyz +
                    a_xzz * h_xzz +
                    a_yzz * h_yzz) +
                Real{6} * a_xyz * h_xyz;
            const Real reconstructed_neq =
                static_cast<Real>(D3Q27::weights[index]) *
                hermite_projection_prefactor<Real> *
                (second_order_contraction + third_order_contraction);

            pops[index] = feq[index] + (Real{1} - omega) * reconstructed_neq;
        }
    } else {
        static_assert(
            detail::unsupported_regularized_lattice_v<Lattice>,
            "Regularized collision is currently supported only for D2Q9, D3Q19, and D3Q27.");
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

/**
 * @brief Compute the exact local source increment for `A + B -> C` over one step.
 *
 * The scalar collision accepts a source increment integrated over one lattice
 * time step. For the second-order batch reaction
 * `dC_A/dt = dC_B/dt = -k C_A C_B`, the concentration difference
 * `C_B - C_A` is invariant. This helper uses that closed-form solution to
 * avoid the time-discretization error of an explicit Euler source term while
 * preserving the fused no-extra-storage reaction update.
 *
 * @tparam Real Floating-point precision used for concentrations and rate.
 * @param concentration_a Local concentration of reactant A before reaction.
 * @param concentration_b Local concentration of reactant B before reaction.
 * @param k_react Second-order reaction-rate constant in lattice units.
 * @return Negative concentration increment to apply to both reactants.
 */
template <std::floating_point Real>
__host__ __device__ inline Real compute_reaction_ab_source(
    Real concentration_a,
    Real concentration_b,
    Real k_react) {
    const Real abs_a = concentration_a < Real{} ? -concentration_a : concentration_a;
    const Real abs_b = concentration_b < Real{} ? -concentration_b : concentration_b;
    const Real difference = concentration_b - concentration_a;
    const Real abs_difference = difference < Real{} ? -difference : difference;
    const Real tolerance =
        static_cast<Real>(64.0) * std::numeric_limits<Real>::epsilon() *
        (static_cast<Real>(1.0) + abs_a + abs_b);

    Real concentration_a_after{};
    if (abs_difference <= tolerance) {
        concentration_a_after =
            concentration_a /
            (static_cast<Real>(1.0) + k_react * concentration_a);
    } else {
#if defined(__CUDA_ARCH__)
        const Real attenuation = exp(-k_react * difference);
#else
        using std::exp;
        const Real attenuation = exp(-k_react * difference);
#endif
        concentration_a_after =
            concentration_a * difference * attenuation /
            (concentration_b - concentration_a * attenuation);
    }

    return concentration_a_after - concentration_a;
}

} // namespace lbm

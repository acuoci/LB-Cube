#pragma once
/**
 * @file lattice_mrt.hpp
 * @brief Algebraic MRT moment transformations for supported lattice models.
 *
 * This header starts the Multiple-Relaxation-Time implementation with the D2Q9
 * Lallemand-Luo moment basis. The transforms are written as explicit algebraic
 * expansions so they are usable in CPU and CUDA kernels without allocating dense
 * matrices or invoking runtime linear algebra.
 */

#include <array>

#include "lattice_memory.hpp"
#include "lattice_traits.hpp"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace lbm::mrt {

/**
 * @brief Diagonal MRT relaxation rates for the D2Q9 Lallemand-Luo basis.
 *
 * The conserved moments `rho`, `j_x`, and `j_y` are not relaxed by the collision
 * routine. This structure stores only the non-conserved groups used by the
 * diagonal collision matrix: bulk/energy modes, energy-squared mode, energy
 * fluxes, and viscous stresses.
 *
 * @tparam Real Floating-point precision used for relaxation rates.
 */
template <typename Real>
struct MrtRelaxationRates_D2Q9 {
    /**
     * @brief Construct stable non-hydrodynamic rates with unset viscosity rate.
     *
     * `s_nu` is left at zero because it is tied to the physical viscosity and is
     * normally supplied from the chosen relaxation time for the run.
     */
    __host__ __device__ constexpr MrtRelaxationRates_D2Q9() = default;

    /**
     * @brief Construct stable MRT rates with a supplied viscous relaxation rate.
     *
     * @param viscous_relaxation_rate Relaxation rate for stress moments 7 and 8.
     */
    __host__ __device__ constexpr explicit MrtRelaxationRates_D2Q9(
        Real viscous_relaxation_rate)
        : s_nu(viscous_relaxation_rate) {}

    /** @brief Relaxation rate for the energy mode `e` (moment 1). */
    Real s_e{Real{1.63}};
    /** @brief Relaxation rate for the energy-squared mode `epsilon` (moment 2). */
    Real s_eps{Real{1.14}};
    /** @brief Relaxation rate for energy-flux modes `q_x` and `q_y` (moments 4, 6). */
    Real s_q{Real{1.92}};
    /** @brief Relaxation rate for viscous stress modes `p_xx` and `p_xy` (moments 7, 8). */
    Real s_nu{};
};

/**
 * @brief Transform D2Q9 populations into Lallemand-Luo raw moments.
 *
 * The population ordering matches `lbm::D2Q9`: rest, east, north, west, south,
 * northeast, northwest, southwest, southeast. The returned moments are ordered
 * as `rho, e, epsilon, j_x, q_x, j_y, q_y, p_xx, p_xy`.
 *
 * @tparam Real Floating-point precision used for populations and moments.
 * @param f D2Q9 population vector.
 * @return Moment vector `m = M f` in the Lallemand-Luo orthogonal basis.
 */
template <typename Real>
__host__ __device__ inline std::array<Real, 9> compute_moments_d2q9(
    const std::array<Real, 9>& f) {
    return {
        f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8],
        -Real{4} * f[0] - f[1] - f[2] - f[3] - f[4] +
            Real{2} * (f[5] + f[6] + f[7] + f[8]),
        Real{4} * f[0] - Real{2} * (f[1] + f[2] + f[3] + f[4]) +
            f[5] + f[6] + f[7] + f[8],
        f[1] - f[3] + f[5] - f[6] - f[7] + f[8],
        -Real{2} * f[1] + Real{2} * f[3] + f[5] - f[6] - f[7] + f[8],
        f[2] - f[4] + f[5] + f[6] - f[7] - f[8],
        -Real{2} * f[2] + Real{2} * f[4] + f[5] + f[6] - f[7] - f[8],
        f[1] - f[2] + f[3] - f[4],
        f[5] - f[6] + f[7] - f[8]
    };
}

/**
 * @brief Transform D2Q9 Lallemand-Luo raw moments back to populations.
 *
 * This is the explicit inverse of the standard D2Q9 MRT matrix. The fractional
 * coefficients include the inverse basis normalization, so applying
 * `compute_populations_d2q9(compute_moments_d2q9(f))` recovers `f` to
 * floating-point roundoff.
 *
 * @tparam Real Floating-point precision used for moments and populations.
 * @param m Moment vector ordered as `rho, e, epsilon, j_x, q_x, j_y, q_y,
 * p_xx, p_xy`.
 * @return Population vector `f = M^{-1} m` in `lbm::D2Q9` ordering.
 */
template <typename Real>
__host__ __device__ inline std::array<Real, 9> compute_populations_d2q9(
    const std::array<Real, 9>& m) {
    return {
        Real{1} / Real{9} * m[0] -
            Real{1} / Real{9} * m[1] +
            Real{1} / Real{9} * m[2],
        Real{1} / Real{9} * m[0] -
            Real{1} / Real{36} * m[1] -
            Real{1} / Real{18} * m[2] +
            Real{1} / Real{6} * m[3] -
            Real{1} / Real{6} * m[4] +
            Real{1} / Real{4} * m[7],
        Real{1} / Real{9} * m[0] -
            Real{1} / Real{36} * m[1] -
            Real{1} / Real{18} * m[2] +
            Real{1} / Real{6} * m[5] -
            Real{1} / Real{6} * m[6] -
            Real{1} / Real{4} * m[7],
        Real{1} / Real{9} * m[0] -
            Real{1} / Real{36} * m[1] -
            Real{1} / Real{18} * m[2] -
            Real{1} / Real{6} * m[3] +
            Real{1} / Real{6} * m[4] +
            Real{1} / Real{4} * m[7],
        Real{1} / Real{9} * m[0] -
            Real{1} / Real{36} * m[1] -
            Real{1} / Real{18} * m[2] -
            Real{1} / Real{6} * m[5] +
            Real{1} / Real{6} * m[6] -
            Real{1} / Real{4} * m[7],
        Real{1} / Real{9} * m[0] +
            Real{1} / Real{18} * m[1] +
            Real{1} / Real{36} * m[2] +
            Real{1} / Real{6} * m[3] +
            Real{1} / Real{12} * m[4] +
            Real{1} / Real{6} * m[5] +
            Real{1} / Real{12} * m[6] +
            Real{1} / Real{4} * m[8],
        Real{1} / Real{9} * m[0] +
            Real{1} / Real{18} * m[1] +
            Real{1} / Real{36} * m[2] -
            Real{1} / Real{6} * m[3] -
            Real{1} / Real{12} * m[4] +
            Real{1} / Real{6} * m[5] +
            Real{1} / Real{12} * m[6] -
            Real{1} / Real{4} * m[8],
        Real{1} / Real{9} * m[0] +
            Real{1} / Real{18} * m[1] +
            Real{1} / Real{36} * m[2] -
            Real{1} / Real{6} * m[3] -
            Real{1} / Real{12} * m[4] -
            Real{1} / Real{6} * m[5] -
            Real{1} / Real{12} * m[6] +
            Real{1} / Real{4} * m[8],
        Real{1} / Real{9} * m[0] +
            Real{1} / Real{18} * m[1] +
            Real{1} / Real{36} * m[2] +
            Real{1} / Real{6} * m[3] +
            Real{1} / Real{12} * m[4] -
            Real{1} / Real{6} * m[5] -
            Real{1} / Real{12} * m[6] -
            Real{1} / Real{4} * m[8]
    };
}

/**
 * @brief Compute standard D2Q9 MRT equilibrium moments from macroscopic fields.
 *
 * The moment ordering is the Lallemand-Luo basis used by
 * `compute_moments_d2q9`: `rho, e, epsilon, j_x, q_x, j_y, q_y, p_xx, p_xy`.
 * Momentum moments are conserved, while the remaining equilibrium moments define
 * the weakly compressible isothermal Navier-Stokes closure.
 *
 * @tparam Real Floating-point precision used for the macroscopic state.
 * @param macro D2Q9 macroscopic density and velocity.
 * @return Equilibrium moment vector `m_eq`.
 */
template <typename Real>
__host__ __device__ inline std::array<Real, 9> compute_equilibrium_moments_d2q9(
    const lbm::MacroState<lbm::D2Q9, Real>& macro) {
    const Real rho = macro.density;
    const Real ux = macro.velocity[0];
    const Real uy = macro.velocity[1];
    const Real jx = rho * ux;
    const Real jy = rho * uy;
    const Real usq = ux * ux + uy * uy;

    return {
        rho,
        -Real{2} * rho + Real{3} * rho * usq,
        rho - Real{3} * rho * usq,
        jx,
        -jx,
        jy,
        -jy,
        rho * (ux * ux - uy * uy),
        rho * ux * uy
    };
}

/**
 * @brief Apply one D2Q9 Multiple-Relaxation-Time collision in moment space.
 *
 * Populations are transformed to the orthogonal MRT moment basis, non-conserved
 * moments are relaxed independently toward their equilibria using the diagonal
 * rates in `S`, and the post-collision moments are transformed back to
 * populations. Conserved moments `rho`, `j_x`, and `j_y` are intentionally left
 * unchanged.
 *
 * @tparam Real Floating-point precision used for populations and moments.
 * @param pops D2Q9 local population vector. Values are overwritten with
 * post-collision populations.
 * @param macro Macroscopic state used to construct equilibrium moments.
 * @param S Diagonal non-conserved relaxation rates for the MRT collision.
 */
template <typename Real>
__host__ __device__ inline void collide_mrt_d2q9(
    std::array<Real, 9>& pops,
    const lbm::MacroState<lbm::D2Q9, Real>& macro,
    const MrtRelaxationRates_D2Q9<Real>& S) {
    std::array<Real, 9> moments = compute_moments_d2q9<Real>(pops);
    const std::array<Real, 9> equilibrium =
        compute_equilibrium_moments_d2q9<Real>(macro);

    moments[1] -= S.s_e * (moments[1] - equilibrium[1]);
    moments[2] -= S.s_eps * (moments[2] - equilibrium[2]);
    moments[4] -= S.s_q * (moments[4] - equilibrium[4]);
    moments[6] -= S.s_q * (moments[6] - equilibrium[6]);
    moments[7] -= S.s_nu * (moments[7] - equilibrium[7]);
    moments[8] -= S.s_nu * (moments[8] - equilibrium[8]);

    pops = compute_populations_d2q9<Real>(moments);
}

} // namespace lbm::mrt

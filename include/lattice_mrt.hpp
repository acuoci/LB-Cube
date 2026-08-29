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
 * @brief Diagonal MRT relaxation rates for the D3Q19 d'Humieres basis.
 *
 * Conserved moments `rho`, `j_x`, `j_y`, and `j_z` are not relaxed by the
 * collision routine. The remaining modes are grouped by their hydrodynamic role:
 * energy modes, energy fluxes, viscous stresses, higher-order symmetric stress
 * modes, and anti-symmetric ghost modes.
 *
 * @tparam Real Floating-point precision used for relaxation rates.
 */
template <typename Real>
struct MrtRelaxationRates_D3Q19 {
    /**
     * @brief Construct stable non-hydrodynamic rates with unset viscosity rate.
     *
     * `s_nu` is left at zero because it is tied to the physical shear viscosity
     * and is normally supplied from the run's relaxation time.
     */
    __host__ __device__ constexpr MrtRelaxationRates_D3Q19() = default;

    /**
     * @brief Construct stable MRT rates with a supplied viscous relaxation rate.
     *
     * @param viscous_relaxation_rate Relaxation rate for stress moments 9-13.
     */
    __host__ __device__ constexpr explicit MrtRelaxationRates_D3Q19(
        Real viscous_relaxation_rate)
        : s_nu(viscous_relaxation_rate) {}

    /** @brief Relaxation rate for the energy mode `e` (moment 1). */
    Real s_e{Real{1.19}};
    /** @brief Relaxation rate for the energy-squared mode `epsilon` (moment 2). */
    Real s_eps{Real{1.4}};
    /** @brief Relaxation rate for energy-flux modes `q_x`, `q_y`, and `q_z`. */
    Real s_q{Real{1.2}};
    /** @brief Relaxation rate for viscous stress modes 9-13. */
    Real s_nu{};
    /** @brief Relaxation rate for anti-symmetric stress / higher-order modes 14-15. */
    Real s_pi{Real{1.4}};
    /** @brief Relaxation rate for higher-order kinetic modes 16-18. */
    Real s_m{Real{1.98}};
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

/**
 * @brief Transform D3Q19 populations into an orthogonal MRT moment basis.
 *
 * The population ordering matches `lbm::D3Q19`: rest, axial links, then face
 * diagonals in the order declared by the lattice trait. The returned moments are
 * ordered as `rho, e, epsilon, j_x, q_x, j_y, q_y, j_z, q_z, 3p_xx, p_ww,
 * p_xy, p_yz, p_xz, pi_xx, pi_ww, m_x, m_y, m_z`.
 *
 * @tparam Real Floating-point precision used for populations and moments.
 * @param f D3Q19 population vector.
 * @return Moment vector `m = M f` in a d'Humieres-style orthogonal basis.
 */
template <typename Real>
__host__ __device__ inline std::array<Real, 19> compute_moments_d3q19(
    const std::array<Real, 19>& f) {
    return {
        f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8] +
            f[9] + f[10] + f[11] + f[12] + f[13] + f[14] + f[15] +
            f[16] + f[17] + f[18],
        -Real{30} * f[0] -
            Real{11} * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6]) +
            Real{8} * (f[7] + f[8] + f[9] + f[10] + f[11] + f[12] +
                f[13] + f[14] + f[15] + f[16] + f[17] + f[18]),
        Real{12} * f[0] -
            Real{4} * (f[1] + f[2] + f[3] + f[4] + f[5] + f[6]) +
            f[7] + f[8] + f[9] + f[10] + f[11] + f[12] + f[13] +
            f[14] + f[15] + f[16] + f[17] + f[18],
        f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] +
            f[13] - f[14],
        -Real{4} * f[1] + Real{4} * f[2] + f[7] - f[8] + f[9] - f[10] +
            f[11] - f[12] + f[13] - f[14],
        f[3] - f[4] + f[7] + f[8] - f[9] - f[10] + f[15] - f[16] +
            f[17] - f[18],
        -Real{4} * f[3] + Real{4} * f[4] + f[7] + f[8] - f[9] -
            f[10] + f[15] - f[16] + f[17] - f[18],
        f[5] - f[6] + f[11] + f[12] - f[13] - f[14] + f[15] + f[16] -
            f[17] - f[18],
        -Real{4} * f[5] + Real{4} * f[6] + f[11] + f[12] - f[13] -
            f[14] + f[15] + f[16] - f[17] - f[18],
        Real{2} * (f[1] + f[2]) - f[3] - f[4] - f[5] - f[6] +
            f[7] + f[8] + f[9] + f[10] + f[11] + f[12] + f[13] +
            f[14] - Real{2} * (f[15] + f[16] + f[17] + f[18]),
        f[3] + f[4] - f[5] - f[6] + f[7] + f[8] + f[9] + f[10] -
            f[11] - f[12] - f[13] - f[14],
        f[7] - f[8] - f[9] + f[10],
        f[15] - f[16] - f[17] + f[18],
        f[11] - f[12] - f[13] + f[14],
        -Real{4} * (f[1] + f[2]) + Real{2} * (f[3] + f[4] + f[5] + f[6]) +
            f[7] + f[8] + f[9] + f[10] + f[11] + f[12] + f[13] +
            f[14] - Real{2} * (f[15] + f[16] + f[17] + f[18]),
        -Real{2} * (f[3] + f[4]) + Real{2} * (f[5] + f[6]) +
            f[7] + f[8] + f[9] + f[10] - f[11] - f[12] - f[13] - f[14],
        f[7] - f[8] + f[9] - f[10] - f[11] + f[12] - f[13] + f[14],
        -f[7] - f[8] + f[9] + f[10] + f[15] - f[16] + f[17] - f[18],
        f[11] + f[12] - f[13] - f[14] - f[15] - f[16] + f[17] + f[18]
    };
}

/**
 * @brief Transform D3Q19 MRT moments back to populations.
 *
 * The inverse uses the orthogonality relation `M^{-1} = M^T D^{-1}`, where
 * `D` contains the row norms of the hardcoded D3Q19 basis. This keeps the
 * scaling explicit and avoids storing or multiplying dense matrices.
 *
 * @tparam Real Floating-point precision used for moments and populations.
 * @param m Moment vector ordered as in `compute_moments_d3q19`.
 * @return Population vector `f = M^{-1} m` in `lbm::D3Q19` ordering.
 */
template <typename Real>
__host__ __device__ inline std::array<Real, 19> compute_populations_d3q19(
    const std::array<Real, 19>& m) {
    return {
        Real{1} / Real{19} * m[0] -
            Real{5} / Real{399} * m[1] +
            Real{1} / Real{21} * m[2],
        Real{1} / Real{19} * m[0] -
            Real{11} / Real{2394} * m[1] -
            Real{1} / Real{63} * m[2] +
            Real{1} / Real{10} * m[3] -
            Real{1} / Real{10} * m[4] +
            Real{1} / Real{18} * m[9] -
            Real{1} / Real{18} * m[14],
        Real{1} / Real{19} * m[0] -
            Real{11} / Real{2394} * m[1] -
            Real{1} / Real{63} * m[2] -
            Real{1} / Real{10} * m[3] +
            Real{1} / Real{10} * m[4] +
            Real{1} / Real{18} * m[9] -
            Real{1} / Real{18} * m[14],
        Real{1} / Real{19} * m[0] -
            Real{11} / Real{2394} * m[1] -
            Real{1} / Real{63} * m[2] +
            Real{1} / Real{10} * m[5] -
            Real{1} / Real{10} * m[6] -
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{36} * m[14] +
            Real{1} / Real{12} * m[10] -
            Real{1} / Real{12} * m[15],
        Real{1} / Real{19} * m[0] -
            Real{11} / Real{2394} * m[1] -
            Real{1} / Real{63} * m[2] -
            Real{1} / Real{10} * m[5] +
            Real{1} / Real{10} * m[6] -
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{36} * m[14] +
            Real{1} / Real{12} * m[10] -
            Real{1} / Real{12} * m[15],
        Real{1} / Real{19} * m[0] -
            Real{11} / Real{2394} * m[1] -
            Real{1} / Real{63} * m[2] +
            Real{1} / Real{10} * m[7] -
            Real{1} / Real{10} * m[8] -
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{36} * m[14] -
            Real{1} / Real{12} * m[10] +
            Real{1} / Real{12} * m[15],
        Real{1} / Real{19} * m[0] -
            Real{11} / Real{2394} * m[1] -
            Real{1} / Real{63} * m[2] -
            Real{1} / Real{10} * m[7] +
            Real{1} / Real{10} * m[8] -
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{36} * m[14] -
            Real{1} / Real{12} * m[10] +
            Real{1} / Real{12} * m[15],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] +
            Real{1} / Real{10} * m[3] +
            Real{1} / Real{40} * m[4] +
            Real{1} / Real{10} * m[5] +
            Real{1} / Real{40} * m[6] +
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{72} * m[14] +
            Real{1} / Real{12} * m[10] +
            Real{1} / Real{24} * m[15] +
            Real{1} / Real{4} * m[11] +
            Real{1} / Real{8} * m[16] -
            Real{1} / Real{8} * m[17],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] -
            Real{1} / Real{10} * m[3] -
            Real{1} / Real{40} * m[4] +
            Real{1} / Real{10} * m[5] +
            Real{1} / Real{40} * m[6] +
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{72} * m[14] +
            Real{1} / Real{12} * m[10] +
            Real{1} / Real{24} * m[15] -
            Real{1} / Real{4} * m[11] -
            Real{1} / Real{8} * m[16] -
            Real{1} / Real{8} * m[17],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] +
            Real{1} / Real{10} * m[3] +
            Real{1} / Real{40} * m[4] -
            Real{1} / Real{10} * m[5] -
            Real{1} / Real{40} * m[6] +
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{72} * m[14] +
            Real{1} / Real{12} * m[10] +
            Real{1} / Real{24} * m[15] -
            Real{1} / Real{4} * m[11] +
            Real{1} / Real{8} * m[16] +
            Real{1} / Real{8} * m[17],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] -
            Real{1} / Real{10} * m[3] -
            Real{1} / Real{40} * m[4] -
            Real{1} / Real{10} * m[5] -
            Real{1} / Real{40} * m[6] +
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{72} * m[14] +
            Real{1} / Real{12} * m[10] +
            Real{1} / Real{24} * m[15] +
            Real{1} / Real{4} * m[11] -
            Real{1} / Real{8} * m[16] +
            Real{1} / Real{8} * m[17],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] +
            Real{1} / Real{10} * m[3] +
            Real{1} / Real{40} * m[4] +
            Real{1} / Real{10} * m[7] +
            Real{1} / Real{40} * m[8] +
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{72} * m[14] -
            Real{1} / Real{12} * m[10] -
            Real{1} / Real{24} * m[15] +
            Real{1} / Real{4} * m[13] -
            Real{1} / Real{8} * m[16] +
            Real{1} / Real{8} * m[18],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] -
            Real{1} / Real{10} * m[3] -
            Real{1} / Real{40} * m[4] +
            Real{1} / Real{10} * m[7] +
            Real{1} / Real{40} * m[8] +
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{72} * m[14] -
            Real{1} / Real{12} * m[10] -
            Real{1} / Real{24} * m[15] -
            Real{1} / Real{4} * m[13] +
            Real{1} / Real{8} * m[16] +
            Real{1} / Real{8} * m[18],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] +
            Real{1} / Real{10} * m[3] +
            Real{1} / Real{40} * m[4] -
            Real{1} / Real{10} * m[7] -
            Real{1} / Real{40} * m[8] +
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{72} * m[14] -
            Real{1} / Real{12} * m[10] -
            Real{1} / Real{24} * m[15] -
            Real{1} / Real{4} * m[13] -
            Real{1} / Real{8} * m[16] -
            Real{1} / Real{8} * m[18],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] -
            Real{1} / Real{10} * m[3] -
            Real{1} / Real{40} * m[4] -
            Real{1} / Real{10} * m[7] -
            Real{1} / Real{40} * m[8] +
            Real{1} / Real{36} * m[9] +
            Real{1} / Real{72} * m[14] -
            Real{1} / Real{12} * m[10] -
            Real{1} / Real{24} * m[15] +
            Real{1} / Real{4} * m[13] +
            Real{1} / Real{8} * m[16] -
            Real{1} / Real{8} * m[18],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] +
            Real{1} / Real{10} * m[5] +
            Real{1} / Real{40} * m[6] +
            Real{1} / Real{10} * m[7] +
            Real{1} / Real{40} * m[8] -
            Real{1} / Real{18} * m[9] -
            Real{1} / Real{36} * m[14] +
            Real{1} / Real{4} * m[12] +
            Real{1} / Real{8} * m[17] -
            Real{1} / Real{8} * m[18],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] -
            Real{1} / Real{10} * m[5] -
            Real{1} / Real{40} * m[6] +
            Real{1} / Real{10} * m[7] +
            Real{1} / Real{40} * m[8] -
            Real{1} / Real{18} * m[9] -
            Real{1} / Real{36} * m[14] -
            Real{1} / Real{4} * m[12] -
            Real{1} / Real{8} * m[17] -
            Real{1} / Real{8} * m[18],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] +
            Real{1} / Real{10} * m[5] +
            Real{1} / Real{40} * m[6] -
            Real{1} / Real{10} * m[7] -
            Real{1} / Real{40} * m[8] -
            Real{1} / Real{18} * m[9] -
            Real{1} / Real{36} * m[14] -
            Real{1} / Real{4} * m[12] +
            Real{1} / Real{8} * m[17] +
            Real{1} / Real{8} * m[18],
        Real{1} / Real{19} * m[0] +
            Real{4} / Real{1197} * m[1] +
            Real{1} / Real{252} * m[2] -
            Real{1} / Real{10} * m[5] -
            Real{1} / Real{40} * m[6] -
            Real{1} / Real{10} * m[7] -
            Real{1} / Real{40} * m[8] -
            Real{1} / Real{18} * m[9] -
            Real{1} / Real{36} * m[14] +
            Real{1} / Real{4} * m[12] -
            Real{1} / Real{8} * m[17] +
            Real{1} / Real{8} * m[18]
    };
}

/**
 * @brief Compute standard D3Q19 MRT equilibrium moments from macroscopic fields.
 *
 * The moment ordering is the d'Humieres-style basis used by
 * `compute_moments_d3q19`: `rho, e, epsilon, j_x, q_x, j_y, q_y, j_z, q_z,
 * 3p_xx, p_ww, p_xy, p_yz, p_xz, pi_xx, pi_ww, m_x, m_y, m_z`.
 * Momentum moments are conserved; stress moments 9-13 encode the isothermal
 * weakly compressible Navier-Stokes closure, while ghost modes 14-18 have zero
 * equilibrium in the incompressible limit.
 *
 * @tparam Real Floating-point precision used for the macroscopic state.
 * @param macro D3Q19 macroscopic density and velocity.
 * @return Equilibrium moment vector `m_eq`.
 */
template <typename Real>
__host__ __device__ inline std::array<Real, 19> compute_equilibrium_moments_d3q19(
    const lbm::MacroState<lbm::D3Q19, Real>& macro) {
    const Real rho = macro.density;
    const Real ux = macro.velocity[0];
    const Real uy = macro.velocity[1];
    const Real uz = macro.velocity[2];
    const Real jx = rho * ux;
    const Real jy = rho * uy;
    const Real jz = rho * uz;
    const Real ux2 = ux * ux;
    const Real uy2 = uy * uy;
    const Real uz2 = uz * uz;
    const Real usq = ux2 + uy2 + uz2;

    const Real m9_eq = Real{3} * rho * (Real{2} * ux2 - uy2 - uz2);
    const Real m10_eq = rho * (uy2 - uz2);

    return {
        rho,
        -Real{11} * rho + Real{19} * rho * usq,
        Real{3} * rho - (Real{11} / Real{2}) * rho * usq,
        jx,
        -(Real{2} / Real{3}) * jx,
        jy,
        -(Real{2} / Real{3}) * jy,
        jz,
        -(Real{2} / Real{3}) * jz,
        m9_eq,
        m10_eq,
        rho * ux * uy,
        rho * uy * uz,
        rho * ux * uz,
        Real{0},
        Real{0},
        Real{0},
        Real{0},
        Real{0}
    };
}

/**
 * @brief Apply one D3Q19 Multiple-Relaxation-Time collision in moment space.
 *
 * Populations are transformed into the D3Q19 orthogonal moment basis, each
 * non-conserved mode is relaxed with its own diagonal rate, and the resulting
 * post-collision moments are transformed back to populations. The density and
 * three momentum components are skipped so collision preserves local mass and
 * momentum exactly up to floating-point roundoff.
 *
 * @tparam Real Floating-point precision used for populations and moments.
 * @param pops D3Q19 local population vector. Values are overwritten with
 * post-collision populations.
 * @param macro Macroscopic state used to construct equilibrium moments.
 * @param S Diagonal non-conserved relaxation rates for the D3Q19 MRT collision.
 */
template <typename Real>
__host__ __device__ inline void collide_mrt_d3q19(
    std::array<Real, 19>& pops,
    const lbm::MacroState<lbm::D3Q19, Real>& macro,
    const MrtRelaxationRates_D3Q19<Real>& S) {
    std::array<Real, 19> moments = compute_moments_d3q19<Real>(pops);
    const std::array<Real, 19> equilibrium =
        compute_equilibrium_moments_d3q19<Real>(macro);

    moments[1] -= S.s_e * (moments[1] - equilibrium[1]);
    moments[2] -= S.s_eps * (moments[2] - equilibrium[2]);
    moments[4] -= S.s_q * (moments[4] - equilibrium[4]);
    moments[6] -= S.s_q * (moments[6] - equilibrium[6]);
    moments[8] -= S.s_q * (moments[8] - equilibrium[8]);
    moments[9] -= S.s_nu * (moments[9] - equilibrium[9]);
    moments[10] -= S.s_nu * (moments[10] - equilibrium[10]);
    moments[11] -= S.s_nu * (moments[11] - equilibrium[11]);
    moments[12] -= S.s_nu * (moments[12] - equilibrium[12]);
    moments[13] -= S.s_nu * (moments[13] - equilibrium[13]);
    moments[14] -= S.s_pi * (moments[14] - equilibrium[14]);
    moments[15] -= S.s_pi * (moments[15] - equilibrium[15]);
    moments[16] -= S.s_m * (moments[16] - equilibrium[16]);
    moments[17] -= S.s_m * (moments[17] - equilibrium[17]);
    moments[18] -= S.s_m * (moments[18] - equilibrium[18]);

    pops = compute_populations_d3q19<Real>(moments);
}

} // namespace lbm::mrt

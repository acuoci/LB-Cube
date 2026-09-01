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
#include <cstddef>

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
 * @brief Diagonal MRT relaxation rates for the D3Q27 tensor-product basis.
 *
 * The D3Q27 basis below starts with conserved density and momentum, then groups
 * the second-order hydrodynamic stress modes, third-order energy-flux-like
 * modes, and progressively higher-order kinetic ghost modes. `s_nu` controls
 * the physical shear viscosity and is normally set from the runtime relaxation
 * frequency `omega`.
 *
 * @tparam Real Floating-point precision used for relaxation rates.
 */
template <typename Real>
struct MrtRelaxationRates_D3Q27 {
    /** @brief Construct stable non-hydrodynamic rates with unset viscosity rate. */
    __host__ __device__ constexpr MrtRelaxationRates_D3Q27() = default;

    /**
     * @brief Construct stable MRT rates with a supplied viscous relaxation rate.
     *
     * @param viscous_relaxation_rate Relaxation rate for moments carrying
     * second-order stress.
     */
    __host__ __device__ constexpr explicit MrtRelaxationRates_D3Q27(
        Real viscous_relaxation_rate)
        : s_nu(viscous_relaxation_rate),
          s_b(viscous_relaxation_rate) {}

    /** @brief Bulk/trace relaxation rate for the diagonal second-order mode. */
    Real s_b{Real{1}};
    /** @brief Relaxation rate for energy-flux-like third-order modes. */
    Real s_e{Real{1.1}};
    /** @brief Relaxation rate for mixed third-order and fourth-order kinetic modes. */
    Real s_eps{Real{1.2}};
    /** @brief Relaxation rate for high-order ghost modes. */
    Real s_q{Real{1.2}};
    /** @brief Relaxation rate for shear and normal-stress modes. */
    Real s_nu{};

    /**
     * @brief Expand grouped rates into the 27-entry diagonal collision vector.
     *
     * Entries 0-3 are density and momentum and are conserved. Entries 4-9 are
     * second-order bulk/shear stresses; entries 10-16 are third-order
     * energy-flux-like modes; entries 17-26 are higher-order kinetic modes.
     *
     * @return Diagonal relaxation vector ordered like `transform_to_moments_d3q27`.
     */
    [[nodiscard]] __host__ __device__ constexpr std::array<Real, 27> as_array() const {
        return {
            Real{0}, Real{0}, Real{0}, Real{0},
            s_b, s_nu, s_nu, s_nu, s_nu, s_nu,
            s_e, s_e, s_e, s_e, s_e, s_e, s_eps,
            s_q, s_q, s_q, s_q, s_q, s_q, s_q, s_q, s_q, s_q
        };
    }
};

/**
 * @brief Evaluate the one-dimensional D3Q27 orthogonal polynomial factor.
 *
 * The tensor-product MRT basis is built from `P0 = 1`, `P1 = c`, and
 * `P2 = 3 c^2 - 2`, which are mutually orthogonal under the unweighted
 * standard inner product over `c in {-1, 0, 1}`.
 *
 * @tparam Real Floating-point precision used for moment arithmetic.
 * @param order Polynomial order selector in `{0, 1, 2}`.
 * @param c Integer lattice velocity component.
 * @return Polynomial value at `c`.
 */
template <typename Real>
[[nodiscard]] __host__ __device__ constexpr Real d3q27_axis_basis(
    int order,
    int c) {
    const Real value = static_cast<Real>(c);
    if (order == 0) {
        return Real{1};
    }
    if (order == 1) {
        return value;
    }
    return Real{3} * value * value - Real{2};
}

/**
 * @brief Evaluate one row of the orthogonal D3Q27 MRT basis.
 *
 * Rows 0-9 are ordered as density, momentum, bulk/normal stresses, and shear
 * stresses. Remaining rows are tensor-product higher-order modes. The basis is
 * orthogonal by construction, so the inverse transform needs only row norms.
 *
 * @tparam Real Floating-point precision used for moment arithmetic.
 * @param row Moment row index.
 * @param cx X component of the lattice velocity.
 * @param cy Y component of the lattice velocity.
 * @param cz Z component of the lattice velocity.
 * @return Basis-row value at the supplied lattice direction.
 */
template <typename Real>
[[nodiscard]] __host__ __device__ constexpr Real d3q27_basis_value(
    int row,
    int cx,
    int cy,
    int cz) {
    const Real bx0 = d3q27_axis_basis<Real>(0, cx);
    const Real by0 = d3q27_axis_basis<Real>(0, cy);
    const Real bz0 = d3q27_axis_basis<Real>(0, cz);
    const Real bx1 = d3q27_axis_basis<Real>(1, cx);
    const Real by1 = d3q27_axis_basis<Real>(1, cy);
    const Real bz1 = d3q27_axis_basis<Real>(1, cz);
    const Real bx2 = d3q27_axis_basis<Real>(2, cx);
    const Real by2 = d3q27_axis_basis<Real>(2, cy);
    const Real bz2 = d3q27_axis_basis<Real>(2, cz);

    switch (row) {
    case 0:
        return bx0 * by0 * bz0;
    case 1:
        return bx1 * by0 * bz0;
    case 2:
        return bx0 * by1 * bz0;
    case 3:
        return bx0 * by0 * bz1;
    case 4:
        return bx2 * by0 * bz0 + bx0 * by2 * bz0 + bx0 * by0 * bz2;
    case 5:
        return Real{2} * bx2 * by0 * bz0 - bx0 * by2 * bz0 - bx0 * by0 * bz2;
    case 6:
        return bx0 * by2 * bz0 - bx0 * by0 * bz2;
    case 7:
        return bx1 * by1 * bz0;
    case 8:
        return bx0 * by1 * bz1;
    case 9:
        return bx1 * by0 * bz1;
    case 10:
        return bx2 * by1 * bz0;
    case 11:
        return bx2 * by0 * bz1;
    case 12:
        return bx1 * by2 * bz0;
    case 13:
        return bx0 * by2 * bz1;
    case 14:
        return bx1 * by0 * bz2;
    case 15:
        return bx0 * by1 * bz2;
    case 16:
        return bx1 * by1 * bz1;
    case 17:
        return bx2 * by2 * bz0;
    case 18:
        return bx2 * by0 * bz2;
    case 19:
        return bx0 * by2 * bz2;
    case 20:
        return bx2 * by1 * bz1;
    case 21:
        return bx1 * by2 * bz1;
    case 22:
        return bx1 * by1 * bz2;
    case 23:
        return bx2 * by2 * bz1;
    case 24:
        return bx2 * by1 * bz2;
    case 25:
        return bx1 * by2 * bz2;
    default:
        return bx2 * by2 * bz2;
    }
}

/**
 * @brief Return the squared norm of one D3Q27 orthogonal basis row.
 *
 * These values are the exact unweighted inner products of the hardcoded basis
 * rows over all 27 lattice directions. They implement
 * `M^{-1} = M^T D^{-1}` without storing an inverse matrix.
 *
 * @tparam Real Floating-point precision used for inverse scaling.
 * @param row Moment row index.
 * @return Squared norm of the selected basis row.
 */
template <typename Real>
[[nodiscard]] __host__ __device__ constexpr Real d3q27_basis_norm(int row) {
    switch (row) {
    case 0:
        return Real{27};
    case 1:
    case 2:
    case 3:
        return Real{18};
    case 4:
        return Real{162};
    case 5:
        return Real{324};
    case 6:
        return Real{108};
    case 7:
    case 8:
    case 9:
        return Real{12};
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
        return Real{36};
    case 16:
        return Real{8};
    case 17:
    case 18:
    case 19:
        return Real{108};
    case 20:
    case 21:
    case 22:
        return Real{24};
    case 23:
    case 24:
    case 25:
        return Real{72};
    default:
        return Real{216};
    }
}

/**
 * @brief Flatten a 3x3x3 tensor index used by separable D3Q27 transforms.
 *
 * @param a First tensor index.
 * @param b Second tensor index.
 * @param c Third tensor index.
 * @return Flat index in row-major tensor order.
 */
[[nodiscard]] __host__ __device__ constexpr std::size_t d3q27_tensor_index(
    int a,
    int b,
    int c) {
    return static_cast<std::size_t>((a * 3 + b) * 3 + c);
}

/**
 * @brief Convert a D3Q27 velocity component in `{-1, 0, 1}` to tensor index.
 *
 * @param c Integer lattice velocity component.
 * @return Tensor index in `{0, 1, 2}`.
 */
[[nodiscard]] __host__ __device__ constexpr int d3q27_axis_index(int c) {
    return c + 1;
}

/**
 * @brief Convert a tensor axis index to the associated D3Q27 velocity component.
 *
 * @param index Tensor index in `{0, 1, 2}`.
 * @return Integer lattice velocity component in `{-1, 0, 1}`.
 */
[[nodiscard]] __host__ __device__ constexpr int d3q27_axis_velocity(int index) {
    return index - 1;
}

/**
 * @brief Return the 1D squared norm for `P0`, `P1`, or `P2`.
 *
 * @param order Polynomial order selector.
 * @return Exact unweighted norm over the three D1Q3 abscissae.
 */
template <typename Real>
[[nodiscard]] __host__ __device__ constexpr Real d3q27_axis_norm(int order) {
    if (order == 0) {
        return Real{3};
    }
    if (order == 1) {
        return Real{2};
    }
    return Real{6};
}

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

/**
 * @brief Transform D3Q27 populations into the orthogonal tensor-product MRT basis.
 *
 * The first ten moments are ordered as `rho, j_x, j_y, j_z, e, p_xx, p_ww,
 * p_xy, p_yz, p_xz`; rows 10-26 are higher-order orthogonal polynomial modes.
 * The transform is evaluated from compact basis polynomials instead of a stored
 * dense matrix, keeping the implementation exact and device friendly.
 *
 * @tparam Real Floating-point precision used for populations and moments.
 * @param f D3Q27 population vector in `lbm::D3Q27` ordering.
 * @return Moment vector `m = M f`.
 */
template <typename Real>
__host__ __device__ inline std::array<Real, 27> transform_to_moments_d3q27(
    const std::array<Real, 27>& f) {
    std::array<Real, 27> populations_by_tensor{};
    for (int i = 0; i < lbm::D3Q27::Q; ++i) {
        const auto population_index = static_cast<std::size_t>(i);
        const auto direction_offset =
            static_cast<std::size_t>(i * lbm::D3Q27::D);
        populations_by_tensor[d3q27_tensor_index(
            d3q27_axis_index(lbm::D3Q27::c[direction_offset]),
            d3q27_axis_index(lbm::D3Q27::c[direction_offset + 1]),
            d3q27_axis_index(lbm::D3Q27::c[direction_offset + 2]))] =
            f[population_index];
    }

    std::array<Real, 27> x_transformed{};
    for (int ax = 0; ax < 3; ++ax) {
        for (int iy = 0; iy < 3; ++iy) {
            for (int iz = 0; iz < 3; ++iz) {
                Real value{};
                for (int ix = 0; ix < 3; ++ix) {
                    value +=
                        d3q27_axis_basis<Real>(ax, d3q27_axis_velocity(ix)) *
                        populations_by_tensor[d3q27_tensor_index(ix, iy, iz)];
                }
                x_transformed[d3q27_tensor_index(ax, iy, iz)] = value;
            }
        }
    }

    std::array<Real, 27> xy_transformed{};
    for (int ax = 0; ax < 3; ++ax) {
        for (int ay = 0; ay < 3; ++ay) {
            for (int iz = 0; iz < 3; ++iz) {
                Real value{};
                for (int iy = 0; iy < 3; ++iy) {
                    value +=
                        d3q27_axis_basis<Real>(ay, d3q27_axis_velocity(iy)) *
                        x_transformed[d3q27_tensor_index(ax, iy, iz)];
                }
                xy_transformed[d3q27_tensor_index(ax, ay, iz)] = value;
            }
        }
    }

    std::array<Real, 27> raw{};
    for (int ax = 0; ax < 3; ++ax) {
        for (int ay = 0; ay < 3; ++ay) {
            for (int az = 0; az < 3; ++az) {
                Real value{};
                for (int iz = 0; iz < 3; ++iz) {
                    value +=
                        d3q27_axis_basis<Real>(az, d3q27_axis_velocity(iz)) *
                        xy_transformed[d3q27_tensor_index(ax, ay, iz)];
                }
                raw[d3q27_tensor_index(ax, ay, az)] = value;
            }
        }
    }

    std::array<Real, 27> moments{};
    const Real r200 = raw[d3q27_tensor_index(2, 0, 0)];
    const Real r020 = raw[d3q27_tensor_index(0, 2, 0)];
    const Real r002 = raw[d3q27_tensor_index(0, 0, 2)];

    moments[0] = raw[d3q27_tensor_index(0, 0, 0)];
    moments[1] = raw[d3q27_tensor_index(1, 0, 0)];
    moments[2] = raw[d3q27_tensor_index(0, 1, 0)];
    moments[3] = raw[d3q27_tensor_index(0, 0, 1)];
    moments[4] = r200 + r020 + r002;
    moments[5] = Real{2} * r200 - r020 - r002;
    moments[6] = r020 - r002;
    moments[7] = raw[d3q27_tensor_index(1, 1, 0)];
    moments[8] = raw[d3q27_tensor_index(0, 1, 1)];
    moments[9] = raw[d3q27_tensor_index(1, 0, 1)];
    moments[10] = raw[d3q27_tensor_index(2, 1, 0)];
    moments[11] = raw[d3q27_tensor_index(2, 0, 1)];
    moments[12] = raw[d3q27_tensor_index(1, 2, 0)];
    moments[13] = raw[d3q27_tensor_index(0, 2, 1)];
    moments[14] = raw[d3q27_tensor_index(1, 0, 2)];
    moments[15] = raw[d3q27_tensor_index(0, 1, 2)];
    moments[16] = raw[d3q27_tensor_index(1, 1, 1)];
    moments[17] = raw[d3q27_tensor_index(2, 2, 0)];
    moments[18] = raw[d3q27_tensor_index(2, 0, 2)];
    moments[19] = raw[d3q27_tensor_index(0, 2, 2)];
    moments[20] = raw[d3q27_tensor_index(2, 1, 1)];
    moments[21] = raw[d3q27_tensor_index(1, 2, 1)];
    moments[22] = raw[d3q27_tensor_index(1, 1, 2)];
    moments[23] = raw[d3q27_tensor_index(2, 2, 1)];
    moments[24] = raw[d3q27_tensor_index(2, 1, 2)];
    moments[25] = raw[d3q27_tensor_index(1, 2, 2)];
    moments[26] = raw[d3q27_tensor_index(2, 2, 2)];

    return moments;
}

/**
 * @brief Transform D3Q27 MRT moments back to populations.
 *
 * Since the basis rows are orthogonal under the standard inner product, the
 * inverse projection is `f_i = sum_r m_r phi_r(c_i) / ||phi_r||^2`. The row
 * norms are hardcoded by `d3q27_basis_norm`, avoiding dense inverse storage.
 *
 * @tparam Real Floating-point precision used for moments and populations.
 * @param moments Moment vector ordered as in `transform_to_moments_d3q27`.
 * @return Population vector in `lbm::D3Q27` ordering.
 */
template <typename Real>
__host__ __device__ inline std::array<Real, 27> transform_to_populations_d3q27(
    const std::array<Real, 27>& moments) {
    std::array<Real, 27> raw{};

    raw[d3q27_tensor_index(0, 0, 0)] = moments[0];
    raw[d3q27_tensor_index(1, 0, 0)] = moments[1];
    raw[d3q27_tensor_index(0, 1, 0)] = moments[2];
    raw[d3q27_tensor_index(0, 0, 1)] = moments[3];

    raw[d3q27_tensor_index(2, 0, 0)] = (moments[4] + moments[5]) / Real{3};
    raw[d3q27_tensor_index(0, 2, 0)] =
        (Real{2} * moments[4] - moments[5] + Real{3} * moments[6]) / Real{6};
    raw[d3q27_tensor_index(0, 0, 2)] =
        (Real{2} * moments[4] - moments[5] - Real{3} * moments[6]) / Real{6};

    raw[d3q27_tensor_index(1, 1, 0)] = moments[7];
    raw[d3q27_tensor_index(0, 1, 1)] = moments[8];
    raw[d3q27_tensor_index(1, 0, 1)] = moments[9];
    raw[d3q27_tensor_index(2, 1, 0)] = moments[10];
    raw[d3q27_tensor_index(2, 0, 1)] = moments[11];
    raw[d3q27_tensor_index(1, 2, 0)] = moments[12];
    raw[d3q27_tensor_index(0, 2, 1)] = moments[13];
    raw[d3q27_tensor_index(1, 0, 2)] = moments[14];
    raw[d3q27_tensor_index(0, 1, 2)] = moments[15];
    raw[d3q27_tensor_index(1, 1, 1)] = moments[16];
    raw[d3q27_tensor_index(2, 2, 0)] = moments[17];
    raw[d3q27_tensor_index(2, 0, 2)] = moments[18];
    raw[d3q27_tensor_index(0, 2, 2)] = moments[19];
    raw[d3q27_tensor_index(2, 1, 1)] = moments[20];
    raw[d3q27_tensor_index(1, 2, 1)] = moments[21];
    raw[d3q27_tensor_index(1, 1, 2)] = moments[22];
    raw[d3q27_tensor_index(2, 2, 1)] = moments[23];
    raw[d3q27_tensor_index(2, 1, 2)] = moments[24];
    raw[d3q27_tensor_index(1, 2, 2)] = moments[25];
    raw[d3q27_tensor_index(2, 2, 2)] = moments[26];

    std::array<Real, 27> z_inverted{};
    for (int ax = 0; ax < 3; ++ax) {
        for (int ay = 0; ay < 3; ++ay) {
            for (int iz = 0; iz < 3; ++iz) {
                Real value{};
                for (int az = 0; az < 3; ++az) {
                    value += raw[d3q27_tensor_index(ax, ay, az)] *
                        d3q27_axis_basis<Real>(az, d3q27_axis_velocity(iz)) /
                        d3q27_axis_norm<Real>(az);
                }
                z_inverted[d3q27_tensor_index(ax, ay, iz)] = value;
            }
        }
    }

    std::array<Real, 27> yz_inverted{};
    for (int ax = 0; ax < 3; ++ax) {
        for (int iy = 0; iy < 3; ++iy) {
            for (int iz = 0; iz < 3; ++iz) {
                Real value{};
                for (int ay = 0; ay < 3; ++ay) {
                    value += z_inverted[d3q27_tensor_index(ax, ay, iz)] *
                        d3q27_axis_basis<Real>(ay, d3q27_axis_velocity(iy)) /
                        d3q27_axis_norm<Real>(ay);
                }
                yz_inverted[d3q27_tensor_index(ax, iy, iz)] = value;
            }
        }
    }

    std::array<Real, 27> populations_by_tensor{};
    for (int ix = 0; ix < 3; ++ix) {
        for (int iy = 0; iy < 3; ++iy) {
            for (int iz = 0; iz < 3; ++iz) {
                Real value{};
                for (int ax = 0; ax < 3; ++ax) {
                    value += yz_inverted[d3q27_tensor_index(ax, iy, iz)] *
                        d3q27_axis_basis<Real>(ax, d3q27_axis_velocity(ix)) /
                        d3q27_axis_norm<Real>(ax);
                }
                populations_by_tensor[d3q27_tensor_index(ix, iy, iz)] = value;
            }
        }
    }

    std::array<Real, 27> populations{};
    for (int i = 0; i < lbm::D3Q27::Q; ++i) {
        const auto population_index = static_cast<std::size_t>(i);
        const auto direction_offset =
            static_cast<std::size_t>(i * lbm::D3Q27::D);
        populations[population_index] =
            populations_by_tensor[d3q27_tensor_index(
                d3q27_axis_index(lbm::D3Q27::c[direction_offset]),
                d3q27_axis_index(lbm::D3Q27::c[direction_offset + 1]),
                d3q27_axis_index(lbm::D3Q27::c[direction_offset + 2]))];
    }

    return populations;
}

/**
 * @brief Compute D3Q27 MRT equilibrium moments from macroscopic fields.
 *
 * The equilibrium is the exact projection of the standard second-order
 * isothermal D3Q27 equilibrium populations onto the orthogonal MRT basis. This
 * keeps all hydrodynamic and ghost equilibria consistent with the population
 * equilibrium used elsewhere in the solver.
 *
 * @tparam Real Floating-point precision used for macroscopic fields.
 * @param rho Density.
 * @param ux X velocity.
 * @param uy Y velocity.
 * @param uz Z velocity.
 * @return Equilibrium moment vector ordered as in `transform_to_moments_d3q27`.
 */
template <typename Real>
__host__ __device__ inline std::array<Real, 27> compute_equilibrium_moments_d3q27(
    Real rho,
    Real ux,
    Real uy,
    Real uz) {
    std::array<Real, 27> equilibrium_populations{};
    const Real usq = ux * ux + uy * uy + uz * uz;

    for (int i = 0; i < lbm::D3Q27::Q; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const auto direction_offset =
            static_cast<std::size_t>(i * lbm::D3Q27::D);
        const Real cx = static_cast<Real>(lbm::D3Q27::c[direction_offset]);
        const Real cy = static_cast<Real>(lbm::D3Q27::c[direction_offset + 1]);
        const Real cz = static_cast<Real>(lbm::D3Q27::c[direction_offset + 2]);
        const Real cu = cx * ux + cy * uy + cz * uz;

        equilibrium_populations[index] =
            static_cast<Real>(lbm::D3Q27::weights[index]) *
            rho *
            (Real{1} + Real{3} * cu + Real{4.5} * cu * cu -
                Real{1.5} * usq);
    }

    return transform_to_moments_d3q27<Real>(equilibrium_populations);
}

/**
 * @brief Apply one D3Q27 Multiple-Relaxation-Time collision in moment space.
 *
 * The collision transforms the local population vector to the orthogonal D3Q27
 * moment basis, relaxes every non-conserved mode with its configured diagonal
 * rate, and projects the result back to populations. Density and momentum have
 * zero relaxation rates in `S.as_array()`, preserving local invariants.
 *
 * @tparam Real Floating-point precision used for populations and moments.
 * @param pops D3Q27 local population vector. Values are overwritten with
 * post-collision populations.
 * @param macro Macroscopic state used to construct equilibrium moments.
 * @param S Grouped diagonal relaxation rates for the D3Q27 MRT collision.
 */
template <typename Real>
__host__ __device__ inline void collide_mrt_d3q27(
    std::array<Real, 27>& pops,
    const lbm::MacroState<lbm::D3Q27, Real>& macro,
    const MrtRelaxationRates_D3Q27<Real>& S) {
    std::array<Real, 27> moments = transform_to_moments_d3q27<Real>(pops);
    const std::array<Real, 27> equilibrium =
        compute_equilibrium_moments_d3q27<Real>(
            macro.density,
            macro.velocity[0],
            macro.velocity[1],
            macro.velocity[2]);
    const std::array<Real, 27> relaxation_rates = S.as_array();

    for (int i = 0; i < lbm::D3Q27::Q; ++i) {
        const auto index = static_cast<std::size_t>(i);
        moments[index] -=
            relaxation_rates[index] * (moments[index] - equilibrium[index]);
    }

    pops = transform_to_populations_d3q27<Real>(moments);
}

} // namespace lbm::mrt

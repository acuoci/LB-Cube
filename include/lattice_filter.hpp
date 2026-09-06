#pragma once
/**
 * @file lattice_filter.hpp
 * @brief Reusable periodic filtering operations for scalar fields.
 *
 * The routines in this header operate on plain 3D scalar mdspan views rather
 * than on lattice populations. This keeps the filtering layer independent of
 * the LBM collision and streaming machinery while allowing the same operation
 * to be reused later for reactants, mixture fraction, products, and composite
 * scalar quantities.
 */

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "mdspan_compat.hpp"

namespace lbm {

/**
 * @brief Mutable 3D scalar view with logical extents `[Nz, Ny, Nx]`.
 *
 * @tparam Real Floating-point scalar precision.
 */
template <std::floating_point Real>
using ScalarField3DView = lbm::mdspan<
    Real,
    lbm::dextents<std::size_t, 3>,
    lbm::layout_right>;

/**
 * @brief Read-only 3D scalar view with logical extents `[Nz, Ny, Nx]`.
 *
 * @tparam Real Floating-point scalar precision.
 */
template <std::floating_point Real>
using ConstScalarField3DView = lbm::mdspan<
    const Real,
    lbm::dextents<std::size_t, 3>,
    lbm::layout_right>;

/**
 * @brief Validate that a box-filter width defines a centered odd stencil.
 *
 * @param n_filter Number of points per coordinate direction.
 *
 * @throws std::invalid_argument if `n_filter` is zero or even.
 */
inline void validate_box_filter_width(std::size_t n_filter) {
    if (n_filter == 0 || n_filter % 2 == 0) {
        throw std::invalid_argument(
            "box_filter_3d requires a positive odd filter width");
    }
}

/**
 * @brief Precompute periodic stencil indices for one coordinate direction.
 *
 * The returned table stores `n_filter` wrapped neighbors for every grid index,
 * ordered as offsets `[-h, ..., h]`, where `h = (n_filter - 1) / 2`.
 * Precomputing these indices avoids repeated modulo arithmetic in the innermost
 * accumulation loop while preserving exact periodic wrapping.
 *
 * @param extent Number of cells in the selected coordinate direction.
 * @param n_filter Positive odd filter width.
 * @return Flattened table indexed as `table[cell * n_filter + stencil_index]`.
 */
[[nodiscard]] inline std::vector<std::size_t> make_periodic_stencil_indices(
    std::size_t extent,
    std::size_t n_filter) {
    validate_box_filter_width(n_filter);
    if (extent == 0) {
        throw std::invalid_argument(
            "box_filter_3d requires strictly positive grid extents");
    }

    const auto half_width = static_cast<std::ptrdiff_t>((n_filter - 1) / 2);
    std::vector<std::size_t> indices(extent * n_filter);
    for (std::size_t cell = 0; cell < extent; ++cell) {
        for (std::size_t stencil = 0; stencil < n_filter; ++stencil) {
            const auto offset =
                static_cast<std::ptrdiff_t>(stencil) - half_width;
            const auto wrapped = static_cast<std::ptrdiff_t>(cell) + offset;
            const auto period = static_cast<std::ptrdiff_t>(extent);
            const auto positive_modulo = (wrapped % period + period) % period;
            indices[cell * n_filter + stencil] =
                static_cast<std::size_t>(positive_modulo);
        }
    }
    return indices;
}

/**
 * @brief Apply a normalized periodic 3D box filter to a scalar field.
 *
 * For an odd filter width `n_filter = 2h + 1`, the operation is
 *
 * \f[
 * \bar{\phi}(x,y,z) =
 * \frac{1}{n_\mathrm{filter}^3}
 * \sum_{r=-h}^{h}\sum_{q=-h}^{h}\sum_{p=-h}^{h}
 * \phi(x+p, y+q, z+r),
 * \f]
 *
 * where every neighbor index is wrapped periodically. The input and output
 * views must have identical extents and must refer to separate storage; this
 * conservative first implementation is intentionally out-of-place so that each
 * filtered value is computed from the unmodified input field.
 *
 * @tparam Real Floating-point scalar precision.
 * @param input Read-only scalar field with extents `[Nz, Ny, Nx]`.
 * @param output Mutable scalar field with extents `[Nz, Ny, Nx]`.
 * @param n_filter Positive odd number of stencil points per direction.
 *
 * @throws std::invalid_argument if the filter width or view extents are invalid.
 */
template <std::floating_point Real>
inline void box_filter_3d(
    ConstScalarField3DView<Real> input,
    ScalarField3DView<Real> output,
    std::size_t n_filter) {
    validate_box_filter_width(n_filter);

    const std::size_t nz = input.extent(0);
    const std::size_t ny = input.extent(1);
    const std::size_t nx = input.extent(2);
    if (nx == 0 || ny == 0 || nz == 0) {
        throw std::invalid_argument(
            "box_filter_3d requires strictly positive grid extents");
    }
    if (output.extent(0) != nz || output.extent(1) != ny ||
        output.extent(2) != nx) {
        throw std::invalid_argument(
            "box_filter_3d input and output extents must match");
    }
    if (input.data_handle() == static_cast<const Real*>(output.data_handle())) {
        throw std::invalid_argument(
            "box_filter_3d is an out-of-place filter; input and output storage must differ");
    }

    const auto x_indices = make_periodic_stencil_indices(nx, n_filter);
    const auto y_indices = make_periodic_stencil_indices(ny, n_filter);
    const auto z_indices = make_periodic_stencil_indices(nz, n_filter);
    const Real filter_width = static_cast<Real>(n_filter);
    const Real inverse_count =
        Real{1} / (filter_width * filter_width * filter_width);

#pragma omp parallel for collapse(3) schedule(static)
    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                Real sum{};
                for (std::size_t sz = 0; sz < n_filter; ++sz) {
                    const std::size_t zz = z_indices[z * n_filter + sz];
                    for (std::size_t sy = 0; sy < n_filter; ++sy) {
                        const std::size_t yy = y_indices[y * n_filter + sy];
                        for (std::size_t sx = 0; sx < n_filter; ++sx) {
                            const std::size_t xx = x_indices[x * n_filter + sx];
                            sum += input[zz, yy, xx];
                        }
                    }
                }
                output[z, y, x] = sum * inverse_count;
            }
        }
    }
}

} // namespace lbm

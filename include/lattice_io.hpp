#pragma once
/**
 * @file lattice_io.hpp
 * @brief Host-side diagnostics logging and legacy VTK visualization output.
 *
 * Output routines reconstruct macroscopic fields directly from the population
 * buffers via stateless physics functions. This avoids persistent density or
 * velocity allocations and keeps IO separate from the core stepping algorithms.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <fstream>
#include <format>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "lattice_memory.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

namespace lbm {

namespace detail {

/**
 * @brief Gather all populations of one cell from a dimension-dependent mdspan.
 *
 * The returned local array is the format expected by `compute_macro_state`.
 * Centralizing this indexing keeps diagnostics and VTK export consistent with
 * the `[Q, Y, X]` and `[Q, Z, Y, X]` memory contracts.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param view Read-only population view for the current time level.
 * @param x Cell x coordinate.
 * @param y Cell y coordinate.
 * @param z Cell z coordinate, ignored for 2D views.
 * @return Local population array ordered by lattice direction.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
[[nodiscard]] inline std::array<Real, static_cast<std::size_t>(Lattice::Q)> gather_cell_populations(
    typename LatticeMemory<Lattice, Real>::ConstView view,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    std::array<Real, static_cast<std::size_t>(Lattice::Q)> local_pops{};

    for (int i = 0; i < Lattice::Q; ++i) {
        const auto q = static_cast<std::size_t>(i);
        if constexpr (Lattice::D == 2) {
            local_pops[q] = view[q, y, x];
        } else {
            local_pops[q] = view[q, z, y, x];
        }
    }

    return local_pops;
}

/**
 * @brief Reconstruct macroscopic state at one grid point.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision used for populations and macros.
 * @param view Read-only population view for the current time level.
 * @param x Cell x coordinate.
 * @param y Cell y coordinate.
 * @param z Cell z coordinate, ignored for 2D views.
 * @return Density and velocity reconstructed from the cell populations.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
[[nodiscard]] inline MacroState<Lattice, Real> macro_at(
    typename LatticeMemory<Lattice, Real>::ConstView view,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    return compute_macro_state<Lattice, Real>(
        gather_cell_populations<Lattice, Real>(view, x, y, z));
}

/**
 * @brief Read a velocity component while padding 2D vectors to 3D for VTK.
 *
 * Legacy VTK vector fields require three components. For 2D lattices this helper
 * returns zero for the z component.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point velocity precision.
 * @param macro Macroscopic state whose velocity is being written.
 * @param component Component index in `[0, 2]`.
 * @return Requested velocity component, or zero for the 2D z component.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
[[nodiscard]] inline Real velocity_component(
    const MacroState<Lattice, Real>& macro,
    int component) {
    if constexpr (Lattice::D == 2) {
        return component < 2 ? macro.velocity[component] : Real{};
    } else {
        return macro.velocity[component];
    }
}

/**
 * @brief Reconstruct scalar concentration at one grid point.
 *
 * Scalar diagnostics use the same current-buffer gather path as VTK export, but
 * reduce the populations with `compute_concentration` rather than constructing a
 * fluid macroscopic state.
 *
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision used for scalar populations.
 * @param view Read-only scalar population view for the current time level.
 * @param x Cell x coordinate.
 * @param y Cell y coordinate.
 * @param z Cell z coordinate, ignored for 2D views.
 * @return Local scalar concentration.
 */
template <IsLatticeModel ScalarLattice, std::floating_point Real>
[[nodiscard]] inline Real concentration_at(
    typename LatticeMemory<ScalarLattice, Real>::ConstView view,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    return compute_concentration<ScalarLattice, Real>(
        gather_cell_populations<ScalarLattice, Real>(view, x, y, z));
}

} // namespace detail

/**
 * @brief Spatial scalar statistics for reactive-mixing diagnostics.
 *
 * These quantities are intended for lightweight, high-frequency logging during
 * DNS runs where full VTK output would be too expensive. The fields are spatial
 * expectations or rates, not integrated totals.
 *
 * @tparam Real Floating-point precision used for diagnostic values. Production
 * runs should instantiate this with `double` to reduce cancellation in variance
 * and covariance calculations.
 */
template <std::floating_point Real>
struct ReactiveDiagnostics {
    /** @brief Spatial mean of species A concentration, `<C_A>`. */
    Real mean_A{};
    /** @brief Spatial mean of species B concentration, `<C_B>`. */
    Real mean_B{};
    /** @brief Spatial variance of species A concentration. */
    Real var_A{};
    /** @brief Spatial variance of species B concentration. */
    Real var_B{};
    /** @brief Spatial covariance `<C_A C_B> - <C_A><C_B>`. */
    Real covariance{};
    /** @brief Normalized covariance, measuring scalar segregation. */
    Real segregation_intensity{};
    /** @brief Actual mean reaction rate `k_react <C_A C_B>`. */
    Real true_reaction_rate{};
    /** @brief Ideal perfectly mixed rate `k_react <C_A><C_B>`. */
    Real mixed_reaction_rate{};
};

/**
 * @brief Append integral diagnostics for the current population field to a CSV stream.
 *
 * The function reconstructs macros at each node and accumulates total mass,
 * total kinetic energy `0.5 * rho * |u|^2`, and maximum velocity magnitude. The
 * caller owns the stream so long-running simulations can keep one diagnostics
 * file open and append at selected output intervals.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param view Read-only view of the current population buffer.
 * @param time_step Simulation time step to write in the first CSV column.
 * @param csv_stream Open output stream receiving one diagnostics row.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
inline void log_diagnostics(
    typename LatticeMemory<Lattice, Real>::ConstView view,
    std::size_t time_step,
    std::ofstream& csv_stream) {
    Real total_mass{};
    Real total_kinetic_energy{};
    Real max_velocity_magnitude{};

    if constexpr (Lattice::D == 2) {
        const std::size_t y_extent = view.extent(1);
        const std::size_t x_extent = view.extent(2);

        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                const MacroState<Lattice, Real> macro =
                    detail::macro_at<Lattice, Real>(view, x, y, 0);
                const Real speed_squared = macro.velocity.squaredNorm();

                total_mass += macro.density;
                total_kinetic_energy += static_cast<Real>(0.5) * macro.density * speed_squared;
                max_velocity_magnitude =
                    std::max(max_velocity_magnitude, std::sqrt(speed_squared));
            }
        }
    } else {
        const std::size_t z_extent = view.extent(1);
        const std::size_t y_extent = view.extent(2);
        const std::size_t x_extent = view.extent(3);

        for (std::size_t z = 0; z < z_extent; ++z) {
            for (std::size_t y = 0; y < y_extent; ++y) {
                for (std::size_t x = 0; x < x_extent; ++x) {
                    const MacroState<Lattice, Real> macro =
                        detail::macro_at<Lattice, Real>(view, x, y, z);
                    const Real speed_squared = macro.velocity.squaredNorm();

                    total_mass += macro.density;
                    total_kinetic_energy += static_cast<Real>(0.5) * macro.density * speed_squared;
                    max_velocity_magnitude =
                        std::max(max_velocity_magnitude, std::sqrt(speed_squared));
                }
            }
        }
    }

    csv_stream << std::format(
        "{},{:.17g},{:.17g},{:.17g}\n",
        time_step,
        static_cast<double>(total_mass),
        static_cast<double>(total_kinetic_energy),
        static_cast<double>(max_velocity_magnitude));
}

/**
 * @brief Compute spatial reactive scalar statistics from current population buffers.
 *
 * This host implementation performs a single-pass reduction over the full
 * periodic domain and accumulates the five moments needed for reactive-mixing
 * diagnostics: `sum C_A`, `sum C_B`, `sum C_A^2`, `sum C_B^2`, and
 * `sum C_A C_B`. For a CUDA-resident production path, the same five-component
 * reduction should be moved to a parallel backend using a tuple-valued
 * `thrust::transform_reduce` or a custom CUB block/grid reduction to avoid
 * device-to-host field transfers.
 *
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point precision used for scalar populations and
 * diagnostic math. Prefer `double` for production statistics.
 * @param memA Current population memory for species A.
 * @param memB Current population memory for species B.
 * @param k_react Second-order reaction-rate constant in lattice units.
 * @return Spatial means, variances, covariance, segregation intensity, and
 * reaction-rate diagnostics.
 */
template <IsLatticeModel ScalarLattice, std::floating_point Real>
[[nodiscard]] inline ReactiveDiagnostics<Real> compute_reactive_stats(
    const LatticeMemory<ScalarLattice, Real>& memA,
    const LatticeMemory<ScalarLattice, Real>& memB,
    Real k_react) {
    const auto view_a = memA.get_current_view();
    const auto view_b = memB.get_current_view();

    Real sum_a{};
    Real sum_b{};
    Real sum_a2{};
    Real sum_b2{};
    Real sum_ab{};
    std::size_t cell_count{};

    if constexpr (ScalarLattice::D == 2) {
        const std::size_t y_extent = view_a.extent(1);
        const std::size_t x_extent = view_a.extent(2);
        cell_count = x_extent * y_extent;

        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                const Real concentration_a =
                    detail::concentration_at<ScalarLattice, Real>(view_a, x, y, 0);
                const Real concentration_b =
                    detail::concentration_at<ScalarLattice, Real>(view_b, x, y, 0);

                sum_a += concentration_a;
                sum_b += concentration_b;
                sum_a2 += concentration_a * concentration_a;
                sum_b2 += concentration_b * concentration_b;
                sum_ab += concentration_a * concentration_b;
            }
        }
    } else {
        const std::size_t z_extent = view_a.extent(1);
        const std::size_t y_extent = view_a.extent(2);
        const std::size_t x_extent = view_a.extent(3);
        cell_count = x_extent * y_extent * z_extent;

        for (std::size_t z = 0; z < z_extent; ++z) {
            for (std::size_t y = 0; y < y_extent; ++y) {
                for (std::size_t x = 0; x < x_extent; ++x) {
                    const Real concentration_a =
                        detail::concentration_at<ScalarLattice, Real>(view_a, x, y, z);
                    const Real concentration_b =
                        detail::concentration_at<ScalarLattice, Real>(view_b, x, y, z);

                    sum_a += concentration_a;
                    sum_b += concentration_b;
                    sum_a2 += concentration_a * concentration_a;
                    sum_b2 += concentration_b * concentration_b;
                    sum_ab += concentration_a * concentration_b;
                }
            }
        }
    }

    const Real inv_cell_count = Real{1} / static_cast<Real>(cell_count);
    const Real mean_a = sum_a * inv_cell_count;
    const Real mean_b = sum_b * inv_cell_count;
    const Real mean_a2 = sum_a2 * inv_cell_count;
    const Real mean_b2 = sum_b2 * inv_cell_count;
    const Real mean_ab = sum_ab * inv_cell_count;
    const Real mixed_reaction_rate = k_react * mean_a * mean_b;
    const Real covariance = mean_ab - mean_a * mean_b;
    const Real mean_product = mean_a * mean_b;
    const Real segregation_intensity =
        std::abs(mean_product) > std::numeric_limits<Real>::epsilon()
            ? covariance / mean_product
            : Real{};

    ReactiveDiagnostics<Real> diagnostics{};
    diagnostics.mean_A = mean_a;
    diagnostics.mean_B = mean_b;
    diagnostics.var_A = mean_a2 - mean_a * mean_a;
    diagnostics.var_B = mean_b2 - mean_b * mean_b;
    diagnostics.covariance = covariance;
    diagnostics.segregation_intensity = segregation_intensity;
    diagnostics.true_reaction_rate = k_react * mean_ab;
    diagnostics.mixed_reaction_rate = mixed_reaction_rate;
    return diagnostics;
}

/**
 * @brief Append reactive scalar statistics to a CSV diagnostics stream.
 *
 * A suitable CSV header is:
 * `step,mean_A,mean_B,var_A,var_B,covariance,segregation_intensity,true_reaction_rate,mixed_reaction_rate`.
 *
 * @tparam Real Floating-point precision stored in the diagnostics object.
 * @param csv_stream Open output stream receiving one CSV row.
 * @param time_step Simulation time step written in the first column.
 * @param diagnostics Reactive scalar statistics returned by
 * `compute_reactive_stats`.
 */
template <std::floating_point Real>
inline void log_reactive_diagnostics(
    std::ofstream& csv_stream,
    std::size_t time_step,
    const ReactiveDiagnostics<Real>& diagnostics) {
    csv_stream << std::format(
        "{},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g}\n",
        time_step,
        static_cast<double>(diagnostics.mean_A),
        static_cast<double>(diagnostics.mean_B),
        static_cast<double>(diagnostics.var_A),
        static_cast<double>(diagnostics.var_B),
        static_cast<double>(diagnostics.covariance),
        static_cast<double>(diagnostics.segregation_intensity),
        static_cast<double>(diagnostics.true_reaction_rate),
        static_cast<double>(diagnostics.mixed_reaction_rate));
}

/**
 * @brief Write density and velocity fields to a legacy ASCII VTK file.
 *
 * The exporter creates `lbm_XXXXXX.vtk` files using the `STRUCTURED_POINTS`
 * dataset. Density, velocity magnitude, and velocity vectors are reconstructed on
 * demand from the current populations so visualization does not impose separate
 * macroscopic storage in the solver state.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param mem Host memory object containing the current population buffer.
 * @param x_extent Number of nodes in x to write.
 * @param y_extent Number of nodes in y to write.
 * @param z_extent Number of nodes in z to write; forced to 1 for 2D lattices.
 * @param time_step Time step encoded into the file name and VTK title.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
inline void write_vtk(
    const LatticeMemory<Lattice, Real>& mem,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    std::size_t time_step) {
    if constexpr (Lattice::D == 2) {
        z_extent = 1;
    }

    const std::string filename = std::format("lbm_{:06}.vtk", time_step);
    std::ofstream vtk{filename};
    const auto view = mem.get_current_view();
    const std::size_t point_count = x_extent * y_extent * z_extent;

    vtk << "# vtk DataFile Version 3.0\n";
    vtk << std::format("LBM output step {}\n", time_step);
    vtk << "ASCII\n";
    vtk << "DATASET STRUCTURED_POINTS\n";
    vtk << std::format("DIMENSIONS {} {} {}\n", x_extent, y_extent, z_extent);
    vtk << "ORIGIN 0 0 0\n";
    vtk << "SPACING 1 1 1\n";
    vtk << std::format("POINT_DATA {}\n", point_count);

    vtk << "SCALARS density double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t z = 0; z < z_extent; ++z) {
        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                const MacroState<Lattice, Real> macro =
                    detail::macro_at<Lattice, Real>(view, x, y, z);
                vtk << std::format("{:.17g}\n", static_cast<double>(macro.density));
            }
        }
    }

    vtk << "SCALARS velocity_magnitude double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t z = 0; z < z_extent; ++z) {
        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                const MacroState<Lattice, Real> macro =
                    detail::macro_at<Lattice, Real>(view, x, y, z);
                vtk << std::format(
                    "{:.17g}\n",
                    static_cast<double>(std::sqrt(macro.velocity.squaredNorm())));
            }
        }
    }

    vtk << "VECTORS velocity double\n";
    for (std::size_t z = 0; z < z_extent; ++z) {
        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                const MacroState<Lattice, Real> macro =
                    detail::macro_at<Lattice, Real>(view, x, y, z);
                vtk << std::format(
                    "{:.17g} {:.17g} {:.17g}\n",
                    static_cast<double>(detail::velocity_component<Lattice, Real>(macro, 0)),
                    static_cast<double>(detail::velocity_component<Lattice, Real>(macro, 1)),
                    static_cast<double>(detail::velocity_component<Lattice, Real>(macro, 2)));
            }
        }
    }
}

} // namespace lbm

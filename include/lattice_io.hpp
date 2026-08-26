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

} // namespace detail

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

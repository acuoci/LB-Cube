#pragma once

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

template <IsLatticeModel Lattice, std::floating_point Real>
[[nodiscard]] inline MacroState<Lattice, Real> macro_at(
    typename LatticeMemory<Lattice, Real>::ConstView view,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    return compute_macro_state<Lattice, Real>(
        gather_cell_populations<Lattice, Real>(view, x, y, z));
}

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

/**
 * @file compare_operators.cpp
 * @brief Compare BGK, TRT, and MRT collision operators on the same 2D reactive-mixing case.
 *
 * The executable runs three independent D2Q9/D2Q5 simulations with identical
 * initial conditions. Each run writes scalar mixing diagnostics at high
 * frequency and periodic VTK snapshots so the operators can be compared in
 * terms of mixing, segregation, and reaction progress.
 */

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <format>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>

#include "lattice_core.hpp"
#include "lattice_io.hpp"
#include "lattice_memory.hpp"
#include "lattice_mrt.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

namespace {

using FluidLattice = lbm::D2Q9;
using ScalarLattice = lbm::D2Q5;
using Real = double;
using lbm::CollisionType;

constexpr std::size_t nx = 256;
constexpr std::size_t ny = 256;
constexpr int total_steps = 5000;
constexpr int stats_frequency = 10;
constexpr int vtk_frequency = 500;
constexpr Real fluid_relaxation_time = Real{0.6};
constexpr Real scalar_relaxation_time = Real{0.8};
constexpr Real initial_velocity = Real{0.04};
constexpr Real reaction_rate = Real{0.05};

[[nodiscard]] std::string_view operator_name(CollisionType collision_type) {
    switch (collision_type) {
    case CollisionType::BGK:
        return "BGK";
    case CollisionType::TRT:
        return "TRT";
    case CollisionType::MRT:
        return "MRT";
    }

    return "UNKNOWN";
}

[[nodiscard]] lbm::MacroState<FluidLattice, Real> taylor_green_macro(
    std::size_t x,
    std::size_t y) {
    const Real kx = Real{2} * std::numbers::pi_v<Real> / static_cast<Real>(nx);
    const Real ky = Real{2} * std::numbers::pi_v<Real> / static_cast<Real>(ny);
    const Real phase_x = kx * static_cast<Real>(x);
    const Real phase_y = ky * static_cast<Real>(y);
    const Real density_perturbation =
        (initial_velocity * initial_velocity) /
        (Real{4} * static_cast<Real>(FluidLattice::cs2)) *
        (std::cos(Real{2} * phase_x) + std::cos(Real{2} * phase_y));

    lbm::MacroState<FluidLattice, Real> macro{};
    macro.density = Real{1} - density_perturbation;
    macro.velocity << -initial_velocity * std::cos(phase_x) * std::sin(phase_y),
        initial_velocity * std::sin(phase_x) * std::cos(phase_y);
    return macro;
}

void initialize_fields(
    lbm::LatticeMemory<FluidLattice, Real>& fluid,
    lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    lbm::LatticeMemory<ScalarLattice, Real>& species_b) {
    auto fluid_view = fluid.get_current_view();
    auto a_view = species_a.get_current_view();
    auto b_view = species_b.get_current_view();

    for (std::size_t y_index = 0; y_index < ny; ++y_index) {
        for (std::size_t x_index = 0; x_index < nx; ++x_index) {
            const lbm::MacroState<FluidLattice, Real> macro =
                taylor_green_macro(x_index, y_index);

            for (int i = 0; i < FluidLattice::Q; ++i) {
                fluid_view[static_cast<std::size_t>(i), y_index, x_index] =
                    lbm::compute_equilibrium<FluidLattice, Real>(i, macro);
            }

            const Real concentration_a = x_index < nx / 2 ? Real{1} : Real{0};
            const Real concentration_b = x_index < nx / 2 ? Real{0} : Real{1};

            for (int i = 0; i < ScalarLattice::Q; ++i) {
                const auto q = static_cast<std::size_t>(i);
                a_view[q, y_index, x_index] =
                    lbm::compute_scalar_equilibrium<ScalarLattice, Real>(
                        i,
                        concentration_a,
                        macro.velocity);
                b_view[q, y_index, x_index] =
                    lbm::compute_scalar_equilibrium<ScalarLattice, Real>(
                        i,
                        concentration_b,
                        macro.velocity);
            }
        }
    }
}

[[nodiscard]] lbm::MacroState<FluidLattice, Real> fluid_macro_at(
    typename lbm::LatticeMemory<FluidLattice, Real>::ConstView view,
    std::size_t x_index,
    std::size_t y_index) {
    std::array<Real, static_cast<std::size_t>(FluidLattice::Q)> populations{};

    for (int i = 0; i < FluidLattice::Q; ++i) {
        populations[static_cast<std::size_t>(i)] =
            view[static_cast<std::size_t>(i), y_index, x_index];
    }

    return lbm::compute_macro_state<FluidLattice, Real>(populations);
}

[[nodiscard]] Real scalar_concentration_at(
    typename lbm::LatticeMemory<ScalarLattice, Real>::ConstView view,
    std::size_t x_index,
    std::size_t y_index) {
    std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> populations{};

    for (int i = 0; i < ScalarLattice::Q; ++i) {
        populations[static_cast<std::size_t>(i)] =
            view[static_cast<std::size_t>(i), y_index, x_index];
    }

    return lbm::compute_concentration<ScalarLattice, Real>(populations);
}

void write_comparison_vtk(
    const std::filesystem::path& output_dir,
    std::string_view collision_name,
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b,
    std::size_t time_step) {
    const std::filesystem::path filename =
        output_dir / std::format("comparison_{}_{:06}.vtk", collision_name, time_step);
    std::ofstream vtk{filename};
    if (!vtk) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    const auto fluid_view = fluid.get_current_view();
    const auto a_view = species_a.get_current_view();
    const auto b_view = species_b.get_current_view();
    constexpr std::size_t point_count = nx * ny;

    vtk << "# vtk DataFile Version 3.0\n";
    vtk << std::format("LB-Cube {} comparison step {}\n", collision_name, time_step);
    vtk << "ASCII\n";
    vtk << "DATASET STRUCTURED_POINTS\n";
    vtk << std::format("DIMENSIONS {} {} 1\n", nx, ny);
    vtk << "ORIGIN 0 0 0\n";
    vtk << "SPACING 1 1 1\n";
    vtk << std::format("POINT_DATA {}\n", point_count);

    vtk << "VECTORS velocity double\n";
    for (std::size_t y_index = 0; y_index < ny; ++y_index) {
        for (std::size_t x_index = 0; x_index < nx; ++x_index) {
            const lbm::MacroState<FluidLattice, Real> macro =
                fluid_macro_at(fluid_view, x_index, y_index);
            vtk << std::format(
                "{:.17g} {:.17g} 0\n",
                static_cast<double>(macro.velocity[0]),
                static_cast<double>(macro.velocity[1]));
        }
    }

    vtk << "SCALARS C_A double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t y_index = 0; y_index < ny; ++y_index) {
        for (std::size_t x_index = 0; x_index < nx; ++x_index) {
            vtk << std::format(
                "{:.17g}\n",
                static_cast<double>(scalar_concentration_at(a_view, x_index, y_index)));
        }
    }

    vtk << "SCALARS C_B double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t y_index = 0; y_index < ny; ++y_index) {
        for (std::size_t x_index = 0; x_index < nx; ++x_index) {
            vtk << std::format(
                "{:.17g}\n",
                static_cast<double>(scalar_concentration_at(b_view, x_index, y_index)));
        }
    }

    vtk << "SCALARS reaction_rate double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t y_index = 0; y_index < ny; ++y_index) {
        for (std::size_t x_index = 0; x_index < nx; ++x_index) {
            const Real concentration_a = scalar_concentration_at(a_view, x_index, y_index);
            const Real concentration_b = scalar_concentration_at(b_view, x_index, y_index);
            vtk << std::format(
                "{:.17g}\n",
                static_cast<double>(reaction_rate * concentration_a * concentration_b));
        }
    }
}

template <CollisionType CT>
void run_operator_case(std::string_view collision_name) {
    const std::filesystem::path output_dir = std::format("output_{}", collision_name);
    std::filesystem::create_directories(output_dir);

    const std::filesystem::path stats_filename =
        output_dir / std::format("stats_{}.csv", collision_name);
    std::ofstream stats{stats_filename};
    if (!stats) {
        throw std::runtime_error("failed to open " + stats_filename.string());
    }

    stats
        << "step,mean_A,mean_B,var_A,var_B,covariance,"
        << "segregation_intensity,true_reaction_rate,mixed_reaction_rate\n";

    lbm::LatticeMemory<FluidLattice, Real> fluid{nx, ny};
    lbm::LatticeMemory<ScalarLattice, Real> species_a{nx, ny};
    lbm::LatticeMemory<ScalarLattice, Real> species_b{nx, ny};
    initialize_fields(fluid, species_a, species_b);

    const Real omega = Real{1} / fluid_relaxation_time;
    const Real omega_c = Real{1} / scalar_relaxation_time;

    std::cout << "Running " << collision_name << " comparison case\n" << std::flush;

    lbm::log_reactive_diagnostics(
        stats,
        0,
        lbm::compute_reactive_stats<ScalarLattice, Real>(
            species_a,
            species_b,
            reaction_rate));
    write_comparison_vtk(output_dir, collision_name, fluid, species_a, species_b, 0);

    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 1; step <= total_steps; ++step) {
        lbm::step_cpu<FluidLattice, Real, CT>(fluid, omega);
        lbm::step_reaction_AB<FluidLattice, ScalarLattice, Real>(
            fluid,
            species_a,
            species_b,
            omega_c,
            reaction_rate);

        if (step % stats_frequency == 0) {
            lbm::log_reactive_diagnostics(
                stats,
                static_cast<std::size_t>(step),
                lbm::compute_reactive_stats<ScalarLattice, Real>(
                    species_a,
                    species_b,
                    reaction_rate));
        }

        if (step % vtk_frequency == 0) {
            write_comparison_vtk(
                output_dir,
                collision_name,
                fluid,
                species_a,
                species_b,
                static_cast<std::size_t>(step));
            std::cout << collision_name << ": step " << step << " / " << total_steps
                      << " complete\n"
                      << std::flush;
        }
    }

    const auto stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    const double updates =
        static_cast<double>(nx) * static_cast<double>(ny) * static_cast<double>(total_steps);
    const double mlups = updates / elapsed.count() / 1.0e6;

    std::cout << collision_name << " complete in " << elapsed.count() << " s ("
              << mlups << " MLUPS fluid step basis)\n"
              << std::flush;
}

void run_operator_case(CollisionType collision_type) {
    switch (collision_type) {
    case CollisionType::BGK:
        run_operator_case<CollisionType::BGK>(operator_name(collision_type));
        break;
    case CollisionType::TRT:
        run_operator_case<CollisionType::TRT>(operator_name(collision_type));
        break;
    case CollisionType::MRT:
        run_operator_case<CollisionType::MRT>(operator_name(collision_type));
        break;
    }
}

} // namespace

int main() {
    try {
        const auto operators = {CollisionType::BGK, CollisionType::TRT, CollisionType::MRT};

        std::cout << "LB-Cube collision operator comparison\n"
                  << "Grid: " << nx << " x " << ny << '\n'
                  << "Steps per operator: " << total_steps << '\n'
                  << "Stats frequency: " << stats_frequency << '\n'
                  << "VTK frequency: " << vtk_frequency << '\n'
                  << "Backend: CPU\n"
                  << std::flush;

        for (const CollisionType collision_type : operators) {
            run_operator_case(collision_type);
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

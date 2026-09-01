/**
 * @file turbulent_shear_layer.cpp
 * @brief High-Reynolds-number doubly periodic shear-layer benchmark.
 *
 * This executable compares BGK, TRT, and MRT on a challenging Kelvin-Helmholtz
 * instability setup. The case is intentionally configured near the stability
 * limit so BGK can fail gracefully while TRT and MRT can be inspected under the
 * same initial conditions.
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
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

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

constexpr std::size_t nx = 512;
constexpr std::size_t ny = 512;
constexpr int total_steps = 30000;
constexpr int stability_check_frequency = 100;
constexpr int vtk_frequency = 1000;
constexpr Real u0 = Real{0.04};
constexpr Real perturbation_amplitude = Real{0.05};
constexpr Real shear_steepness = Real{80.0};
constexpr Real fluid_relaxation_time = Real{0.5005};
constexpr Real scalar_relaxation_time = Real{0.8};
constexpr Real crash_velocity_threshold = Real{1.0};

[[nodiscard]] lbm::MacroState<FluidLattice, Real> shear_layer_macro(
    std::size_t x_index,
    std::size_t y_index) {
    const Real x_normalized = static_cast<Real>(x_index) / static_cast<Real>(nx);
    const Real y_normalized = static_cast<Real>(y_index) / static_cast<Real>(ny);

    const Real ux = y_normalized <= Real{0.5}
        ? u0 * std::tanh(shear_steepness * (y_normalized - Real{0.25}))
        : u0 * std::tanh(shear_steepness * (Real{0.75} - y_normalized));
    const Real uy = u0 * perturbation_amplitude *
        std::sin(Real{2} * std::numbers::pi_v<Real> * (x_normalized + Real{0.25}));

    lbm::MacroState<FluidLattice, Real> macro{};
    macro.density = Real{1};
    macro.velocity << ux, uy;
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
        const Real y_normalized = static_cast<Real>(y_index) / static_cast<Real>(ny);
        const Real concentration_a = y_normalized > Real{0.5} ? Real{1} : Real{0};
        const Real concentration_b = y_normalized <= Real{0.5} ? Real{1} : Real{0};

        for (std::size_t x_index = 0; x_index < nx; ++x_index) {
            const lbm::MacroState<FluidLattice, Real> macro =
                shear_layer_macro(x_index, y_index);

            for (int i = 0; i < FluidLattice::Q; ++i) {
                fluid_view[static_cast<std::size_t>(i), y_index, x_index] =
                    lbm::compute_equilibrium<FluidLattice, Real>(i, macro);
            }

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

[[nodiscard]] Real compute_max_velocity(
    const lbm::LatticeMemory<FluidLattice, Real>& fluid) {
    const auto view = fluid.get_current_view();
    Real max_velocity{};

    for (std::size_t y_index = 0; y_index < ny; ++y_index) {
        for (std::size_t x_index = 0; x_index < nx; ++x_index) {
            const lbm::MacroState<FluidLattice, Real> macro =
                fluid_macro_at(view, x_index, y_index);
            const Real speed = std::sqrt(macro.velocity.squaredNorm());

            if (!std::isfinite(speed)) {
                return std::numeric_limits<Real>::infinity();
            }

            max_velocity = std::max(max_velocity, speed);
        }
    }

    return max_velocity;
}

void write_shear_layer_vtk(
    const std::filesystem::path& output_dir,
    const std::string& name,
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b,
    std::size_t time_step) {
    const std::filesystem::path filename =
        output_dir / std::format("shear_{}_{:06}.vtk", name, time_step);
    std::ofstream vtk{filename};
    if (!vtk) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    const auto fluid_view = fluid.get_current_view();
    const auto a_view = species_a.get_current_view();
    const auto b_view = species_b.get_current_view();
    constexpr std::size_t point_count = nx * ny;

    vtk << "# vtk DataFile Version 3.0\n";
    vtk << std::format("LB-Cube turbulent shear layer {} step {}\n", name, time_step);
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

    vtk << "SCALARS velocity_magnitude double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t y_index = 0; y_index < ny; ++y_index) {
        for (std::size_t x_index = 0; x_index < nx; ++x_index) {
            const lbm::MacroState<FluidLattice, Real> macro =
                fluid_macro_at(fluid_view, x_index, y_index);
            vtk << std::format(
                "{:.17g}\n",
                static_cast<double>(std::sqrt(macro.velocity.squaredNorm())));
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
}

template <CollisionType CT>
void run_shear_layer_impl(const std::string& name) {
    const std::filesystem::path output_dir = std::format("output_shear_{}", name);
    std::filesystem::create_directories(output_dir);

    const std::filesystem::path stability_filename =
        output_dir / std::format("stability_{}.csv", name);
    std::ofstream stability_log{stability_filename};
    if (!stability_log) {
        throw std::runtime_error("failed to open " + stability_filename.string());
    }
    stability_log << "step,max_velocity\n";

    lbm::LatticeMemory<FluidLattice, Real> fluid{nx, ny};
    lbm::LatticeMemory<ScalarLattice, Real> species_a{nx, ny};
    lbm::LatticeMemory<ScalarLattice, Real> species_b{nx, ny};
    initialize_fields(fluid, species_a, species_b);

    const Real omega = Real{1} / fluid_relaxation_time;
    const Real omega_c = Real{1} / scalar_relaxation_time;

    std::cout << "Running " << name << " turbulent shear-layer benchmark\n"
              << "Output: " << output_dir.string() << '\n'
              << std::flush;

    write_shear_layer_vtk(output_dir, name, fluid, species_a, species_b, 0);
    stability_log << "0," << compute_max_velocity(fluid) << '\n';

    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 1; step <= total_steps; ++step) {
        lbm::step_cpu<FluidLattice, Real, CT>(fluid, omega);
        lbm::step_scalar_cpu<FluidLattice, ScalarLattice, Real>(
            species_a,
            fluid,
            omega_c);
        lbm::step_scalar_cpu<FluidLattice, ScalarLattice, Real>(
            species_b,
            fluid,
            omega_c);

        if (step % stability_check_frequency == 0) {
            const Real max_velocity = compute_max_velocity(fluid);
            stability_log << step << ',' << std::format("{:.17g}", max_velocity) << '\n';

            if (!std::isfinite(max_velocity) || max_velocity > crash_velocity_threshold) {
                std::cout << name << ": Solver Crashed due to instability! "
                          << "step=" << step << ", Umax=" << max_velocity << '\n'
                          << std::flush;
                write_shear_layer_vtk(
                    output_dir,
                    name,
                    fluid,
                    species_a,
                    species_b,
                    static_cast<std::size_t>(step));
                break;
            }
        }

        if (step % vtk_frequency == 0) {
            write_shear_layer_vtk(
                output_dir,
                name,
                fluid,
                species_a,
                species_b,
                static_cast<std::size_t>(step));
            std::cout << name << ": step " << step << " / " << total_steps
                      << " complete\n"
                      << std::flush;
        }
    }

    const auto stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    std::cout << name << " elapsed time: " << elapsed.count() << " s\n" << std::flush;
}

void run_shear_layer(CollisionType collision_type, const std::string& name) {
    switch (collision_type) {
    case CollisionType::BGK:
        run_shear_layer_impl<CollisionType::BGK>(name);
        break;
    case CollisionType::TRT:
        run_shear_layer_impl<CollisionType::TRT>(name);
        break;
    case CollisionType::MRT:
        run_shear_layer_impl<CollisionType::MRT>(name);
        break;
    }
}

} // namespace

int main() {
    try {
        std::cout << "LB-Cube turbulent doubly periodic shear-layer benchmark\n"
                  << "Grid: " << nx << " x " << ny << '\n'
                  << "Steps per operator: " << total_steps << '\n'
                  << "U0: " << u0 << '\n'
                  << "delta: " << perturbation_amplitude << '\n'
                  << "kappa: " << shear_steepness << '\n'
                  << "tau: " << fluid_relaxation_time << '\n'
                  << "nu: "
                  << static_cast<Real>(FluidLattice::cs2) *
                         (fluid_relaxation_time - Real{0.5})
                  << '\n'
                  << std::flush;

        run_shear_layer(CollisionType::BGK, "BGK");
    //  run_shear_layer(CollisionType::TRT, "TRT");
        run_shear_layer(CollisionType::MRT, "MRT");

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

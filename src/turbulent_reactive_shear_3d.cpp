/**
 * @file turbulent_reactive_shear_3d.cpp
 * @brief Production 3D turbulent reactive shear-layer executable.
 *
 * This executable runs a D3Q27/RLBM liquid-phase shear layer coupled to two
 * D3Q7 scalar reactants with fused non-ODE `A + B -> C` kinetics. It provides
 * command-line control over the numerical parameters, high-frequency CSV
 * diagnostics, live console telemetry, and an automatic binary VTK burst mode
 * triggered by kinetic-energy decay.
 */

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
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
#include <string_view>

#include "lattice_core.hpp"
#include "lattice_io.hpp"
#include "lattice_memory.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

namespace {

using FluidLattice = lbm::D3Q27;
using ScalarLattice = lbm::D3Q7;
using Real = double;

struct Config {
    std::size_t nx{128};
    std::size_t ny{128};
    std::size_t nz{128};
    Real tau_f{0.505};
    Real tau_s{0.5005};
    Real u0{0.05};
    Real k_react{0.1};
    int steps{10000};
    int stat_freq{10};
    int screen_freq{100};
    int vtk_burst_length{1000};
    int vtk_burst_freq{100};
};

struct FlowDiagnostics {
    Real u_max{};
    Real mean_kinetic_energy{};
};

[[nodiscard]] std::size_t cell_count(const Config& config) {
    return config.nx * config.ny * config.nz;
}

[[nodiscard]] std::string require_value(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::format("missing value for {}", argv[index]));
    }

    ++index;
    return argv[index];
}

[[nodiscard]] std::size_t parse_size(std::string_view flag, std::string_view value) {
    std::size_t parsed_chars{};
    const std::size_t parsed = std::stoull(std::string{value}, &parsed_chars);
    if (parsed_chars != value.size() || parsed == 0) {
        throw std::runtime_error(std::format("{} must be a positive integer", flag));
    }
    return parsed;
}

[[nodiscard]] int parse_int(std::string_view flag, std::string_view value) {
    std::size_t parsed_chars{};
    const int parsed = std::stoi(std::string{value}, &parsed_chars);
    if (parsed_chars != value.size() || parsed <= 0) {
        throw std::runtime_error(std::format("{} must be a positive integer", flag));
    }
    return parsed;
}

[[nodiscard]] Real parse_real(std::string_view flag, std::string_view value) {
    std::size_t parsed_chars{};
    const Real parsed = std::stod(std::string{value}, &parsed_chars);
    if (parsed_chars != value.size() || !std::isfinite(parsed)) {
        throw std::runtime_error(std::format("{} must be a finite floating-point value", flag));
    }
    return parsed;
}

void print_usage(std::ostream& stream, std::string_view executable) {
    stream
        << "Usage: " << executable << " [options]\n"
        << "  --Nx <n>                 Grid nodes in x (default 128)\n"
        << "  --Ny <n>                 Grid nodes in y (default 128)\n"
        << "  --Nz <n>                 Grid nodes in z (default 128)\n"
        << "  --tau_f <value>          Fluid relaxation time (default 0.505)\n"
        << "  --tau_s <value>          Scalar relaxation time (default 0.5005)\n"
        << "  --U0 <value>             Shear velocity amplitude (default 0.05)\n"
        << "  --k_react <value>        Reaction rate constant (default 0.1)\n"
        << "  --steps <n>              Total simulation steps (default 10000)\n"
        << "  --stat_freq <n>          CSV statistics interval (default 10)\n"
        << "  --screen_freq <n>        Console telemetry interval (default 100)\n"
        << "  --vtk_burst_length <n>   Steps covered by burst output (default 1000)\n"
        << "  --vtk_burst_freq <n>     VTK interval during burst (default 100)\n"
        << "  --help                   Show this message\n";
}

[[nodiscard]] Config parse_arguments(int argc, char** argv) {
    Config config{};

    for (int index = 1; index < argc; ++index) {
        const std::string_view flag{argv[index]};

        if (flag == "--help" || flag == "-h") {
            print_usage(std::cout, argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (flag == "--Nx") {
            config.nx = parse_size(flag, require_value(index, argc, argv));
        } else if (flag == "--Ny") {
            config.ny = parse_size(flag, require_value(index, argc, argv));
        } else if (flag == "--Nz") {
            config.nz = parse_size(flag, require_value(index, argc, argv));
        } else if (flag == "--tau_f") {
            config.tau_f = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--tau_s") {
            config.tau_s = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--U0") {
            config.u0 = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--k_react") {
            config.k_react = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--steps") {
            config.steps = parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--stat_freq") {
            config.stat_freq = parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--screen_freq") {
            config.screen_freq = parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--vtk_burst_length") {
            config.vtk_burst_length = parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--vtk_burst_freq") {
            config.vtk_burst_freq = parse_int(flag, require_value(index, argc, argv));
        } else {
            throw std::runtime_error(std::format("unknown option: {}", flag));
        }
    }

    if (config.tau_f <= Real{0.5}) {
        throw std::runtime_error("--tau_f must be greater than 0.5");
    }
    if (config.tau_s <= Real{0.5}) {
        throw std::runtime_error("--tau_s must be greater than 0.5");
    }
    if (config.u0 <= Real{}) {
        throw std::runtime_error("--U0 must be positive");
    }
    if (config.k_react < Real{}) {
        throw std::runtime_error("--k_react must be non-negative");
    }

    return config;
}

void print_recap(const Config& config) {
    const Real viscosity = FluidLattice::cs2 * (config.tau_f - Real{0.5});
    const Real scalar_diffusivity = ScalarLattice::cs2 * (config.tau_s - Real{0.5});
    const Real schmidt = viscosity / scalar_diffusivity;
    const Real damkohler = config.k_react * static_cast<Real>(config.ny) / config.u0;

    std::cout
        << "LB-Cube production 3D reactive shear layer\n"
        << "Grid: " << config.nx << " x " << config.ny << " x " << config.nz
        << " (" << cell_count(config) << " cells)\n"
        << "Fluid: D3Q27, Collision: RLBM\n"
        << "Scalars: two D3Q7 reactant fields, A + B -> C\n"
        << "tau_f: " << config.tau_f << ", omega_f: " << Real{1} / config.tau_f << '\n'
        << "tau_s: " << config.tau_s << ", omega_s: " << Real{1} / config.tau_s << '\n'
        << "U0: " << config.u0 << ", k_react: " << config.k_react << '\n'
        << "nu: " << viscosity << ", D: " << scalar_diffusivity << '\n'
        << "Sc: " << schmidt << ", Da: " << damkohler << '\n'
        << "steps: " << config.steps
        << ", stat_freq: " << config.stat_freq
        << ", screen_freq: " << config.screen_freq << '\n'
        << "VTK burst length: " << config.vtk_burst_length
        << ", VTK burst frequency: " << config.vtk_burst_freq << '\n'
        << std::flush;
}

[[nodiscard]] lbm::MacroState<FluidLattice, Real> initial_fluid_macro(
    const Config& config,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    const Real y1 = static_cast<Real>(config.ny) / Real{4};
    const Real y2 = Real{3} * static_cast<Real>(config.ny) / Real{4};
    const Real kappa = Real{80} / static_cast<Real>(config.ny);
    const Real y_real = static_cast<Real>(y);

    const Real phase_x =
        Real{2} * std::numbers::pi_v<Real> *
        static_cast<Real>(x) / static_cast<Real>(config.nx);
    const Real phase_y =
        Real{2} * std::numbers::pi_v<Real> *
        static_cast<Real>(y) / static_cast<Real>(config.ny);
    const Real phase_z =
        Real{2} * std::numbers::pi_v<Real> *
        static_cast<Real>(z) / static_cast<Real>(config.nz);
    const Real perturbation = Real{0.05} * config.u0;

    lbm::MacroState<FluidLattice, Real> macro{};
    macro.density = Real{1};
    macro.velocity <<
        config.u0 *
            (std::tanh(kappa * (y_real - y1)) -
             std::tanh(kappa * (y_real - y2)) -
             Real{1}),
        perturbation * std::sin(phase_x) * std::cos(phase_z),
        perturbation * std::cos(phase_x) * std::sin(phase_z) * std::cos(phase_y);

    return macro;
}

void initialize_fields(
    const Config& config,
    lbm::LatticeMemory<FluidLattice, Real>& fluid,
    lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    lbm::LatticeMemory<ScalarLattice, Real>& species_b) {
    auto fluid_view = fluid.get_current_view();
    auto a_view = species_a.get_current_view();
    auto b_view = species_b.get_current_view();

    const Real y1 = static_cast<Real>(config.ny) / Real{4};
    const Real y2 = Real{3} * static_cast<Real>(config.ny) / Real{4};

    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            const Real y_real = static_cast<Real>(y);
            const bool inside_layer = y_real >= y1 && y_real <= y2;
            const Real concentration_a = inside_layer ? Real{1} : Real{0};
            const Real concentration_b = inside_layer ? Real{0} : Real{1};

            for (std::size_t x = 0; x < config.nx; ++x) {
                const lbm::MacroState<FluidLattice, Real> macro =
                    initial_fluid_macro(config, x, y, z);

                for (int i = 0; i < FluidLattice::Q; ++i) {
                    fluid_view[static_cast<std::size_t>(i), z, y, x] =
                        lbm::compute_equilibrium<FluidLattice, Real>(i, macro);
                }

                for (int i = 0; i < ScalarLattice::Q; ++i) {
                    const auto q = static_cast<std::size_t>(i);
                    a_view[q, z, y, x] =
                        lbm::compute_scalar_equilibrium<ScalarLattice, Real>(
                            i,
                            concentration_a,
                            macro.velocity);
                    b_view[q, z, y, x] =
                        lbm::compute_scalar_equilibrium<ScalarLattice, Real>(
                            i,
                            concentration_b,
                            macro.velocity);
                }
            }
        }
    }
}

template <lbm::IsLatticeModel Lattice>
[[nodiscard]] lbm::MacroState<Lattice, Real> macro_at(
    typename lbm::LatticeMemory<Lattice, Real>::ConstView view,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    std::array<Real, static_cast<std::size_t>(Lattice::Q)> populations{};
    for (int i = 0; i < Lattice::Q; ++i) {
        populations[static_cast<std::size_t>(i)] =
            view[static_cast<std::size_t>(i), z, y, x];
    }

    return lbm::compute_macro_state<Lattice, Real>(populations);
}

[[nodiscard]] Real concentration_at(
    typename lbm::LatticeMemory<ScalarLattice, Real>::ConstView view,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> populations{};
    for (int i = 0; i < ScalarLattice::Q; ++i) {
        populations[static_cast<std::size_t>(i)] =
            view[static_cast<std::size_t>(i), z, y, x];
    }

    return lbm::compute_concentration<ScalarLattice, Real>(populations);
}

[[nodiscard]] FlowDiagnostics compute_flow_diagnostics(
    const Config& config,
    const lbm::LatticeMemory<FluidLattice, Real>& fluid) {
    const auto view = fluid.get_current_view();
    long double kinetic_energy = 0.0L;
    Real u_max{};

    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const lbm::MacroState<FluidLattice, Real> macro =
                    macro_at<FluidLattice>(view, x, y, z);
                const Real speed_squared = macro.velocity.squaredNorm();
                const Real speed = std::sqrt(speed_squared);

                if (!std::isfinite(speed) || !std::isfinite(macro.density)) {
                    return {
                        std::numeric_limits<Real>::infinity(),
                        std::numeric_limits<Real>::infinity()};
                }

                kinetic_energy +=
                    static_cast<long double>(Real{0.5} * macro.density * speed_squared);
                u_max = std::max(u_max, speed);
            }
        }
    }

    return {
        u_max,
        static_cast<Real>(kinetic_energy / static_cast<long double>(cell_count(config)))};
}

[[nodiscard]] double total_concentration(
    const Config& config,
    const lbm::LatticeMemory<ScalarLattice, Real>& scalar) {
    const auto view = scalar.get_current_view();
    long double total = 0.0L;

    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                total += static_cast<long double>(concentration_at(view, x, y, z));
            }
        }
    }

    return static_cast<double>(total);
}

[[nodiscard]] double step_reaction_ab_accumulate_product(
    lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    lbm::LatticeMemory<ScalarLattice, Real>& species_b,
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    Real omega_c,
    Real k_react) {
    auto fluid_current = fluid.get_current_view();
    auto a_current = species_a.get_current_view();
    auto a_next = species_a.get_next_view();
    auto b_current = species_b.get_current_view();
    auto b_next = species_b.get_next_view();

    const std::size_t z_extent = a_current.extent(1);
    const std::size_t y_extent = a_current.extent(2);
    const std::size_t x_extent = a_current.extent(3);
    long double product_increment = 0.0L;

    for (std::size_t z = 0; z < z_extent; ++z) {
        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                std::array<Real, static_cast<std::size_t>(FluidLattice::Q)> fluid_pops{};
                for (int i = 0; i < FluidLattice::Q; ++i) {
                    fluid_pops[static_cast<std::size_t>(i)] =
                        fluid_current[static_cast<std::size_t>(i), z, y, x];
                }
                const lbm::MacroState<FluidLattice, Real> fluid_macro =
                    lbm::compute_macro_state<FluidLattice, Real>(fluid_pops);

                std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> a_pops{};
                std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> b_pops{};
                for (int i = 0; i < ScalarLattice::Q; ++i) {
                    const auto direction_offset = static_cast<std::size_t>(i * ScalarLattice::D);
                    const int cx = ScalarLattice::c[direction_offset];
                    const int cy = ScalarLattice::c[direction_offset + 1];
                    const int cz = ScalarLattice::c[direction_offset + 2];
                    const std::size_t nx = lbm::detail::periodic_pull_index(x, cx, x_extent);
                    const std::size_t ny = lbm::detail::periodic_pull_index(y, cy, y_extent);
                    const std::size_t nz = lbm::detail::periodic_pull_index(z, cz, z_extent);
                    const auto q = static_cast<std::size_t>(i);

                    a_pops[q] = a_current[q, nz, ny, nx];
                    b_pops[q] = b_current[q, nz, ny, nx];
                }

                const Real concentration_a =
                    lbm::compute_concentration<ScalarLattice, Real>(a_pops);
                const Real concentration_b =
                    lbm::compute_concentration<ScalarLattice, Real>(b_pops);
                const Real reaction_source =
                    lbm::compute_reaction_ab_source<Real>(
                        concentration_a,
                        concentration_b,
                        k_react);

                product_increment += static_cast<long double>(-reaction_source);

                lbm::collide_scalar_max_dissipation<ScalarLattice, Real>(
                    a_pops,
                    fluid_macro.velocity,
                    omega_c,
                    reaction_source);
                lbm::collide_scalar_max_dissipation<ScalarLattice, Real>(
                    b_pops,
                    fluid_macro.velocity,
                    omega_c,
                    reaction_source);

                for (int i = 0; i < ScalarLattice::Q; ++i) {
                    const auto q = static_cast<std::size_t>(i);
                    a_next[q, z, y, x] = a_pops[q];
                    b_next[q, z, y, x] = b_pops[q];
                }
            }
        }
    }

    species_a.swap_buffers();
    species_b.swap_buffers();
    return static_cast<double>(product_increment);
}

void write_big_endian_double(std::ostream& stream, double value) {
    std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    if constexpr (std::endian::native == std::endian::little) {
        bits = std::byteswap(bits);
    }

    stream.write(reinterpret_cast<const char*>(&bits), sizeof(bits));
}

void write_binary_vtk(
    const Config& config,
    const std::filesystem::path& output_dir,
    int step,
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b) {
    std::filesystem::create_directories(output_dir);
    const std::filesystem::path filename =
        output_dir / std::format("reactive_shear3d_{:06}.vtk", step);
    std::ofstream vtk{filename, std::ios::binary};
    if (!vtk) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    const auto fluid_view = fluid.get_current_view();
    const auto a_view = species_a.get_current_view();
    const auto b_view = species_b.get_current_view();
    const std::size_t points = cell_count(config);

    vtk << "# vtk DataFile Version 3.0\n";
    vtk << std::format("LB-Cube production reactive shear layer step {}\n", step);
    vtk << "BINARY\n";
    vtk << "DATASET STRUCTURED_POINTS\n";
    vtk << std::format("DIMENSIONS {} {} {}\n", config.nx, config.ny, config.nz);
    vtk << "ORIGIN 0 0 0\n";
    vtk << "SPACING 1 1 1\n";
    vtk << std::format("POINT_DATA {}\n", points);

    vtk << "VECTORS velocity double\n";
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const lbm::MacroState<FluidLattice, Real> macro =
                    macro_at<FluidLattice>(fluid_view, x, y, z);
                write_big_endian_double(vtk, static_cast<double>(macro.velocity[0]));
                write_big_endian_double(vtk, static_cast<double>(macro.velocity[1]));
                write_big_endian_double(vtk, static_cast<double>(macro.velocity[2]));
            }
        }
    }
    vtk << '\n';

    vtk << "SCALARS C_A double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                write_big_endian_double(
                    vtk,
                    static_cast<double>(concentration_at(a_view, x, y, z)));
            }
        }
    }
    vtk << '\n';

    vtk << "SCALARS C_B double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                write_big_endian_double(
                    vtk,
                    static_cast<double>(concentration_at(b_view, x, y, z)));
            }
        }
    }
    vtk << '\n';
}

void append_statistics(
    std::ofstream& statistics,
    const Config& config,
    int step,
    Real simulation_time,
    const FlowDiagnostics& flow,
    Real dissipation_rate,
    const lbm::ReactiveDiagnostics<Real>& reactive,
    double total_product_c_formed) {
    const Real mean_c =
        static_cast<Real>(total_product_c_formed / static_cast<double>(cell_count(config)));
    const Real global_rate =
        reactive.true_reaction_rate * static_cast<Real>(cell_count(config));

    statistics << std::format(
        "{},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g}\n",
        step,
        static_cast<double>(simulation_time),
        static_cast<double>(flow.u_max),
        static_cast<double>(flow.mean_kinetic_energy),
        static_cast<double>(dissipation_rate),
        static_cast<double>(reactive.mean_A),
        static_cast<double>(reactive.var_A),
        static_cast<double>(reactive.mean_B),
        static_cast<double>(reactive.var_B),
        static_cast<double>(mean_c),
        static_cast<double>(global_rate));
}

void run_simulation(const Config& config) {
    lbm::LatticeMemory<FluidLattice, Real> fluid{config.nx, config.ny, config.nz};
    lbm::LatticeMemory<ScalarLattice, Real> species_a{config.nx, config.ny, config.nz};
    lbm::LatticeMemory<ScalarLattice, Real> species_b{config.nx, config.ny, config.nz};

    initialize_fields(config, fluid, species_a, species_b);

    std::ofstream statistics{"statistics_shear_3d.csv"};
    if (!statistics) {
        throw std::runtime_error("failed to open statistics_shear_3d.csv");
    }
    statistics
        << "step,time,u_max,E_k,dissipation_rate,"
        << "mean_Ca,var_Ca,mean_Cb,var_Cb,mean_Cc,global_rate\n";

    const Real omega_f = Real{1} / config.tau_f;
    const Real omega_s = Real{1} / config.tau_s;
    const std::filesystem::path burst_dir{"vtk_burst_shear_3d"};
    const auto start = std::chrono::high_resolution_clock::now();

    double total_product_c_formed = 0.0;
    FlowDiagnostics flow = compute_flow_diagnostics(config, fluid);
    const FlowDiagnostics initial_flow = flow;
    Real kinetic_energy_previous = flow.mean_kinetic_energy;
    Real latest_dissipation_rate{};
    bool burst_active = false;
    int burst_steps_recorded = 0;

    const lbm::ReactiveDiagnostics<Real> initial_reactive =
        lbm::compute_reactive_stats<ScalarLattice, Real>(
            species_a,
            species_b,
            config.k_react);
    append_statistics(
        statistics,
        config,
        0,
        Real{},
        flow,
        latest_dissipation_rate,
        initial_reactive,
        total_product_c_formed);

    for (int step = 1; step <= config.steps; ++step) {
        lbm::step_cpu<FluidLattice, Real, lbm::CollisionType::RLBM>(fluid, omega_f);
        total_product_c_formed += step_reaction_ab_accumulate_product(
            species_a,
            species_b,
            fluid,
            omega_s,
            config.k_react);

        if (step % config.stat_freq == 0) {
            flow = compute_flow_diagnostics(config, fluid);
            latest_dissipation_rate =
                (kinetic_energy_previous - flow.mean_kinetic_energy) /
                static_cast<Real>(config.stat_freq);
            kinetic_energy_previous = flow.mean_kinetic_energy;

            const lbm::ReactiveDiagnostics<Real> reactive =
                lbm::compute_reactive_stats<ScalarLattice, Real>(
                    species_a,
                    species_b,
                    config.k_react);
            append_statistics(
                statistics,
                config,
                step,
                static_cast<Real>(step),
                flow,
                latest_dissipation_rate,
                reactive,
                total_product_c_formed);

            if (!burst_active &&
                flow.mean_kinetic_energy < Real{0.95} * initial_flow.mean_kinetic_energy) {
                burst_active = true;
                burst_steps_recorded = 0;
                std::cout << "[VTK burst] triggered at step " << step
                          << ", E_k=" << flow.mean_kinetic_energy << '\n'
                          << std::flush;
            }
        }

        if (step % config.screen_freq == 0) {
            const auto now = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> elapsed = now - start;
            const double updates =
                static_cast<double>(cell_count(config)) *
                static_cast<double>(step) *
                3.0;
            const double mlups = updates / elapsed.count() / 1.0e6;

            std::cout << std::format(
                "Step [{} / {}] Umax: {:.6g} Ek: {:.6g} Dissipation: {:.6g} Burst: {} MLUPS: {:.6g}\n",
                step,
                config.steps,
                static_cast<double>(flow.u_max),
                static_cast<double>(flow.mean_kinetic_energy),
                static_cast<double>(latest_dissipation_rate),
                burst_active ? "ON" : "OFF",
                mlups)
                      << std::flush;
        }

        if (burst_active && burst_steps_recorded < config.vtk_burst_length) {
            ++burst_steps_recorded;
            if (burst_steps_recorded == 1 ||
                burst_steps_recorded % config.vtk_burst_freq == 0) {
                write_binary_vtk(config, burst_dir, step, fluid, species_a, species_b);
            }
        }

        if (!std::isfinite(flow.u_max) || !std::isfinite(flow.mean_kinetic_energy)) {
            std::cout << "[D3Q27_RLBM] Solver produced non-finite diagnostics at step "
                      << step << '\n'
                      << std::flush;
            break;
        }
    }

    const auto stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    std::cout << "Simulation complete in " << elapsed.count()
              << " s. Total product C formed: " << total_product_c_formed
              << "\nStatistics: statistics_shear_3d.csv"
              << "\nVTK burst directory: " << burst_dir.string() << '\n'
              << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_arguments(argc, argv);
        print_recap(config);
        run_simulation(config);
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        print_usage(std::cerr, argc > 0 ? argv[0] : "lbm_turbulent_reactive_shear_3d");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

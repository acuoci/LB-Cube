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
    int vtk_freq{1000};
    int vtk_burst_length{1000};
    int vtk_burst_freq{100};
};

struct FlowDiagnostics {
    Real u_max{};
    Real mean_kinetic_energy{};
};

struct ScalarDiagnostics {
    Real mean_ca{};
    Real var_ca{};
    Real min_ca{};
    Real max_ca{};
    Real mean_cb{};
    Real var_cb{};
    Real mean_cc{};
    Real var_cc{};
    Real rate_true{};
    Real rate_mixed{};
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
        << "  --vtk_freq <n>           Regular binary VTK interval (default 1000)\n"
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
        } else if (flag == "--vtk_freq") {
            config.vtk_freq = parse_int(flag, require_value(index, argc, argv));
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
        << "VTK regular frequency: " << config.vtk_freq << '\n'
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
    int invalid_count = 0;

#pragma omp parallel for collapse(3) schedule(static) reduction(+ : kinetic_energy, invalid_count) reduction(max : u_max)
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const lbm::MacroState<FluidLattice, Real> macro =
                    macro_at<FluidLattice>(view, x, y, z);
                const Real speed_squared = macro.velocity.squaredNorm();
                const Real speed = std::sqrt(speed_squared);

                if (!std::isfinite(speed) || !std::isfinite(macro.density)) {
                    ++invalid_count;
                    continue;
                }

                kinetic_energy +=
                    static_cast<long double>(Real{0.5} * macro.density * speed_squared);
                u_max = std::max(u_max, speed);
            }
        }
    }

    if (invalid_count > 0) {
        return {
            std::numeric_limits<Real>::infinity(),
            std::numeric_limits<Real>::infinity()};
    }

    return {
        u_max,
        static_cast<Real>(kinetic_energy / static_cast<long double>(cell_count(config)))};
}

[[nodiscard]] ScalarDiagnostics compute_scalar_diagnostics(
    const Config& config,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b) {
    const auto a_view = species_a.get_current_view();
    const auto b_view = species_b.get_current_view();

    long double sum_ca = 0.0L;
    long double sum_cb = 0.0L;
    long double sum_cc = 0.0L;
    long double sum_ca2 = 0.0L;
    long double sum_cb2 = 0.0L;
    long double sum_cc2 = 0.0L;
    long double sum_rate = 0.0L;
    Real min_ca = std::numeric_limits<Real>::infinity();
    Real max_ca = -std::numeric_limits<Real>::infinity();

#pragma omp parallel for collapse(3) schedule(static) reduction(+ : sum_ca, sum_cb, sum_cc, sum_ca2, sum_cb2, sum_cc2, sum_rate) reduction(min : min_ca) reduction(max : max_ca)
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const Real concentration_a = concentration_at(a_view, x, y, z);
                const Real concentration_b = concentration_at(b_view, x, y, z);
                const Real concentration_c =
                    Real{0.5} * (Real{1} - concentration_a - concentration_b);
                const Real local_rate = config.k_react * concentration_a * concentration_b;

                sum_ca += static_cast<long double>(concentration_a);
                sum_cb += static_cast<long double>(concentration_b);
                sum_cc += static_cast<long double>(concentration_c);
                sum_ca2 += static_cast<long double>(concentration_a * concentration_a);
                sum_cb2 += static_cast<long double>(concentration_b * concentration_b);
                sum_cc2 += static_cast<long double>(concentration_c * concentration_c);
                sum_rate += static_cast<long double>(local_rate);
                min_ca = std::min(min_ca, concentration_a);
                max_ca = std::max(max_ca, concentration_a);
            }
        }
    }

    const long double inv_cells = 1.0L / static_cast<long double>(cell_count(config));
    const Real mean_ca = static_cast<Real>(sum_ca * inv_cells);
    const Real mean_cb = static_cast<Real>(sum_cb * inv_cells);
    const Real mean_cc = static_cast<Real>(sum_cc * inv_cells);
    const Real mean_ca2 = static_cast<Real>(sum_ca2 * inv_cells);
    const Real mean_cb2 = static_cast<Real>(sum_cb2 * inv_cells);
    const Real mean_cc2 = static_cast<Real>(sum_cc2 * inv_cells);

    return {
        mean_ca,
        mean_ca2 - mean_ca * mean_ca,
        min_ca,
        max_ca,
        mean_cb,
        mean_cb2 - mean_cb * mean_cb,
        mean_cc,
        mean_cc2 - mean_cc * mean_cc,
        static_cast<Real>(sum_rate * inv_cells),
        config.k_react * mean_ca * mean_cb};
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

    vtk << "SCALARS C_C double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const Real concentration_a = concentration_at(a_view, x, y, z);
                const Real concentration_b = concentration_at(b_view, x, y, z);
                const Real concentration_c =
                    Real{0.5} * (Real{1} - concentration_a - concentration_b);
                write_big_endian_double(vtk, static_cast<double>(concentration_c));
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
    const ScalarDiagnostics& scalar) {
    statistics << std::format(
        "{},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g}\n",
        step,
        static_cast<double>(simulation_time),
        static_cast<double>(flow.u_max),
        static_cast<double>(flow.mean_kinetic_energy),
        static_cast<double>(dissipation_rate),
        static_cast<double>(scalar.mean_ca),
        static_cast<double>(scalar.var_ca),
        static_cast<double>(scalar.min_ca),
        static_cast<double>(scalar.max_ca),
        static_cast<double>(scalar.mean_cb),
        static_cast<double>(scalar.var_cb),
        static_cast<double>(scalar.mean_cc),
        static_cast<double>(scalar.var_cc),
        static_cast<double>(scalar.rate_true),
        static_cast<double>(scalar.rate_mixed));
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
        << "mean_Ca,var_Ca,min_Ca,max_Ca,"
        << "mean_Cb,var_Cb,mean_Cc,var_Cc,rate_true,rate_mixed\n";

    const Real omega_f = Real{1} / config.tau_f;
    const Real omega_s = Real{1} / config.tau_s;
    const std::filesystem::path vtk_dir{"vtk_shear_3d"};
    const auto start = std::chrono::high_resolution_clock::now();

    FlowDiagnostics flow = compute_flow_diagnostics(config, fluid);
    const FlowDiagnostics initial_flow = flow;
    Real kinetic_energy_previous = flow.mean_kinetic_energy;
    Real latest_dissipation_rate{};
    bool burst_active = false;
    int burst_steps_recorded = 0;

    const ScalarDiagnostics initial_scalar =
        compute_scalar_diagnostics(config, species_a, species_b);
    append_statistics(
        statistics,
        config,
        0,
        Real{},
        flow,
        latest_dissipation_rate,
        initial_scalar);
    write_binary_vtk(config, vtk_dir, 0, fluid, species_a, species_b);

    for (int step = 1; step <= config.steps; ++step) {
        lbm::step_cpu<FluidLattice, Real, lbm::CollisionType::RLBM>(fluid, omega_f);
        lbm::step_reaction_AB<FluidLattice, ScalarLattice, Real>(
            fluid,
            species_a,
            species_b,
            omega_s,
            config.k_react);

        if (step % config.stat_freq == 0) {
            flow = compute_flow_diagnostics(config, fluid);
            latest_dissipation_rate =
                (kinetic_energy_previous - flow.mean_kinetic_energy) /
                static_cast<Real>(config.stat_freq);
            kinetic_energy_previous = flow.mean_kinetic_energy;

            const ScalarDiagnostics scalar =
                compute_scalar_diagnostics(config, species_a, species_b);
            append_statistics(
                statistics,
                config,
                step,
                static_cast<Real>(step),
                flow,
                latest_dissipation_rate,
                scalar);

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

        const bool is_regular_vtk = step % config.vtk_freq == 0;
        const bool is_burst_vtk =
            burst_active &&
            burst_steps_recorded < config.vtk_burst_length &&
            step % config.vtk_burst_freq == 0;
        if (is_regular_vtk || is_burst_vtk) {
            write_binary_vtk(config, vtk_dir, step, fluid, species_a, species_b);
        }
        if (is_burst_vtk) {
            ++burst_steps_recorded;
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
              << " s.\nStatistics: statistics_shear_3d.csv"
              << "\nVTK directory: " << vtk_dir.string() << '\n'
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

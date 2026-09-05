/**
 * @file turbulent_reactive_double_shear_3d.cpp
 * @brief Production 3D turbulent reactive double-shear-layer executable.
 *
 * This executable runs a triply periodic D3Q27/RLBM double shear layer coupled
 * to two D3Q7 scalar reactants with fused non-ODE `A + B -> C` kinetics. It
 * mirrors the production reactive shear-layer driver while using the canonical
 * unperturbed two-interface initial condition for systematic mixing studies.
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

#include "double_shear_parameters.hpp"
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
    lbm::double_shear::ParameterizationMode mode{lbm::double_shear::ParameterizationMode::DirectLbm};
    std::size_t nx{128};
    std::size_t ny{128};
    std::size_t nz{128};
    Real tau_f{0.505};
    Real tau_s{0.5005};
    Real u0{0.05};
    Real c0{1.0};
    Real delta_ratio{0.0625};
    Real k_react{0.1};
    Real re_delta{};
    Real sc{};
    Real da_delta{};
    Real delta0{};
    Real delta_u{};
    Real viscosity{};
    Real scalar_diffusivity{};
    Real tau_delta{};
    Real tau_chem{std::numeric_limits<Real>::infinity()};
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
    Real min_cb{};
    Real max_cb{};
    Real mean_cc{};
    Real var_cc{};
    Real rate_true{};
    Real rate_mixed{};
    Real reaction_efficiency{};
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
        << "\n"
        << "Parameterization modes are mutually exclusive:\n"
        << "  Direct LBM mode: --tau_f, --tau_s, --k_react\n"
        << "  Physical mode:   --Re_delta, --Sc, --Da_delta\n"
        << "\n"
        << "Definitions:\n"
        << "  delta0   = delta_ratio * Ny\n"
        << "  Re_delta = (2 U0) delta0 / nu\n"
        << "  Sc       = nu / D\n"
        << "  Da_delta = k_react C0 delta0 / (2 U0)\n"
        << "\n"
        << "  --Nx <n>                 Grid nodes in x (default 128)\n"
        << "  --Ny <n>                 Grid nodes in y (default 128)\n"
        << "  --Nz <n>                 Grid nodes in z (default 128)\n"
        << "  --tau_f <value>          Fluid relaxation time (default 0.505)\n"
        << "  --tau_s <value>          Scalar relaxation time (default 0.5005)\n"
        << "  --Re_delta <value>       Shear-layer Reynolds number for physical mode\n"
        << "  --Sc <value>             Schmidt number for physical mode\n"
        << "  --Da_delta <value>       Shear-layer Damkohler number for physical mode\n"
        << "  --U0 <value>             Shear velocity amplitude (default 0.05)\n"
        << "  --C0 <value>             Reference reactant concentration (default 1.0)\n"
        << "  --delta_ratio <value>    Initial shear thickness / Ny (default 0.0625)\n"
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
    lbm::double_shear::ExplicitParameterFlags explicit_parameters{};

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
            explicit_parameters.tau_f = true;
        } else if (flag == "--tau_s") {
            config.tau_s = parse_real(flag, require_value(index, argc, argv));
            explicit_parameters.tau_s = true;
        } else if (flag == "--U0") {
            config.u0 = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--C0") {
            config.c0 = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--delta_ratio") {
            config.delta_ratio = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--k_react") {
            config.k_react = parse_real(flag, require_value(index, argc, argv));
            explicit_parameters.k_react = true;
        } else if (flag == "--Re_delta") {
            config.re_delta = parse_real(flag, require_value(index, argc, argv));
            explicit_parameters.re_delta = true;
        } else if (flag == "--Sc") {
            config.sc = parse_real(flag, require_value(index, argc, argv));
            explicit_parameters.sc = true;
        } else if (flag == "--Da_delta") {
            config.da_delta = parse_real(flag, require_value(index, argc, argv));
            explicit_parameters.da_delta = true;
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

    const lbm::double_shear::ParameterInputs<Real> input{
        .nx = config.nx,
        .ny = config.ny,
        .nz = config.nz,
        .tau_f = config.tau_f,
        .tau_s = config.tau_s,
        .k_react = config.k_react,
        .u0 = config.u0,
        .c0 = config.c0,
        .delta_ratio = config.delta_ratio,
        .re_delta = config.re_delta,
        .sc = config.sc,
        .da_delta = config.da_delta
    };
    const lbm::double_shear::ResolvedParameters<Real> resolved =
        lbm::double_shear::resolve_parameters<Real>(input, explicit_parameters);

    config.mode = resolved.mode;
    config.tau_f = resolved.tau_f;
    config.tau_s = resolved.tau_s;
    config.k_react = resolved.k_react;
    config.re_delta = resolved.re_delta;
    config.sc = resolved.sc;
    config.da_delta = resolved.da_delta;
    config.delta0 = resolved.delta0;
    config.delta_u = resolved.delta_u;
    config.viscosity = resolved.nu;
    config.scalar_diffusivity = resolved.scalar_diffusivity;
    config.tau_delta = resolved.tau_delta;
    config.tau_chem = resolved.tau_chem;

    return config;
}

[[nodiscard]] Real y1(const Config& config) {
    return static_cast<Real>(config.ny) / Real{4};
}

[[nodiscard]] Real y2(const Config& config) {
    return Real{3} * static_cast<Real>(config.ny) / Real{4};
}

[[nodiscard]] Real shear_profile(const Config& config, std::size_t y) {
    const Real y_real = static_cast<Real>(y);
    const Real thickness = config.delta0;
    return std::tanh((y_real - y1(config)) / thickness) -
           std::tanh((y_real - y2(config)) / thickness) -
           Real{1};
}

[[nodiscard]] Real product_concentration(const Config& config, Real concentration_a, Real concentration_b) {
    return Real{0.5} * (config.c0 - concentration_a - concentration_b);
}

void print_recap(const Config& config) {
    const Real mach = config.u0 / std::sqrt(static_cast<Real>(FluidLattice::cs2));

    std::cout
        << "LB-Cube production 3D reactive double shear layer\n"
        << "Parameterization mode: "
        << lbm::double_shear::to_string(config.mode) << '\n'
        << "Grid: " << config.nx << " x " << config.ny << " x " << config.nz
        << " (" << cell_count(config) << " cells)\n"
        << "Domain: triply periodic\n"
        << "Fluid: D3Q27, Collision: RLBM\n"
        << "Scalars: two D3Q7 reactant fields, A + B -> C\n"
        << "tau_f: " << config.tau_f << ", omega_f: " << Real{1} / config.tau_f << '\n'
        << "tau_s: " << config.tau_s << ", omega_s: " << Real{1} / config.tau_s << '\n'
        << "U0: " << config.u0 << ", DeltaU: " << config.delta_u << '\n'
        << "C0: " << config.c0 << ", k_react: " << config.k_react << '\n'
        << "delta_ratio: " << config.delta_ratio << ", delta0: " << config.delta0 << '\n'
        << "interfaces: y1=" << y1(config) << ", y2=" << y2(config) << '\n'
        << "nu: " << config.viscosity << ", D: " << config.scalar_diffusivity << '\n'
        << "Re_delta: " << config.re_delta
        << ", Sc: " << config.sc
        << ", Da_delta: " << config.da_delta << '\n'
        << "tau_delta: " << config.tau_delta
        << ", tau_chem: " << config.tau_chem << '\n'
        << "Mach(U0): " << mach << '\n'
        << "steps: " << config.steps
        << ", stat_freq: " << config.stat_freq
        << ", screen_freq: " << config.screen_freq << '\n'
        << "VTK regular frequency: " << config.vtk_freq << '\n'
        << "VTK burst length: " << config.vtk_burst_length
        << ", VTK burst frequency: " << config.vtk_burst_freq << '\n';

    if (mach > Real{0.1}) {
        std::cout << "Warning: Mach(U0) exceeds 0.1; compressibility artifacts may be significant.\n";
    }
    if (config.tau_f - Real{0.5} < Real{1.0e-3}) {
        std::cout << "Warning: tau_f is very close to 0.5; fluid viscosity is near the stability limit.\n";
    }
    if (config.tau_s - Real{0.5} < Real{1.0e-3}) {
        std::cout << "Warning: tau_s is very close to 0.5; scalar diffusivity is near the stability limit.\n";
    }

    std::cout << std::flush;
}

void write_metadata_json(const Config& config) {
    std::ofstream metadata{"metadata_double_shear_3d.json"};
    if (!metadata) {
        throw std::runtime_error("failed to open metadata_double_shear_3d.json");
    }

    metadata << std::format(
        "{{\n"
        "  \"parameterization_mode\": \"{}\",\n"
        "  \"Nx\": {},\n"
        "  \"Ny\": {},\n"
        "  \"Nz\": {},\n"
        "  \"Re_delta\": {:.17g},\n"
        "  \"Sc\": {:.17g},\n"
        "  \"Da_delta\": {:.17g},\n"
        "  \"tau_f\": {:.17g},\n"
        "  \"tau_s\": {:.17g},\n"
        "  \"k_react\": {:.17g},\n"
        "  \"nu\": {:.17g},\n"
        "  \"D\": {:.17g},\n"
        "  \"U0\": {:.17g},\n"
        "  \"C0\": {:.17g},\n"
        "  \"delta_ratio\": {:.17g},\n"
        "  \"delta0\": {:.17g},\n"
        "  \"DeltaU\": {:.17g},\n"
        "  \"tau_delta\": {:.17g},\n"
        "  \"tau_chem\": {:.17g}\n"
        "}}\n",
        lbm::double_shear::to_string(config.mode),
        config.nx,
        config.ny,
        config.nz,
        static_cast<double>(config.re_delta),
        static_cast<double>(config.sc),
        static_cast<double>(config.da_delta),
        static_cast<double>(config.tau_f),
        static_cast<double>(config.tau_s),
        static_cast<double>(config.k_react),
        static_cast<double>(config.viscosity),
        static_cast<double>(config.scalar_diffusivity),
        static_cast<double>(config.u0),
        static_cast<double>(config.c0),
        static_cast<double>(config.delta_ratio),
        static_cast<double>(config.delta0),
        static_cast<double>(config.delta_u),
        static_cast<double>(config.tau_delta),
        static_cast<double>(config.tau_chem));
    metadata.flush();
    metadata.close();
}

[[nodiscard]] lbm::MacroState<FluidLattice, Real> initial_fluid_macro(
    const Config& config,
    std::size_t y) {
    lbm::MacroState<FluidLattice, Real> macro{};
    macro.density = Real{1};
    macro.velocity << config.u0 * shear_profile(config, y), Real{}, Real{};
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

#pragma omp parallel for collapse(3) schedule(static)
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const Real profile = shear_profile(config, y);
                const Real concentration_a =
                    std::clamp(Real{0.5} * config.c0 * (Real{1} + profile), Real{}, config.c0);
                const Real concentration_b =
                    std::clamp(Real{0.5} * config.c0 * (Real{1} - profile), Real{}, config.c0);
                const lbm::MacroState<FluidLattice, Real> macro =
                    initial_fluid_macro(config, y);

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
    Real min_cb = std::numeric_limits<Real>::infinity();
    Real max_cb = -std::numeric_limits<Real>::infinity();

#pragma omp parallel for collapse(3) schedule(static) reduction(+ : sum_ca, sum_cb, sum_cc, sum_ca2, sum_cb2, sum_cc2, sum_rate) reduction(min : min_ca, min_cb) reduction(max : max_ca, max_cb)
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const Real concentration_a = concentration_at(a_view, x, y, z);
                const Real concentration_b = concentration_at(b_view, x, y, z);
                const Real concentration_c =
                    product_concentration(config, concentration_a, concentration_b);
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
                min_cb = std::min(min_cb, concentration_b);
                max_cb = std::max(max_cb, concentration_b);
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
    const Real rate_true = static_cast<Real>(sum_rate * inv_cells);
    const Real rate_mixed = config.k_react * mean_ca * mean_cb;
    const Real reaction_efficiency =
        rate_mixed > Real{} ? rate_true / rate_mixed : Real{};

    return {
        mean_ca,
        mean_ca2 - mean_ca * mean_ca,
        min_ca,
        max_ca,
        mean_cb,
        mean_cb2 - mean_cb * mean_cb,
        min_cb,
        max_cb,
        mean_cc,
        mean_cc2 - mean_cc * mean_cc,
        rate_true,
        rate_mixed,
        reaction_efficiency};
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
        output_dir / std::format("double_shear3d_{:06}.vtk", step);
    std::ofstream vtk{filename, std::ios::binary};
    if (!vtk) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    const auto fluid_view = fluid.get_current_view();
    const auto a_view = species_a.get_current_view();
    const auto b_view = species_b.get_current_view();
    const std::size_t points = cell_count(config);

    vtk << "# vtk DataFile Version 3.0\n";
    vtk << std::format("LB-Cube reactive double shear layer step {}\n", step);
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
                    product_concentration(config, concentration_a, concentration_b);
                write_big_endian_double(vtk, static_cast<double>(concentration_c));
            }
        }
    }
    vtk << '\n';
}

void append_statistics(
    std::ofstream& statistics,
    int step,
    Real simulation_time,
    const FlowDiagnostics& flow,
    Real dissipation_rate,
    const ScalarDiagnostics& scalar) {
    statistics << std::format(
        "{},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g},{:.17g}",
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
        static_cast<double>(scalar.min_cb),
        static_cast<double>(scalar.max_cb),
        static_cast<double>(scalar.mean_cc),
        static_cast<double>(scalar.var_cc),
        static_cast<double>(scalar.rate_true),
        static_cast<double>(scalar.rate_mixed),
        static_cast<double>(scalar.reaction_efficiency))
               << std::endl;
    statistics.flush();
}

void run_simulation(const Config& config) {
    write_metadata_json(config);

    lbm::LatticeMemory<FluidLattice, Real> fluid{config.nx, config.ny, config.nz};
    lbm::LatticeMemory<ScalarLattice, Real> species_a{config.nx, config.ny, config.nz};
    lbm::LatticeMemory<ScalarLattice, Real> species_b{config.nx, config.ny, config.nz};

    initialize_fields(config, fluid, species_a, species_b);

    std::ofstream statistics{"statistics_double_shear_3d.csv"};
    if (!statistics) {
        throw std::runtime_error("failed to open statistics_double_shear_3d.csv");
    }
    statistics
        << "step,time,u_max,E_k,dissipation_rate,"
        << "mean_Ca,var_Ca,min_Ca,max_Ca,"
        << "mean_Cb,var_Cb,min_Cb,max_Cb,"
        << "mean_Cc,var_Cc,rate_true,rate_mixed,reaction_efficiency"
        << std::endl;
    statistics.flush();

    const Real omega_f = Real{1} / config.tau_f;
    const Real omega_s = Real{1} / config.tau_s;
    const std::filesystem::path vtk_dir{"vtk_double_shear_3d"};
    const auto start = std::chrono::high_resolution_clock::now();

    FlowDiagnostics flow = compute_flow_diagnostics(config, fluid);
    const FlowDiagnostics initial_flow = flow;
    Real kinetic_energy_previous = flow.mean_kinetic_energy;
    Real latest_dissipation_rate{};
    bool burst_active = false;
    int burst_steps_recorded = 0;

    const ScalarDiagnostics initial_scalar =
        compute_scalar_diagnostics(config, species_a, species_b);
    std::cout << "Initial profile check: "
              << "ux(y=0)=" << config.u0 * shear_profile(config, 0)
              << ", ux(y=Ny/2)=" << config.u0 * shear_profile(config, config.ny / 2)
              << ", min_Ca=" << initial_scalar.min_ca
              << ", max_Ca=" << initial_scalar.max_ca
              << ", min_Cb=" << initial_scalar.min_cb
              << ", max_Cb=" << initial_scalar.max_cb
              << ", mean_Ca=" << initial_scalar.mean_ca
              << ", mean_Cb=" << initial_scalar.mean_cb << '\n'
              << std::flush;
    append_statistics(
        statistics,
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
            config.k_react,
            config.c0);

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
    statistics.flush();
    statistics.close();

    std::cout << "Simulation complete in " << elapsed.count()
              << " s.\nStatistics: statistics_double_shear_3d.csv"
              << "\nMetadata: metadata_double_shear_3d.json"
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
        print_usage(
            std::cerr,
            argc > 0 ? argv[0] : "lbm_turbulent_reactive_double_shear_3d");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

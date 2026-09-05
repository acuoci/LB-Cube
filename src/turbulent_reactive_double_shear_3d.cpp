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
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
    Real perturb_amplitude{0.02};
    std::uint64_t perturb_seed{12345};
    int perturb_kmin{1};
    int perturb_kmax{4};
    Real perturb_width{2.0};
    int steps{10000};
    int stat_freq{10};
    int screen_freq{100};
    int vtk_freq{1000};
    int vtk_burst_length{1000};
    int vtk_burst_freq{100};
    int profile_freq{};
};

struct PerturbationMode {
    int kx{};
    int ky{};
    int kz{};
    Real k_norm{};
    std::array<Real, 3> amplitude{};
    std::array<Real, 3> phase{};
};

struct PerturbationVelocity {
    Real ux{};
    Real uy{};
    Real uz{};
};

struct PerturbationDiagnostics {
    bool enabled{};
    std::size_t number_of_modes{};
    Real target_rms{};
    Real achieved_rms{};
    Real mean_ux{};
    Real mean_uy{};
    Real mean_uz{};
    Real rms_ux{};
    Real rms_uy{};
    Real rms_uz{};
    Real max_plane_mean_abs_ux{};
    Real max_plane_mean_abs_uy{};
    Real max_plane_mean_abs_uz{};
    Real max_plane_rms{};
    std::size_t max_plane_rms_y{};
    Real y1_plane_rms{};
    Real y2_plane_rms{};
    Real y1_plane_rms_ux{};
    Real y1_plane_rms_uy{};
    Real y1_plane_rms_uz{};
    Real y2_plane_rms_ux{};
    Real y2_plane_rms_uy{};
    Real y2_plane_rms_uz{};
    Real divergence_rms{};
    Real normalized_divergence{};
    Real localization_energy_max{};
    std::size_t localization_peak_y1{};
    std::size_t localization_peak_y2{};
    Real localization_layer_to_bulk_ratio{};
};

struct PerturbationDefinition {
    std::vector<PerturbationMode> modes{};
    std::array<Real, 3> raw_mean{};
    Real scale{};
    PerturbationDiagnostics diagnostics{};
};

struct FlowDiagnostics {
    Real u_max{};
    Real mean_kinetic_energy{};
    Real transverse_kinetic_energy{};
    Real fluctuation_kinetic_energy{};
    Real uy_rms{};
    Real uz_rms{};
    Real uperp_rms{};
    Real enstrophy{};
    Real epsilon{};
    Real theta_1{};
    Real theta_2{};
    Real theta_avg{};
    Real re_theta{};
    Real delta_omega_1{};
    Real delta_omega_2{};
    Real delta_omega_avg{};
    Real mean_rho{};
    Real min_rho{};
    Real max_rho{};
    Real rho_rms_fluct{};
    Real mach_max{};
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
    Real mean_z{};
    Real var_z{};
    Real min_z{};
    Real max_z{};
    Real mean_chi_z{};
    Real rms_chi_z{};
    Real max_chi_z{};
    Real var_chi_z{};
    Real tau_mix{};
    Real tau_mix_star{};
    Real da_mix{};
    Real cov_ab{};
    Real segregation_index{};
    Real rho_ab{};
    Real mean_grad_z2{};
    Real mean_grad_z4{};
    Real grad_z_flatness{};
    Real delta_z_1{};
    Real delta_z_2{};
    Real delta_z_avg{};
    Real rms_reaction_rate{};
    Real max_reaction_rate{};
    Real reaction_effective_volume_fraction{};
    Real corr_r_chi_z{};
};

struct ResolutionDiagnostics {
    Real eta_k{};
    Real eta_b{};
    Real dx_over_eta_k{};
    Real dx_over_eta_b{};
    Real tau_eta{};
    Real tau_eta_star{};
    Real da_eta{};
};

struct ScalarBudgetDiagnostics {
    Real variance_decay_rate{};
    Real budget_ratio{};
    Real numerical_dissipation_fraction{};
    Real tau_eff{};
    Real tau_eff_star{};
};

struct ProfilePlaneSums {
    long double ux{};
    long double uy{};
    long double uz{};
    long double ux2{};
    long double uy2{};
    long double uz2{};
    long double ca{};
    long double cb{};
    long double cc{};
    long double z{};
    long double z2{};
    long double chi_z{};
    long double reaction_rate{};
    long double ca_cb{};
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

[[nodiscard]] int parse_nonnegative_int(std::string_view flag, std::string_view value) {
    std::size_t parsed_chars{};
    const int parsed = std::stoi(std::string{value}, &parsed_chars);
    if (parsed_chars != value.size() || parsed < 0) {
        throw std::runtime_error(std::format("{} must be a non-negative integer", flag));
    }
    return parsed;
}

[[nodiscard]] std::uint64_t parse_uint64(std::string_view flag, std::string_view value) {
    std::size_t parsed_chars{};
    const std::uint64_t parsed = std::stoull(std::string{value}, &parsed_chars);
    if (parsed_chars != value.size()) {
        throw std::runtime_error(std::format("{} must be an unsigned integer", flag));
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
        << "  --perturb_amplitude <a>  Target perturbation RMS / DeltaU (default 0.02)\n"
        << "  --perturb_seed <n>       Deterministic perturbation seed (default 12345)\n"
        << "  --perturb_kmin <n>       Minimum integer Fourier mode norm (default 1)\n"
        << "  --perturb_kmax <n>       Maximum integer Fourier mode norm (default 4)\n"
        << "  --perturb_width <value>  Localization width / delta0 (default 2.0)\n"
        << "  --steps <n>              Total simulation steps (default 10000)\n"
        << "  --stat_freq <n>          CSV statistics interval (default 10)\n"
        << "  --screen_freq <n>        Console telemetry interval (default 100)\n"
        << "  --vtk_freq <n>           Regular binary VTK interval (default 1000)\n"
        << "  --vtk_burst_length <n>   Steps covered by burst output (default 1000)\n"
        << "  --vtk_burst_freq <n>     VTK interval during burst (default 100)\n"
        << "  --profile_freq <n>       y-profile CSV interval; 0 disables output (default 0)\n"
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
        } else if (flag == "--perturb_amplitude") {
            config.perturb_amplitude = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--perturb_seed") {
            config.perturb_seed = parse_uint64(flag, require_value(index, argc, argv));
        } else if (flag == "--perturb_kmin") {
            config.perturb_kmin = parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--perturb_kmax") {
            config.perturb_kmax = parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--perturb_width") {
            config.perturb_width = parse_real(flag, require_value(index, argc, argv));
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
        } else if (flag == "--profile_freq") {
            config.profile_freq =
                parse_nonnegative_int(flag, require_value(index, argc, argv));
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

    if (config.perturb_amplitude < Real{}) {
        throw std::runtime_error("--perturb_amplitude must be non-negative");
    }
    if (config.perturb_kmin < 1) {
        throw std::runtime_error("--perturb_kmin must be at least 1");
    }
    if (config.perturb_kmax < config.perturb_kmin) {
        throw std::runtime_error("--perturb_kmax must be greater than or equal to --perturb_kmin");
    }
    if (config.perturb_width <= Real{}) {
        throw std::runtime_error("--perturb_width must be positive");
    }

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

[[nodiscard]] Real periodic_distance_abs(Real coordinate, Real center, Real period) {
    Real distance = std::abs(coordinate - center);
    if (distance > Real{0.5} * period) {
        distance = period - distance;
    }
    return distance;
}

[[nodiscard]] bool is_canonical_fourier_half_space(int kx, int ky, int kz) {
    return kx > 0 || (kx == 0 && ky > 0) || (kx == 0 && ky == 0 && kz > 0);
}

[[nodiscard]] std::vector<PerturbationMode> generate_perturbation_modes(const Config& config) {
    std::vector<PerturbationMode> modes{};
    if (config.perturb_amplitude == Real{}) {
        return modes;
    }

    std::mt19937_64 rng{config.perturb_seed};
    std::normal_distribution<Real> normal{Real{}, Real{1}};
    std::uniform_real_distribution<Real> uniform_phase{
        Real{},
        Real{2} * std::numbers::pi_v<Real>};

    for (int kx = -config.perturb_kmax; kx <= config.perturb_kmax; ++kx) {
        for (int ky = -config.perturb_kmax; ky <= config.perturb_kmax; ++ky) {
            for (int kz = -config.perturb_kmax; kz <= config.perturb_kmax; ++kz) {
                // A single real cosine with phase represents the +/- k pair; this
                // canonical half-space avoids double-counting conjugate modes.
                if (kx == 0 && ky == 0 && kz == 0) {
                    continue;
                }
                if (kx == 0 && kz == 0) {
                    continue;
                }
                if (!is_canonical_fourier_half_space(kx, ky, kz)) {
                    continue;
                }

                const Real k_norm = std::sqrt(
                    static_cast<Real>(kx * kx + ky * ky + kz * kz));
                if (k_norm < static_cast<Real>(config.perturb_kmin) ||
                    k_norm > static_cast<Real>(config.perturb_kmax)) {
                    continue;
                }

                PerturbationMode mode{};
                mode.kx = kx;
                mode.ky = ky;
                mode.kz = kz;
                mode.k_norm = k_norm;
                for (int component = 0; component < 3; ++component) {
                    mode.amplitude[static_cast<std::size_t>(component)] = normal(rng);
                    mode.phase[static_cast<std::size_t>(component)] = uniform_phase(rng);
                }
                modes.push_back(mode);
            }
        }
    }

    if (modes.empty()) {
        throw std::runtime_error(
            "the selected perturbation mode band contains no valid modes after exclusions");
    }

    return modes;
}

[[nodiscard]] std::pair<Real, Real> perturbation_envelope_and_derivative(
    const Config& config,
    std::size_t y) {
    const Real eta = static_cast<Real>(y) / static_cast<Real>(config.ny);
    const Real sigma = config.perturb_width * config.delta0;
    const Real sigma_over_ly = sigma / static_cast<Real>(config.ny);
    // Periodic Gaussian written through cos(2*pi*eta) to avoid a derivative
    // branch cut while preserving the requested width in lattice units.
    const Real beta = Real{1} /
        (Real{2} * std::numbers::pi_v<Real> * std::numbers::pi_v<Real> *
         sigma_over_ly * sigma_over_ly);
    const Real theta1 = Real{2} * std::numbers::pi_v<Real> * (eta - Real{0.25});
    const Real theta2 = Real{2} * std::numbers::pi_v<Real> * (eta - Real{0.75});
    const Real exp1 = std::exp(beta * (std::cos(theta1) - Real{1}));
    const Real exp2 = std::exp(beta * (std::cos(theta2) - Real{1}));
    const Real dtheta_dy =
        Real{2} * std::numbers::pi_v<Real> / static_cast<Real>(config.ny);
    const Real derivative =
        -beta * dtheta_dy * (exp1 * std::sin(theta1) + exp2 * std::sin(theta2));

    return {exp1 + exp2, derivative};
}

[[nodiscard]] PerturbationVelocity evaluate_raw_perturbation(
    const Config& config,
    const std::vector<PerturbationMode>& modes,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    if (modes.empty()) {
        return {};
    }

    const Real xi = static_cast<Real>(x) / static_cast<Real>(config.nx);
    const Real eta = static_cast<Real>(y) / static_cast<Real>(config.ny);
    const Real zeta = static_cast<Real>(z) / static_cast<Real>(config.nz);
    const auto [envelope, envelope_derivative] =
        perturbation_envelope_and_derivative(config, y);

    std::array<Real, 3> f{};
    std::array<Real, 3> dfdx{};
    std::array<Real, 3> dfdy{};
    std::array<Real, 3> dfdz{};

    for (const PerturbationMode& mode : modes) {
        const Real base_phase =
            Real{2} * std::numbers::pi_v<Real> *
            (static_cast<Real>(mode.kx) * xi +
             static_cast<Real>(mode.ky) * eta +
             static_cast<Real>(mode.kz) * zeta);
        const Real dtheta_dx =
            Real{2} * std::numbers::pi_v<Real> *
            static_cast<Real>(mode.kx) / static_cast<Real>(config.nx);
        const Real dtheta_dy =
            Real{2} * std::numbers::pi_v<Real> *
            static_cast<Real>(mode.ky) / static_cast<Real>(config.ny);
        const Real dtheta_dz =
            Real{2} * std::numbers::pi_v<Real> *
            static_cast<Real>(mode.kz) / static_cast<Real>(config.nz);

        for (int component = 0; component < 3; ++component) {
            const auto c = static_cast<std::size_t>(component);
            const Real coefficient = mode.amplitude[c] / mode.k_norm;
            const Real angle = base_phase + mode.phase[c];
            const Real cosine = std::cos(angle);
            const Real sine = std::sin(angle);

            f[c] += coefficient * cosine;
            dfdx[c] -= coefficient * sine * dtheta_dx;
            dfdy[c] -= coefficient * sine * dtheta_dy;
            dfdz[c] -= coefficient * sine * dtheta_dz;
        }
    }

    const std::array<Real, 3> dadx{
        envelope * dfdx[0],
        envelope * dfdx[1],
        envelope * dfdx[2]};
    const std::array<Real, 3> dady{
        envelope_derivative * f[0] + envelope * dfdy[0],
        envelope_derivative * f[1] + envelope * dfdy[1],
        envelope_derivative * f[2] + envelope * dfdy[2]};
    const std::array<Real, 3> dadz{
        envelope * dfdz[0],
        envelope * dfdz[1],
        envelope * dfdz[2]};

    return {
        dady[2] - dadz[1],
        dadz[0] - dadx[2],
        dadx[1] - dady[0]};
}

[[nodiscard]] PerturbationVelocity evaluate_scaled_perturbation(
    const Config& config,
    const PerturbationDefinition& perturbation,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    if (!perturbation.diagnostics.enabled) {
        return {};
    }

    const PerturbationVelocity raw =
        evaluate_raw_perturbation(config, perturbation.modes, x, y, z);
    return {
        perturbation.scale * (raw.ux - perturbation.raw_mean[0]),
        perturbation.scale * (raw.uy - perturbation.raw_mean[1]),
        perturbation.scale * (raw.uz - perturbation.raw_mean[2])};
}

[[nodiscard]] PerturbationDefinition prepare_perturbation(const Config& config) {
    PerturbationDefinition perturbation{};
    perturbation.diagnostics.enabled = config.perturb_amplitude > Real{};
    perturbation.diagnostics.target_rms = config.perturb_amplitude * config.delta_u;

    if (!perturbation.diagnostics.enabled) {
        return perturbation;
    }

    perturbation.modes = generate_perturbation_modes(config);
    perturbation.diagnostics.number_of_modes = perturbation.modes.size();

    long double sum_ux = 0.0L;
    long double sum_uy = 0.0L;
    long double sum_uz = 0.0L;
    long double sum_u2 = 0.0L;

#pragma omp parallel for collapse(3) schedule(static) reduction(+ : sum_ux, sum_uy, sum_uz, sum_u2)
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const PerturbationVelocity raw =
                    evaluate_raw_perturbation(config, perturbation.modes, x, y, z);
                sum_ux += static_cast<long double>(raw.ux);
                sum_uy += static_cast<long double>(raw.uy);
                sum_uz += static_cast<long double>(raw.uz);
                sum_u2 += static_cast<long double>(
                    raw.ux * raw.ux + raw.uy * raw.uy + raw.uz * raw.uz);
            }
        }
    }

    const long double inv_cells = 1.0L / static_cast<long double>(cell_count(config));
    perturbation.raw_mean = {
        static_cast<Real>(sum_ux * inv_cells),
        static_cast<Real>(sum_uy * inv_cells),
        static_cast<Real>(sum_uz * inv_cells)};
    const long double mean_square =
        sum_u2 * inv_cells -
        static_cast<long double>(perturbation.raw_mean[0] * perturbation.raw_mean[0]) -
        static_cast<long double>(perturbation.raw_mean[1] * perturbation.raw_mean[1]) -
        static_cast<long double>(perturbation.raw_mean[2] * perturbation.raw_mean[2]);
    const Real raw_rms = std::sqrt(static_cast<Real>(std::max(mean_square, 0.0L)));
    if (raw_rms <= Real{} || !std::isfinite(raw_rms)) {
        throw std::runtime_error("raw perturbation has zero or non-finite RMS");
    }
    perturbation.scale = perturbation.diagnostics.target_rms / raw_rms;

    std::vector<long double> plane_sum_ux(config.ny, 0.0L);
    std::vector<long double> plane_sum_uy(config.ny, 0.0L);
    std::vector<long double> plane_sum_uz(config.ny, 0.0L);
    std::vector<long double> plane_sum_ux2(config.ny, 0.0L);
    std::vector<long double> plane_sum_uy2(config.ny, 0.0L);
    std::vector<long double> plane_sum_uz2(config.ny, 0.0L);
    std::vector<long double> plane_energy(config.ny, 0.0L);

    long double final_sum_ux = 0.0L;
    long double final_sum_uy = 0.0L;
    long double final_sum_uz = 0.0L;
    long double final_sum_ux2 = 0.0L;
    long double final_sum_uy2 = 0.0L;
    long double final_sum_uz2 = 0.0L;
    long double divergence_sum2 = 0.0L;

#pragma omp parallel for schedule(static) reduction(+ : final_sum_ux, final_sum_uy, final_sum_uz, final_sum_ux2, final_sum_uy2, final_sum_uz2, divergence_sum2)
    for (std::size_t y = 0; y < config.ny; ++y) {
        long double local_sum_ux = 0.0L;
        long double local_sum_uy = 0.0L;
        long double local_sum_uz = 0.0L;
        long double local_sum_ux2 = 0.0L;
        long double local_sum_uy2 = 0.0L;
        long double local_sum_uz2 = 0.0L;
        long double local_energy = 0.0L;

        for (std::size_t z = 0; z < config.nz; ++z) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const PerturbationVelocity velocity =
                    evaluate_scaled_perturbation(config, perturbation, x, y, z);
                const Real velocity_squared =
                    velocity.ux * velocity.ux +
                    velocity.uy * velocity.uy +
                    velocity.uz * velocity.uz;
                const std::size_t xp = (x + 1) % config.nx;
                const std::size_t xm = (x + config.nx - 1) % config.nx;
                const std::size_t yp = (y + 1) % config.ny;
                const std::size_t ym = (y + config.ny - 1) % config.ny;
                const std::size_t zp = (z + 1) % config.nz;
                const std::size_t zm = (z + config.nz - 1) % config.nz;
                const PerturbationVelocity vx_plus =
                    evaluate_scaled_perturbation(config, perturbation, xp, y, z);
                const PerturbationVelocity vx_minus =
                    evaluate_scaled_perturbation(config, perturbation, xm, y, z);
                const PerturbationVelocity vy_plus =
                    evaluate_scaled_perturbation(config, perturbation, x, yp, z);
                const PerturbationVelocity vy_minus =
                    evaluate_scaled_perturbation(config, perturbation, x, ym, z);
                const PerturbationVelocity vz_plus =
                    evaluate_scaled_perturbation(config, perturbation, x, y, zp);
                const PerturbationVelocity vz_minus =
                    evaluate_scaled_perturbation(config, perturbation, x, y, zm);
                const Real divergence =
                    Real{0.5} * (vx_plus.ux - vx_minus.ux) +
                    Real{0.5} * (vy_plus.uy - vy_minus.uy) +
                    Real{0.5} * (vz_plus.uz - vz_minus.uz);

                local_sum_ux += static_cast<long double>(velocity.ux);
                local_sum_uy += static_cast<long double>(velocity.uy);
                local_sum_uz += static_cast<long double>(velocity.uz);
                local_sum_ux2 += static_cast<long double>(velocity.ux * velocity.ux);
                local_sum_uy2 += static_cast<long double>(velocity.uy * velocity.uy);
                local_sum_uz2 += static_cast<long double>(velocity.uz * velocity.uz);
                local_energy += static_cast<long double>(Real{0.5} * velocity_squared);
                final_sum_ux += static_cast<long double>(velocity.ux);
                final_sum_uy += static_cast<long double>(velocity.uy);
                final_sum_uz += static_cast<long double>(velocity.uz);
                final_sum_ux2 += static_cast<long double>(velocity.ux * velocity.ux);
                final_sum_uy2 += static_cast<long double>(velocity.uy * velocity.uy);
                final_sum_uz2 += static_cast<long double>(velocity.uz * velocity.uz);
                divergence_sum2 += static_cast<long double>(divergence * divergence);
            }
        }

        plane_sum_ux[y] = local_sum_ux;
        plane_sum_uy[y] = local_sum_uy;
        plane_sum_uz[y] = local_sum_uz;
        plane_sum_ux2[y] = local_sum_ux2;
        plane_sum_uy2[y] = local_sum_uy2;
        plane_sum_uz2[y] = local_sum_uz2;
        plane_energy[y] =
            local_energy / static_cast<long double>(config.nx * config.nz);
    }

    const long double inv_plane = 1.0L /
        static_cast<long double>(config.nx * config.nz);
    for (std::size_t y = 0; y < config.ny; ++y) {
        perturbation.diagnostics.max_plane_mean_abs_ux = std::max(
            perturbation.diagnostics.max_plane_mean_abs_ux,
            static_cast<Real>(std::abs(plane_sum_ux[y] * inv_plane)));
        perturbation.diagnostics.max_plane_mean_abs_uy = std::max(
            perturbation.diagnostics.max_plane_mean_abs_uy,
            static_cast<Real>(std::abs(plane_sum_uy[y] * inv_plane)));
        perturbation.diagnostics.max_plane_mean_abs_uz = std::max(
            perturbation.diagnostics.max_plane_mean_abs_uz,
            static_cast<Real>(std::abs(plane_sum_uz[y] * inv_plane)));
    }

    perturbation.diagnostics.mean_ux = static_cast<Real>(final_sum_ux * inv_cells);
    perturbation.diagnostics.mean_uy = static_cast<Real>(final_sum_uy * inv_cells);
    perturbation.diagnostics.mean_uz = static_cast<Real>(final_sum_uz * inv_cells);
    perturbation.diagnostics.rms_ux =
        std::sqrt(static_cast<Real>(final_sum_ux2 * inv_cells));
    perturbation.diagnostics.rms_uy =
        std::sqrt(static_cast<Real>(final_sum_uy2 * inv_cells));
    perturbation.diagnostics.rms_uz =
        std::sqrt(static_cast<Real>(final_sum_uz2 * inv_cells));
    perturbation.diagnostics.achieved_rms = std::sqrt(static_cast<Real>(
        (final_sum_ux2 + final_sum_uy2 + final_sum_uz2) * inv_cells));
    perturbation.diagnostics.divergence_rms =
        std::sqrt(static_cast<Real>(divergence_sum2 * inv_cells));
    perturbation.diagnostics.normalized_divergence =
        perturbation.diagnostics.achieved_rms > Real{}
            ? perturbation.diagnostics.divergence_rms * config.delta0 /
                  perturbation.diagnostics.achieved_rms
            : Real{};

    auto lower_peak = std::max_element(plane_energy.begin(), plane_energy.begin() + static_cast<std::ptrdiff_t>(config.ny / 2));
    auto upper_peak = std::max_element(plane_energy.begin() + static_cast<std::ptrdiff_t>(config.ny / 2), plane_energy.end());
    perturbation.diagnostics.localization_energy_max =
        static_cast<Real>(*std::max_element(plane_energy.begin(), plane_energy.end()));
    perturbation.diagnostics.max_plane_rms =
        std::sqrt(Real{2} * perturbation.diagnostics.localization_energy_max);
    perturbation.diagnostics.max_plane_rms_y =
        static_cast<std::size_t>(std::distance(
            plane_energy.begin(),
            std::max_element(plane_energy.begin(), plane_energy.end())));
    perturbation.diagnostics.localization_peak_y1 =
        static_cast<std::size_t>(std::distance(plane_energy.begin(), lower_peak));
    perturbation.diagnostics.localization_peak_y2 =
        static_cast<std::size_t>(std::distance(plane_energy.begin(), upper_peak));

    const std::size_t y_lower = config.ny / 4;
    const std::size_t y_upper = 3 * config.ny / 4;
    const auto plane_component_rms = [inv_plane](long double component_square_sum) {
        return std::sqrt(static_cast<Real>(component_square_sum * inv_plane));
    };
    perturbation.diagnostics.y1_plane_rms_ux =
        plane_component_rms(plane_sum_ux2[y_lower]);
    perturbation.diagnostics.y1_plane_rms_uy =
        plane_component_rms(plane_sum_uy2[y_lower]);
    perturbation.diagnostics.y1_plane_rms_uz =
        plane_component_rms(plane_sum_uz2[y_lower]);
    perturbation.diagnostics.y2_plane_rms_ux =
        plane_component_rms(plane_sum_ux2[y_upper]);
    perturbation.diagnostics.y2_plane_rms_uy =
        plane_component_rms(plane_sum_uy2[y_upper]);
    perturbation.diagnostics.y2_plane_rms_uz =
        plane_component_rms(plane_sum_uz2[y_upper]);
    perturbation.diagnostics.y1_plane_rms = std::sqrt(
        perturbation.diagnostics.y1_plane_rms_ux *
            perturbation.diagnostics.y1_plane_rms_ux +
        perturbation.diagnostics.y1_plane_rms_uy *
            perturbation.diagnostics.y1_plane_rms_uy +
        perturbation.diagnostics.y1_plane_rms_uz *
            perturbation.diagnostics.y1_plane_rms_uz);
    perturbation.diagnostics.y2_plane_rms = std::sqrt(
        perturbation.diagnostics.y2_plane_rms_ux *
            perturbation.diagnostics.y2_plane_rms_ux +
        perturbation.diagnostics.y2_plane_rms_uy *
            perturbation.diagnostics.y2_plane_rms_uy +
        perturbation.diagnostics.y2_plane_rms_uz *
            perturbation.diagnostics.y2_plane_rms_uz);

    long double layer_energy = 0.0L;
    long double bulk_energy = 0.0L;
    std::size_t layer_count = 0;
    std::size_t bulk_count = 0;
    const Real sigma = config.perturb_width * config.delta0;
    for (std::size_t y = 0; y < config.ny; ++y) {
        const Real y_real = static_cast<Real>(y);
        const Real nearest_layer = std::min(
            periodic_distance_abs(y_real, y1(config), static_cast<Real>(config.ny)),
            periodic_distance_abs(y_real, y2(config), static_cast<Real>(config.ny)));
        if (nearest_layer <= sigma) {
            layer_energy += plane_energy[y];
            ++layer_count;
        } else if (nearest_layer >= Real{2} * sigma) {
            bulk_energy += plane_energy[y];
            ++bulk_count;
        }
    }
    const long double mean_layer_energy =
        layer_count > 0 ? layer_energy / static_cast<long double>(layer_count) : 0.0L;
    const long double mean_bulk_energy =
        bulk_count > 0 ? bulk_energy / static_cast<long double>(bulk_count) : 0.0L;
    perturbation.diagnostics.localization_layer_to_bulk_ratio =
        mean_bulk_energy > 0.0L
            ? static_cast<Real>(mean_layer_energy / mean_bulk_energy)
            : std::numeric_limits<Real>::infinity();

    return perturbation;
}

[[nodiscard]] Real product_concentration(const Config& config, Real concentration_a, Real concentration_b) {
    return Real{0.5} * (config.c0 - concentration_a - concentration_b);
}

void print_recap(const Config& config, const PerturbationDefinition& perturbation) {
    const Real mach = config.u0 / std::sqrt(static_cast<Real>(FluidLattice::cs2));
    const PerturbationDiagnostics& pert = perturbation.diagnostics;

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
        << ", VTK burst frequency: " << config.vtk_burst_freq << '\n'
        << "Profile frequency: " << config.profile_freq
        << (config.profile_freq > 0 ? "" : " (disabled)") << '\n'
        << "Initial perturbation:\n"
        << "  enabled:                  " << (pert.enabled ? "yes" : "no") << '\n'
        << "  type:                     curl(periodic Gaussian localized vector potential)\n"
        << "  amplitude/DeltaU:         " << config.perturb_amplitude << '\n'
        << "  target RMS velocity:      " << pert.target_rms << '\n'
        << "  achieved RMS velocity:    " << pert.achieved_rms << '\n'
        << "  seed:                     " << config.perturb_seed << '\n'
        << "  kmin:                     " << config.perturb_kmin << '\n'
        << "  kmax:                     " << config.perturb_kmax << '\n'
        << "  number of modes:          " << pert.number_of_modes << '\n'
        << "  localization width/delta: " << config.perturb_width << '\n'
        << "  mean perturbation ux:     " << pert.mean_ux << '\n'
        << "  mean perturbation uy:     " << pert.mean_uy << '\n'
        << "  mean perturbation uz:     " << pert.mean_uz << '\n'
        << "  RMS perturbation ux:      " << pert.rms_ux << '\n'
        << "  RMS perturbation uy:      " << pert.rms_uy << '\n'
        << "  RMS perturbation uz:      " << pert.rms_uz << '\n'
        << "  RMS perturbation total:   " << pert.achieved_rms << '\n'
        << "  max_y RMS/DeltaU:         " << pert.max_plane_rms / config.delta_u
        << " at y=" << pert.max_plane_rms_y << '\n'
        << "  y1 RMS/DeltaU:            " << pert.y1_plane_rms / config.delta_u << '\n'
        << "  y2 RMS/DeltaU:            " << pert.y2_plane_rms / config.delta_u << '\n'
        << "  y1 comp RMS/DeltaU:       "
        << pert.y1_plane_rms_ux / config.delta_u << ", "
        << pert.y1_plane_rms_uy / config.delta_u << ", "
        << pert.y1_plane_rms_uz / config.delta_u << '\n'
        << "  y2 comp RMS/DeltaU:       "
        << pert.y2_plane_rms_ux / config.delta_u << ", "
        << pert.y2_plane_rms_uy / config.delta_u << ", "
        << pert.y2_plane_rms_uz / config.delta_u << '\n'
        << "  max plane-mean |ux'|:     " << pert.max_plane_mean_abs_ux << '\n'
        << "  max plane-mean |uy'|:     " << pert.max_plane_mean_abs_uy << '\n'
        << "  max plane-mean |uz'|:     " << pert.max_plane_mean_abs_uz << '\n'
        << "  divergence RMS:           " << pert.divergence_rms << '\n'
        << "  div_rms * delta0 / RMS:   " << pert.normalized_divergence << '\n'
        << "  max plane Epert:          " << pert.localization_energy_max << '\n'
        << "  lower peak y:             " << pert.localization_peak_y1 << '\n'
        << "  upper peak y:             " << pert.localization_peak_y2 << '\n'
        << "  layer/bulk Epert ratio:   " << pert.localization_layer_to_bulk_ratio << '\n';

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

[[nodiscard]] std::string json_number(Real value) {
    if (!std::isfinite(value)) {
        return "null";
    }

    return std::format("{:.17g}", static_cast<double>(value));
}

void write_metadata_json(const Config& config, const PerturbationDefinition& perturbation) {
    std::ofstream metadata{"metadata_double_shear_3d.json"};
    if (!metadata) {
        throw std::runtime_error("failed to open metadata_double_shear_3d.json");
    }

    const PerturbationDiagnostics& pert = perturbation.diagnostics;
    metadata
        << "{\n"
        << "  \"parameterization_mode\": \"" << lbm::double_shear::to_string(config.mode) << "\",\n"
        << "  \"Nx\": " << config.nx << ",\n"
        << "  \"Ny\": " << config.ny << ",\n"
        << "  \"Nz\": " << config.nz << ",\n"
        << "  \"Re_delta\": " << json_number(config.re_delta) << ",\n"
        << "  \"Sc\": " << json_number(config.sc) << ",\n"
        << "  \"Da_delta\": " << json_number(config.da_delta) << ",\n"
        << "  \"tau_f\": " << json_number(config.tau_f) << ",\n"
        << "  \"tau_s\": " << json_number(config.tau_s) << ",\n"
        << "  \"k_react\": " << json_number(config.k_react) << ",\n"
        << "  \"nu\": " << json_number(config.viscosity) << ",\n"
        << "  \"D\": " << json_number(config.scalar_diffusivity) << ",\n"
        << "  \"U0\": " << json_number(config.u0) << ",\n"
        << "  \"C0\": " << json_number(config.c0) << ",\n"
        << "  \"delta_ratio\": " << json_number(config.delta_ratio) << ",\n"
        << "  \"delta0\": " << json_number(config.delta0) << ",\n"
        << "  \"DeltaU\": " << json_number(config.delta_u) << ",\n"
        << "  \"tau_delta\": " << json_number(config.tau_delta) << ",\n"
        << "  \"tau_chem\": " << json_number(config.tau_chem) << ",\n"
        << "  \"profile_freq\": " << config.profile_freq << ",\n"
        << "  \"perturbation_type\": \"" << (pert.enabled ? "curl_localized_vector_potential" : "none") << "\",\n"
        << "  \"perturb_amplitude\": " << json_number(config.perturb_amplitude) << ",\n"
        << "  \"perturb_seed\": " << config.perturb_seed << ",\n"
        << "  \"perturb_kmin\": " << config.perturb_kmin << ",\n"
        << "  \"perturb_kmax\": " << config.perturb_kmax << ",\n"
        << "  \"perturb_width\": " << json_number(config.perturb_width) << ",\n"
        << "  \"perturb_number_of_modes\": " << pert.number_of_modes << ",\n"
        << "  \"target_upert_rms\": " << json_number(pert.target_rms) << ",\n"
        << "  \"achieved_upert_rms\": " << json_number(pert.achieved_rms) << ",\n"
        << "  \"mean_upert_x\": " << json_number(pert.mean_ux) << ",\n"
        << "  \"mean_upert_y\": " << json_number(pert.mean_uy) << ",\n"
        << "  \"mean_upert_z\": " << json_number(pert.mean_uz) << ",\n"
        << "  \"rms_upert_x\": " << json_number(pert.rms_ux) << ",\n"
        << "  \"rms_upert_y\": " << json_number(pert.rms_uy) << ",\n"
        << "  \"rms_upert_z\": " << json_number(pert.rms_uz) << ",\n"
        << "  \"max_plane_rms_upert\": " << json_number(pert.max_plane_rms) << ",\n"
        << "  \"max_plane_rms_upert_y\": " << pert.max_plane_rms_y << ",\n"
        << "  \"y1_plane_rms_upert\": " << json_number(pert.y1_plane_rms) << ",\n"
        << "  \"y2_plane_rms_upert\": " << json_number(pert.y2_plane_rms) << ",\n"
        << "  \"y1_plane_rms_upert_x\": " << json_number(pert.y1_plane_rms_ux) << ",\n"
        << "  \"y1_plane_rms_upert_y\": " << json_number(pert.y1_plane_rms_uy) << ",\n"
        << "  \"y1_plane_rms_upert_z\": " << json_number(pert.y1_plane_rms_uz) << ",\n"
        << "  \"y2_plane_rms_upert_x\": " << json_number(pert.y2_plane_rms_ux) << ",\n"
        << "  \"y2_plane_rms_upert_y\": " << json_number(pert.y2_plane_rms_uy) << ",\n"
        << "  \"y2_plane_rms_upert_z\": " << json_number(pert.y2_plane_rms_uz) << ",\n"
        << "  \"max_plane_mean_abs_upert_x\": " << json_number(pert.max_plane_mean_abs_ux) << ",\n"
        << "  \"max_plane_mean_abs_upert_y\": " << json_number(pert.max_plane_mean_abs_uy) << ",\n"
        << "  \"max_plane_mean_abs_upert_z\": " << json_number(pert.max_plane_mean_abs_uz) << ",\n"
        << "  \"divergence_rms\": " << json_number(pert.divergence_rms) << ",\n"
        << "  \"normalized_divergence\": " << json_number(pert.normalized_divergence) << ",\n"
        << "  \"localization_energy_max\": " << json_number(pert.localization_energy_max) << ",\n"
        << "  \"localization_peak_y1\": " << pert.localization_peak_y1 << ",\n"
        << "  \"localization_peak_y2\": " << pert.localization_peak_y2 << ",\n"
        << "  \"localization_layer_to_bulk_ratio\": "
        << json_number(pert.localization_layer_to_bulk_ratio) << "\n"
        << "}\n";
    metadata.flush();
    metadata.close();
}

[[nodiscard]] lbm::MacroState<FluidLattice, Real> initial_fluid_macro(
    const Config& config,
    const PerturbationDefinition& perturbation,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    lbm::MacroState<FluidLattice, Real> macro{};
    const PerturbationVelocity perturb =
        evaluate_scaled_perturbation(config, perturbation, x, y, z);
    macro.density = Real{1};
    macro.velocity << config.u0 * shear_profile(config, y) + perturb.ux,
        perturb.uy,
        perturb.uz;
    return macro;
}

void initialize_fields(
    const Config& config,
    const PerturbationDefinition& perturbation,
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
                    initial_fluid_macro(config, perturbation, x, y, z);

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
    std::vector<long double> plane_sum_ux(config.ny, 0.0L);
    std::vector<long double> plane_sum_uy(config.ny, 0.0L);
    std::vector<long double> plane_sum_uz(config.ny, 0.0L);
    long double kinetic_energy = 0.0L;
    long double transverse_kinetic_energy = 0.0L;
    long double sum_uy2 = 0.0L;
    long double sum_uz2 = 0.0L;
    long double sum_rho = 0.0L;
    long double sum_rho2 = 0.0L;
    Real u_max{};
    Real min_rho = std::numeric_limits<Real>::infinity();
    Real max_rho = -std::numeric_limits<Real>::infinity();
    int invalid_count = 0;

    // First pass: global velocity moments and x-z plane means for each y.
#pragma omp parallel for schedule(static) reduction(+ : kinetic_energy, transverse_kinetic_energy, sum_uy2, sum_uz2, sum_rho, sum_rho2, invalid_count) reduction(max : u_max, max_rho) reduction(min : min_rho)
    for (std::size_t y = 0; y < config.ny; ++y) {
        long double local_sum_ux = 0.0L;
        long double local_sum_uy = 0.0L;
        long double local_sum_uz = 0.0L;

        for (std::size_t z = 0; z < config.nz; ++z) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const lbm::MacroState<FluidLattice, Real> macro =
                    macro_at<FluidLattice>(view, x, y, z);
                const Real ux = macro.velocity[0];
                const Real uy = macro.velocity[1];
                const Real uz = macro.velocity[2];
                const Real speed_squared = macro.velocity.squaredNorm();
                const Real speed = std::sqrt(speed_squared);

                if (!std::isfinite(speed) || !std::isfinite(macro.density)) {
                    ++invalid_count;
                    continue;
                }

                kinetic_energy +=
                    static_cast<long double>(Real{0.5} * macro.density * speed_squared);
                transverse_kinetic_energy +=
                    static_cast<long double>(Real{0.5} * (uy * uy + uz * uz));
                sum_uy2 += static_cast<long double>(uy * uy);
                sum_uz2 += static_cast<long double>(uz * uz);
                sum_rho += static_cast<long double>(macro.density);
                sum_rho2 += static_cast<long double>(macro.density * macro.density);
                min_rho = std::min(min_rho, macro.density);
                max_rho = std::max(max_rho, macro.density);
                local_sum_ux += static_cast<long double>(ux);
                local_sum_uy += static_cast<long double>(uy);
                local_sum_uz += static_cast<long double>(uz);
                u_max = std::max(u_max, speed);
            }
        }

        plane_sum_ux[y] = local_sum_ux;
        plane_sum_uy[y] = local_sum_uy;
        plane_sum_uz[y] = local_sum_uz;
    }

    const long double inv_cells = 1.0L / static_cast<long double>(cell_count(config));
    const long double inv_plane =
        1.0L / static_cast<long double>(config.nx * config.nz);
    std::vector<Real> plane_mean_ux(config.ny, Real{});
    std::vector<Real> plane_mean_uy(config.ny, Real{});
    std::vector<Real> plane_mean_uz(config.ny, Real{});
    for (std::size_t y = 0; y < config.ny; ++y) {
        plane_mean_ux[y] = static_cast<Real>(plane_sum_ux[y] * inv_plane);
        plane_mean_uy[y] = static_cast<Real>(plane_sum_uy[y] * inv_plane);
        plane_mean_uz[y] = static_cast<Real>(plane_sum_uz[y] * inv_plane);
    }

    long double theta_1 = 0.0L;
    long double theta_2 = 0.0L;
    Real max_abs_dubar_dy_1{};
    Real max_abs_dubar_dy_2{};
    const std::size_t half_y = config.ny / 2;
    for (std::size_t y = 0; y < config.ny; ++y) {
        const Real normalized_velocity = plane_mean_ux[y] / config.delta_u;
        const Real theta_integrand =
            Real{0.25} - normalized_velocity * normalized_velocity;
        const std::size_t yp = (y + 1) % config.ny;
        const std::size_t ym = (y + config.ny - 1) % config.ny;
        const Real dubar_dy =
            Real{0.5} * (plane_mean_ux[yp] - plane_mean_ux[ym]);
        const long double endpoint_weight =
            (y == 0 || y == half_y) ? 0.5L : 1.0L;

        if (y <= half_y) {
            theta_1 += endpoint_weight * static_cast<long double>(theta_integrand);
            max_abs_dubar_dy_1 =
                std::max(max_abs_dubar_dy_1, std::abs(dubar_dy));
        }
        if (y == 0 || y >= half_y) {
            theta_2 += endpoint_weight * static_cast<long double>(theta_integrand);
            max_abs_dubar_dy_2 =
                std::max(max_abs_dubar_dy_2, std::abs(dubar_dy));
        }
    }

    long double fluctuation_kinetic_energy = 0.0L;
    long double enstrophy = 0.0L;
    long double viscous_dissipation = 0.0L;

    // Second pass: fluctuations relative to instantaneous plane means and
    // vorticity from centered periodic finite differences with dx=dy=dz=1.
#pragma omp parallel for collapse(3) schedule(static) reduction(+ : fluctuation_kinetic_energy, enstrophy, viscous_dissipation, invalid_count)
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const lbm::MacroState<FluidLattice, Real> macro =
                    macro_at<FluidLattice>(view, x, y, z);
                const Real ux_fluc = macro.velocity[0] - plane_mean_ux[y];
                const Real uy_fluc = macro.velocity[1] - plane_mean_uy[y];
                const Real uz_fluc = macro.velocity[2] - plane_mean_uz[y];
                const Real fluctuation_speed_squared =
                    ux_fluc * ux_fluc + uy_fluc * uy_fluc + uz_fluc * uz_fluc;

                if (!std::isfinite(fluctuation_speed_squared)) {
                    ++invalid_count;
                    continue;
                }

                const std::size_t xp = (x + 1) % config.nx;
                const std::size_t xm = (x + config.nx - 1) % config.nx;
                const std::size_t yp = (y + 1) % config.ny;
                const std::size_t ym = (y + config.ny - 1) % config.ny;
                const std::size_t zp = (z + 1) % config.nz;
                const std::size_t zm = (z + config.nz - 1) % config.nz;
                const auto vx_plus = macro_at<FluidLattice>(view, xp, y, z).velocity;
                const auto vx_minus = macro_at<FluidLattice>(view, xm, y, z).velocity;
                const auto vy_plus = macro_at<FluidLattice>(view, x, yp, z).velocity;
                const auto vy_minus = macro_at<FluidLattice>(view, x, ym, z).velocity;
                const auto vz_plus = macro_at<FluidLattice>(view, x, y, zp).velocity;
                const auto vz_minus = macro_at<FluidLattice>(view, x, y, zm).velocity;

                const Real d_uz_dy = Real{0.5} * (vy_plus[2] - vy_minus[2]);
                const Real d_uy_dz = Real{0.5} * (vz_plus[1] - vz_minus[1]);
                const Real d_ux_dz = Real{0.5} * (vz_plus[0] - vz_minus[0]);
                const Real d_uz_dx = Real{0.5} * (vx_plus[2] - vx_minus[2]);
                const Real d_uy_dx = Real{0.5} * (vx_plus[1] - vx_minus[1]);
                const Real d_ux_dy = Real{0.5} * (vy_plus[0] - vy_minus[0]);
                const Real d_ux_dx = Real{0.5} * (vx_plus[0] - vx_minus[0]);
                const Real d_uy_dy = Real{0.5} * (vy_plus[1] - vy_minus[1]);
                const Real d_uz_dz = Real{0.5} * (vz_plus[2] - vz_minus[2]);

                const Real omega_x = d_uz_dy - d_uy_dz;
                const Real omega_y = d_ux_dz - d_uz_dx;
                const Real omega_z = d_uy_dx - d_ux_dy;
                const Real vorticity_squared =
                    omega_x * omega_x + omega_y * omega_y + omega_z * omega_z;
                const Real local_epsilon =
                    config.viscosity *
                    (Real{2} * d_ux_dx * d_ux_dx +
                     Real{2} * d_uy_dy * d_uy_dy +
                     Real{2} * d_uz_dz * d_uz_dz +
                     (d_ux_dy + d_uy_dx) * (d_ux_dy + d_uy_dx) +
                     (d_ux_dz + d_uz_dx) * (d_ux_dz + d_uz_dx) +
                     (d_uy_dz + d_uz_dy) * (d_uy_dz + d_uz_dy));

                if (!std::isfinite(vorticity_squared) || !std::isfinite(local_epsilon)) {
                    ++invalid_count;
                    continue;
                }

                fluctuation_kinetic_energy +=
                    static_cast<long double>(Real{0.5} * fluctuation_speed_squared);
                enstrophy += static_cast<long double>(Real{0.5} * vorticity_squared);
                viscous_dissipation += static_cast<long double>(local_epsilon);
            }
        }
    }

    if (invalid_count > 0) {
        FlowDiagnostics invalid{};
        invalid.u_max = std::numeric_limits<Real>::infinity();
        invalid.mean_kinetic_energy = std::numeric_limits<Real>::infinity();
        invalid.transverse_kinetic_energy = std::numeric_limits<Real>::infinity();
        invalid.fluctuation_kinetic_energy = std::numeric_limits<Real>::infinity();
        invalid.uy_rms = std::numeric_limits<Real>::infinity();
        invalid.uz_rms = std::numeric_limits<Real>::infinity();
        invalid.uperp_rms = std::numeric_limits<Real>::infinity();
        invalid.enstrophy = std::numeric_limits<Real>::infinity();
        invalid.epsilon = std::numeric_limits<Real>::infinity();
        invalid.theta_1 = std::numeric_limits<Real>::infinity();
        invalid.theta_2 = std::numeric_limits<Real>::infinity();
        invalid.theta_avg = std::numeric_limits<Real>::infinity();
        invalid.re_theta = std::numeric_limits<Real>::infinity();
        invalid.delta_omega_1 = std::numeric_limits<Real>::infinity();
        invalid.delta_omega_2 = std::numeric_limits<Real>::infinity();
        invalid.delta_omega_avg = std::numeric_limits<Real>::infinity();
        invalid.mean_rho = std::numeric_limits<Real>::infinity();
        invalid.min_rho = std::numeric_limits<Real>::infinity();
        invalid.max_rho = std::numeric_limits<Real>::infinity();
        invalid.rho_rms_fluct = std::numeric_limits<Real>::infinity();
        invalid.mach_max = std::numeric_limits<Real>::infinity();
        return invalid;
    }

    const Real mean_rho = static_cast<Real>(sum_rho * inv_cells);
    const Real mean_rho2 = static_cast<Real>(sum_rho2 * inv_cells);
    const Real theta_1_real = static_cast<Real>(theta_1);
    const Real theta_2_real = static_cast<Real>(theta_2);
    const Real theta_avg = Real{0.5} * (theta_1_real + theta_2_real);
    const Real delta_omega_1 =
        max_abs_dubar_dy_1 > Real{} ? config.delta_u / max_abs_dubar_dy_1 : Real{};
    const Real delta_omega_2 =
        max_abs_dubar_dy_2 > Real{} ? config.delta_u / max_abs_dubar_dy_2 : Real{};
    const Real delta_omega_avg = Real{0.5} * (delta_omega_1 + delta_omega_2);
    const Real cs = std::sqrt(static_cast<Real>(FluidLattice::cs2));

    return {
        u_max,
        static_cast<Real>(kinetic_energy * inv_cells),
        static_cast<Real>(transverse_kinetic_energy * inv_cells),
        static_cast<Real>(fluctuation_kinetic_energy * inv_cells),
        std::sqrt(static_cast<Real>(sum_uy2 * inv_cells)),
        std::sqrt(static_cast<Real>(sum_uz2 * inv_cells)),
        std::sqrt(static_cast<Real>((sum_uy2 + sum_uz2) * inv_cells)),
        static_cast<Real>(enstrophy * inv_cells),
        static_cast<Real>(viscous_dissipation * inv_cells),
        theta_1_real,
        theta_2_real,
        theta_avg,
        config.viscosity > Real{} ? config.delta_u * theta_avg / config.viscosity : Real{},
        delta_omega_1,
        delta_omega_2,
        delta_omega_avg,
        mean_rho,
        min_rho,
        max_rho,
        std::sqrt(std::max(Real{}, mean_rho2 - mean_rho * mean_rho)),
        cs > Real{} ? u_max / cs : Real{}};
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
    long double sum_cacb = 0.0L;
    long double sum_z = 0.0L;
    long double sum_z2 = 0.0L;
    long double sum_chi_z = 0.0L;
    long double sum_chi_z2 = 0.0L;
    long double sum_grad_z2 = 0.0L;
    long double sum_grad_z4 = 0.0L;
    long double sum_rate2 = 0.0L;
    long double sum_rate_chi_z = 0.0L;
    std::vector<long double> plane_sum_z(config.ny, 0.0L);
    Real min_ca = std::numeric_limits<Real>::infinity();
    Real max_ca = -std::numeric_limits<Real>::infinity();
    Real min_cb = std::numeric_limits<Real>::infinity();
    Real max_cb = -std::numeric_limits<Real>::infinity();
    Real min_z = std::numeric_limits<Real>::infinity();
    Real max_z = -std::numeric_limits<Real>::infinity();
    Real max_chi_z{};
    Real max_reaction_rate{};

    // Parallelize by y so the x-z plane average of Z can be accumulated
    // without atomics while the global scalar moments still use reductions.
#pragma omp parallel for schedule(static) reduction(+ : sum_ca, sum_cb, sum_cc, sum_ca2, sum_cb2, sum_cc2, sum_rate, sum_cacb, sum_z, sum_z2, sum_chi_z, sum_chi_z2, sum_grad_z2, sum_grad_z4, sum_rate2, sum_rate_chi_z) reduction(min : min_ca, min_cb, min_z) reduction(max : max_ca, max_cb, max_z, max_chi_z, max_reaction_rate)
    for (std::size_t y = 0; y < config.ny; ++y) {
        long double local_plane_sum_z = 0.0L;

        for (std::size_t z = 0; z < config.nz; ++z) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const Real concentration_a = concentration_at(a_view, x, y, z);
                const Real concentration_b = concentration_at(b_view, x, y, z);
                const Real concentration_c =
                    product_concentration(config, concentration_a, concentration_b);
                const Real local_rate = config.k_react * concentration_a * concentration_b;
                const Real mixture_fraction =
                    Real{0.5} * (Real{1} + (concentration_a - concentration_b) / config.c0);
                const std::size_t xp = (x + 1) % config.nx;
                const std::size_t xm = (x + config.nx - 1) % config.nx;
                const std::size_t yp = (y + 1) % config.ny;
                const std::size_t ym = (y + config.ny - 1) % config.ny;
                const std::size_t zp = (z + 1) % config.nz;
                const std::size_t zm = (z + config.nz - 1) % config.nz;
                const auto z_at = [&](std::size_t xi, std::size_t yi, std::size_t zi) {
                    const Real ca = concentration_at(a_view, xi, yi, zi);
                    const Real cb = concentration_at(b_view, xi, yi, zi);
                    return Real{0.5} * (Real{1} + (ca - cb) / config.c0);
                };
                const Real dz_dx = Real{0.5} * (z_at(xp, y, z) - z_at(xm, y, z));
                const Real dz_dy = Real{0.5} * (z_at(x, yp, z) - z_at(x, ym, z));
                const Real dz_dz = Real{0.5} * (z_at(x, y, zp) - z_at(x, y, zm));
                const Real grad_z2 = dz_dx * dz_dx + dz_dy * dz_dy + dz_dz * dz_dz;
                const Real chi_z = Real{2} * config.scalar_diffusivity * grad_z2;

                sum_ca += static_cast<long double>(concentration_a);
                sum_cb += static_cast<long double>(concentration_b);
                sum_cc += static_cast<long double>(concentration_c);
                sum_ca2 += static_cast<long double>(concentration_a * concentration_a);
                sum_cb2 += static_cast<long double>(concentration_b * concentration_b);
                sum_cc2 += static_cast<long double>(concentration_c * concentration_c);
                sum_rate += static_cast<long double>(local_rate);
                sum_cacb += static_cast<long double>(concentration_a * concentration_b);
                sum_z += static_cast<long double>(mixture_fraction);
                sum_z2 += static_cast<long double>(mixture_fraction * mixture_fraction);
                sum_chi_z += static_cast<long double>(chi_z);
                sum_chi_z2 += static_cast<long double>(chi_z * chi_z);
                sum_grad_z2 += static_cast<long double>(grad_z2);
                sum_grad_z4 += static_cast<long double>(grad_z2 * grad_z2);
                sum_rate2 += static_cast<long double>(local_rate * local_rate);
                sum_rate_chi_z += static_cast<long double>(local_rate * chi_z);
                local_plane_sum_z += static_cast<long double>(mixture_fraction);
                min_ca = std::min(min_ca, concentration_a);
                max_ca = std::max(max_ca, concentration_a);
                min_cb = std::min(min_cb, concentration_b);
                max_cb = std::max(max_cb, concentration_b);
                min_z = std::min(min_z, mixture_fraction);
                max_z = std::max(max_z, mixture_fraction);
                max_chi_z = std::max(max_chi_z, chi_z);
                max_reaction_rate = std::max(max_reaction_rate, local_rate);
            }
        }

        plane_sum_z[y] = local_plane_sum_z;
    }

    const long double inv_cells = 1.0L / static_cast<long double>(cell_count(config));
    const long double inv_plane =
        1.0L / static_cast<long double>(config.nx * config.nz);
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
    const Real mean_cacb = static_cast<Real>(sum_cacb * inv_cells);
    const Real mean_z = static_cast<Real>(sum_z * inv_cells);
    const Real mean_z2 = static_cast<Real>(sum_z2 * inv_cells);
    const Real var_z = std::max(Real{}, mean_z2 - mean_z * mean_z);
    const Real mean_chi_z = static_cast<Real>(sum_chi_z * inv_cells);
    const Real mean_chi_z2 = static_cast<Real>(sum_chi_z2 * inv_cells);
    const Real var_chi_z = std::max(Real{}, mean_chi_z2 - mean_chi_z * mean_chi_z);
    const Real mean_rate2 = static_cast<Real>(sum_rate2 * inv_cells);
    const Real mean_rate_chi_z = static_cast<Real>(sum_rate_chi_z * inv_cells);
    const Real var_rate = std::max(Real{}, mean_rate2 - rate_true * rate_true);
    const Real tau_mix =
        mean_chi_z > Real{} ? var_z / mean_chi_z : std::numeric_limits<Real>::infinity();
    const Real tau_mix_star =
        std::isfinite(tau_mix) ? tau_mix * config.delta_u / config.delta0
                               : std::numeric_limits<Real>::infinity();
    const Real da_mix = config.k_react > Real{} ? config.k_react * config.c0 * tau_mix
                                                : Real{};
    const Real var_ca = std::max(Real{}, mean_ca2 - mean_ca * mean_ca);
    const Real var_cb = std::max(Real{}, mean_cb2 - mean_cb * mean_cb);
    const Real cov_ab = mean_cacb - mean_ca * mean_cb;
    const Real segregation_index =
        mean_ca * mean_cb > Real{} ? -cov_ab / (mean_ca * mean_cb) : Real{};
    const Real rho_ab =
        var_ca > Real{} && var_cb > Real{} ? cov_ab / std::sqrt(var_ca * var_cb) : Real{};
    const Real mean_grad_z2 = static_cast<Real>(sum_grad_z2 * inv_cells);
    const Real mean_grad_z4 = static_cast<Real>(sum_grad_z4 * inv_cells);
    const Real grad_z_flatness =
        mean_grad_z2 > Real{} ? mean_grad_z4 / (mean_grad_z2 * mean_grad_z2) : Real{};
    long double delta_z_1 = 0.0L;
    long double delta_z_2 = 0.0L;
    const std::size_t half_y = config.ny / 2;
    for (std::size_t y = 0; y < config.ny; ++y) {
        const Real zbar = static_cast<Real>(plane_sum_z[y] * inv_plane);
        const Real integrand = Real{4} * zbar * (Real{1} - zbar);
        const long double endpoint_weight =
            (y == 0 || y == half_y) ? 0.5L : 1.0L;
        if (y <= half_y) {
            delta_z_1 += endpoint_weight * static_cast<long double>(integrand);
        }
        if (y == 0 || y >= half_y) {
            delta_z_2 += endpoint_weight * static_cast<long double>(integrand);
        }
    }
    const Real delta_z_1_real = static_cast<Real>(delta_z_1);
    const Real delta_z_2_real = static_cast<Real>(delta_z_2);
    const Real delta_z_avg = Real{0.5} * (delta_z_1_real + delta_z_2_real);
    const Real rms_reaction_rate = std::sqrt(mean_rate2);
    const Real reaction_effective_volume_fraction =
        mean_rate2 > Real{} ? (rate_true * rate_true) / mean_rate2 : Real{};
    const Real corr_r_chi_z =
        var_rate > Real{} && var_chi_z > Real{}
            ? (mean_rate_chi_z - rate_true * mean_chi_z) /
                  std::sqrt(var_rate * var_chi_z)
            : Real{};

    return {
        mean_ca,
        var_ca,
        min_ca,
        max_ca,
        mean_cb,
        var_cb,
        min_cb,
        max_cb,
        mean_cc,
        std::max(Real{}, mean_cc2 - mean_cc * mean_cc),
        rate_true,
        rate_mixed,
        reaction_efficiency,
        mean_z,
        var_z,
        min_z,
        max_z,
        mean_chi_z,
        std::sqrt(mean_chi_z2),
        max_chi_z,
        var_chi_z,
        tau_mix,
        tau_mix_star,
        da_mix,
        cov_ab,
        segregation_index,
        rho_ab,
        mean_grad_z2,
        mean_grad_z4,
        grad_z_flatness,
        delta_z_1_real,
        delta_z_2_real,
        delta_z_avg,
        rms_reaction_rate,
        max_reaction_rate,
        reaction_effective_volume_fraction,
        corr_r_chi_z};
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

void write_y_profile_csv(
    const Config& config,
    const std::filesystem::path& output_dir,
    int step,
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b) {
    std::filesystem::create_directories(output_dir);
    const std::filesystem::path filename =
        output_dir / std::format("profile_{:08}.csv", step);
    std::ofstream profile{filename};
    if (!profile) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    const auto fluid_view = fluid.get_current_view();
    const auto a_view = species_a.get_current_view();
    const auto b_view = species_b.get_current_view();
    std::vector<ProfilePlaneSums> planes(config.ny);

#pragma omp parallel for schedule(static)
    for (std::size_t y = 0; y < config.ny; ++y) {
        ProfilePlaneSums sums{};

        for (std::size_t z = 0; z < config.nz; ++z) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const lbm::MacroState<FluidLattice, Real> macro =
                    macro_at<FluidLattice>(fluid_view, x, y, z);
                const Real ux = macro.velocity[0];
                const Real uy = macro.velocity[1];
                const Real uz = macro.velocity[2];
                const Real concentration_a = concentration_at(a_view, x, y, z);
                const Real concentration_b = concentration_at(b_view, x, y, z);
                const Real concentration_c =
                    product_concentration(config, concentration_a, concentration_b);
                const Real mixture_fraction =
                    Real{0.5} * (Real{1} + (concentration_a - concentration_b) / config.c0);
                const Real local_rate = config.k_react * concentration_a * concentration_b;

                const std::size_t xp = (x + 1) % config.nx;
                const std::size_t xm = (x + config.nx - 1) % config.nx;
                const std::size_t yp = (y + 1) % config.ny;
                const std::size_t ym = (y + config.ny - 1) % config.ny;
                const std::size_t zp = (z + 1) % config.nz;
                const std::size_t zm = (z + config.nz - 1) % config.nz;
                const auto z_at = [&](std::size_t xi, std::size_t yi, std::size_t zi) {
                    const Real ca = concentration_at(a_view, xi, yi, zi);
                    const Real cb = concentration_at(b_view, xi, yi, zi);
                    return Real{0.5} * (Real{1} + (ca - cb) / config.c0);
                };
                const Real dz_dx = Real{0.5} * (z_at(xp, y, z) - z_at(xm, y, z));
                const Real dz_dy = Real{0.5} * (z_at(x, yp, z) - z_at(x, ym, z));
                const Real dz_dz = Real{0.5} * (z_at(x, y, zp) - z_at(x, y, zm));
                const Real grad_z2 = dz_dx * dz_dx + dz_dy * dz_dy + dz_dz * dz_dz;
                const Real chi_z = Real{2} * config.scalar_diffusivity * grad_z2;

                sums.ux += static_cast<long double>(ux);
                sums.uy += static_cast<long double>(uy);
                sums.uz += static_cast<long double>(uz);
                sums.ux2 += static_cast<long double>(ux * ux);
                sums.uy2 += static_cast<long double>(uy * uy);
                sums.uz2 += static_cast<long double>(uz * uz);
                sums.ca += static_cast<long double>(concentration_a);
                sums.cb += static_cast<long double>(concentration_b);
                sums.cc += static_cast<long double>(concentration_c);
                sums.z += static_cast<long double>(mixture_fraction);
                sums.z2 += static_cast<long double>(mixture_fraction * mixture_fraction);
                sums.chi_z += static_cast<long double>(chi_z);
                sums.reaction_rate += static_cast<long double>(local_rate);
                sums.ca_cb += static_cast<long double>(concentration_a * concentration_b);
            }
        }

        planes[y] = sums;
    }

    profile
        << "y,y_over_L,"
        << "mean_ux,mean_uy,mean_uz,"
        << "ux_rms_fluct,uy_rms_fluct,uz_rms_fluct,k_fluc,"
        << "mean_Ca,mean_Cb,mean_Cc,"
        << "mean_Z,var_Z,"
        << "mean_chi_Z,"
        << "mean_reaction_rate,"
        << "cov_AB\n";

    const long double inv_plane =
        1.0L / static_cast<long double>(config.nx * config.nz);
    const auto finite_variance = [](Real mean_square, Real mean) {
        return std::max(Real{}, mean_square - mean * mean);
    };

    for (std::size_t y = 0; y < config.ny; ++y) {
        const ProfilePlaneSums& sums = planes[y];
        const Real mean_ux = static_cast<Real>(sums.ux * inv_plane);
        const Real mean_uy = static_cast<Real>(sums.uy * inv_plane);
        const Real mean_uz = static_cast<Real>(sums.uz * inv_plane);
        const Real ux_var =
            finite_variance(static_cast<Real>(sums.ux2 * inv_plane), mean_ux);
        const Real uy_var =
            finite_variance(static_cast<Real>(sums.uy2 * inv_plane), mean_uy);
        const Real uz_var =
            finite_variance(static_cast<Real>(sums.uz2 * inv_plane), mean_uz);
        const Real mean_ca = static_cast<Real>(sums.ca * inv_plane);
        const Real mean_cb = static_cast<Real>(sums.cb * inv_plane);
        const Real mean_z = static_cast<Real>(sums.z * inv_plane);

        profile
            << y << ','
            << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(y) / static_cast<Real>(config.ny)))
            << ',' << std::format("{:.17g}", static_cast<double>(mean_ux))
            << ',' << std::format("{:.17g}", static_cast<double>(mean_uy))
            << ',' << std::format("{:.17g}", static_cast<double>(mean_uz))
            << ',' << std::format("{:.17g}", static_cast<double>(std::sqrt(ux_var)))
            << ',' << std::format("{:.17g}", static_cast<double>(std::sqrt(uy_var)))
            << ',' << std::format("{:.17g}", static_cast<double>(std::sqrt(uz_var)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   Real{0.5} * (ux_var + uy_var + uz_var)))
            << ',' << std::format("{:.17g}", static_cast<double>(mean_ca))
            << ',' << std::format("{:.17g}", static_cast<double>(mean_cb))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(sums.cc * inv_plane)))
            << ',' << std::format("{:.17g}", static_cast<double>(mean_z))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   finite_variance(static_cast<Real>(sums.z2 * inv_plane), mean_z)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(sums.chi_z * inv_plane)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(sums.reaction_rate * inv_plane)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(sums.ca_cb * inv_plane) - mean_ca * mean_cb))
            << '\n';
    }

    profile.flush();
    profile.close();
}

[[nodiscard]] ResolutionDiagnostics compute_resolution_diagnostics(
    const Config& config,
    const FlowDiagnostics& flow) {
    if (flow.epsilon <= Real{} || !std::isfinite(flow.epsilon)) {
        return {
            std::numeric_limits<Real>::infinity(),
            std::numeric_limits<Real>::infinity(),
            Real{},
            Real{},
            std::numeric_limits<Real>::infinity(),
            std::numeric_limits<Real>::infinity(),
            config.k_react > Real{} ? std::numeric_limits<Real>::infinity() : Real{}};
    }

    const Real eta_k =
        std::pow(config.viscosity * config.viscosity * config.viscosity / flow.epsilon,
                 Real{0.25});
    const Real eta_b = eta_k / std::sqrt(config.sc);
    const Real tau_eta = std::sqrt(config.viscosity / flow.epsilon);
    return {
        eta_k,
        eta_b,
        Real{1} / eta_k,
        Real{1} / eta_b,
        tau_eta,
        tau_eta * config.delta_u / config.delta0,
        config.k_react > Real{} ? config.k_react * config.c0 * tau_eta : Real{}};
}

[[nodiscard]] ScalarBudgetDiagnostics compute_scalar_budget_diagnostics(
    const Config& config,
    int previous_statistics_step,
    Real previous_var_z,
    int current_step,
    const ScalarDiagnostics& scalar) {
    if (current_step <= previous_statistics_step) {
        return {};
    }

    const Real delta_t =
        static_cast<Real>(current_step - previous_statistics_step);
    const Real variance_decay_rate = -(scalar.var_z - previous_var_z) / delta_t;
    const Real budget_ratio =
        scalar.mean_chi_z > Real{} ? variance_decay_rate / scalar.mean_chi_z : Real{};
    const Real numerical_dissipation_fraction =
        variance_decay_rate > Real{}
            ? Real{1} - scalar.mean_chi_z / variance_decay_rate
            : Real{};
    const Real tau_eff =
        variance_decay_rate > Real{} ? scalar.var_z / variance_decay_rate : Real{};
    const Real tau_eff_star =
        variance_decay_rate > Real{} ? tau_eff * config.delta_u / config.delta0 : Real{};

    return {
        variance_decay_rate,
        budget_ratio,
        numerical_dissipation_fraction,
        tau_eff,
        tau_eff_star};
}

void append_statistics(
    std::ofstream& statistics,
    const Config& config,
    int step,
    Real simulation_time,
    const FlowDiagnostics& flow,
    Real kinetic_energy_decay_rate,
    const ScalarDiagnostics& scalar,
    const ScalarBudgetDiagnostics& scalar_budget,
    Real initial_mean_z,
    Real initial_mean_rho) {
    const ResolutionDiagnostics resolution = compute_resolution_diagnostics(config, flow);
    const Real mean_z_drift = scalar.mean_z - initial_mean_z;
    const Real relative_mass_drift =
        initial_mean_rho != Real{} ? (flow.mean_rho - initial_mean_rho) / initial_mean_rho
                                   : Real{};

    const auto write_real = [&statistics](Real value) {
        statistics << ',' << std::format("{:.17g}", static_cast<double>(value));
    };

    statistics << step;
    write_real(simulation_time);
    write_real(flow.u_max);
    write_real(flow.mean_kinetic_energy);
    write_real(kinetic_energy_decay_rate);
    write_real(flow.transverse_kinetic_energy);
    write_real(flow.fluctuation_kinetic_energy);
    write_real(flow.uy_rms);
    write_real(flow.uz_rms);
    write_real(flow.uperp_rms);
    write_real(flow.enstrophy);
    write_real(flow.epsilon);
    write_real(scalar.mean_ca);
    write_real(scalar.var_ca);
    write_real(scalar.min_ca);
    write_real(scalar.max_ca);
    write_real(scalar.mean_cb);
    write_real(scalar.var_cb);
    write_real(scalar.min_cb);
    write_real(scalar.max_cb);
    write_real(scalar.mean_cc);
    write_real(scalar.var_cc);
    write_real(scalar.rate_true);
    write_real(scalar.rate_mixed);
    write_real(scalar.reaction_efficiency);
    write_real(scalar.mean_z);
    write_real(scalar.var_z);
    write_real(scalar_budget.variance_decay_rate);
    write_real(scalar_budget.budget_ratio);
    write_real(scalar.min_z);
    write_real(scalar.max_z);
    write_real(scalar.mean_chi_z);
    write_real(scalar.rms_chi_z);
    write_real(scalar.max_chi_z);
    write_real(scalar.var_chi_z);
    write_real(scalar.tau_mix);
    write_real(scalar.tau_mix_star);
    write_real(scalar_budget.numerical_dissipation_fraction);
    write_real(scalar_budget.tau_eff);
    write_real(scalar_budget.tau_eff_star);
    write_real(scalar.da_mix);
    write_real(scalar.cov_ab);
    write_real(scalar.segregation_index);
    write_real(scalar.rho_ab);
    write_real(resolution.eta_k);
    write_real(resolution.eta_b);
    write_real(resolution.dx_over_eta_k);
    write_real(resolution.dx_over_eta_b);
    write_real(scalar.mean_grad_z2);
    write_real(scalar.mean_grad_z4);
    write_real(scalar.grad_z_flatness);
    write_real(flow.theta_1);
    write_real(flow.theta_2);
    write_real(flow.theta_avg);
    write_real(flow.re_theta);
    write_real(flow.delta_omega_1);
    write_real(flow.delta_omega_2);
    write_real(flow.delta_omega_avg);
    write_real(scalar.delta_z_1);
    write_real(scalar.delta_z_2);
    write_real(scalar.delta_z_avg);
    write_real(scalar.rms_reaction_rate);
    write_real(scalar.max_reaction_rate);
    write_real(scalar.reaction_effective_volume_fraction);
    write_real(scalar.corr_r_chi_z);
    write_real(resolution.tau_eta);
    write_real(resolution.tau_eta_star);
    write_real(resolution.da_eta);
    write_real(flow.mean_rho);
    write_real(flow.min_rho);
    write_real(flow.max_rho);
    write_real(flow.rho_rms_fluct);
    write_real(flow.mach_max);
    write_real(mean_z_drift);
    write_real(relative_mass_drift);
    statistics << std::endl;
    statistics.flush();
}

void run_simulation(const Config& config, const PerturbationDefinition& perturbation) {
    write_metadata_json(config, perturbation);

    lbm::LatticeMemory<FluidLattice, Real> fluid{config.nx, config.ny, config.nz};
    lbm::LatticeMemory<ScalarLattice, Real> species_a{config.nx, config.ny, config.nz};
    lbm::LatticeMemory<ScalarLattice, Real> species_b{config.nx, config.ny, config.nz};

    initialize_fields(config, perturbation, fluid, species_a, species_b);

    std::ofstream statistics{"statistics_double_shear_3d.csv"};
    if (!statistics) {
        throw std::runtime_error("failed to open statistics_double_shear_3d.csv");
    }
    statistics
        << "step,time,u_max,E_k,kinetic_energy_decay_rate,"
        << "E_perp,E_fluc,uy_rms,uz_rms,uperp_rms,enstrophy,epsilon,"
        << "mean_Ca,var_Ca,min_Ca,max_Ca,"
        << "mean_Cb,var_Cb,min_Cb,max_Cb,"
        << "mean_Cc,var_Cc,rate_true,rate_mixed,reaction_efficiency,"
        << "mean_Z,var_Z,scalar_variance_decay_rate,scalar_budget_ratio,"
        << "min_Z,max_Z,"
        << "mean_chi_Z,rms_chi_Z,max_chi_Z,var_chi_Z,"
        << "tau_mix,tau_mix_star,scalar_numerical_dissipation_fraction,"
        << "tau_eff,tau_eff_star,Da_mix,"
        << "cov_AB,I_seg,rho_AB,"
        << "eta_K,eta_B,dx_over_etaK,dx_over_etaB,"
        << "mean_gradZ2,mean_gradZ4,F_gradZ,"
        << "theta_1,theta_2,theta_avg,Re_theta,"
        << "delta_omega_1,delta_omega_2,delta_omega_avg,"
        << "delta_Z_1,delta_Z_2,delta_Z_avg,"
        << "rms_reaction_rate,max_reaction_rate,"
        << "reaction_effective_volume_fraction,corr_R_chiZ,"
        << "tau_eta,tau_eta_star,Da_eta,"
        << "mean_rho,min_rho,max_rho,rho_rms_fluct,Mach_max,"
        << "mean_Z_drift,relative_mass_drift"
        << std::endl;
    statistics.flush();

    const Real omega_f = Real{1} / config.tau_f;
    const Real omega_s = Real{1} / config.tau_s;
    const std::filesystem::path vtk_dir{"vtk_double_shear_3d"};
    const std::filesystem::path profile_dir{"profiles_double_shear_3d"};
    const auto start = std::chrono::high_resolution_clock::now();

    FlowDiagnostics flow = compute_flow_diagnostics(config, fluid);
    const FlowDiagnostics initial_flow = flow;
    Real kinetic_energy_previous = flow.mean_kinetic_energy;
    Real latest_kinetic_energy_decay_rate{};
    ScalarBudgetDiagnostics scalar_budget{};
    bool burst_active = false;
    int burst_steps_recorded = 0;

    ScalarDiagnostics scalar =
        compute_scalar_diagnostics(config, species_a, species_b);
    Real previous_var_z = scalar.var_z;
    int previous_statistics_step = 0;
    const Real initial_mean_z = scalar.mean_z;
    const Real initial_mean_rho = flow.mean_rho;
    std::cout << "Initial profile check: "
              << "ux(y=0)=" << config.u0 * shear_profile(config, 0)
              << ", ux(y=Ny/2)=" << config.u0 * shear_profile(config, config.ny / 2)
              << ", min_Ca=" << scalar.min_ca
              << ", max_Ca=" << scalar.max_ca
              << ", min_Cb=" << scalar.min_cb
              << ", max_Cb=" << scalar.max_cb
              << ", mean_Ca=" << scalar.mean_ca
              << ", mean_Cb=" << scalar.mean_cb << '\n'
              << std::flush;
    append_statistics(
        statistics,
        config,
        0,
        Real{},
        flow,
        latest_kinetic_energy_decay_rate,
        scalar,
        scalar_budget,
        initial_mean_z,
        initial_mean_rho);
    write_binary_vtk(config, vtk_dir, 0, fluid, species_a, species_b);
    if (config.profile_freq > 0) {
        write_y_profile_csv(config, profile_dir, 0, fluid, species_a, species_b);
    }

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
            latest_kinetic_energy_decay_rate =
                (kinetic_energy_previous - flow.mean_kinetic_energy) /
                static_cast<Real>(config.stat_freq);
            kinetic_energy_previous = flow.mean_kinetic_energy;

            scalar = compute_scalar_diagnostics(config, species_a, species_b);
            scalar_budget = compute_scalar_budget_diagnostics(
                config,
                previous_statistics_step,
                previous_var_z,
                step,
                scalar);
            append_statistics(
                statistics,
                config,
                step,
                static_cast<Real>(step),
                flow,
                latest_kinetic_energy_decay_rate,
                scalar,
                scalar_budget,
                initial_mean_z,
                initial_mean_rho);
            previous_var_z = scalar.var_z;
            previous_statistics_step = step;

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
            const ResolutionDiagnostics resolution =
                compute_resolution_diagnostics(config, flow);

            std::cout << std::format(
                "Step [{} / {}] Umax: {:.6g} Ek: {:.6g} Eperp: {:.6g} Efluc: {:.6g} Enst: {:.6g} eps: {:.6g} chiZ: {:.6g} budget: {:.6g} tauMix*: {:.6g} DaMix: {:.6g} dx/etaB: {:.6g} theta: {:.6g} dZ: {:.6g} ReTheta: {:.6g} Mach: {:.6g} dE/dt: {:.6g} Burst: {} MLUPS: {:.6g}\n",
                step,
                config.steps,
                static_cast<double>(flow.u_max),
                static_cast<double>(flow.mean_kinetic_energy),
                static_cast<double>(flow.transverse_kinetic_energy),
                static_cast<double>(flow.fluctuation_kinetic_energy),
                static_cast<double>(flow.enstrophy),
                static_cast<double>(flow.epsilon),
                static_cast<double>(scalar.mean_chi_z),
                static_cast<double>(scalar_budget.budget_ratio),
                static_cast<double>(scalar.tau_mix_star),
                static_cast<double>(scalar.da_mix),
                static_cast<double>(resolution.dx_over_eta_b),
                static_cast<double>(flow.theta_avg),
                static_cast<double>(scalar.delta_z_avg),
                static_cast<double>(flow.re_theta),
                static_cast<double>(flow.mach_max),
                static_cast<double>(latest_kinetic_energy_decay_rate),
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
        if (config.profile_freq > 0 && step % config.profile_freq == 0) {
            write_y_profile_csv(config, profile_dir, step, fluid, species_a, species_b);
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
              << "Profile directory: "
              << (config.profile_freq > 0 ? profile_dir.string() : "disabled") << '\n'
              << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_arguments(argc, argv);
        const PerturbationDefinition perturbation = prepare_perturbation(config);
        print_recap(config, perturbation);
        run_simulation(config, perturbation);
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        print_usage(
            std::cerr,
            argc > 0 ? argv[0] : "lbm_turbulent_reactive_double_shear_3d");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

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
#include <cstring>
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
#include <utility>
#include <vector>

#include "double_shear_parameters.hpp"
#include "lattice_core.hpp"
#include "lattice_filter.hpp"
#include "lattice_io.hpp"
#include "lattice_memory.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

#ifdef LB_CUBE_HAS_FFTW
#include <fftw3.h>
#endif

namespace {

using FluidLattice = lbm::D3Q27;
using ScalarLattice = lbm::D3Q7;
using Real = double;

constexpr std::uint32_t checkpoint_current_format_version = 2;
constexpr std::uint32_t checkpoint_legacy_full_ping_pong_format_version = 1;

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
    int profile_freq{};
    int pdf_freq{};
    int pdf_bins{128};
    int joint_pdf_bins{128};
    Real pdf_log_chi_min{-12};
    Real pdf_log_chi_max{2};
    Real pdf_log_r_min{-12};
    Real pdf_log_r_max{2};
    int spectrum_freq{};
    std::vector<int> filter_widths{};
    int filter_freq{};
    int filter_conditional_bins{64};
    Real filter_log_tau_zz_min{-12};
    Real filter_log_tau_zz_max{0};
    int filter_pdf_bins{128};
    int filter_joint_pdf_bins{128};
    Real filter_tau_ab_min{-0.5};
    Real filter_tau_ab_max{0.5};
    int checkpoint_freq{};
    Real checkpoint_walltime_hours{};
    int checkpoint_keep{2};
    Real max_walltime_hours{};
    Real walltime_safety_margin_hours{0.25};
    std::string restart_from{};
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

struct RestartState {
    bool enabled{};
    int step{};
    Real kinetic_energy_previous{};
    int previous_kinetic_energy_step{};
    Real latest_kinetic_energy_decay_rate{};
    Real previous_var_z{};
    int previous_statistics_step{};
    Real initial_mean_z{};
    Real initial_mean_rho{};
    std::filesystem::path source_path{};
};

struct CheckpointWriteResult {
    std::filesystem::path path{};
    double elapsed_seconds{};
};

struct CheckpointHeader {
    char magic[16]{};
    std::uint32_t version{};
    std::uint32_t header_size{};
    std::uint64_t step{};
    std::uint64_t nx{};
    std::uint64_t ny{};
    std::uint64_t nz{};
    std::uint32_t real_size{};
    std::uint32_t fluid_q{};
    std::uint32_t scalar_q{};
    std::uint32_t fluid_d{};
    std::uint32_t scalar_d{};
    std::uint32_t fluid_current_buffer{};
    std::uint32_t species_a_current_buffer{};
    std::uint32_t species_b_current_buffer{};
    std::uint32_t reserved{};
    std::uint64_t fluid_buffer_size{};
    std::uint64_t scalar_buffer_size{};
    Real tau_f{};
    Real tau_s{};
    Real k_react{};
    Real u0{};
    Real c0{};
    Real delta_ratio{};
    Real re_delta{};
    Real sc{};
    Real da_delta{};
    Real delta0{};
    Real delta_u{};
    Real viscosity{};
    Real scalar_diffusivity{};
    Real kinetic_energy_previous{};
    std::int64_t previous_kinetic_energy_step{};
    Real latest_kinetic_energy_decay_rate{};
    Real previous_var_z{};
    std::int64_t previous_statistics_step{};
    Real initial_mean_z{};
    Real initial_mean_rho{};
    std::uint64_t perturb_number_of_modes{};
    std::uint32_t perturb_enabled{};
    Real perturb_target_rms{};
    Real perturb_achieved_rms{};
    Real perturb_mean_ux{};
    Real perturb_mean_uy{};
    Real perturb_mean_uz{};
    Real perturb_rms_ux{};
    Real perturb_rms_uy{};
    Real perturb_rms_uz{};
    Real perturb_max_plane_mean_abs_ux{};
    Real perturb_max_plane_mean_abs_uy{};
    Real perturb_max_plane_mean_abs_uz{};
    Real perturb_max_plane_rms{};
    std::uint64_t perturb_max_plane_rms_y{};
    Real perturb_y1_plane_rms{};
    Real perturb_y2_plane_rms{};
    Real perturb_y1_plane_rms_ux{};
    Real perturb_y1_plane_rms_uy{};
    Real perturb_y1_plane_rms_uz{};
    Real perturb_y2_plane_rms_ux{};
    Real perturb_y2_plane_rms_uy{};
    Real perturb_y2_plane_rms_uz{};
    Real perturb_divergence_rms{};
    Real perturb_normalized_divergence{};
    Real perturb_localization_energy_max{};
    std::uint64_t perturb_localization_peak_y1{};
    std::uint64_t perturb_localization_peak_y2{};
    Real perturb_localization_layer_to_bulk_ratio{};
    std::uint64_t payload_checksum{};
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

struct PdfCounters {
    std::uint64_t z_underflow{};
    std::uint64_t z_overflow{};
    std::uint64_t chi_zero{};
    std::uint64_t chi_underflow{};
    std::uint64_t chi_overflow{};
    std::uint64_t chi_included{};
    std::uint64_t r_zero{};
    std::uint64_t r_underflow{};
    std::uint64_t r_overflow{};
    std::uint64_t r_included{};
    std::uint64_t joint_z_chi_excluded{};
    std::uint64_t joint_r_chi_excluded{};
    std::uint64_t joint_r_chi_included{};
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

[[nodiscard]] std::vector<int> parse_filter_widths(
    std::string_view flag,
    std::string_view value) {
    if (value.empty()) {
        throw std::runtime_error(std::format("{} must not be empty", flag));
    }

    std::vector<int> widths;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        const std::size_t comma = value.find(',', begin);
        const std::size_t end =
            comma == std::string_view::npos ? value.size() : comma;
        const std::string_view token = value.substr(begin, end - begin);
        if (token.empty()) {
            throw std::runtime_error(
                std::format("{} contains an empty list entry", flag));
        }

        const int width = parse_int(flag, token);
        if (width <= 0 || width % 2 == 0) {
            throw std::runtime_error(
                std::format("{} entries must be positive odd integers", flag));
        }
        widths.push_back(width);

        if (comma == std::string_view::npos) {
            break;
        }
        begin = comma + 1;
    }

    std::ranges::sort(widths);
    if (std::ranges::adjacent_find(widths) != widths.end()) {
        throw std::runtime_error(
            std::format("{} must not contain duplicate widths", flag));
    }

    return widths;
}

[[nodiscard]] bool filtering_enabled(const Config& config) {
    return !config.filter_widths.empty();
}

[[nodiscard]] std::string filter_widths_to_string(const Config& config) {
    if (config.filter_widths.empty()) {
        return "disabled";
    }

    std::string result;
    for (std::size_t i = 0; i < config.filter_widths.size(); ++i) {
        if (i > 0) {
            result += ',';
        }
        result += std::to_string(config.filter_widths[i]);
    }
    return result;
}

[[nodiscard]] std::string filter_widths_to_json(const Config& config) {
    std::string result = "[";
    for (std::size_t i = 0; i < config.filter_widths.size(); ++i) {
        if (i > 0) {
            result += ", ";
        }
        result += std::to_string(config.filter_widths[i]);
    }
    result += "]";
    return result;
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
        << "  --vtk_freq <n>           Binary VTK interval; 0 disables output (default 1000)\n"
        << "  --profile_freq <n>       y-profile CSV interval; 0 disables output (default 0)\n"
        << "  --pdf_freq <n>           PDF output interval; 0 disables output (default 0)\n"
        << "  --pdf_bins <n>           Marginal PDF bin count (default 128)\n"
        << "  --joint_pdf_bins <n>     Joint PDF bin count per dimension (default 128)\n"
        << "  --pdf_log_chi_min <v>    Minimum log10(chi*) bin edge (default -12)\n"
        << "  --pdf_log_chi_max <v>    Maximum log10(chi*) bin edge (default 2)\n"
        << "  --pdf_log_R_min <v>      Minimum log10(R*) bin edge (default -12)\n"
        << "  --pdf_log_R_max <v>      Maximum log10(R*) bin edge (default 2)\n"
        << "  --spectrum_freq <n>      2D x-z scalar-spectrum interval; 0 disables output (default 0)\n"
        << "  --filter_widths <list>   Comma-separated odd box-filter widths, e.g. 3,5,9 (default disabled)\n"
        << "  --filter_width <n>       Backward-compatible single-width alias; reject with --filter_widths\n"
        << "  --filter_freq <n>        Diagnostic scalar filtering interval; 0 disables filtering (default 0)\n"
        << "  --filter_conditional_bins <n>  Filtered conditional bin count (default 64)\n"
        << "  --filter_log_tauZZ_min <v>     Minimum log10(tau_ZZ) conditional edge (default -12)\n"
        << "  --filter_log_tauZZ_max <v>     Maximum log10(tau_ZZ) conditional edge (default 0)\n"
        << "  --filter_pdf_bins <n>          SGS marginal PDF bin count (default 128)\n"
        << "  --filter_joint_pdf_bins <n>    SGS joint PDF bin count per dimension (default 128)\n"
        << "  --filter_tauAB_min <v>         Minimum tau_AB PDF bin edge (default -0.5)\n"
        << "  --filter_tauAB_max <v>         Maximum tau_AB PDF bin edge (default 0.5)\n"
        << "  --checkpoint_freq <n>          Checkpoint interval in completed steps; 0 disables (default 0)\n"
        << "  --checkpoint_walltime <hours>  Checkpoint interval in elapsed wall-clock hours; 0 disables (default 0)\n"
        << "  --checkpoint_keep <n>          Number of completed checkpoints to retain (default 2)\n"
        << "  --max_walltime <hours>         Gracefully checkpoint and exit before this elapsed walltime; 0 disables (default 0)\n"
        << "  --walltime_safety_margin <h>   Safety margin reserved before max walltime (default 0.25)\n"
        << "  --restart_from <file>          Restart from a binary checkpoint; --steps is the absolute final step\n"
        << "  --help                   Show this message\n";
}

[[nodiscard]] Config parse_arguments(int argc, char** argv) {
    Config config{};
    lbm::double_shear::ExplicitParameterFlags explicit_parameters{};
    bool filter_widths_explicit = false;
    bool legacy_filter_width_explicit = false;

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
            config.vtk_freq =
                parse_nonnegative_int(flag, require_value(index, argc, argv));
        } else if (flag == "--profile_freq") {
            config.profile_freq =
                parse_nonnegative_int(flag, require_value(index, argc, argv));
        } else if (flag == "--pdf_freq") {
            config.pdf_freq =
                parse_nonnegative_int(flag, require_value(index, argc, argv));
        } else if (flag == "--pdf_bins") {
            config.pdf_bins = parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--joint_pdf_bins") {
            config.joint_pdf_bins = parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--pdf_log_chi_min") {
            config.pdf_log_chi_min = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--pdf_log_chi_max") {
            config.pdf_log_chi_max = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--pdf_log_R_min") {
            config.pdf_log_r_min = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--pdf_log_R_max") {
            config.pdf_log_r_max = parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--spectrum_freq") {
            config.spectrum_freq =
                parse_nonnegative_int(flag, require_value(index, argc, argv));
        } else if (flag == "--filter_widths") {
            if (filter_widths_explicit) {
                throw std::runtime_error("--filter_widths was specified more than once");
            }
            config.filter_widths =
                parse_filter_widths(flag, require_value(index, argc, argv));
            filter_widths_explicit = true;
        } else if (flag == "--filter_width") {
            if (legacy_filter_width_explicit) {
                throw std::runtime_error("--filter_width was specified more than once");
            }
            const int width =
                parse_nonnegative_int(flag, require_value(index, argc, argv));
            if (width > 0 && width % 2 == 0) {
                throw std::runtime_error(
                    "--filter_width must be 0 or a positive odd integer");
            }
            if (width > 0) {
                config.filter_widths = {width};
            }
            legacy_filter_width_explicit = true;
        } else if (flag == "--filter_freq") {
            config.filter_freq =
                parse_nonnegative_int(flag, require_value(index, argc, argv));
        } else if (flag == "--filter_conditional_bins") {
            config.filter_conditional_bins =
                parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--filter_log_tauZZ_min") {
            config.filter_log_tau_zz_min =
                parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--filter_log_tauZZ_max") {
            config.filter_log_tau_zz_max =
                parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--filter_pdf_bins") {
            config.filter_pdf_bins = parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--filter_joint_pdf_bins") {
            config.filter_joint_pdf_bins =
                parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--filter_tauAB_min") {
            config.filter_tau_ab_min =
                parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--filter_tauAB_max") {
            config.filter_tau_ab_max =
                parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--checkpoint_freq") {
            config.checkpoint_freq =
                parse_nonnegative_int(flag, require_value(index, argc, argv));
        } else if (flag == "--checkpoint_walltime") {
            config.checkpoint_walltime_hours =
                parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--checkpoint_keep") {
            config.checkpoint_keep =
                parse_int(flag, require_value(index, argc, argv));
        } else if (flag == "--max_walltime") {
            config.max_walltime_hours =
                parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--walltime_safety_margin") {
            config.walltime_safety_margin_hours =
                parse_real(flag, require_value(index, argc, argv));
        } else if (flag == "--restart_from") {
            config.restart_from = require_value(index, argc, argv);
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
    if (config.pdf_log_chi_max <= config.pdf_log_chi_min) {
        throw std::runtime_error("--pdf_log_chi_max must be greater than --pdf_log_chi_min");
    }
    if (config.pdf_log_r_max <= config.pdf_log_r_min) {
        throw std::runtime_error("--pdf_log_R_max must be greater than --pdf_log_R_min");
    }
    if (filter_widths_explicit && legacy_filter_width_explicit) {
        throw std::runtime_error(
            "choose either --filter_widths or the backward-compatible --filter_width alias, not both");
    }
    if (filtering_enabled(config) != (config.filter_freq > 0)) {
        throw std::runtime_error(
            "--filter_widths/--filter_width and --filter_freq must either both enable filtering or both be disabled");
    }
    if (config.filter_log_tau_zz_max <= config.filter_log_tau_zz_min) {
        throw std::runtime_error(
            "--filter_log_tauZZ_max must be greater than --filter_log_tauZZ_min");
    }
    if (config.filter_tau_ab_max <= config.filter_tau_ab_min) {
        throw std::runtime_error(
            "--filter_tauAB_max must be greater than --filter_tauAB_min");
    }
    if (config.checkpoint_walltime_hours < Real{}) {
        throw std::runtime_error("--checkpoint_walltime must be non-negative");
    }
    if (config.max_walltime_hours < Real{}) {
        throw std::runtime_error("--max_walltime must be non-negative");
    }
    if (config.walltime_safety_margin_hours < Real{}) {
        throw std::runtime_error("--walltime_safety_margin must be non-negative");
    }
#ifndef LB_CUBE_HAS_FFTW
    if (config.spectrum_freq > 0) {
        throw std::runtime_error(
            "--spectrum_freq requested, but this executable was built without FFTW. "
            "Reconfigure with -DLB_CUBE_ENABLE_FFTW=ON.");
    }
#endif

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
        << "VTK frequency: " << config.vtk_freq
        << (config.vtk_freq > 0 ? "" : " (disabled)") << '\n'
        << "Profile frequency: " << config.profile_freq
        << (config.profile_freq > 0 ? "" : " (disabled)") << '\n'
        << "PDF frequency: " << config.pdf_freq
        << (config.pdf_freq > 0 ? "" : " (disabled)") << '\n'
        << "PDF bins: " << config.pdf_bins
        << ", joint PDF bins: " << config.joint_pdf_bins << '\n'
        << "log10(chi*) range: [" << config.pdf_log_chi_min
        << ", " << config.pdf_log_chi_max << "]\n"
        << "log10(R*) range: [" << config.pdf_log_r_min
        << ", " << config.pdf_log_r_max << "]\n"
        << "Spectrum frequency: " << config.spectrum_freq
        << (config.spectrum_freq > 0 ? "" : " (disabled)") << '\n'
        << "Filter widths: " << filter_widths_to_string(config)
        << (filtering_enabled(config) ? "" : " (disabled)") << '\n'
        << "Filter frequency: " << config.filter_freq
        << (config.filter_freq > 0 ? "" : " (disabled)") << '\n'
        << "Filter conditional bins: " << config.filter_conditional_bins << '\n'
        << "log10(tau_ZZ) conditional range: ["
        << config.filter_log_tau_zz_min << ", "
        << config.filter_log_tau_zz_max << "]\n"
        << "Filter PDF bins: " << config.filter_pdf_bins
        << ", joint filter PDF bins: " << config.filter_joint_pdf_bins << '\n'
        << "tau_AB PDF range: [" << config.filter_tau_ab_min
        << ", " << config.filter_tau_ab_max << "]\n"
        << "Checkpoint frequency: " << config.checkpoint_freq
        << (config.checkpoint_freq > 0 ? "" : " (disabled)") << '\n'
        << "Checkpoint walltime: " << config.checkpoint_walltime_hours
        << " h" << (config.checkpoint_walltime_hours > Real{} ? "" : " (disabled)") << '\n'
        << "Checkpoint keep: " << config.checkpoint_keep << '\n'
        << "Maximum walltime: " << config.max_walltime_hours
        << " h" << (config.max_walltime_hours > Real{} ? "" : " (disabled)") << '\n'
        << "Walltime safety margin: " << config.walltime_safety_margin_hours
        << " h\n"
        << "Restart from: "
        << (config.restart_from.empty() ? "none" : config.restart_from) << '\n'
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

[[nodiscard]] std::string json_string(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped.push_back(ch);
            break;
        }
    }
    escaped.push_back('"');
    return escaped;
}

void write_metadata_json(
    const Config& config,
    const PerturbationDefinition& perturbation,
    const RestartState& restart_state) {
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
        << "  \"pdf_freq\": " << config.pdf_freq << ",\n"
        << "  \"pdf_bins\": " << config.pdf_bins << ",\n"
        << "  \"joint_pdf_bins\": " << config.joint_pdf_bins << ",\n"
        << "  \"pdf_log_chi_min\": " << json_number(config.pdf_log_chi_min) << ",\n"
        << "  \"pdf_log_chi_max\": " << json_number(config.pdf_log_chi_max) << ",\n"
        << "  \"pdf_log_R_min\": " << json_number(config.pdf_log_r_min) << ",\n"
        << "  \"pdf_log_R_max\": " << json_number(config.pdf_log_r_max) << ",\n"
        << "  \"spectrum_freq\": " << config.spectrum_freq << ",\n"
        << "  \"filter_width\": "
        << (config.filter_widths.empty() ? 0 : config.filter_widths.front()) << ",\n"
        << "  \"filter_widths\": " << filter_widths_to_json(config) << ",\n"
        << "  \"filter_freq\": " << config.filter_freq << ",\n"
        << "  \"filter_conditional_bins\": " << config.filter_conditional_bins << ",\n"
        << "  \"filter_log_tauZZ_min\": " << json_number(config.filter_log_tau_zz_min) << ",\n"
        << "  \"filter_log_tauZZ_max\": " << json_number(config.filter_log_tau_zz_max) << ",\n"
        << "  \"filter_pdf_bins\": " << config.filter_pdf_bins << ",\n"
        << "  \"filter_joint_pdf_bins\": " << config.filter_joint_pdf_bins << ",\n"
        << "  \"filter_tauAB_min\": " << json_number(config.filter_tau_ab_min) << ",\n"
        << "  \"filter_tauAB_max\": " << json_number(config.filter_tau_ab_max) << ",\n"
        << "  \"checkpoint_freq\": " << config.checkpoint_freq << ",\n"
        << "  \"checkpoint_walltime\": " << json_number(config.checkpoint_walltime_hours) << ",\n"
        << "  \"checkpoint_keep\": " << config.checkpoint_keep << ",\n"
        << "  \"max_walltime\": " << json_number(config.max_walltime_hours) << ",\n"
        << "  \"walltime_safety_margin\": "
        << json_number(config.walltime_safety_margin_hours) << ",\n"
        << "  \"restart_from\": "
        << (restart_state.enabled ? json_string(restart_state.source_path.string()) : "null")
        << ",\n"
        << "  \"restart_step\": "
        << (restart_state.enabled ? std::to_string(restart_state.step) : "null")
        << ",\n"
        << "  \"checkpoint_format_version\": " << checkpoint_current_format_version << ",\n"
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

[[nodiscard]] constexpr std::array<char, 16> checkpoint_magic() {
    return {'L', 'B', 'C', 'U', 'B', 'E', '_', 'D',
            'S', '3', 'D', '_', 'C', 'K', 'P', 'T'};
}

[[nodiscard]] constexpr std::uint32_t checkpoint_format_version() {
    return checkpoint_current_format_version;
}

[[nodiscard]] std::filesystem::path checkpoint_directory() {
    return "checkpoints_double_shear_3d";
}

[[nodiscard]] std::filesystem::path checkpoint_path_for_step(int step) {
    return checkpoint_directory() / std::format("checkpoint_{:08}.bin", step);
}

void checksum_update(std::uint64_t& checksum, const void* data, std::size_t size) {
    constexpr std::uint64_t prime = 1099511628211ULL;
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        checksum ^= static_cast<std::uint64_t>(bytes[i]);
        checksum *= prime;
    }
}

template <typename T>
void checksum_value(std::uint64_t& checksum, const T& value) {
    checksum_update(checksum, &value, sizeof(T));
}

template <typename T>
void checksum_vector(std::uint64_t& checksum, const std::vector<T>& values) {
    if (!values.empty()) {
        checksum_update(checksum, values.data(), values.size() * sizeof(T));
    }
}

[[nodiscard]] std::uint64_t checkpoint_payload_checksum_common(
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b,
    const RestartState& state) {
    std::uint64_t checksum = 1469598103934665603ULL;
    checksum_value(checksum, state.step);
    checksum_value(checksum, state.kinetic_energy_previous);
    checksum_value(checksum, state.previous_kinetic_energy_step);
    checksum_value(checksum, state.latest_kinetic_energy_decay_rate);
    checksum_value(checksum, state.previous_var_z);
    checksum_value(checksum, state.previous_statistics_step);
    checksum_value(checksum, state.initial_mean_z);
    checksum_value(checksum, state.initial_mean_rho);
    checksum_value(checksum, fluid.current_buffer_index());
    checksum_value(checksum, species_a.current_buffer_index());
    checksum_value(checksum, species_b.current_buffer_index());
    return checksum;
}

[[nodiscard]] std::uint64_t checkpoint_payload_checksum_compact(
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b,
    const RestartState& state) {
    std::uint64_t checksum =
        checkpoint_payload_checksum_common(fluid, species_a, species_b, state);
    checksum_vector(checksum, fluid.raw_buffer(fluid.current_buffer_index()));
    checksum_vector(checksum, species_a.raw_buffer(species_a.current_buffer_index()));
    checksum_vector(checksum, species_b.raw_buffer(species_b.current_buffer_index()));
    return checksum;
}

[[nodiscard]] std::uint64_t checkpoint_payload_checksum_full_ping_pong(
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b,
    const RestartState& state) {
    std::uint64_t checksum =
        checkpoint_payload_checksum_common(fluid, species_a, species_b, state);
    checksum_vector(checksum, fluid.raw_buffer(0));
    checksum_vector(checksum, fluid.raw_buffer(1));
    checksum_vector(checksum, species_a.raw_buffer(0));
    checksum_vector(checksum, species_a.raw_buffer(1));
    checksum_vector(checksum, species_b.raw_buffer(0));
    checksum_vector(checksum, species_b.raw_buffer(1));
    return checksum;
}

void write_bytes(std::ofstream& stream, const void* data, std::size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint data");
    }
}

void read_bytes(std::ifstream& stream, void* data, std::size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("checkpoint is truncated or unreadable");
    }
}

template <typename T>
void write_vector(std::ofstream& stream, const std::vector<T>& values) {
    if (!values.empty()) {
        write_bytes(stream, values.data(), values.size() * sizeof(T));
    }
}

template <typename T>
void read_vector(std::ifstream& stream, std::vector<T>& values) {
    if (!values.empty()) {
        read_bytes(stream, values.data(), values.size() * sizeof(T));
    }
}

[[nodiscard]] CheckpointHeader make_checkpoint_header(
    const Config& config,
    const PerturbationDefinition& perturbation,
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b,
    const RestartState& state) {
    CheckpointHeader header{};
    std::memset(&header, 0, sizeof(header));
    const auto magic = checkpoint_magic();
    std::copy(magic.begin(), magic.end(), std::begin(header.magic));
    header.version = checkpoint_format_version();
    header.header_size = sizeof(CheckpointHeader);
    header.step = static_cast<std::uint64_t>(state.step);
    header.nx = config.nx;
    header.ny = config.ny;
    header.nz = config.nz;
    header.real_size = sizeof(Real);
    header.fluid_q = FluidLattice::Q;
    header.scalar_q = ScalarLattice::Q;
    header.fluid_d = FluidLattice::D;
    header.scalar_d = ScalarLattice::D;
    header.fluid_current_buffer =
        static_cast<std::uint32_t>(fluid.current_buffer_index());
    header.species_a_current_buffer =
        static_cast<std::uint32_t>(species_a.current_buffer_index());
    header.species_b_current_buffer =
        static_cast<std::uint32_t>(species_b.current_buffer_index());
    header.fluid_buffer_size = fluid.raw_buffer(0).size();
    header.scalar_buffer_size = species_a.raw_buffer(0).size();
    header.tau_f = config.tau_f;
    header.tau_s = config.tau_s;
    header.k_react = config.k_react;
    header.u0 = config.u0;
    header.c0 = config.c0;
    header.delta_ratio = config.delta_ratio;
    header.re_delta = config.re_delta;
    header.sc = config.sc;
    header.da_delta = config.da_delta;
    header.delta0 = config.delta0;
    header.delta_u = config.delta_u;
    header.viscosity = config.viscosity;
    header.scalar_diffusivity = config.scalar_diffusivity;
    header.kinetic_energy_previous = state.kinetic_energy_previous;
    header.previous_kinetic_energy_step = state.previous_kinetic_energy_step;
    header.latest_kinetic_energy_decay_rate = state.latest_kinetic_energy_decay_rate;
    header.previous_var_z = state.previous_var_z;
    header.previous_statistics_step = state.previous_statistics_step;
    header.initial_mean_z = state.initial_mean_z;
    header.initial_mean_rho = state.initial_mean_rho;
    const PerturbationDiagnostics& pert = perturbation.diagnostics;
    header.perturb_number_of_modes = pert.number_of_modes;
    header.perturb_enabled = pert.enabled ? 1U : 0U;
    header.perturb_target_rms = pert.target_rms;
    header.perturb_achieved_rms = pert.achieved_rms;
    header.perturb_mean_ux = pert.mean_ux;
    header.perturb_mean_uy = pert.mean_uy;
    header.perturb_mean_uz = pert.mean_uz;
    header.perturb_rms_ux = pert.rms_ux;
    header.perturb_rms_uy = pert.rms_uy;
    header.perturb_rms_uz = pert.rms_uz;
    header.perturb_max_plane_mean_abs_ux = pert.max_plane_mean_abs_ux;
    header.perturb_max_plane_mean_abs_uy = pert.max_plane_mean_abs_uy;
    header.perturb_max_plane_mean_abs_uz = pert.max_plane_mean_abs_uz;
    header.perturb_max_plane_rms = pert.max_plane_rms;
    header.perturb_max_plane_rms_y = pert.max_plane_rms_y;
    header.perturb_y1_plane_rms = pert.y1_plane_rms;
    header.perturb_y2_plane_rms = pert.y2_plane_rms;
    header.perturb_y1_plane_rms_ux = pert.y1_plane_rms_ux;
    header.perturb_y1_plane_rms_uy = pert.y1_plane_rms_uy;
    header.perturb_y1_plane_rms_uz = pert.y1_plane_rms_uz;
    header.perturb_y2_plane_rms_ux = pert.y2_plane_rms_ux;
    header.perturb_y2_plane_rms_uy = pert.y2_plane_rms_uy;
    header.perturb_y2_plane_rms_uz = pert.y2_plane_rms_uz;
    header.perturb_divergence_rms = pert.divergence_rms;
    header.perturb_normalized_divergence = pert.normalized_divergence;
    header.perturb_localization_energy_max = pert.localization_energy_max;
    header.perturb_localization_peak_y1 = pert.localization_peak_y1;
    header.perturb_localization_peak_y2 = pert.localization_peak_y2;
    header.perturb_localization_layer_to_bulk_ratio =
        pert.localization_layer_to_bulk_ratio;
    header.payload_checksum =
        checkpoint_payload_checksum_compact(fluid, species_a, species_b, state);
    return header;
}

void restore_perturbation_diagnostics(
    PerturbationDefinition& perturbation,
    const CheckpointHeader& header) {
    PerturbationDiagnostics& pert = perturbation.diagnostics;
    pert.enabled = header.perturb_enabled != 0U;
    pert.number_of_modes = header.perturb_number_of_modes;
    pert.target_rms = header.perturb_target_rms;
    pert.achieved_rms = header.perturb_achieved_rms;
    pert.mean_ux = header.perturb_mean_ux;
    pert.mean_uy = header.perturb_mean_uy;
    pert.mean_uz = header.perturb_mean_uz;
    pert.rms_ux = header.perturb_rms_ux;
    pert.rms_uy = header.perturb_rms_uy;
    pert.rms_uz = header.perturb_rms_uz;
    pert.max_plane_mean_abs_ux = header.perturb_max_plane_mean_abs_ux;
    pert.max_plane_mean_abs_uy = header.perturb_max_plane_mean_abs_uy;
    pert.max_plane_mean_abs_uz = header.perturb_max_plane_mean_abs_uz;
    pert.max_plane_rms = header.perturb_max_plane_rms;
    pert.max_plane_rms_y = header.perturb_max_plane_rms_y;
    pert.y1_plane_rms = header.perturb_y1_plane_rms;
    pert.y2_plane_rms = header.perturb_y2_plane_rms;
    pert.y1_plane_rms_ux = header.perturb_y1_plane_rms_ux;
    pert.y1_plane_rms_uy = header.perturb_y1_plane_rms_uy;
    pert.y1_plane_rms_uz = header.perturb_y1_plane_rms_uz;
    pert.y2_plane_rms_ux = header.perturb_y2_plane_rms_ux;
    pert.y2_plane_rms_uy = header.perturb_y2_plane_rms_uy;
    pert.y2_plane_rms_uz = header.perturb_y2_plane_rms_uz;
    pert.divergence_rms = header.perturb_divergence_rms;
    pert.normalized_divergence = header.perturb_normalized_divergence;
    pert.localization_energy_max = header.perturb_localization_energy_max;
    pert.localization_peak_y1 = header.perturb_localization_peak_y1;
    pert.localization_peak_y2 = header.perturb_localization_peak_y2;
    pert.localization_layer_to_bulk_ratio =
        header.perturb_localization_layer_to_bulk_ratio;
}

void validate_checkpoint_header(const Config& config, const CheckpointHeader& header) {
    const auto magic = checkpoint_magic();
    if (!std::equal(magic.begin(), magic.end(), std::begin(header.magic))) {
        throw std::runtime_error("checkpoint magic identifier does not match LB-Cube double-shear checkpoints");
    }
    if (header.version != checkpoint_current_format_version &&
        header.version != checkpoint_legacy_full_ping_pong_format_version) {
        throw std::runtime_error(
            std::format("unsupported checkpoint format version {}", header.version));
    }
    if (header.header_size != sizeof(CheckpointHeader)) {
        throw std::runtime_error("checkpoint header size is incompatible with this executable");
    }
    if (header.real_size != sizeof(Real)) {
        throw std::runtime_error("checkpoint precision does not match this executable");
    }
    if (header.fluid_q != FluidLattice::Q || header.fluid_d != FluidLattice::D ||
        header.scalar_q != ScalarLattice::Q || header.scalar_d != ScalarLattice::D) {
        throw std::runtime_error("checkpoint lattice/model identifiers are incompatible");
    }
    if (header.nx != config.nx || header.ny != config.ny || header.nz != config.nz) {
        throw std::runtime_error("checkpoint grid dimensions do not match --Nx/--Ny/--Nz");
    }
    if (header.fluid_buffer_size !=
            static_cast<std::uint64_t>(FluidLattice::Q) * config.nx * config.ny * config.nz ||
        header.scalar_buffer_size !=
            static_cast<std::uint64_t>(ScalarLattice::Q) * config.nx * config.ny * config.nz) {
        throw std::runtime_error("checkpoint stored array sizes do not match the requested grid/lattices");
    }
    if (header.fluid_current_buffer > 1 || header.species_a_current_buffer > 1 ||
        header.species_b_current_buffer > 1) {
        throw std::runtime_error("checkpoint contains an invalid active buffer index");
    }

    const auto check_close = [](std::string_view name, Real actual, Real expected) {
        const Real scale = std::max<Real>({Real{1}, std::abs(actual), std::abs(expected)});
        if (std::abs(actual - expected) > Real{1.0e-11} * scale) {
            throw std::runtime_error(
                std::format(
                    "checkpoint parameter {} is incompatible: checkpoint={}, command-line={}",
                    name,
                    actual,
                    expected));
        }
    };

    check_close("tau_f", header.tau_f, config.tau_f);
    check_close("tau_s", header.tau_s, config.tau_s);
    check_close("k_react", header.k_react, config.k_react);
    check_close("U0", header.u0, config.u0);
    check_close("C0", header.c0, config.c0);
    check_close("delta_ratio", header.delta_ratio, config.delta_ratio);
    check_close("Re_delta", header.re_delta, config.re_delta);
    check_close("Sc", header.sc, config.sc);
    check_close("Da_delta", header.da_delta, config.da_delta);
    check_close("delta0", header.delta0, config.delta0);
    check_close("DeltaU", header.delta_u, config.delta_u);
    check_close("nu", header.viscosity, config.viscosity);
    check_close("D", header.scalar_diffusivity, config.scalar_diffusivity);
}

[[nodiscard]] CheckpointWriteResult write_checkpoint(
    const Config& config,
    const PerturbationDefinition& perturbation,
    int step,
    Real kinetic_energy_previous,
    int previous_kinetic_energy_step,
    Real latest_kinetic_energy_decay_rate,
    Real previous_var_z,
    int previous_statistics_step,
    Real initial_mean_z,
    Real initial_mean_rho,
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b) {
    std::filesystem::create_directories(checkpoint_directory());
    const std::filesystem::path final_path = checkpoint_path_for_step(step);
    const std::filesystem::path tmp_path = final_path.string() + ".tmp";
    const auto checkpoint_start = std::chrono::high_resolution_clock::now();

    RestartState state{};
    state.step = step;
    state.kinetic_energy_previous = kinetic_energy_previous;
    state.previous_kinetic_energy_step = previous_kinetic_energy_step;
    state.latest_kinetic_energy_decay_rate = latest_kinetic_energy_decay_rate;
    state.previous_var_z = previous_var_z;
    state.previous_statistics_step = previous_statistics_step;
    state.initial_mean_z = initial_mean_z;
    state.initial_mean_rho = initial_mean_rho;
    const CheckpointHeader header =
        make_checkpoint_header(config, perturbation, fluid, species_a, species_b, state);

    {
        std::ofstream checkpoint{tmp_path, std::ios::binary | std::ios::trunc};
        if (!checkpoint) {
            throw std::runtime_error(
                std::format("failed to open temporary checkpoint {}", tmp_path.string()));
        }
        write_bytes(checkpoint, &header, sizeof(header));
        write_vector(checkpoint, fluid.raw_buffer(fluid.current_buffer_index()));
        write_vector(checkpoint, species_a.raw_buffer(species_a.current_buffer_index()));
        write_vector(checkpoint, species_b.raw_buffer(species_b.current_buffer_index()));
        checkpoint.flush();
        if (!checkpoint) {
            throw std::runtime_error("failed to flush checkpoint file");
        }
    }

    std::filesystem::rename(tmp_path, final_path);
    const auto checkpoint_stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> checkpoint_elapsed =
        checkpoint_stop - checkpoint_start;

    std::vector<std::pair<int, std::filesystem::path>> checkpoints;
    if (std::filesystem::exists(checkpoint_directory())) {
        for (const auto& entry : std::filesystem::directory_iterator(checkpoint_directory())) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string filename = entry.path().filename().string();
            constexpr std::string_view prefix{"checkpoint_"};
            constexpr std::string_view suffix{".bin"};
            if (!filename.starts_with(prefix) || !filename.ends_with(suffix)) {
                continue;
            }
            const std::string step_text = filename.substr(
                prefix.size(),
                filename.size() - prefix.size() - suffix.size());
            try {
                checkpoints.emplace_back(std::stoi(step_text), entry.path());
            } catch (const std::exception&) {
            }
        }
    }
    std::ranges::sort(checkpoints, {}, &std::pair<int, std::filesystem::path>::first);
    while (checkpoints.size() > static_cast<std::size_t>(config.checkpoint_keep)) {
        std::filesystem::remove(checkpoints.front().second);
        checkpoints.erase(checkpoints.begin());
    }

    std::cout << "Checkpoint written: " << final_path.string()
              << " (" << std::filesystem::file_size(final_path)
              << " bytes, " << checkpoint_elapsed.count() << " s)\n"
              << std::flush;
    return {.path = final_path, .elapsed_seconds = checkpoint_elapsed.count()};
}

[[nodiscard]] RestartState read_checkpoint(
    const Config& config,
    PerturbationDefinition& perturbation,
    lbm::LatticeMemory<FluidLattice, Real>& fluid,
    lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    lbm::LatticeMemory<ScalarLattice, Real>& species_b) {
    if (std::filesystem::path{config.restart_from}.extension() == ".tmp") {
        throw std::runtime_error("refusing to restart from a temporary checkpoint file");
    }
    const auto checkpoint_start = std::chrono::high_resolution_clock::now();
    std::ifstream checkpoint{config.restart_from, std::ios::binary};
    if (!checkpoint) {
        throw std::runtime_error(
            std::format("failed to open restart checkpoint {}", config.restart_from));
    }

    CheckpointHeader header{};
    read_bytes(checkpoint, &header, sizeof(header));
    validate_checkpoint_header(config, header);
    restore_perturbation_diagnostics(perturbation, header);

    fluid.set_current_buffer_index(static_cast<int>(header.fluid_current_buffer));
    species_a.set_current_buffer_index(static_cast<int>(header.species_a_current_buffer));
    species_b.set_current_buffer_index(static_cast<int>(header.species_b_current_buffer));

    if (header.version == checkpoint_legacy_full_ping_pong_format_version) {
        read_vector(checkpoint, fluid.raw_buffer(0));
        read_vector(checkpoint, fluid.raw_buffer(1));
        read_vector(checkpoint, species_a.raw_buffer(0));
        read_vector(checkpoint, species_a.raw_buffer(1));
        read_vector(checkpoint, species_b.raw_buffer(0));
        read_vector(checkpoint, species_b.raw_buffer(1));
    } else {
        read_vector(checkpoint, fluid.raw_buffer(fluid.current_buffer_index()));
        read_vector(checkpoint, species_a.raw_buffer(species_a.current_buffer_index()));
        read_vector(checkpoint, species_b.raw_buffer(species_b.current_buffer_index()));
    }

    char trailing_byte{};
    checkpoint.read(&trailing_byte, 1);
    if (checkpoint.gcount() != 0) {
        throw std::runtime_error("checkpoint contains unexpected trailing bytes");
    }

    RestartState state{};
    state.enabled = true;
    state.step = static_cast<int>(header.step);
    state.kinetic_energy_previous = header.kinetic_energy_previous;
    state.previous_kinetic_energy_step =
        static_cast<int>(header.previous_kinetic_energy_step);
    state.latest_kinetic_energy_decay_rate = header.latest_kinetic_energy_decay_rate;
    state.previous_var_z = header.previous_var_z;
    state.previous_statistics_step =
        static_cast<int>(header.previous_statistics_step);
    state.initial_mean_z = header.initial_mean_z;
    state.initial_mean_rho = header.initial_mean_rho;
    state.source_path = config.restart_from;

    const std::uint64_t checksum =
        header.version == checkpoint_legacy_full_ping_pong_format_version
            ? checkpoint_payload_checksum_full_ping_pong(fluid, species_a, species_b, state)
            : checkpoint_payload_checksum_compact(fluid, species_a, species_b, state);
    if (checksum != header.payload_checksum) {
        throw std::runtime_error("checkpoint payload checksum mismatch");
    }
    const auto checkpoint_stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> checkpoint_elapsed =
        checkpoint_stop - checkpoint_start;
    std::cout << "Checkpoint read: " << config.restart_from
              << " (" << std::filesystem::file_size(config.restart_from)
              << " bytes, " << checkpoint_elapsed.count() << " s)\n"
              << std::flush;
    return state;
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

[[nodiscard]] std::size_t clamped_unit_bin(Real value, int bins) {
    if (!(value > Real{})) {
        return 0;
    }
    if (!(value < Real{1})) {
        return static_cast<std::size_t>(bins - 1);
    }

    return std::min(
        static_cast<std::size_t>(value * static_cast<Real>(bins)),
        static_cast<std::size_t>(bins - 1));
}

[[nodiscard]] bool log_bin(
    Real value,
    Real log_min,
    Real log_max,
    int bins,
    std::size_t& bin) {
    if (value <= Real{} || !std::isfinite(value)) {
        return false;
    }

    const Real log_value = std::log10(value);
    if (log_value < log_min || log_value > log_max) {
        return false;
    }

    const Real fraction = (log_value - log_min) / (log_max - log_min);
    bin = std::min(
        static_cast<std::size_t>(fraction * static_cast<Real>(bins)),
        static_cast<std::size_t>(bins - 1));
    return true;
}

void write_histogram_csv(
    const std::filesystem::path& filename,
    std::string_view variable,
    Real range_min,
    Real range_max,
    const std::vector<std::uint64_t>& histogram,
    std::uint64_t total_samples) {
    std::ofstream file{filename};
    if (!file) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    file << "variable,bin,bin_min,bin_max,bin_center,count,probability\n";
    const Real width =
        (range_max - range_min) / static_cast<Real>(histogram.size());
    for (std::size_t bin = 0; bin < histogram.size(); ++bin) {
        const Real bin_min = range_min + static_cast<Real>(bin) * width;
        const Real bin_max = bin_min + width;
        const Real bin_center = Real{0.5} * (bin_min + bin_max);
        const Real probability =
            static_cast<Real>(histogram[bin]) / static_cast<Real>(total_samples);
        file << variable << ',' << bin
             << ',' << std::format("{:.17g}", static_cast<double>(bin_min))
             << ',' << std::format("{:.17g}", static_cast<double>(bin_max))
             << ',' << std::format("{:.17g}", static_cast<double>(bin_center))
             << ',' << histogram[bin]
             << ',' << std::format("{:.17g}", static_cast<double>(probability))
             << '\n';
    }
}

void write_joint_histogram_csv(
    const std::filesystem::path& filename,
    std::string_view x_name,
    std::string_view y_name,
    Real x_min,
    Real x_max,
    Real y_min,
    Real y_max,
    int bins,
    const std::vector<std::uint64_t>& histogram,
    std::uint64_t total_samples) {
    std::ofstream file{filename};
    if (!file) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    file << x_name << "_center," << y_name
         << "_center,count,probability\n";
    const Real x_width = (x_max - x_min) / static_cast<Real>(bins);
    const Real y_width = (y_max - y_min) / static_cast<Real>(bins);
    for (int y_bin = 0; y_bin < bins; ++y_bin) {
        const Real y_center =
            y_min + (static_cast<Real>(y_bin) + Real{0.5}) * y_width;
        for (int x_bin = 0; x_bin < bins; ++x_bin) {
            const Real x_center =
                x_min + (static_cast<Real>(x_bin) + Real{0.5}) * x_width;
            const std::size_t index =
                static_cast<std::size_t>(y_bin * bins + x_bin);
            const Real probability =
                static_cast<Real>(histogram[index]) /
                static_cast<Real>(total_samples);
            file << std::format("{:.17g}", static_cast<double>(x_center))
                 << ',' << std::format("{:.17g}", static_cast<double>(y_center))
                 << ',' << histogram[index]
                 << ',' << std::format("{:.17g}", static_cast<double>(probability))
                 << '\n';
        }
    }
}

void write_pdf_metadata_json(
    const Config& config,
    const std::filesystem::path& filename,
    int step,
    const PdfCounters& counters,
    const std::vector<std::uint64_t>& pdf_z,
    const std::vector<std::uint64_t>& pdf_log_chi,
    const std::vector<std::uint64_t>& pdf_log_r,
    const std::vector<std::uint64_t>& joint_ab,
    const std::vector<std::uint64_t>& joint_z_chi,
    const std::vector<std::uint64_t>& joint_r_chi) {
    const auto sum_histogram = [](const std::vector<std::uint64_t>& histogram) {
        std::uint64_t sum{};
        for (std::uint64_t count : histogram) {
            sum += count;
        }
        return sum;
    };
    const auto fraction = [](std::uint64_t count, std::uint64_t total) {
        return total > 0 ? static_cast<Real>(count) / static_cast<Real>(total)
                         : Real{};
    };

    const std::uint64_t total = static_cast<std::uint64_t>(cell_count(config));
    std::ofstream metadata{filename};
    if (!metadata) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    metadata
        << "{\n"
        << "  \"step\": " << step << ",\n"
        << "  \"time\": " << json_number(static_cast<Real>(step)) << ",\n"
        << "  \"total_samples\": " << total << ",\n"
        << "  \"normalization\": \"probability = count / total_domain_cells\",\n"
        << "  \"pdf_bins\": " << config.pdf_bins << ",\n"
        << "  \"joint_pdf_bins\": " << config.joint_pdf_bins << ",\n"
        << "  \"Z_range\": [0, 1],\n"
        << "  \"a_range\": [0, 1],\n"
        << "  \"b_range\": [0, 1],\n"
        << "  \"log10_chi_star_range\": ["
        << json_number(config.pdf_log_chi_min) << ", "
        << json_number(config.pdf_log_chi_max) << "],\n"
        << "  \"log10_R_star_range\": ["
        << json_number(config.pdf_log_r_min) << ", "
        << json_number(config.pdf_log_r_max) << "],\n"
        << "  \"Z_underflow_fraction\": "
        << json_number(fraction(counters.z_underflow, total)) << ",\n"
        << "  \"Z_overflow_fraction\": "
        << json_number(fraction(counters.z_overflow, total)) << ",\n"
        << "  \"Z_included_probability\": "
        << json_number(fraction(sum_histogram(pdf_z), total)) << ",\n"
        << "  \"chi_zero_fraction\": "
        << json_number(fraction(counters.chi_zero, total)) << ",\n"
        << "  \"chi_underflow_fraction\": "
        << json_number(fraction(counters.chi_underflow, total)) << ",\n"
        << "  \"chi_overflow_fraction\": "
        << json_number(fraction(counters.chi_overflow, total)) << ",\n"
        << "  \"chi_included_probability\": "
        << json_number(fraction(counters.chi_included, total)) << ",\n"
        << "  \"R_zero_fraction\": "
        << json_number(fraction(counters.r_zero, total)) << ",\n"
        << "  \"R_underflow_fraction\": "
        << json_number(fraction(counters.r_underflow, total)) << ",\n"
        << "  \"R_overflow_fraction\": "
        << json_number(fraction(counters.r_overflow, total)) << ",\n"
        << "  \"R_included_probability\": "
        << json_number(fraction(counters.r_included, total)) << ",\n"
        << "  \"joint_Ca_Cb_included_probability\": "
        << json_number(fraction(sum_histogram(joint_ab), total)) << ",\n"
        << "  \"joint_Z_log_chi_included_probability\": "
        << json_number(fraction(sum_histogram(joint_z_chi), total)) << ",\n"
        << "  \"joint_Z_log_chi_excluded_fraction\": "
        << json_number(fraction(counters.joint_z_chi_excluded, total)) << ",\n"
        << "  \"joint_log_R_log_chi_included_probability\": "
        << json_number(fraction(counters.joint_r_chi_included, total)) << ",\n"
        << "  \"joint_log_R_log_chi_excluded_fraction\": "
        << json_number(fraction(counters.joint_r_chi_excluded, total)) << ",\n"
        << "  \"dimensionless_variables\": {\n"
        << "    \"Z\": \"0.5 * (1 + (C_A - C_B) / C0)\",\n"
        << "    \"a\": \"C_A / C0\",\n"
        << "    \"b\": \"C_B / C0\",\n"
        << "    \"chi_star\": \"chi_Z * delta0 / DeltaU\",\n"
        << "    \"R_star\": \"k_react * C_A * C_B * delta0 / (C0 * DeltaU)\"\n"
        << "  },\n"
        << "  \"histogram_counts\": {\n"
        << "    \"pdf_Z\": " << sum_histogram(pdf_z) << ",\n"
        << "    \"pdf_log_chi\": " << sum_histogram(pdf_log_chi) << ",\n"
        << "    \"pdf_log_R\": " << sum_histogram(pdf_log_r) << ",\n"
        << "    \"joint_pdf_Ca_Cb\": " << sum_histogram(joint_ab) << ",\n"
        << "    \"joint_pdf_Z_log_chi\": " << sum_histogram(joint_z_chi) << ",\n"
        << "    \"joint_pdf_log_R_log_chi\": " << sum_histogram(joint_r_chi) << "\n"
        << "  },\n"
        << "  \"conditional_statistics\": {\n"
        << "    \"enabled\": true,\n"
        << "    \"conditional_Z_bins\": " << config.pdf_bins << ",\n"
        << "    \"conditional_Z_range\": [0, 1],\n"
        << "    \"conditional_log_chi_bins\": " << config.pdf_bins << ",\n"
        << "    \"conditional_log10_chi_star_range\": ["
        << json_number(config.pdf_log_chi_min) << ", "
        << json_number(config.pdf_log_chi_max) << "]\n"
        << "  }\n"
        << "}\n";
}

void write_conditional_chi_given_z_csv(
    const std::filesystem::path& filename,
    const std::vector<std::uint64_t>& counts,
    const std::vector<long double>& sum_chi_z,
    const std::vector<long double>& sum_chi_star,
    std::uint64_t total_samples) {
    std::ofstream file{filename};
    if (!file) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    file << "Z_center,count,probability,mean_chi_Z_given_Z,"
            "mean_chi_star_given_Z\n";
    const Real width = Real{1} / static_cast<Real>(counts.size());
    for (std::size_t bin = 0; bin < counts.size(); ++bin) {
        const Real center = (static_cast<Real>(bin) + Real{0.5}) * width;
        const Real probability =
            static_cast<Real>(counts[bin]) / static_cast<Real>(total_samples);
        const Real mean_chi_z =
            counts[bin] > 0
                ? static_cast<Real>(sum_chi_z[bin] / static_cast<long double>(counts[bin]))
                : Real{};
        const Real mean_chi_star =
            counts[bin] > 0
                ? static_cast<Real>(sum_chi_star[bin] / static_cast<long double>(counts[bin]))
                : Real{};

        file << std::format("{:.17g}", static_cast<double>(center))
             << ',' << counts[bin]
             << ',' << std::format("{:.17g}", static_cast<double>(probability))
             << ',' << std::format("{:.17g}", static_cast<double>(mean_chi_z))
             << ',' << std::format("{:.17g}", static_cast<double>(mean_chi_star))
             << '\n';
    }
}

void write_conditional_r_given_z_csv(
    const std::filesystem::path& filename,
    const std::vector<std::uint64_t>& counts,
    const std::vector<long double>& sum_reaction_rate,
    const std::vector<long double>& sum_reaction_rate_star,
    const std::vector<long double>& sum_ca,
    const std::vector<long double>& sum_cb,
    const std::vector<long double>& sum_cacb,
    std::uint64_t total_samples) {
    std::ofstream file{filename};
    if (!file) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    file << "Z_center,count,probability,mean_R_given_Z,mean_R_star_given_Z,"
            "reaction_efficiency_given_Z\n";
    const Real width = Real{1} / static_cast<Real>(counts.size());
    for (std::size_t bin = 0; bin < counts.size(); ++bin) {
        const Real center = (static_cast<Real>(bin) + Real{0.5}) * width;
        const Real probability =
            static_cast<Real>(counts[bin]) / static_cast<Real>(total_samples);
        const long double inv_count =
            counts[bin] > 0 ? 1.0L / static_cast<long double>(counts[bin]) : 0.0L;
        const Real mean_r =
            counts[bin] > 0 ? static_cast<Real>(sum_reaction_rate[bin] * inv_count) : Real{};
        const Real mean_r_star =
            counts[bin] > 0
                ? static_cast<Real>(sum_reaction_rate_star[bin] * inv_count)
                : Real{};
        const Real mean_a =
            counts[bin] > 0 ? static_cast<Real>(sum_ca[bin] * inv_count) : Real{};
        const Real mean_b =
            counts[bin] > 0 ? static_cast<Real>(sum_cb[bin] * inv_count) : Real{};
        const Real mean_ab =
            counts[bin] > 0 ? static_cast<Real>(sum_cacb[bin] * inv_count) : Real{};
        const Real efficiency =
            mean_a * mean_b > Real{} ? mean_ab / (mean_a * mean_b) : Real{};

        file << std::format("{:.17g}", static_cast<double>(center))
             << ',' << counts[bin]
             << ',' << std::format("{:.17g}", static_cast<double>(probability))
             << ',' << std::format("{:.17g}", static_cast<double>(mean_r))
             << ',' << std::format("{:.17g}", static_cast<double>(mean_r_star))
             << ',' << std::format("{:.17g}", static_cast<double>(efficiency))
             << '\n';
    }
}

void write_conditional_r_given_log_chi_csv(
    const Config& config,
    const std::filesystem::path& filename,
    const std::vector<std::uint64_t>& counts,
    const std::vector<long double>& sum_reaction_rate,
    const std::vector<long double>& sum_reaction_rate_star,
    std::uint64_t total_samples) {
    std::ofstream file{filename};
    if (!file) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    file << "log10_chi_center,chi_star_center,count,probability,mean_R,"
            "mean_R_star\n";
    const Real width =
        (config.pdf_log_chi_max - config.pdf_log_chi_min) /
        static_cast<Real>(counts.size());
    for (std::size_t bin = 0; bin < counts.size(); ++bin) {
        const Real log_center =
            config.pdf_log_chi_min + (static_cast<Real>(bin) + Real{0.5}) * width;
        const Real chi_star_center = std::pow(Real{10}, log_center);
        const Real probability =
            static_cast<Real>(counts[bin]) / static_cast<Real>(total_samples);
        const long double inv_count =
            counts[bin] > 0 ? 1.0L / static_cast<long double>(counts[bin]) : 0.0L;
        const Real mean_r =
            counts[bin] > 0 ? static_cast<Real>(sum_reaction_rate[bin] * inv_count) : Real{};
        const Real mean_r_star =
            counts[bin] > 0
                ? static_cast<Real>(sum_reaction_rate_star[bin] * inv_count)
                : Real{};

        file << std::format("{:.17g}", static_cast<double>(log_center))
             << ',' << std::format("{:.17g}", static_cast<double>(chi_star_center))
             << ',' << counts[bin]
             << ',' << std::format("{:.17g}", static_cast<double>(probability))
             << ',' << std::format("{:.17g}", static_cast<double>(mean_r))
             << ',' << std::format("{:.17g}", static_cast<double>(mean_r_star))
             << '\n';
    }
}

void write_pdf_outputs(
    const Config& config,
    const std::filesystem::path& output_root,
    int step,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b) {
    const std::filesystem::path output_dir =
        output_root / std::format("step_{:08}", step);
    std::filesystem::create_directories(output_dir);

    const auto a_view = species_a.get_current_view();
    const auto b_view = species_b.get_current_view();
    const std::size_t pdf_bins = static_cast<std::size_t>(config.pdf_bins);
    const std::size_t joint_bins = static_cast<std::size_t>(config.joint_pdf_bins);
    std::vector<std::uint64_t> pdf_z(pdf_bins);
    std::vector<std::uint64_t> pdf_log_chi(pdf_bins);
    std::vector<std::uint64_t> pdf_log_r(pdf_bins);
    std::vector<std::uint64_t> joint_ab(joint_bins * joint_bins);
    std::vector<std::uint64_t> joint_z_chi(joint_bins * joint_bins);
    std::vector<std::uint64_t> joint_r_chi(joint_bins * joint_bins);
    std::vector<std::uint64_t> conditional_z_counts(pdf_bins);
    std::vector<long double> conditional_z_sum_chi(pdf_bins);
    std::vector<long double> conditional_z_sum_chi_star(pdf_bins);
    std::vector<long double> conditional_z_sum_r(pdf_bins);
    std::vector<long double> conditional_z_sum_r_star(pdf_bins);
    std::vector<long double> conditional_z_sum_ca(pdf_bins);
    std::vector<long double> conditional_z_sum_cb(pdf_bins);
    std::vector<long double> conditional_z_sum_cacb(pdf_bins);
    std::vector<std::uint64_t> conditional_log_chi_counts(pdf_bins);
    std::vector<long double> conditional_log_chi_sum_r(pdf_bins);
    std::vector<long double> conditional_log_chi_sum_r_star(pdf_bins);
    PdfCounters counters{};

#pragma omp parallel
    {
        std::vector<std::uint64_t> local_pdf_z(pdf_bins);
        std::vector<std::uint64_t> local_pdf_log_chi(pdf_bins);
        std::vector<std::uint64_t> local_pdf_log_r(pdf_bins);
        std::vector<std::uint64_t> local_joint_ab(joint_bins * joint_bins);
        std::vector<std::uint64_t> local_joint_z_chi(joint_bins * joint_bins);
        std::vector<std::uint64_t> local_joint_r_chi(joint_bins * joint_bins);
        std::vector<std::uint64_t> local_conditional_z_counts(pdf_bins);
        std::vector<long double> local_conditional_z_sum_chi(pdf_bins);
        std::vector<long double> local_conditional_z_sum_chi_star(pdf_bins);
        std::vector<long double> local_conditional_z_sum_r(pdf_bins);
        std::vector<long double> local_conditional_z_sum_r_star(pdf_bins);
        std::vector<long double> local_conditional_z_sum_ca(pdf_bins);
        std::vector<long double> local_conditional_z_sum_cb(pdf_bins);
        std::vector<long double> local_conditional_z_sum_cacb(pdf_bins);
        std::vector<std::uint64_t> local_conditional_log_chi_counts(pdf_bins);
        std::vector<long double> local_conditional_log_chi_sum_r(pdf_bins);
        std::vector<long double> local_conditional_log_chi_sum_r_star(pdf_bins);
        PdfCounters local{};

#pragma omp for collapse(3) schedule(static) nowait
        for (std::size_t z = 0; z < config.nz; ++z) {
            for (std::size_t y = 0; y < config.ny; ++y) {
                for (std::size_t x = 0; x < config.nx; ++x) {
                    const Real concentration_a = concentration_at(a_view, x, y, z);
                    const Real concentration_b = concentration_at(b_view, x, y, z);
                    const Real a = concentration_a / config.c0;
                    const Real b = concentration_b / config.c0;
                    const Real mixture_fraction =
                        Real{0.5} * (Real{1} + a - b);
                    const Real reaction_rate =
                        config.k_react * concentration_a * concentration_b;
                    const Real reaction_rate_star =
                        reaction_rate * config.delta0 / (config.c0 * config.delta_u);

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
                    const Real grad_z2 =
                        dz_dx * dz_dx + dz_dy * dz_dy + dz_dz * dz_dz;
                    const Real chi_z =
                        Real{2} * config.scalar_diffusivity * grad_z2;
                    const Real chi_star =
                        chi_z * config.delta0 / config.delta_u;

                    if (mixture_fraction < Real{}) {
                        ++local.z_underflow;
                    } else if (mixture_fraction > Real{1}) {
                        ++local.z_overflow;
                    }
                    const std::size_t z_pdf_bin =
                        clamped_unit_bin(mixture_fraction, config.pdf_bins);
                    ++local_pdf_z[z_pdf_bin];
                    ++local_conditional_z_counts[z_pdf_bin];
                    local_conditional_z_sum_chi[z_pdf_bin] +=
                        static_cast<long double>(chi_z);
                    local_conditional_z_sum_chi_star[z_pdf_bin] +=
                        static_cast<long double>(chi_star);
                    local_conditional_z_sum_r[z_pdf_bin] +=
                        static_cast<long double>(reaction_rate);
                    local_conditional_z_sum_r_star[z_pdf_bin] +=
                        static_cast<long double>(reaction_rate_star);
                    local_conditional_z_sum_ca[z_pdf_bin] +=
                        static_cast<long double>(concentration_a);
                    local_conditional_z_sum_cb[z_pdf_bin] +=
                        static_cast<long double>(concentration_b);
                    local_conditional_z_sum_cacb[z_pdf_bin] +=
                        static_cast<long double>(concentration_a * concentration_b);

                    const std::size_t a_bin =
                        clamped_unit_bin(a, config.joint_pdf_bins);
                    const std::size_t b_bin =
                        clamped_unit_bin(b, config.joint_pdf_bins);
                    ++local_joint_ab[b_bin * joint_bins + a_bin];

                    std::size_t chi_pdf_bin{};
                    const bool chi_included = log_bin(
                        chi_star,
                        config.pdf_log_chi_min,
                        config.pdf_log_chi_max,
                        config.pdf_bins,
                        chi_pdf_bin);
                    if (chi_included) {
                        ++local.chi_included;
                        ++local_pdf_log_chi[chi_pdf_bin];
                        ++local_conditional_log_chi_counts[chi_pdf_bin];
                        local_conditional_log_chi_sum_r[chi_pdf_bin] +=
                            static_cast<long double>(reaction_rate);
                        local_conditional_log_chi_sum_r_star[chi_pdf_bin] +=
                            static_cast<long double>(reaction_rate_star);
                    } else if (chi_star <= Real{} || !std::isfinite(chi_star)) {
                        ++local.chi_zero;
                    } else if (std::log10(chi_star) < config.pdf_log_chi_min) {
                        ++local.chi_underflow;
                    } else {
                        ++local.chi_overflow;
                    }

                    std::size_t r_pdf_bin{};
                    const bool r_included = log_bin(
                        reaction_rate_star,
                        config.pdf_log_r_min,
                        config.pdf_log_r_max,
                        config.pdf_bins,
                        r_pdf_bin);
                    if (r_included) {
                        ++local.r_included;
                        ++local_pdf_log_r[r_pdf_bin];
                    } else if (reaction_rate_star <= Real{} ||
                               !std::isfinite(reaction_rate_star)) {
                        ++local.r_zero;
                    } else if (std::log10(reaction_rate_star) < config.pdf_log_r_min) {
                        ++local.r_underflow;
                    } else {
                        ++local.r_overflow;
                    }

                    std::size_t chi_joint_bin{};
                    const bool chi_joint_included = log_bin(
                        chi_star,
                        config.pdf_log_chi_min,
                        config.pdf_log_chi_max,
                        config.joint_pdf_bins,
                        chi_joint_bin);
                    if (chi_joint_included) {
                        const std::size_t z_joint_bin =
                            clamped_unit_bin(mixture_fraction, config.joint_pdf_bins);
                        ++local_joint_z_chi[chi_joint_bin * joint_bins + z_joint_bin];
                    } else {
                        ++local.joint_z_chi_excluded;
                    }

                    std::size_t r_joint_bin{};
                    const bool r_joint_included = log_bin(
                        reaction_rate_star,
                        config.pdf_log_r_min,
                        config.pdf_log_r_max,
                        config.joint_pdf_bins,
                        r_joint_bin);
                    if (r_joint_included && chi_joint_included) {
                        ++local.joint_r_chi_included;
                        ++local_joint_r_chi[chi_joint_bin * joint_bins + r_joint_bin];
                    } else {
                        ++local.joint_r_chi_excluded;
                    }
                }
            }
        }

#pragma omp critical
        {
            const auto merge_histogram =
                [](std::vector<std::uint64_t>& target,
                   const std::vector<std::uint64_t>& source) {
                    for (std::size_t i = 0; i < target.size(); ++i) {
                        target[i] += source[i];
                    }
                };
            const auto merge_sums =
                [](std::vector<long double>& target,
                   const std::vector<long double>& source) {
                    for (std::size_t i = 0; i < target.size(); ++i) {
                        target[i] += source[i];
                    }
                };
            merge_histogram(pdf_z, local_pdf_z);
            merge_histogram(pdf_log_chi, local_pdf_log_chi);
            merge_histogram(pdf_log_r, local_pdf_log_r);
            merge_histogram(joint_ab, local_joint_ab);
            merge_histogram(joint_z_chi, local_joint_z_chi);
            merge_histogram(joint_r_chi, local_joint_r_chi);
            merge_histogram(conditional_z_counts, local_conditional_z_counts);
            merge_sums(conditional_z_sum_chi, local_conditional_z_sum_chi);
            merge_sums(conditional_z_sum_chi_star, local_conditional_z_sum_chi_star);
            merge_sums(conditional_z_sum_r, local_conditional_z_sum_r);
            merge_sums(conditional_z_sum_r_star, local_conditional_z_sum_r_star);
            merge_sums(conditional_z_sum_ca, local_conditional_z_sum_ca);
            merge_sums(conditional_z_sum_cb, local_conditional_z_sum_cb);
            merge_sums(conditional_z_sum_cacb, local_conditional_z_sum_cacb);
            merge_histogram(
                conditional_log_chi_counts,
                local_conditional_log_chi_counts);
            merge_sums(
                conditional_log_chi_sum_r,
                local_conditional_log_chi_sum_r);
            merge_sums(
                conditional_log_chi_sum_r_star,
                local_conditional_log_chi_sum_r_star);
            counters.z_underflow += local.z_underflow;
            counters.z_overflow += local.z_overflow;
            counters.chi_zero += local.chi_zero;
            counters.chi_underflow += local.chi_underflow;
            counters.chi_overflow += local.chi_overflow;
            counters.chi_included += local.chi_included;
            counters.r_zero += local.r_zero;
            counters.r_underflow += local.r_underflow;
            counters.r_overflow += local.r_overflow;
            counters.r_included += local.r_included;
            counters.joint_z_chi_excluded += local.joint_z_chi_excluded;
            counters.joint_r_chi_excluded += local.joint_r_chi_excluded;
            counters.joint_r_chi_included += local.joint_r_chi_included;
        }
    }

    const std::uint64_t total_samples = static_cast<std::uint64_t>(cell_count(config));
    write_histogram_csv(
        output_dir / "pdf_Z.csv",
        "Z",
        Real{},
        Real{1},
        pdf_z,
        total_samples);
    write_histogram_csv(
        output_dir / "pdf_log_chi.csv",
        "log10_chi_star",
        config.pdf_log_chi_min,
        config.pdf_log_chi_max,
        pdf_log_chi,
        total_samples);
    write_histogram_csv(
        output_dir / "pdf_log_R.csv",
        "log10_R_star",
        config.pdf_log_r_min,
        config.pdf_log_r_max,
        pdf_log_r,
        total_samples);
    write_joint_histogram_csv(
        output_dir / "joint_pdf_Ca_Cb.csv",
        "a",
        "b",
        Real{},
        Real{1},
        Real{},
        Real{1},
        config.joint_pdf_bins,
        joint_ab,
        total_samples);
    write_joint_histogram_csv(
        output_dir / "joint_pdf_Z_log_chi.csv",
        "Z",
        "log10_chi_star",
        Real{},
        Real{1},
        config.pdf_log_chi_min,
        config.pdf_log_chi_max,
        config.joint_pdf_bins,
        joint_z_chi,
        total_samples);
    write_joint_histogram_csv(
        output_dir / "joint_pdf_log_R_log_chi.csv",
        "log10_R_star",
        "log10_chi_star",
        config.pdf_log_r_min,
        config.pdf_log_r_max,
        config.pdf_log_chi_min,
        config.pdf_log_chi_max,
        config.joint_pdf_bins,
        joint_r_chi,
        total_samples);
    write_conditional_chi_given_z_csv(
        output_dir / "conditional_chi_given_Z.csv",
        conditional_z_counts,
        conditional_z_sum_chi,
        conditional_z_sum_chi_star,
        total_samples);
    write_conditional_r_given_z_csv(
        output_dir / "conditional_R_given_Z.csv",
        conditional_z_counts,
        conditional_z_sum_r,
        conditional_z_sum_r_star,
        conditional_z_sum_ca,
        conditional_z_sum_cb,
        conditional_z_sum_cacb,
        total_samples);
    write_conditional_r_given_log_chi_csv(
        config,
        output_dir / "conditional_R_given_log_chi.csv",
        conditional_log_chi_counts,
        conditional_log_chi_sum_r,
        conditional_log_chi_sum_r_star,
        total_samples);
    write_pdf_metadata_json(
        config,
        output_dir / "pdf_metadata.json",
        step,
        counters,
        pdf_z,
        pdf_log_chi,
        pdf_log_r,
        joint_ab,
        joint_z_chi,
        joint_r_chi);
}

void write_filter_statistics_header(std::ofstream& file) {
    file << "step,time,filter_width,Delta_over_delta0,Delta_over_etaK,"
         << "Delta_over_etaB,mean_A,mean_Abar,mean_B,mean_Bbar,mean_AB,"
         << "mean_ABbar,mean_R,mean_Rbar_direct,mean_kABbar,"
         << "max_abs_Rbar_identity_error,relative_Rbar_identity_error,"
         << "mean_AbarBbar,mean_tau_AB,rms_tau_AB,min_tau_AB,max_tau_AB,"
         << "mean_tau_AA,rms_tau_AA,min_tau_AA,max_tau_AA,"
         << "mean_tau_BB,rms_tau_BB,min_tau_BB,max_tau_BB,"
         << "mean_tau_ZZ,rms_tau_ZZ,min_tau_ZZ,max_tau_ZZ,"
         << "mean_rho_AB_SGS,rms_rho_AB_SGS,min_rho_AB_SGS,max_rho_AB_SGS,"
         << "max_abs_tauZZ_identity_error,rms_tauZZ_identity_error,"
         << "relative_tauZZ_identity_error,max_abs_Zbar_linearity_error,"
         << "relative_Zbar_linearity_error,"
         << "mean_R_exact_filtered,mean_R_LES_naive,mean_R_SGS,"
         << "LES_reaction_efficiency,LES_reaction_overprediction_factor,"
         << "SGS_reaction_fraction,SGS_reaction_magnitude_fraction,"
         << "max_abs_reaction_decomposition_error,"
         << "relative_reaction_decomposition_error,covariance_identity_error,"
         << "LES_efficiency_identity_error"
         << std::endl;
    file.flush();
}

[[nodiscard]] lbm::ScalarField3DView<Real> scalar_field_view(
    std::vector<Real>& field,
    const Config& config) {
    return lbm::ScalarField3DView<Real>{
        field.data(),
        config.nz,
        config.ny,
        config.nx};
}

[[nodiscard]] lbm::ConstScalarField3DView<Real> scalar_field_view(
    const std::vector<Real>& field,
    const Config& config) {
    return lbm::ConstScalarField3DView<Real>{
        field.data(),
        config.nz,
        config.ny,
        config.nx};
}

void write_filter_conditional_outputs(
    const Config& config,
    const std::filesystem::path& output_root,
    int step,
    int filter_width_int,
    const ResolutionDiagnostics& resolution,
    Real mean_tau_ab,
    const std::vector<std::uint64_t>& zbar_counts,
    const std::vector<long double>& zbar_sum_tau_ab,
    const std::vector<long double>& zbar_sum_rho_ab_sgs,
    const std::vector<std::uint64_t>& log_tau_zz_counts,
    const std::vector<long double>& log_tau_zz_sum_tau_ab,
    std::uint64_t tau_zz_zero_count,
    std::uint64_t tau_zz_underflow_count,
    std::uint64_t tau_zz_overflow_count,
    long double excluded_log_tau_zz_sum_tau_ab) {
    const std::size_t cells = cell_count(config);
    const long double inv_cells = 1.0L / static_cast<long double>(cells);
    const int bins = config.filter_conditional_bins;
    const Real log_min = config.filter_log_tau_zz_min;
    const Real log_max = config.filter_log_tau_zz_max;
    const Real log_bin_width = (log_max - log_min) / static_cast<Real>(bins);
    const Real filter_delta = static_cast<Real>(filter_width_int);
    const Real delta_over_eta_k =
        std::isfinite(resolution.eta_k) && resolution.eta_k > Real{}
            ? filter_delta / resolution.eta_k
            : Real{};
    const Real delta_over_eta_b =
        std::isfinite(resolution.eta_b) && resolution.eta_b > Real{}
            ? filter_delta / resolution.eta_b
            : Real{};

    std::uint64_t zbar_total_count{};
    std::uint64_t log_tau_zz_included_count{};
    long double reconstructed_tau_ab_from_zbar{};
    long double reconstructed_tau_ab_from_log_tau_zz_included{};
    for (int bin = 0; bin < bins; ++bin) {
        zbar_total_count += zbar_counts[bin];
        log_tau_zz_included_count += log_tau_zz_counts[bin];
        reconstructed_tau_ab_from_zbar += zbar_sum_tau_ab[bin];
        reconstructed_tau_ab_from_log_tau_zz_included +=
            log_tau_zz_sum_tau_ab[bin];
    }
    if (zbar_total_count != cells) {
        throw std::runtime_error(
            "filtered conditional Zbar counts do not sum to the domain size");
    }
    const std::uint64_t tau_zz_classified_count =
        log_tau_zz_included_count + tau_zz_zero_count +
        tau_zz_underflow_count + tau_zz_overflow_count;
    if (tau_zz_classified_count != cells) {
        throw std::runtime_error(
            "filtered conditional tau_ZZ counts do not sum to the domain size");
    }

    reconstructed_tau_ab_from_zbar *= inv_cells;
    reconstructed_tau_ab_from_log_tau_zz_included *= inv_cells;
    const long double reconstructed_tau_ab_from_log_tau_zz =
        reconstructed_tau_ab_from_log_tau_zz_included +
        excluded_log_tau_zz_sum_tau_ab * inv_cells;
    const Real zbar_reconstruction_abs_error = std::abs(
        static_cast<Real>(reconstructed_tau_ab_from_zbar) - mean_tau_ab);
    const Real log_tau_zz_reconstruction_abs_error = std::abs(
        static_cast<Real>(reconstructed_tau_ab_from_log_tau_zz) - mean_tau_ab);

    const std::filesystem::path output_dir =
        output_root / std::format("step_{:08}", step);
    std::filesystem::create_directories(output_dir);
    const std::string width_label = std::format("w{:04}", filter_width_int);

    {
        std::ofstream file{
            output_dir /
            std::format("conditional_tauAB_given_Zbar_{}.csv", width_label)};
        if (!file) {
            throw std::runtime_error(
                "failed to open filtered Zbar conditional output");
        }
        file << "Zbar_center,count,probability,mean_tau_AB_given_Zbar,"
             << "mean_rho_AB_SGS_given_Zbar\n";
        for (int bin = 0; bin < bins; ++bin) {
            const std::uint64_t count = zbar_counts[bin];
            const Real probability = static_cast<Real>(
                static_cast<long double>(count) * inv_cells);
            const Real mean_tau =
                count > 0
                    ? static_cast<Real>(
                          zbar_sum_tau_ab[bin] / static_cast<long double>(count))
                    : Real{};
            const Real mean_rho =
                count > 0
                    ? static_cast<Real>(
                          zbar_sum_rho_ab_sgs[bin] /
                          static_cast<long double>(count))
                    : Real{};
            if (!std::isfinite(mean_tau) || !std::isfinite(mean_rho)) {
                throw std::runtime_error(
                    "non-finite value detected in filtered Zbar conditionals");
            }
            const Real center =
                (static_cast<Real>(bin) + Real{0.5}) / static_cast<Real>(bins);
            file << std::format("{:.17g}", static_cast<double>(center))
                 << ',' << count
                 << ',' << std::format("{:.17g}", static_cast<double>(probability))
                 << ',' << std::format("{:.17g}", static_cast<double>(mean_tau))
                 << ',' << std::format("{:.17g}", static_cast<double>(mean_rho))
                 << '\n';
        }
        file.flush();
    }

    {
        std::ofstream file{
            output_dir /
            std::format(
                "conditional_tauAB_given_log_tauZZ_{}.csv",
                width_label)};
        if (!file) {
            throw std::runtime_error(
                "failed to open filtered tau_ZZ conditional output");
        }
        file << "log10_tauZZ_center,tauZZ_center,count,probability,"
             << "mean_tau_AB_given_tauZZ\n";
        for (int bin = 0; bin < bins; ++bin) {
            const std::uint64_t count = log_tau_zz_counts[bin];
            const Real probability = static_cast<Real>(
                static_cast<long double>(count) * inv_cells);
            const Real mean_tau =
                count > 0
                    ? static_cast<Real>(
                          log_tau_zz_sum_tau_ab[bin] /
                          static_cast<long double>(count))
                    : Real{};
            if (!std::isfinite(mean_tau)) {
                throw std::runtime_error(
                    "non-finite value detected in filtered tau_ZZ conditionals");
            }
            const Real log_center =
                log_min + (static_cast<Real>(bin) + Real{0.5}) * log_bin_width;
            const Real tau_center = std::pow(Real{10}, log_center);
            file << std::format("{:.17g}", static_cast<double>(log_center))
                 << ',' << std::format("{:.17g}", static_cast<double>(tau_center))
                 << ',' << count
                 << ',' << std::format("{:.17g}", static_cast<double>(probability))
                 << ',' << std::format("{:.17g}", static_cast<double>(mean_tau))
                 << '\n';
        }
        file.flush();
    }

    {
        std::ofstream metadata{
            output_dir / std::format("conditional_metadata_{}.json", width_label)};
        if (!metadata) {
            throw std::runtime_error(
                "failed to open filtered conditional metadata output");
        }
        const Real tau_zz_zero_fraction = static_cast<Real>(
            static_cast<long double>(tau_zz_zero_count) * inv_cells);
        const Real tau_zz_underflow_fraction = static_cast<Real>(
            static_cast<long double>(tau_zz_underflow_count) * inv_cells);
        const Real tau_zz_overflow_fraction = static_cast<Real>(
            static_cast<long double>(tau_zz_overflow_count) * inv_cells);
        const Real tau_zz_included_fraction = static_cast<Real>(
            static_cast<long double>(log_tau_zz_included_count) * inv_cells);
        metadata
            << "{\n"
            << "  \"step\": " << step << ",\n"
            << "  \"time\": " << json_number(static_cast<Real>(step)) << ",\n"
            << "  \"filter_width\": " << filter_width_int << ",\n"
            << "  \"Delta_over_delta0\": "
            << json_number(filter_delta / config.delta0) << ",\n"
            << "  \"Delta_over_etaK\": " << json_number(delta_over_eta_k) << ",\n"
            << "  \"Delta_over_etaB\": " << json_number(delta_over_eta_b) << ",\n"
            << "  \"filter_conditional_bins\": " << bins << ",\n"
            << "  \"Zbar_range\": [0, 1],\n"
            << "  \"log10_tauZZ_range\": ["
            << json_number(log_min) << ", " << json_number(log_max) << "],\n"
            << "  \"tauZZ_zero_fraction\": "
            << json_number(tau_zz_zero_fraction) << ",\n"
            << "  \"tauZZ_underflow_fraction\": "
            << json_number(tau_zz_underflow_fraction) << ",\n"
            << "  \"tauZZ_overflow_fraction\": "
            << json_number(tau_zz_overflow_fraction) << ",\n"
            << "  \"tauZZ_included_fraction\": "
            << json_number(tau_zz_included_fraction) << ",\n"
            << "  \"mean_tau_AB\": " << json_number(mean_tau_ab) << ",\n"
            << "  \"reconstructed_mean_tau_AB_from_Zbar_conditionals\": "
            << json_number(static_cast<Real>(reconstructed_tau_ab_from_zbar))
            << ",\n"
            << "  \"Zbar_reconstruction_abs_error\": "
            << json_number(zbar_reconstruction_abs_error) << ",\n"
            << "  \"reconstructed_mean_tau_AB_from_log_tauZZ_conditionals\": "
            << json_number(
                   static_cast<Real>(reconstructed_tau_ab_from_log_tau_zz))
            << ",\n"
            << "  \"included_mean_tau_AB_from_log_tauZZ_conditionals\": "
            << json_number(static_cast<Real>(
                   reconstructed_tau_ab_from_log_tau_zz_included))
            << ",\n"
            << "  \"excluded_mean_tau_AB_from_log_tauZZ_conditionals\": "
            << json_number(
                   static_cast<Real>(excluded_log_tau_zz_sum_tau_ab * inv_cells))
            << ",\n"
            << "  \"log_tauZZ_reconstruction_abs_error\": "
            << json_number(log_tau_zz_reconstruction_abs_error) << "\n"
            << "}\n";
        metadata.flush();
    }
}

void write_filter_pdf_outputs(
    const Config& config,
    const std::filesystem::path& output_root,
    int step,
    int filter_width_int,
    const ResolutionDiagnostics& resolution,
    Real mean_tau_ab,
    Real mean_tau_zz,
    Real min_tau_ab,
    Real max_tau_ab,
    Real min_tau_zz,
    Real max_tau_zz,
    const std::vector<std::uint64_t>& tau_ab_counts,
    const std::vector<std::uint64_t>& log_tau_zz_counts,
    const std::vector<std::uint64_t>& joint_counts,
    std::uint64_t tau_ab_underflow_count,
    std::uint64_t tau_ab_overflow_count,
    std::uint64_t tau_zz_zero_count,
    std::uint64_t tau_zz_underflow_count,
    std::uint64_t tau_zz_overflow_count) {
    const std::size_t cells = cell_count(config);
    const long double inv_cells = 1.0L / static_cast<long double>(cells);
    const int pdf_bins = config.filter_pdf_bins;
    const int joint_bins = config.filter_joint_pdf_bins;
    const Real tau_ab_min = config.filter_tau_ab_min;
    const Real tau_ab_max = config.filter_tau_ab_max;
    const Real tau_ab_bin_width =
        (tau_ab_max - tau_ab_min) / static_cast<Real>(pdf_bins);
    const Real joint_tau_ab_bin_width =
        (tau_ab_max - tau_ab_min) / static_cast<Real>(joint_bins);
    const Real log_tau_zz_min = config.filter_log_tau_zz_min;
    const Real log_tau_zz_max = config.filter_log_tau_zz_max;
    const Real log_tau_zz_bin_width =
        (log_tau_zz_max - log_tau_zz_min) / static_cast<Real>(pdf_bins);
    const Real joint_log_tau_zz_bin_width =
        (log_tau_zz_max - log_tau_zz_min) / static_cast<Real>(joint_bins);
    const Real filter_delta = static_cast<Real>(filter_width_int);
    const Real delta_over_eta_k =
        std::isfinite(resolution.eta_k) && resolution.eta_k > Real{}
            ? filter_delta / resolution.eta_k
            : Real{};
    const Real delta_over_eta_b =
        std::isfinite(resolution.eta_b) && resolution.eta_b > Real{}
            ? filter_delta / resolution.eta_b
            : Real{};

    std::uint64_t tau_ab_included_count{};
    std::uint64_t tau_zz_included_count{};
    std::uint64_t joint_included_count{};
    long double mean_tau_ab_from_pdf{};
    long double mean_tau_zz_from_pdf{};
    for (int bin = 0; bin < pdf_bins; ++bin) {
        const Real tau_ab_center =
            tau_ab_min + (static_cast<Real>(bin) + Real{0.5}) * tau_ab_bin_width;
        const Real log_tau_zz_center =
            log_tau_zz_min +
            (static_cast<Real>(bin) + Real{0.5}) * log_tau_zz_bin_width;
        const Real tau_zz_center = std::pow(Real{10}, log_tau_zz_center);
        tau_ab_included_count += tau_ab_counts[bin];
        tau_zz_included_count += log_tau_zz_counts[bin];
        mean_tau_ab_from_pdf +=
            static_cast<long double>(tau_ab_counts[bin]) * tau_ab_center;
        mean_tau_zz_from_pdf +=
            static_cast<long double>(log_tau_zz_counts[bin]) * tau_zz_center;
    }
    mean_tau_ab_from_pdf *= inv_cells;
    mean_tau_zz_from_pdf *= inv_cells;
    for (const std::uint64_t count : joint_counts) {
        joint_included_count += count;
    }

    const std::uint64_t tau_ab_classified_count =
        tau_ab_included_count + tau_ab_underflow_count + tau_ab_overflow_count;
    const std::uint64_t tau_zz_classified_count =
        tau_zz_included_count + tau_zz_zero_count +
        tau_zz_underflow_count + tau_zz_overflow_count;
    if (tau_ab_classified_count != cells) {
        throw std::runtime_error(
            "filtered tau_AB PDF counts do not sum to the domain size");
    }
    if (tau_zz_classified_count != cells) {
        throw std::runtime_error(
            "filtered tau_ZZ PDF counts do not sum to the domain size");
    }

    const Real tau_ab_underflow_fraction = static_cast<Real>(
        static_cast<long double>(tau_ab_underflow_count) * inv_cells);
    const Real tau_ab_overflow_fraction = static_cast<Real>(
        static_cast<long double>(tau_ab_overflow_count) * inv_cells);
    const Real tau_ab_included_probability = static_cast<Real>(
        static_cast<long double>(tau_ab_included_count) * inv_cells);
    const Real tau_zz_zero_fraction = static_cast<Real>(
        static_cast<long double>(tau_zz_zero_count) * inv_cells);
    const Real tau_zz_underflow_fraction = static_cast<Real>(
        static_cast<long double>(tau_zz_underflow_count) * inv_cells);
    const Real tau_zz_overflow_fraction = static_cast<Real>(
        static_cast<long double>(tau_zz_overflow_count) * inv_cells);
    const Real tau_zz_included_probability = static_cast<Real>(
        static_cast<long double>(tau_zz_included_count) * inv_cells);
    const Real joint_included_probability = static_cast<Real>(
        static_cast<long double>(joint_included_count) * inv_cells);
    const Real joint_excluded_probability =
        Real{1} - joint_included_probability;
    const Real tau_ab_normalization_error = std::abs(
        tau_ab_included_probability + tau_ab_underflow_fraction +
        tau_ab_overflow_fraction - Real{1});
    const Real tau_zz_normalization_error = std::abs(
        tau_zz_included_probability + tau_zz_zero_fraction +
        tau_zz_underflow_fraction + tau_zz_overflow_fraction - Real{1});
    const Real joint_probability_sum_error = std::abs(
        joint_included_probability -
        static_cast<Real>(static_cast<long double>(joint_included_count) *
                          inv_cells));
    const Real mean_tau_ab_pdf_error =
        std::abs(static_cast<Real>(mean_tau_ab_from_pdf) - mean_tau_ab);
    const Real mean_tau_zz_pdf_error =
        std::abs(static_cast<Real>(mean_tau_zz_from_pdf) - mean_tau_zz);

    const std::filesystem::path output_dir =
        output_root / std::format("step_{:08}", step);
    std::filesystem::create_directories(output_dir);
    const std::string width_label = std::format("w{:04}", filter_width_int);

    {
        std::ofstream file{
            output_dir / std::format("pdf_tauAB_{}.csv", width_label)};
        if (!file) {
            throw std::runtime_error("failed to open filtered tau_AB PDF output");
        }
        file << "tauAB_center,count,probability\n";
        for (int bin = 0; bin < pdf_bins; ++bin) {
            const Real center =
                tau_ab_min +
                (static_cast<Real>(bin) + Real{0.5}) * tau_ab_bin_width;
            const Real probability = static_cast<Real>(
                static_cast<long double>(tau_ab_counts[bin]) * inv_cells);
            file << std::format("{:.17g}", static_cast<double>(center))
                 << ',' << tau_ab_counts[bin]
                 << ',' << std::format("{:.17g}", static_cast<double>(probability))
                 << '\n';
        }
        file.flush();
    }

    {
        std::ofstream file{
            output_dir / std::format("pdf_log_tauZZ_{}.csv", width_label)};
        if (!file) {
            throw std::runtime_error(
                "failed to open filtered tau_ZZ PDF output");
        }
        file << "log10_tauZZ_center,tauZZ_center,count,probability\n";
        for (int bin = 0; bin < pdf_bins; ++bin) {
            const Real log_center =
                log_tau_zz_min +
                (static_cast<Real>(bin) + Real{0.5}) * log_tau_zz_bin_width;
            const Real tau_center = std::pow(Real{10}, log_center);
            const Real probability = static_cast<Real>(
                static_cast<long double>(log_tau_zz_counts[bin]) * inv_cells);
            file << std::format("{:.17g}", static_cast<double>(log_center))
                 << ',' << std::format("{:.17g}", static_cast<double>(tau_center))
                 << ',' << log_tau_zz_counts[bin]
                 << ',' << std::format("{:.17g}", static_cast<double>(probability))
                 << '\n';
        }
        file.flush();
    }

    {
        std::ofstream file{
            output_dir /
            std::format("joint_pdf_tauAB_log_tauZZ_{}.csv", width_label)};
        if (!file) {
            throw std::runtime_error(
                "failed to open filtered tau_AB/tau_ZZ joint PDF output");
        }
        file << "tauAB_center,log10_tauZZ_center,tauZZ_center,count,probability\n";
        for (int tau_bin = 0; tau_bin < joint_bins; ++tau_bin) {
            const Real tau_center =
                tau_ab_min +
                (static_cast<Real>(tau_bin) + Real{0.5}) *
                    joint_tau_ab_bin_width;
            for (int log_bin = 0; log_bin < joint_bins; ++log_bin) {
                const Real log_center =
                    log_tau_zz_min +
                    (static_cast<Real>(log_bin) + Real{0.5}) *
                        joint_log_tau_zz_bin_width;
                const Real tau_zz_center = std::pow(Real{10}, log_center);
                const std::size_t index =
                    static_cast<std::size_t>(tau_bin) *
                    static_cast<std::size_t>(joint_bins) +
                    static_cast<std::size_t>(log_bin);
                const Real probability = static_cast<Real>(
                    static_cast<long double>(joint_counts[index]) * inv_cells);
                file << std::format("{:.17g}", static_cast<double>(tau_center))
                     << ','
                     << std::format("{:.17g}", static_cast<double>(log_center))
                     << ','
                     << std::format("{:.17g}", static_cast<double>(tau_zz_center))
                     << ',' << joint_counts[index]
                     << ','
                     << std::format("{:.17g}", static_cast<double>(probability))
                     << '\n';
            }
        }
        file.flush();
    }

    {
        std::ofstream metadata{
            output_dir / std::format("filter_pdf_metadata_{}.json", width_label)};
        if (!metadata) {
            throw std::runtime_error(
                "failed to open filtered SGS PDF metadata output");
        }
        metadata
            << "{\n"
            << "  \"step\": " << step << ",\n"
            << "  \"time\": " << json_number(static_cast<Real>(step)) << ",\n"
            << "  \"filter_width\": " << filter_width_int << ",\n"
            << "  \"Delta_over_delta0\": "
            << json_number(filter_delta / config.delta0) << ",\n"
            << "  \"Delta_over_etaK\": " << json_number(delta_over_eta_k) << ",\n"
            << "  \"Delta_over_etaB\": " << json_number(delta_over_eta_b) << ",\n"
            << "  \"filter_pdf_bins\": " << pdf_bins << ",\n"
            << "  \"filter_joint_pdf_bins\": " << joint_bins << ",\n"
            << "  \"tauAB_range\": ["
            << json_number(tau_ab_min) << ", " << json_number(tau_ab_max)
            << "],\n"
            << "  \"log10_tauZZ_range\": ["
            << json_number(log_tau_zz_min) << ", "
            << json_number(log_tau_zz_max) << "],\n"
            << "  \"tauAB_underflow_fraction\": "
            << json_number(tau_ab_underflow_fraction) << ",\n"
            << "  \"tauAB_overflow_fraction\": "
            << json_number(tau_ab_overflow_fraction) << ",\n"
            << "  \"tauAB_included_probability\": "
            << json_number(tau_ab_included_probability) << ",\n"
            << "  \"tauZZ_zero_fraction\": "
            << json_number(tau_zz_zero_fraction) << ",\n"
            << "  \"tauZZ_underflow_fraction\": "
            << json_number(tau_zz_underflow_fraction) << ",\n"
            << "  \"tauZZ_overflow_fraction\": "
            << json_number(tau_zz_overflow_fraction) << ",\n"
            << "  \"tauZZ_included_probability\": "
            << json_number(tau_zz_included_probability) << ",\n"
            << "  \"joint_included_probability\": "
            << json_number(joint_included_probability) << ",\n"
            << "  \"joint_excluded_probability\": "
            << json_number(joint_excluded_probability) << ",\n"
            << "  \"tauAB_pdf_normalization_error\": "
            << json_number(tau_ab_normalization_error) << ",\n"
            << "  \"tauZZ_pdf_normalization_error\": "
            << json_number(tau_zz_normalization_error) << ",\n"
            << "  \"joint_probability_sum_error\": "
            << json_number(joint_probability_sum_error) << ",\n"
            << "  \"mean_tau_AB\": " << json_number(mean_tau_ab) << ",\n"
            << "  \"mean_tau_AB_from_pdf\": "
            << json_number(static_cast<Real>(mean_tau_ab_from_pdf)) << ",\n"
            << "  \"mean_tau_AB_pdf_error\": "
            << json_number(mean_tau_ab_pdf_error) << ",\n"
            << "  \"mean_tau_ZZ\": " << json_number(mean_tau_zz) << ",\n"
            << "  \"mean_tau_ZZ_from_pdf\": "
            << json_number(static_cast<Real>(mean_tau_zz_from_pdf)) << ",\n"
            << "  \"mean_tau_ZZ_pdf_error\": "
            << json_number(mean_tau_zz_pdf_error) << ",\n"
            << "  \"observed_tauAB_min\": " << json_number(min_tau_ab) << ",\n"
            << "  \"observed_tauAB_max\": " << json_number(max_tau_ab) << ",\n"
            << "  \"observed_tauZZ_min\": " << json_number(min_tau_zz) << ",\n"
            << "  \"observed_tauZZ_max\": " << json_number(max_tau_zz) << "\n"
            << "}\n";
        metadata.flush();
    }
}

void write_filter_statistics(
    std::ofstream& file,
    const std::filesystem::path& conditional_output_root,
    const std::filesystem::path& pdf_output_root,
    const Config& config,
    int step,
    const ResolutionDiagnostics& resolution,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b) {
    const auto a_view = species_a.get_current_view();
    const auto b_view = species_b.get_current_view();
    const std::size_t cells = cell_count(config);
    const long double inv_cells = 1.0L / static_cast<long double>(cells);

    std::vector<Real> field_a(cells);
    std::vector<Real> field_b(cells);
    std::vector<Real> field_extra(cells);
    std::vector<Real> filtered_a(cells);
    std::vector<Real> filtered_b(cells);
    std::vector<Real> filtered_ab(cells);
    std::vector<Real> filtered_extra(cells);
    std::vector<Real> tau_aa(cells);
    std::vector<Real> tau_bb(cells);
    std::vector<Real> filter_tmp1(cells);
    std::vector<Real> filter_tmp2(cells);

    long double sum_a{};
    long double sum_b{};
    long double sum_ab{};

#pragma omp parallel for collapse(3) schedule(static) reduction(+: sum_a, sum_b, sum_ab)
    for (std::size_t z = 0; z < config.nz; ++z) {
        for (std::size_t y = 0; y < config.ny; ++y) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const std::size_t index = (z * config.ny + y) * config.nx + x;
                const Real concentration_a = concentration_at(a_view, x, y, z);
                const Real concentration_b = concentration_at(b_view, x, y, z);
                const Real concentration_ab = concentration_a * concentration_b;
                field_a[index] = concentration_a;
                field_b[index] = concentration_b;
                field_extra[index] = concentration_ab;
                sum_a += static_cast<long double>(concentration_a);
                sum_b += static_cast<long double>(concentration_b);
                sum_ab += static_cast<long double>(concentration_ab);
            }
        }
    }

    for (const int filter_width_int : config.filter_widths) {
    const auto filter_width = static_cast<std::size_t>(filter_width_int);
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < cells; ++index) {
        field_extra[index] = field_a[index] * field_b[index];
    }

    lbm::box_filter_3d_separable<Real>(
        scalar_field_view(std::as_const(field_a), config),
        scalar_field_view(filtered_a, config),
        filter_width,
        filter_tmp1,
        filter_tmp2);
    lbm::box_filter_3d_separable<Real>(
        scalar_field_view(std::as_const(field_b), config),
        scalar_field_view(filtered_b, config),
        filter_width,
        filter_tmp1,
        filter_tmp2);
    lbm::box_filter_3d_separable<Real>(
        scalar_field_view(std::as_const(field_extra), config),
        scalar_field_view(filtered_ab, config),
        filter_width,
        filter_tmp1,
        filter_tmp2);

    long double sum_a_bar{};
    long double sum_b_bar{};
    long double sum_ab_bar{};
    Real max_width_one_identity_error{};
    bool all_finite = true;

#pragma omp parallel for schedule(static) reduction(+: sum_a_bar, sum_b_bar, sum_ab_bar) reduction(max: max_width_one_identity_error) reduction(&&: all_finite)
    for (std::size_t index = 0; index < cells; ++index) {
        sum_a_bar += static_cast<long double>(filtered_a[index]);
        sum_b_bar += static_cast<long double>(filtered_b[index]);
        sum_ab_bar += static_cast<long double>(filtered_ab[index]);
        all_finite = all_finite && std::isfinite(filtered_a[index]) &&
                     std::isfinite(filtered_b[index]) &&
                     std::isfinite(filtered_ab[index]);
        if (filter_width_int == 1) {
            max_width_one_identity_error = std::max(
                max_width_one_identity_error,
                std::abs(filtered_a[index] - field_a[index]));
            max_width_one_identity_error = std::max(
                max_width_one_identity_error,
                std::abs(filtered_b[index] - field_b[index]));
            max_width_one_identity_error = std::max(
                max_width_one_identity_error,
                std::abs(filtered_ab[index] - field_extra[index]));
        }
    }

    if (!all_finite) {
        throw std::runtime_error("non-finite value detected in filtered scalar fields");
    }
    if (filter_width_int == 1 && max_width_one_identity_error != Real{}) {
        throw std::runtime_error("filter_width=1 did not reproduce scalar fields exactly");
    }

    Real max_abs_zbar_linearity_error{};
    Real max_abs_zbar_linearity_scale{};

#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < cells; ++index) {
        filtered_ab[index] -= filtered_a[index] * filtered_b[index];
    }

#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < cells; ++index) {
        const Real z =
            Real{0.5} * (Real{1} + (field_a[index] - field_b[index]) / config.c0);
        field_extra[index] = z * z;
    }
    lbm::box_filter_3d_separable<Real>(
        scalar_field_view(std::as_const(field_extra), config),
        scalar_field_view(filtered_extra, config),
        filter_width,
        filter_tmp1,
        filter_tmp2);
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < cells; ++index) {
        const Real z_bar =
            Real{0.5} *
            (Real{1} + (filtered_a[index] - filtered_b[index]) / config.c0);
        filtered_extra[index] -= z_bar * z_bar;
    }

#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < cells; ++index) {
        field_extra[index] = field_a[index] * field_a[index];
    }
    lbm::box_filter_3d_separable<Real>(
        scalar_field_view(std::as_const(field_extra), config),
        scalar_field_view(tau_aa, config),
        filter_width,
        filter_tmp1,
        filter_tmp2);
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < cells; ++index) {
        tau_aa[index] -= filtered_a[index] * filtered_a[index];
    }

#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < cells; ++index) {
        field_extra[index] = field_b[index] * field_b[index];
    }
    lbm::box_filter_3d_separable<Real>(
        scalar_field_view(std::as_const(field_extra), config),
        scalar_field_view(tau_bb, config),
        filter_width,
        filter_tmp1,
        filter_tmp2);
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < cells; ++index) {
        tau_bb[index] -= filtered_b[index] * filtered_b[index];
    }

    long double sum_r{};
    long double sum_r_bar_direct{};
    long double sum_k_ab_bar{};
    long double sum_a_bar_b_bar{};
    long double sum_tau_ab{};
    long double sum_tau_ab2{};
    long double sum_tau_aa{};
    long double sum_tau_aa2{};
    long double sum_tau_bb{};
    long double sum_tau_bb2{};
    long double sum_tau_zz{};
    long double sum_tau_zz2{};
    long double sum_rho_ab_sgs{};
    long double sum_rho_ab_sgs2{};
    long double sum_r_les_naive{};
    long double sum_r_sgs{};
    Real max_abs_identity_error{};
    Real max_abs_r_bar_direct{};
    Real min_tau_ab{std::numeric_limits<Real>::max()};
    Real max_tau_ab{std::numeric_limits<Real>::lowest()};
    Real min_tau_aa{std::numeric_limits<Real>::max()};
    Real max_tau_aa{std::numeric_limits<Real>::lowest()};
    Real min_tau_bb{std::numeric_limits<Real>::max()};
    Real max_tau_bb{std::numeric_limits<Real>::lowest()};
    Real min_tau_zz{std::numeric_limits<Real>::max()};
    Real max_tau_zz{std::numeric_limits<Real>::lowest()};
    Real min_rho_ab_sgs{std::numeric_limits<Real>::max()};
    Real max_rho_ab_sgs{std::numeric_limits<Real>::lowest()};
    Real max_abs_tau_zz_identity_error{};
    Real max_abs_tau_zz_identity_scale{};
    long double sum_tau_zz_identity_error2{};
    Real max_abs_reaction_decomposition_error{};
    Real max_abs_r_exact_filtered{};
    bool reaction_finite = true;

#pragma omp parallel for schedule(static) reduction(+: sum_r, sum_r_bar_direct, sum_k_ab_bar, sum_a_bar_b_bar, sum_tau_ab, sum_tau_ab2, sum_tau_aa, sum_tau_aa2, sum_tau_bb, sum_tau_bb2, sum_tau_zz, sum_tau_zz2, sum_rho_ab_sgs, sum_rho_ab_sgs2, sum_tau_zz_identity_error2, sum_r_les_naive, sum_r_sgs) reduction(max: max_abs_identity_error, max_abs_r_bar_direct, max_tau_ab, max_tau_aa, max_tau_bb, max_tau_zz, max_rho_ab_sgs, max_abs_tau_zz_identity_error, max_abs_tau_zz_identity_scale, max_abs_reaction_decomposition_error, max_abs_r_exact_filtered) reduction(min: min_tau_ab, min_tau_aa, min_tau_bb, min_tau_zz, min_rho_ab_sgs) reduction(&&: reaction_finite)
    for (std::size_t index = 0; index < cells; ++index) {
        const Real a_bar_b_bar = filtered_a[index] * filtered_b[index];
        const Real tau_ab = filtered_ab[index];
        const Real local_tau_aa = tau_aa[index];
        const Real local_tau_bb = tau_bb[index];
        const Real local_tau_zz = filtered_extra[index];
        const Real tau_zz_from_reactants =
            (local_tau_aa + local_tau_bb - Real{2} * tau_ab) /
            (Real{4} * config.c0 * config.c0);
        const Real tau_zz_identity_error =
            std::abs(local_tau_zz - tau_zz_from_reactants);
        const Real tau_zz_identity_scale =
            std::max(std::abs(local_tau_zz), std::abs(tau_zz_from_reactants));
        const Real variance_floor =
            Real{64} * std::numeric_limits<Real>::epsilon() * config.c0 * config.c0;
        const Real rho_ab_sgs =
            local_tau_aa > variance_floor && local_tau_bb > variance_floor
                ? tau_ab / std::sqrt(local_tau_aa * local_tau_bb)
                : Real{};
        const Real reaction_from_filtered_ab =
            config.k_react * (a_bar_b_bar + tau_ab);
        const Real reaction_bar_direct = reaction_from_filtered_ab;
        const Real reaction = reaction_from_filtered_ab;
        const Real identity_error{};
        const Real reaction_les_naive = config.k_react * a_bar_b_bar;
        const Real reaction_sgs = config.k_react * tau_ab;
        const Real reaction_decomposition_error =
            std::abs(reaction_from_filtered_ab - reaction_les_naive - reaction_sgs);
        sum_r += static_cast<long double>(reaction);
        sum_r_bar_direct += static_cast<long double>(reaction_bar_direct);
        sum_k_ab_bar += static_cast<long double>(reaction_from_filtered_ab);
        sum_a_bar_b_bar += static_cast<long double>(a_bar_b_bar);
        sum_tau_ab += static_cast<long double>(tau_ab);
        sum_tau_ab2 += static_cast<long double>(tau_ab * tau_ab);
        sum_tau_aa += static_cast<long double>(local_tau_aa);
        sum_tau_aa2 += static_cast<long double>(local_tau_aa * local_tau_aa);
        sum_tau_bb += static_cast<long double>(local_tau_bb);
        sum_tau_bb2 += static_cast<long double>(local_tau_bb * local_tau_bb);
        sum_tau_zz += static_cast<long double>(local_tau_zz);
        sum_tau_zz2 += static_cast<long double>(local_tau_zz * local_tau_zz);
        sum_rho_ab_sgs += static_cast<long double>(rho_ab_sgs);
        sum_rho_ab_sgs2 += static_cast<long double>(rho_ab_sgs * rho_ab_sgs);
        sum_tau_zz_identity_error2 +=
            static_cast<long double>(tau_zz_identity_error) *
            static_cast<long double>(tau_zz_identity_error);
        sum_r_les_naive += static_cast<long double>(reaction_les_naive);
        sum_r_sgs += static_cast<long double>(reaction_sgs);
        max_abs_identity_error = std::max(max_abs_identity_error, identity_error);
        max_abs_r_bar_direct =
            std::max(max_abs_r_bar_direct, std::abs(reaction_bar_direct));
        min_tau_ab = std::min(min_tau_ab, tau_ab);
        max_tau_ab = std::max(max_tau_ab, tau_ab);
        min_tau_aa = std::min(min_tau_aa, local_tau_aa);
        max_tau_aa = std::max(max_tau_aa, local_tau_aa);
        min_tau_bb = std::min(min_tau_bb, local_tau_bb);
        max_tau_bb = std::max(max_tau_bb, local_tau_bb);
        min_tau_zz = std::min(min_tau_zz, local_tau_zz);
        max_tau_zz = std::max(max_tau_zz, local_tau_zz);
        min_rho_ab_sgs = std::min(min_rho_ab_sgs, rho_ab_sgs);
        max_rho_ab_sgs = std::max(max_rho_ab_sgs, rho_ab_sgs);
        max_abs_tau_zz_identity_error = std::max(
            max_abs_tau_zz_identity_error,
            tau_zz_identity_error);
        max_abs_tau_zz_identity_scale = std::max(
            max_abs_tau_zz_identity_scale,
            tau_zz_identity_scale);
        max_abs_reaction_decomposition_error = std::max(
            max_abs_reaction_decomposition_error,
            reaction_decomposition_error);
        max_abs_r_exact_filtered = std::max(
            max_abs_r_exact_filtered,
            std::abs(reaction_from_filtered_ab));
        reaction_finite = reaction_finite && std::isfinite(reaction) &&
                          std::isfinite(reaction_bar_direct) &&
                          std::isfinite(reaction_from_filtered_ab) &&
                          std::isfinite(a_bar_b_bar) &&
                          std::isfinite(tau_ab) &&
                          std::isfinite(local_tau_aa) &&
                          std::isfinite(local_tau_bb) &&
                          std::isfinite(local_tau_zz) &&
                          std::isfinite(rho_ab_sgs) &&
                          std::isfinite(reaction_les_naive) &&
                          std::isfinite(reaction_sgs);
    }

    if (!reaction_finite) {
        throw std::runtime_error("non-finite value detected in filtered reaction fields");
    }
    if (filter_width_int == 1) {
        const Real max_width_one_sgs_error = std::max(
            {std::abs(min_tau_ab),
             std::abs(max_tau_ab),
             std::abs(min_tau_aa),
             std::abs(max_tau_aa),
             std::abs(min_tau_bb),
             std::abs(max_tau_bb),
             std::abs(min_tau_zz),
             std::abs(max_tau_zz)});
        const Real width_one_tolerance =
            Real{64} * std::numeric_limits<Real>::epsilon() * config.c0 * config.c0;
        if (max_width_one_sgs_error > width_one_tolerance) {
            throw std::runtime_error(
                "filter_width=1 produced nonzero SGS scalar moments");
        }
    }

    const Real mean_a = static_cast<Real>(sum_a * inv_cells);
    const Real mean_a_bar = static_cast<Real>(sum_a_bar * inv_cells);
    const Real mean_b = static_cast<Real>(sum_b * inv_cells);
    const Real mean_b_bar = static_cast<Real>(sum_b_bar * inv_cells);
    const Real mean_ab = static_cast<Real>(sum_ab * inv_cells);
    const Real mean_ab_bar = static_cast<Real>(sum_ab_bar * inv_cells);
    const Real mean_k_ab_bar = static_cast<Real>(sum_k_ab_bar * inv_cells);
    const Real mean_r = config.k_react * mean_ab;
    const Real mean_r_bar_direct = mean_k_ab_bar;
    const Real mean_a_bar_b_bar = static_cast<Real>(sum_a_bar_b_bar * inv_cells);
    const Real mean_tau_ab = static_cast<Real>(sum_tau_ab * inv_cells);
    const Real rms_tau_ab =
        std::sqrt(std::max(Real{}, static_cast<Real>(sum_tau_ab2 * inv_cells)));
    const Real mean_tau_aa = static_cast<Real>(sum_tau_aa * inv_cells);
    const Real rms_tau_aa =
        std::sqrt(std::max(Real{}, static_cast<Real>(sum_tau_aa2 * inv_cells)));
    const Real mean_tau_bb = static_cast<Real>(sum_tau_bb * inv_cells);
    const Real rms_tau_bb =
        std::sqrt(std::max(Real{}, static_cast<Real>(sum_tau_bb2 * inv_cells)));
    const Real mean_tau_zz = static_cast<Real>(sum_tau_zz * inv_cells);
    const Real rms_tau_zz =
        std::sqrt(std::max(Real{}, static_cast<Real>(sum_tau_zz2 * inv_cells)));
    const Real mean_rho_ab_sgs = static_cast<Real>(sum_rho_ab_sgs * inv_cells);
    const Real rms_rho_ab_sgs = std::sqrt(
        std::max(Real{}, static_cast<Real>(sum_rho_ab_sgs2 * inv_cells)));
    const Real rms_tau_zz_identity_error = std::sqrt(std::max(
        Real{},
        static_cast<Real>(sum_tau_zz_identity_error2 * inv_cells)));
    const Real relative_tau_zz_identity_error =
        max_abs_tau_zz_identity_error /
        std::max(
            {max_abs_tau_zz_identity_scale,
             config.c0 * config.c0,
             std::numeric_limits<Real>::min()});
    const Real relative_zbar_linearity_error =
        max_abs_zbar_linearity_error /
        std::max(max_abs_zbar_linearity_scale, std::numeric_limits<Real>::min());
    const Real mean_r_exact_filtered = mean_k_ab_bar;
    const Real mean_r_les_naive = static_cast<Real>(sum_r_les_naive * inv_cells);
    const Real mean_r_sgs = static_cast<Real>(sum_r_sgs * inv_cells);
    const Real les_reaction_efficiency =
        mean_a_bar_b_bar > Real{} ? mean_ab_bar / mean_a_bar_b_bar : Real{};
    const Real les_reaction_overprediction_factor =
        mean_ab_bar > Real{} ? mean_a_bar_b_bar / mean_ab_bar : Real{};
    const bool has_filtered_reaction =
        std::abs(mean_r_exact_filtered) > std::numeric_limits<Real>::min();
    const Real sgs_reaction_fraction =
        has_filtered_reaction
            ? mean_r_sgs / mean_r_exact_filtered
            : Real{};
    const Real sgs_reaction_magnitude_fraction =
        has_filtered_reaction
            ? std::abs(mean_r_sgs) / std::abs(mean_r_exact_filtered)
            : Real{};
    const Real relative_identity_error =
        max_abs_identity_error /
        std::max(max_abs_r_bar_direct, std::numeric_limits<Real>::min());
    const Real relative_reaction_decomposition_error =
        max_abs_reaction_decomposition_error /
        std::max(max_abs_r_exact_filtered, std::numeric_limits<Real>::min());
    const Real covariance_identity_error =
        std::abs(mean_ab_bar - mean_a_bar_b_bar - mean_tau_ab);
    Real les_efficiency_identity_error{};
    if (les_reaction_efficiency != Real{} &&
        les_reaction_overprediction_factor != Real{}) {
        les_efficiency_identity_error = std::max(
            les_efficiency_identity_error,
            std::abs(les_reaction_efficiency *
                     les_reaction_overprediction_factor - Real{1}));
    }
    if (has_filtered_reaction &&
        std::abs(Real{1} - sgs_reaction_fraction) >
        std::numeric_limits<Real>::min()) {
        const Real efficiency_from_sgs =
            Real{1} / (Real{1} - sgs_reaction_fraction);
        les_efficiency_identity_error = std::max(
            les_efficiency_identity_error,
            std::abs(les_reaction_efficiency - efficiency_from_sgs));
    }
    const Real filter_delta = static_cast<Real>(filter_width_int);
    const Real delta_over_eta_k =
        std::isfinite(resolution.eta_k) && resolution.eta_k > Real{}
            ? filter_delta / resolution.eta_k
            : Real{};
    const Real delta_over_eta_b =
        std::isfinite(resolution.eta_b) && resolution.eta_b > Real{}
            ? filter_delta / resolution.eta_b
            : Real{};

    const int conditional_bins = config.filter_conditional_bins;
    std::vector<std::uint64_t> zbar_counts(conditional_bins);
    std::vector<long double> zbar_sum_tau_ab(conditional_bins);
    std::vector<long double> zbar_sum_rho_ab_sgs(conditional_bins);
    std::vector<std::uint64_t> log_tau_zz_counts(conditional_bins);
    std::vector<long double> log_tau_zz_sum_tau_ab(conditional_bins);
    std::uint64_t tau_zz_zero_count{};
    std::uint64_t tau_zz_underflow_count{};
    std::uint64_t tau_zz_overflow_count{};
    long double excluded_log_tau_zz_sum_tau_ab{};
    bool conditional_finite = true;
    const Real log_tau_zz_min = config.filter_log_tau_zz_min;
    const Real log_tau_zz_max = config.filter_log_tau_zz_max;
    const Real log_tau_zz_bin_width =
        (log_tau_zz_max - log_tau_zz_min) /
        static_cast<Real>(conditional_bins);
    const int pdf_bins = config.filter_pdf_bins;
    const int joint_bins = config.filter_joint_pdf_bins;
    const Real tau_ab_min = config.filter_tau_ab_min;
    const Real tau_ab_max = config.filter_tau_ab_max;
    const Real tau_ab_pdf_bin_width =
        (tau_ab_max - tau_ab_min) / static_cast<Real>(pdf_bins);
    const Real tau_ab_joint_bin_width =
        (tau_ab_max - tau_ab_min) / static_cast<Real>(joint_bins);
    const Real log_tau_zz_pdf_bin_width =
        (log_tau_zz_max - log_tau_zz_min) / static_cast<Real>(pdf_bins);
    const Real log_tau_zz_joint_bin_width =
        (log_tau_zz_max - log_tau_zz_min) / static_cast<Real>(joint_bins);
    std::vector<std::uint64_t> pdf_tau_ab_counts(pdf_bins);
    std::vector<std::uint64_t> pdf_log_tau_zz_counts(pdf_bins);
    std::vector<std::uint64_t> joint_tau_ab_tau_zz_counts(
        static_cast<std::size_t>(joint_bins) *
        static_cast<std::size_t>(joint_bins));
    std::uint64_t tau_ab_underflow_count{};
    std::uint64_t tau_ab_overflow_count{};

#pragma omp parallel
    {
        std::vector<std::uint64_t> local_zbar_counts(conditional_bins);
        std::vector<long double> local_zbar_sum_tau_ab(conditional_bins);
        std::vector<long double> local_zbar_sum_rho_ab_sgs(conditional_bins);
        std::vector<std::uint64_t> local_log_tau_zz_counts(conditional_bins);
        std::vector<long double> local_log_tau_zz_sum_tau_ab(conditional_bins);
        std::uint64_t local_tau_zz_zero_count{};
        std::uint64_t local_tau_zz_underflow_count{};
        std::uint64_t local_tau_zz_overflow_count{};
        long double local_excluded_log_tau_zz_sum_tau_ab{};
        bool local_conditional_finite = true;
        std::vector<std::uint64_t> local_pdf_tau_ab_counts(pdf_bins);
        std::vector<std::uint64_t> local_pdf_log_tau_zz_counts(pdf_bins);
        std::vector<std::uint64_t> local_joint_counts(
            static_cast<std::size_t>(joint_bins) *
            static_cast<std::size_t>(joint_bins));
        std::uint64_t local_tau_ab_underflow_count{};
        std::uint64_t local_tau_ab_overflow_count{};

#pragma omp for schedule(static)
        for (std::size_t index = 0; index < cells; ++index) {
            const Real tau_ab = filtered_ab[index];
            const Real local_tau_aa = tau_aa[index];
            const Real local_tau_bb = tau_bb[index];
            const Real local_tau_zz = filtered_extra[index];
            const Real z_bar =
                Real{0.5} *
                (Real{1} + (filtered_a[index] - filtered_b[index]) / config.c0);
            const Real variance_floor =
                Real{64} * std::numeric_limits<Real>::epsilon() *
                config.c0 * config.c0;
            const Real rho_ab_sgs =
                local_tau_aa > variance_floor && local_tau_bb > variance_floor
                    ? tau_ab / std::sqrt(local_tau_aa * local_tau_bb)
                    : Real{};

            const Real z_clamped = std::clamp(z_bar, Real{}, Real{1});
            int z_bin = static_cast<int>(
                std::floor(z_clamped * static_cast<Real>(conditional_bins)));
            z_bin = std::clamp(z_bin, 0, conditional_bins - 1);
            ++local_zbar_counts[z_bin];
            local_zbar_sum_tau_ab[z_bin] += static_cast<long double>(tau_ab);
            local_zbar_sum_rho_ab_sgs[z_bin] +=
                static_cast<long double>(rho_ab_sgs);

            bool tau_ab_inside = false;
            int tau_ab_joint_bin = 0;
            if (tau_ab < tau_ab_min) {
                ++local_tau_ab_underflow_count;
            } else if (tau_ab > tau_ab_max) {
                ++local_tau_ab_overflow_count;
            } else {
                int tau_ab_pdf_bin = static_cast<int>(
                    std::floor((tau_ab - tau_ab_min) / tau_ab_pdf_bin_width));
                tau_ab_pdf_bin = std::clamp(tau_ab_pdf_bin, 0, pdf_bins - 1);
                ++local_pdf_tau_ab_counts[tau_ab_pdf_bin];
                tau_ab_joint_bin = static_cast<int>(
                    std::floor((tau_ab - tau_ab_min) / tau_ab_joint_bin_width));
                tau_ab_joint_bin =
                    std::clamp(tau_ab_joint_bin, 0, joint_bins - 1);
                tau_ab_inside = true;
            }

            if (local_tau_zz <= Real{}) {
                ++local_tau_zz_zero_count;
                local_excluded_log_tau_zz_sum_tau_ab +=
                    static_cast<long double>(tau_ab);
            } else {
                const Real log_tau_zz = std::log10(local_tau_zz);
                if (log_tau_zz < log_tau_zz_min) {
                    ++local_tau_zz_underflow_count;
                    local_excluded_log_tau_zz_sum_tau_ab +=
                        static_cast<long double>(tau_ab);
                } else if (log_tau_zz > log_tau_zz_max) {
                    ++local_tau_zz_overflow_count;
                    local_excluded_log_tau_zz_sum_tau_ab +=
                        static_cast<long double>(tau_ab);
                } else {
                    int log_bin = static_cast<int>(
                        std::floor((log_tau_zz - log_tau_zz_min) /
                                   log_tau_zz_bin_width));
                    log_bin = std::clamp(log_bin, 0, conditional_bins - 1);
                    ++local_log_tau_zz_counts[log_bin];
                    local_log_tau_zz_sum_tau_ab[log_bin] +=
                        static_cast<long double>(tau_ab);
                    int pdf_log_bin = static_cast<int>(
                        std::floor((log_tau_zz - log_tau_zz_min) /
                                   log_tau_zz_pdf_bin_width));
                    pdf_log_bin = std::clamp(pdf_log_bin, 0, pdf_bins - 1);
                    ++local_pdf_log_tau_zz_counts[pdf_log_bin];
                    if (tau_ab_inside) {
                        int joint_log_bin = static_cast<int>(
                            std::floor((log_tau_zz - log_tau_zz_min) /
                                       log_tau_zz_joint_bin_width));
                        joint_log_bin =
                            std::clamp(joint_log_bin, 0, joint_bins - 1);
                        const std::size_t joint_index =
                            static_cast<std::size_t>(tau_ab_joint_bin) *
                            static_cast<std::size_t>(joint_bins) +
                            static_cast<std::size_t>(joint_log_bin);
                        ++local_joint_counts[joint_index];
                    }
                }
            }

            local_conditional_finite =
                local_conditional_finite && std::isfinite(tau_ab) &&
                std::isfinite(local_tau_aa) && std::isfinite(local_tau_bb) &&
                std::isfinite(local_tau_zz) && std::isfinite(z_bar) &&
                std::isfinite(rho_ab_sgs);
        }

#pragma omp critical(filter_conditionals_merge)
        {
            for (int bin = 0; bin < conditional_bins; ++bin) {
                zbar_counts[bin] += local_zbar_counts[bin];
                zbar_sum_tau_ab[bin] += local_zbar_sum_tau_ab[bin];
                zbar_sum_rho_ab_sgs[bin] += local_zbar_sum_rho_ab_sgs[bin];
                log_tau_zz_counts[bin] += local_log_tau_zz_counts[bin];
                log_tau_zz_sum_tau_ab[bin] +=
                    local_log_tau_zz_sum_tau_ab[bin];
            }
            for (int bin = 0; bin < pdf_bins; ++bin) {
                pdf_tau_ab_counts[bin] += local_pdf_tau_ab_counts[bin];
                pdf_log_tau_zz_counts[bin] +=
                    local_pdf_log_tau_zz_counts[bin];
            }
            for (std::size_t bin = 0; bin < joint_tau_ab_tau_zz_counts.size();
                 ++bin) {
                joint_tau_ab_tau_zz_counts[bin] += local_joint_counts[bin];
            }
            tau_zz_zero_count += local_tau_zz_zero_count;
            tau_zz_underflow_count += local_tau_zz_underflow_count;
            tau_zz_overflow_count += local_tau_zz_overflow_count;
            tau_ab_underflow_count += local_tau_ab_underflow_count;
            tau_ab_overflow_count += local_tau_ab_overflow_count;
            excluded_log_tau_zz_sum_tau_ab +=
                local_excluded_log_tau_zz_sum_tau_ab;
            conditional_finite = conditional_finite && local_conditional_finite;
        }
    }

    if (!conditional_finite) {
        throw std::runtime_error(
            "non-finite value detected in filtered conditional statistics");
    }
    write_filter_conditional_outputs(
        config,
        conditional_output_root,
        step,
        filter_width_int,
        resolution,
        mean_tau_ab,
        zbar_counts,
        zbar_sum_tau_ab,
        zbar_sum_rho_ab_sgs,
        log_tau_zz_counts,
        log_tau_zz_sum_tau_ab,
        tau_zz_zero_count,
        tau_zz_underflow_count,
        tau_zz_overflow_count,
        excluded_log_tau_zz_sum_tau_ab);
    write_filter_pdf_outputs(
        config,
        pdf_output_root,
        step,
        filter_width_int,
        resolution,
        mean_tau_ab,
        mean_tau_zz,
        min_tau_ab,
        max_tau_ab,
        min_tau_zz,
        max_tau_zz,
        pdf_tau_ab_counts,
        pdf_log_tau_zz_counts,
        joint_tau_ab_tau_zz_counts,
        tau_ab_underflow_count,
        tau_ab_overflow_count,
        tau_zz_zero_count,
        tau_zz_underflow_count,
        tau_zz_overflow_count);

    file << step
         << ',' << std::format("{:.17g}", static_cast<double>(step))
         << ',' << filter_width_int
         << ',' << std::format("{:.17g}", static_cast<double>(filter_delta / config.delta0))
         << ',' << std::format("{:.17g}", static_cast<double>(delta_over_eta_k))
         << ',' << std::format("{:.17g}", static_cast<double>(delta_over_eta_b))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_a))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_a_bar))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_b))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_b_bar))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_ab))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_ab_bar))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_r))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_r_bar_direct))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_k_ab_bar))
         << ',' << std::format("{:.17g}", static_cast<double>(max_abs_identity_error))
         << ',' << std::format("{:.17g}", static_cast<double>(relative_identity_error))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_a_bar_b_bar))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_tau_ab))
         << ',' << std::format("{:.17g}", static_cast<double>(rms_tau_ab))
         << ',' << std::format("{:.17g}", static_cast<double>(min_tau_ab))
         << ',' << std::format("{:.17g}", static_cast<double>(max_tau_ab))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_tau_aa))
         << ',' << std::format("{:.17g}", static_cast<double>(rms_tau_aa))
         << ',' << std::format("{:.17g}", static_cast<double>(min_tau_aa))
         << ',' << std::format("{:.17g}", static_cast<double>(max_tau_aa))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_tau_bb))
         << ',' << std::format("{:.17g}", static_cast<double>(rms_tau_bb))
         << ',' << std::format("{:.17g}", static_cast<double>(min_tau_bb))
         << ',' << std::format("{:.17g}", static_cast<double>(max_tau_bb))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_tau_zz))
         << ',' << std::format("{:.17g}", static_cast<double>(rms_tau_zz))
         << ',' << std::format("{:.17g}", static_cast<double>(min_tau_zz))
         << ',' << std::format("{:.17g}", static_cast<double>(max_tau_zz))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_rho_ab_sgs))
         << ',' << std::format("{:.17g}", static_cast<double>(rms_rho_ab_sgs))
         << ',' << std::format("{:.17g}", static_cast<double>(min_rho_ab_sgs))
         << ',' << std::format("{:.17g}", static_cast<double>(max_rho_ab_sgs))
         << ',' << std::format("{:.17g}", static_cast<double>(max_abs_tau_zz_identity_error))
         << ',' << std::format("{:.17g}", static_cast<double>(rms_tau_zz_identity_error))
         << ',' << std::format("{:.17g}", static_cast<double>(relative_tau_zz_identity_error))
         << ',' << std::format("{:.17g}", static_cast<double>(max_abs_zbar_linearity_error))
         << ',' << std::format("{:.17g}", static_cast<double>(relative_zbar_linearity_error))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_r_exact_filtered))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_r_les_naive))
         << ',' << std::format("{:.17g}", static_cast<double>(mean_r_sgs))
         << ',' << std::format("{:.17g}", static_cast<double>(les_reaction_efficiency))
         << ',' << std::format("{:.17g}", static_cast<double>(les_reaction_overprediction_factor))
         << ',' << std::format("{:.17g}", static_cast<double>(sgs_reaction_fraction))
         << ',' << std::format("{:.17g}", static_cast<double>(sgs_reaction_magnitude_fraction))
         << ',' << std::format("{:.17g}", static_cast<double>(max_abs_reaction_decomposition_error))
         << ',' << std::format("{:.17g}", static_cast<double>(relative_reaction_decomposition_error))
         << ',' << std::format("{:.17g}", static_cast<double>(covariance_identity_error))
         << ',' << std::format("{:.17g}", static_cast<double>(les_efficiency_identity_error))
         << std::endl;
    }
    file.flush();
}

void write_scalar_spectrum_outputs(
    const Config& config,
    const std::filesystem::path& output_root,
    int step,
    const ResolutionDiagnostics& resolution,
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b) {
#ifndef LB_CUBE_HAS_FFTW
    (void)config;
    (void)output_root;
    (void)step;
    (void)resolution;
    (void)fluid;
    (void)species_a;
    (void)species_b;
    throw std::runtime_error(
        "scalar spectrum output requested, but LB-Cube was built without FFTW");
#else
    const auto spectrum_start = std::chrono::high_resolution_clock::now();
    const std::filesystem::path output_dir =
        output_root / std::format("step_{:08}", step);
    std::filesystem::create_directories(output_dir);

    const auto fluid_view = fluid.get_current_view();
    const auto a_view = species_a.get_current_view();
    const auto b_view = species_b.get_current_view();
    const int nx = static_cast<int>(config.nx);
    const int nz = static_cast<int>(config.nz);
    const std::size_t real_plane_size = config.nx * config.nz;
    const std::size_t complex_z = config.nz / 2 + 1;
    const std::size_t spectrum_size = config.nx * complex_z;

    std::vector<double> input(real_plane_size);
    std::vector<double> velocity_plane_x(real_plane_size);
    std::vector<double> velocity_plane_y(real_plane_size);
    std::vector<double> velocity_plane_z(real_plane_size);
    std::vector<double> reactant_plane_a(real_plane_size);
    std::vector<double> reactant_plane_b(real_plane_size);
    std::vector<std::array<double, 2>> reactant_a_hat(spectrum_size);
    fftw_complex* output = static_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * spectrum_size));
    if (output == nullptr) {
        throw std::runtime_error("FFTW allocation failed for scalar spectrum output");
    }

    fftw_plan plan =
        fftw_plan_dft_r2c_2d(nx, nz, input.data(), output, FFTW_ESTIMATE);
    if (plan == nullptr) {
        fftw_free(output);
        throw std::runtime_error("FFTW plan creation failed for scalar spectrum output");
    }

    std::vector<long double> weighted_spectrum(spectrum_size, 0.0L);
    std::vector<long double> weighted_spectrum_ux(spectrum_size, 0.0L);
    std::vector<long double> weighted_spectrum_uy(spectrum_size, 0.0L);
    std::vector<long double> weighted_spectrum_uz(spectrum_size, 0.0L);
    std::vector<long double> weighted_cospectrum_ab(spectrum_size, 0.0L);
    long double sum_weights = 0.0L;
    long double weighted_scalar_variance = 0.0L;
    long double weighted_velocity_energy = 0.0L;
    long double weighted_velocity_energy_x = 0.0L;
    long double weighted_velocity_energy_y = 0.0L;
    long double weighted_velocity_energy_z = 0.0L;
    long double weighted_cov_ab = 0.0L;
    const long double inv_plane =
        1.0L / static_cast<long double>(real_plane_size);
    const long double inv_fft_norm =
        1.0L / static_cast<long double>(real_plane_size * real_plane_size);
    const auto hermitian_multiplicity = [nz = config.nz](std::size_t kz) {
        const bool has_distinct_negative_kz =
            kz != 0 && !(nz % 2 == 0 && kz == nz / 2);
        return has_distinct_negative_kz ? 2.0L : 1.0L;
    };
    const auto accumulate_transformed_energy =
        [complex_z, &output, inv_fft_norm, &hermitian_multiplicity](
            Real weight,
            std::vector<long double>& spectrum) {
            const std::size_t nx = spectrum.size() / complex_z;
            for (std::size_t kx = 0; kx < nx; ++kx) {
                for (std::size_t kz = 0; kz < complex_z; ++kz) {
                    const std::size_t index = kx * complex_z + kz;
                    const double real = output[index][0];
                    const double imag = output[index][1];
                    const long double magnitude2 =
                        static_cast<long double>(real * real + imag * imag);
                    const long double energy =
                        0.5L * hermitian_multiplicity(kz) * magnitude2 *
                        inv_fft_norm;
                    spectrum[index] += static_cast<long double>(weight) * energy;
                }
            }
        };
    const auto store_current_transform =
        [complex_z, &output, &reactant_a_hat]() {
            const std::size_t nx = reactant_a_hat.size() / complex_z;
            for (std::size_t kx = 0; kx < nx; ++kx) {
                for (std::size_t kz = 0; kz < complex_z; ++kz) {
                    const std::size_t index = kx * complex_z + kz;
                    reactant_a_hat[index] = {output[index][0], output[index][1]};
                }
            }
        };
    const auto accumulate_transformed_covariance =
        [complex_z,
         &output,
         inv_fft_norm,
         &hermitian_multiplicity,
         &reactant_a_hat](Real weight, std::vector<long double>& cospectrum) {
            const std::size_t nx = cospectrum.size() / complex_z;
            for (std::size_t kx = 0; kx < nx; ++kx) {
                for (std::size_t kz = 0; kz < complex_z; ++kz) {
                    const std::size_t index = kx * complex_z + kz;
                    const long double ar =
                        static_cast<long double>(reactant_a_hat[index][0]);
                    const long double ai =
                        static_cast<long double>(reactant_a_hat[index][1]);
                    const long double br = static_cast<long double>(output[index][0]);
                    const long double bi = static_cast<long double>(output[index][1]);
                    const long double real_cross = ar * br + ai * bi;
                    const long double covariance =
                        hermitian_multiplicity(kz) * real_cross * inv_fft_norm;
                    cospectrum[index] +=
                        static_cast<long double>(weight) * covariance;
                }
            }
        };

    for (std::size_t y = 0; y < config.ny; ++y) {
        long double sum_z = 0.0L;
        long double sum_ca = 0.0L;
        long double sum_cb = 0.0L;
        long double sum_ux = 0.0L;
        long double sum_uy = 0.0L;
        long double sum_uz = 0.0L;
        for (std::size_t z = 0; z < config.nz; ++z) {
            for (std::size_t x = 0; x < config.nx; ++x) {
                const Real concentration_a = concentration_at(a_view, x, y, z);
                const Real concentration_b = concentration_at(b_view, x, y, z);
                const Real mixture_fraction =
                    Real{0.5} *
                    (Real{1} + (concentration_a - concentration_b) / config.c0);
                const lbm::MacroState<FluidLattice, Real> macro =
                    macro_at<FluidLattice>(fluid_view, x, y, z);
                sum_z += static_cast<long double>(mixture_fraction);
                sum_ca += static_cast<long double>(concentration_a);
                sum_cb += static_cast<long double>(concentration_b);
                sum_ux += static_cast<long double>(macro.velocity[0]);
                sum_uy += static_cast<long double>(macro.velocity[1]);
                sum_uz += static_cast<long double>(macro.velocity[2]);
            }
        }

        const Real zbar = static_cast<Real>(sum_z * inv_plane);
        const Real abar = static_cast<Real>(sum_ca * inv_plane);
        const Real bbar = static_cast<Real>(sum_cb * inv_plane);
        const Real uxbar = static_cast<Real>(sum_ux * inv_plane);
        const Real uybar = static_cast<Real>(sum_uy * inv_plane);
        const Real uzbar = static_cast<Real>(sum_uz * inv_plane);
        const Real weight = Real{4} * zbar * (Real{1} - zbar);
        sum_weights += static_cast<long double>(weight);

        long double plane_zprime2 = 0.0L;
        long double plane_uxprime2 = 0.0L;
        long double plane_uyprime2 = 0.0L;
        long double plane_uzprime2 = 0.0L;
        long double plane_abprime = 0.0L;
        for (std::size_t x = 0; x < config.nx; ++x) {
            for (std::size_t z = 0; z < config.nz; ++z) {
                const std::size_t plane_index = x * config.nz + z;
                const Real concentration_a = concentration_at(a_view, x, y, z);
                const Real concentration_b = concentration_at(b_view, x, y, z);
                const Real mixture_fraction =
                    Real{0.5} *
                    (Real{1} + (concentration_a - concentration_b) / config.c0);
                const Real zprime = mixture_fraction - zbar;
                const lbm::MacroState<FluidLattice, Real> macro =
                    macro_at<FluidLattice>(fluid_view, x, y, z);
                const Real uxprime = macro.velocity[0] - uxbar;
                const Real uyprime = macro.velocity[1] - uybar;
                const Real uzprime = macro.velocity[2] - uzbar;
                const Real aprime = concentration_a - abar;
                const Real bprime = concentration_b - bbar;

                input[plane_index] = static_cast<double>(zprime);
                velocity_plane_x[plane_index] = static_cast<double>(uxprime);
                velocity_plane_y[plane_index] = static_cast<double>(uyprime);
                velocity_plane_z[plane_index] = static_cast<double>(uzprime);
                reactant_plane_a[plane_index] = static_cast<double>(aprime);
                reactant_plane_b[plane_index] = static_cast<double>(bprime);
                plane_zprime2 += static_cast<long double>(zprime * zprime);
                plane_uxprime2 += static_cast<long double>(uxprime * uxprime);
                plane_uyprime2 += static_cast<long double>(uyprime * uyprime);
                plane_uzprime2 += static_cast<long double>(uzprime * uzprime);
                plane_abprime += static_cast<long double>(aprime * bprime);
            }
        }

        const long double plane_variance =
            0.5L * plane_zprime2 * inv_plane;
        weighted_scalar_variance += static_cast<long double>(weight) * plane_variance;
        const long double plane_eux = 0.5L * plane_uxprime2 * inv_plane;
        const long double plane_euy = 0.5L * plane_uyprime2 * inv_plane;
        const long double plane_euz = 0.5L * plane_uzprime2 * inv_plane;
        weighted_velocity_energy_x += static_cast<long double>(weight) * plane_eux;
        weighted_velocity_energy_y += static_cast<long double>(weight) * plane_euy;
        weighted_velocity_energy_z += static_cast<long double>(weight) * plane_euz;
        weighted_velocity_energy +=
            static_cast<long double>(weight) * (plane_eux + plane_euy + plane_euz);
        weighted_cov_ab +=
            static_cast<long double>(weight) * plane_abprime * inv_plane;

        fftw_execute(plan);
        accumulate_transformed_energy(weight, weighted_spectrum);

        input = velocity_plane_x;
        fftw_execute(plan);
        accumulate_transformed_energy(weight, weighted_spectrum_ux);

        input = velocity_plane_y;
        fftw_execute(plan);
        accumulate_transformed_energy(weight, weighted_spectrum_uy);

        input = velocity_plane_z;
        fftw_execute(plan);
        accumulate_transformed_energy(weight, weighted_spectrum_uz);

        input = reactant_plane_a;
        fftw_execute(plan);
        store_current_transform();

        input = reactant_plane_b;
        fftw_execute(plan);
        accumulate_transformed_covariance(weight, weighted_cospectrum_ab);
    }

    if (sum_weights > 0.0L) {
        for (long double& value : weighted_spectrum) {
            value /= sum_weights;
        }
        for (long double& value : weighted_spectrum_ux) {
            value /= sum_weights;
        }
        for (long double& value : weighted_spectrum_uy) {
            value /= sum_weights;
        }
        for (long double& value : weighted_spectrum_uz) {
            value /= sum_weights;
        }
        for (long double& value : weighted_cospectrum_ab) {
            value /= sum_weights;
        }
        weighted_scalar_variance /= sum_weights;
        weighted_velocity_energy /= sum_weights;
        weighted_velocity_energy_x /= sum_weights;
        weighted_velocity_energy_y /= sum_weights;
        weighted_velocity_energy_z /= sum_weights;
        weighted_cov_ab /= sum_weights;
    } else {
        weighted_scalar_variance = 0.0L;
        weighted_velocity_energy = 0.0L;
        weighted_velocity_energy_x = 0.0L;
        weighted_velocity_energy_y = 0.0L;
        weighted_velocity_energy_z = 0.0L;
        weighted_cov_ab = 0.0L;
    }

    long double spectral_scalar_variance = 0.0L;
    for (long double value : weighted_spectrum) {
        spectral_scalar_variance += value;
    }

    std::vector<long double> weighted_spectrum_u(spectrum_size, 0.0L);
    long double spectral_velocity_energy_x = 0.0L;
    long double spectral_velocity_energy_y = 0.0L;
    long double spectral_velocity_energy_z = 0.0L;
    long double spectral_velocity_energy = 0.0L;
    for (std::size_t index = 0; index < spectrum_size; ++index) {
        weighted_spectrum_u[index] =
            weighted_spectrum_ux[index] +
            weighted_spectrum_uy[index] +
            weighted_spectrum_uz[index];
        spectral_velocity_energy_x += weighted_spectrum_ux[index];
        spectral_velocity_energy_y += weighted_spectrum_uy[index];
        spectral_velocity_energy_z += weighted_spectrum_uz[index];
        spectral_velocity_energy += weighted_spectrum_u[index];
    }

    long double spectral_cov_ab = 0.0L;
    long double spectral_cov_ab_abs_sum = 0.0L;
    for (long double value : weighted_cospectrum_ab) {
        spectral_cov_ab += value;
        spectral_cov_ab_abs_sum += std::abs(value);
    }

    std::vector<long double> spectrum_kx(config.nx, 0.0L);
    std::vector<long double> spectrum_kz(complex_z, 0.0L);
    std::vector<long double> spectrum_kh{};
    std::vector<long double> cospectrum_ab_kx(config.nx, 0.0L);
    std::vector<long double> cospectrum_ab_kz(complex_z, 0.0L);
    std::vector<long double> cospectrum_ab_kh{};
    std::vector<long double> spectrum_ux_kx(config.nx, 0.0L);
    std::vector<long double> spectrum_uy_kx(config.nx, 0.0L);
    std::vector<long double> spectrum_uz_kx(config.nx, 0.0L);
    std::vector<long double> spectrum_u_kx(config.nx, 0.0L);
    std::vector<long double> spectrum_ux_kz(complex_z, 0.0L);
    std::vector<long double> spectrum_uy_kz(complex_z, 0.0L);
    std::vector<long double> spectrum_uz_kz(complex_z, 0.0L);
    std::vector<long double> spectrum_u_kz(complex_z, 0.0L);
    std::vector<long double> spectrum_ux_kh{};
    std::vector<long double> spectrum_uy_kh{};
    std::vector<long double> spectrum_uz_kh{};
    std::vector<long double> spectrum_u_kh{};
    std::vector<std::size_t> radial_mode_count{};
    const Real radial_dk =
        Real{2} * std::numbers::pi_v<Real> / static_cast<Real>(config.nx);
    for (std::size_t i = 0; i < config.nx; ++i) {
        const int kx_index =
            i <= config.nx / 2 ? static_cast<int>(i)
                               : static_cast<int>(i) - static_cast<int>(config.nx);
        const Real kx =
            Real{2} * std::numbers::pi_v<Real> * static_cast<Real>(kx_index) /
            static_cast<Real>(config.nx);
        for (std::size_t kz_index = 0; kz_index < complex_z; ++kz_index) {
            const Real kz =
                Real{2} * std::numbers::pi_v<Real> *
                static_cast<Real>(kz_index) / static_cast<Real>(config.nz);
            const Real kh = std::sqrt(kx * kx + kz * kz);
            const std::size_t index = i * complex_z + kz_index;
            const long double energy = weighted_spectrum[index];
            const long double energy_ux = weighted_spectrum_ux[index];
            const long double energy_uy = weighted_spectrum_uy[index];
            const long double energy_uz = weighted_spectrum_uz[index];
            const long double energy_u = weighted_spectrum_u[index];
            const long double covariance = weighted_cospectrum_ab[index];
            const std::size_t shell =
                static_cast<std::size_t>(std::floor(kh / radial_dk + Real{0.5}));

            spectrum_kx[i] += energy;
            spectrum_kz[kz_index] += energy;
            cospectrum_ab_kx[i] += covariance;
            cospectrum_ab_kz[kz_index] += covariance;
            spectrum_ux_kx[i] += energy_ux;
            spectrum_uy_kx[i] += energy_uy;
            spectrum_uz_kx[i] += energy_uz;
            spectrum_u_kx[i] += energy_u;
            spectrum_ux_kz[kz_index] += energy_ux;
            spectrum_uy_kz[kz_index] += energy_uy;
            spectrum_uz_kz[kz_index] += energy_uz;
            spectrum_u_kz[kz_index] += energy_u;
            if (shell >= spectrum_kh.size()) {
                spectrum_kh.resize(shell + 1, 0.0L);
                cospectrum_ab_kh.resize(shell + 1, 0.0L);
                spectrum_ux_kh.resize(shell + 1, 0.0L);
                spectrum_uy_kh.resize(shell + 1, 0.0L);
                spectrum_uz_kh.resize(shell + 1, 0.0L);
                spectrum_u_kh.resize(shell + 1, 0.0L);
                radial_mode_count.resize(shell + 1, 0);
            }
            spectrum_kh[shell] += energy;
            cospectrum_ab_kh[shell] += covariance;
            spectrum_ux_kh[shell] += energy_ux;
            spectrum_uy_kh[shell] += energy_uy;
            spectrum_uz_kh[shell] += energy_uz;
            spectrum_u_kh[shell] += energy_u;
            ++radial_mode_count[shell];
        }
    }

    const auto sum_long_double = [](const auto& values) {
        long double sum = 0.0L;
        for (const auto value : values) {
            sum += static_cast<long double>(value);
        }
        return sum;
    };
    const long double sum_spectrum_kx = sum_long_double(spectrum_kx);
    const long double sum_spectrum_kz = sum_long_double(spectrum_kz);
    const long double sum_spectrum_kh = sum_long_double(spectrum_kh);
    const long double sum_cospectrum_ab_kx = sum_long_double(cospectrum_ab_kx);
    const long double sum_cospectrum_ab_kz = sum_long_double(cospectrum_ab_kz);
    const long double sum_cospectrum_ab_kh = sum_long_double(cospectrum_ab_kh);
    const long double sum_velocity_spectrum_kx = sum_long_double(spectrum_u_kx);
    const long double sum_velocity_spectrum_kz = sum_long_double(spectrum_u_kz);
    const long double sum_velocity_spectrum_kh = sum_long_double(spectrum_u_kh);
    const Real small_value = std::numeric_limits<Real>::min();
    const Real weighted_variance_real =
        static_cast<Real>(weighted_scalar_variance);
    const Real spectral_variance_real =
        static_cast<Real>(spectral_scalar_variance);
    const Real parseval_relative_error =
        std::abs(spectral_variance_real - weighted_variance_real) /
        std::max(std::abs(weighted_variance_real), small_value);
    const auto relative_difference_from_2d_sum =
        [spectral_scalar_variance, small_value](long double reduced_sum) {
            return static_cast<Real>(
                std::abs(reduced_sum - spectral_scalar_variance) /
                std::max(std::abs(spectral_scalar_variance),
                         static_cast<long double>(small_value)));
        };
    const Real kx_sum_relative_difference =
        relative_difference_from_2d_sum(sum_spectrum_kx);
    const Real kz_sum_relative_difference =
        relative_difference_from_2d_sum(sum_spectrum_kz);
    const Real kh_sum_relative_difference =
        relative_difference_from_2d_sum(sum_spectrum_kh);
    const auto velocity_relative_difference_from_2d_sum =
        [spectral_velocity_energy, small_value](long double reduced_sum) {
            return static_cast<Real>(
                std::abs(reduced_sum - spectral_velocity_energy) /
                std::max(std::abs(spectral_velocity_energy),
                         static_cast<long double>(small_value)));
        };
    const auto component_parseval_error =
        [small_value](long double spectral_energy, long double physical_energy) {
            return static_cast<Real>(
                std::abs(spectral_energy - physical_energy) /
                std::max(std::abs(physical_energy),
                         static_cast<long double>(small_value)));
        };
    const Real velocity_parseval_relative_error =
        component_parseval_error(spectral_velocity_energy, weighted_velocity_energy);
    const Real velocity_ux_parseval_relative_error =
        component_parseval_error(spectral_velocity_energy_x, weighted_velocity_energy_x);
    const Real velocity_uy_parseval_relative_error =
        component_parseval_error(spectral_velocity_energy_y, weighted_velocity_energy_y);
    const Real velocity_uz_parseval_relative_error =
        component_parseval_error(spectral_velocity_energy_z, weighted_velocity_energy_z);
    const Real velocity_kx_sum_relative_difference =
        velocity_relative_difference_from_2d_sum(sum_velocity_spectrum_kx);
    const Real velocity_kz_sum_relative_difference =
        velocity_relative_difference_from_2d_sum(sum_velocity_spectrum_kz);
    const Real velocity_kh_sum_relative_difference =
        velocity_relative_difference_from_2d_sum(sum_velocity_spectrum_kh);
    long double covariance_error_scale = std::abs(weighted_cov_ab);
    covariance_error_scale =
        std::max(covariance_error_scale, spectral_cov_ab_abs_sum);
    covariance_error_scale =
        std::max(covariance_error_scale, static_cast<long double>(small_value));
    const long double covariance_parseval_absolute_error =
        std::abs(spectral_cov_ab - weighted_cov_ab);
    const Real covariance_parseval_normalized_error =
        static_cast<Real>(covariance_parseval_absolute_error / covariance_error_scale);
    const Real kx_covariance_conservation_error =
        static_cast<Real>(
            std::abs(sum_cospectrum_ab_kx - spectral_cov_ab) /
            covariance_error_scale);
    const Real kz_covariance_conservation_error =
        static_cast<Real>(
            std::abs(sum_cospectrum_ab_kz - spectral_cov_ab) /
            covariance_error_scale);
    const Real kh_covariance_conservation_error =
        static_cast<Real>(
            std::abs(sum_cospectrum_ab_kh - spectral_cov_ab) /
            covariance_error_scale);
    std::vector<long double> cumulative_cospectrum_ab_resolved(
        cospectrum_ab_kh.size(),
        0.0L);
    std::vector<long double> cumulative_cospectrum_ab_unresolved(
        cospectrum_ab_kh.size(),
        0.0L);
    long double cumulative_covariance = 0.0L;
    long double cumulative_covariance_max_conservation_error = 0.0L;
    for (std::size_t shell = 0; shell < cospectrum_ab_kh.size(); ++shell) {
        cumulative_covariance += cospectrum_ab_kh[shell];
        cumulative_cospectrum_ab_resolved[shell] = cumulative_covariance;
        cumulative_cospectrum_ab_unresolved[shell] =
            sum_cospectrum_ab_kh - cumulative_covariance;
        cumulative_covariance_max_conservation_error = std::max(
            cumulative_covariance_max_conservation_error,
            std::abs(
                cumulative_cospectrum_ab_resolved[shell] +
                cumulative_cospectrum_ab_unresolved[shell] -
                sum_cospectrum_ab_kh));
    }
    const long double covariance_fraction_threshold =
        static_cast<long double>(100.0) *
        static_cast<long double>(std::numeric_limits<Real>::epsilon());
    const auto multiply_if_finite = [](Real value, Real scale) {
        return std::isfinite(scale) ? value * scale
                                    : std::numeric_limits<Real>::infinity();
    };

    std::ofstream spectrum{output_dir / "spectrum_Z_2D.csv"};
    if (!spectrum) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "spectrum_Z_2D.csv").string());
    }
    spectrum << "kx_index,kz_index,kx,kz,kh,E_Z\n";
    for (std::size_t i = 0; i < config.nx; ++i) {
        const int kx_index =
            i <= config.nx / 2 ? static_cast<int>(i)
                               : static_cast<int>(i) - static_cast<int>(config.nx);
        const Real kx =
            Real{2} * std::numbers::pi_v<Real> * static_cast<Real>(kx_index) /
            static_cast<Real>(config.nx);
        for (std::size_t kz_index = 0; kz_index < complex_z; ++kz_index) {
            const Real kz =
                Real{2} * std::numbers::pi_v<Real> *
                static_cast<Real>(kz_index) / static_cast<Real>(config.nz);
            const Real kh = std::sqrt(kx * kx + kz * kz);
            const Real energy =
                static_cast<Real>(weighted_spectrum[i * complex_z + kz_index]);
            spectrum
                << kx_index << ',' << kz_index
                << ',' << std::format("{:.17g}", static_cast<double>(kx))
                << ',' << std::format("{:.17g}", static_cast<double>(kz))
                << ',' << std::format("{:.17g}", static_cast<double>(kh))
                << ',' << std::format("{:.17g}", static_cast<double>(energy))
                << '\n';
        }
    }
    spectrum.flush();
    spectrum.close();

    std::ofstream spectrum_kx_file{output_dir / "spectrum_Z_kx.csv"};
    if (!spectrum_kx_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "spectrum_Z_kx.csv").string());
    }
    spectrum_kx_file << "kx_index,kx,kx_etaK,kx_etaB,E_Z\n";
    for (std::size_t i = 0; i < config.nx; ++i) {
        const int kx_index =
            i <= config.nx / 2 ? static_cast<int>(i)
                               : static_cast<int>(i) - static_cast<int>(config.nx);
        const Real kx =
            Real{2} * std::numbers::pi_v<Real> * static_cast<Real>(kx_index) /
            static_cast<Real>(config.nx);
        spectrum_kx_file
            << kx_index
            << ',' << std::format("{:.17g}", static_cast<double>(kx))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kx, resolution.eta_k)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kx, resolution.eta_b)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_kx[i])))
            << '\n';
    }
    spectrum_kx_file.flush();
    spectrum_kx_file.close();

    std::ofstream spectrum_kz_file{output_dir / "spectrum_Z_kz.csv"};
    if (!spectrum_kz_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "spectrum_Z_kz.csv").string());
    }
    spectrum_kz_file << "kz_index,kz,kz_etaK,kz_etaB,E_Z\n";
    for (std::size_t kz_index = 0; kz_index < complex_z; ++kz_index) {
        const Real kz =
            Real{2} * std::numbers::pi_v<Real> *
            static_cast<Real>(kz_index) / static_cast<Real>(config.nz);
        spectrum_kz_file
            << kz_index
            << ',' << std::format("{:.17g}", static_cast<double>(kz))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kz, resolution.eta_k)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kz, resolution.eta_b)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_kz[kz_index])))
            << '\n';
    }
    spectrum_kz_file.flush();
    spectrum_kz_file.close();

    std::ofstream spectrum_kh_file{output_dir / "spectrum_Z_kh.csv"};
    if (!spectrum_kh_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "spectrum_Z_kh.csv").string());
    }
    spectrum_kh_file << "shell_index,kh_center,kh_etaK,kh_etaB,mode_count,E_Z\n";
    for (std::size_t shell = 0; shell < spectrum_kh.size(); ++shell) {
        const Real kh_center = static_cast<Real>(shell) * radial_dk;
        spectrum_kh_file
            << shell
            << ',' << std::format("{:.17g}", static_cast<double>(kh_center))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kh_center, resolution.eta_k)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kh_center, resolution.eta_b)))
            << ',' << radial_mode_count[shell]
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_kh[shell])))
            << '\n';
    }
    spectrum_kh_file.flush();
    spectrum_kh_file.close();

    std::ofstream velocity_2d_file{output_dir / "spectrum_U_2D.csv"};
    if (!velocity_2d_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "spectrum_U_2D.csv").string());
    }
    velocity_2d_file << "kx_index,kz_index,kx,kz,kh,E_ux,E_uy,E_uz,E_u\n";
    for (std::size_t i = 0; i < config.nx; ++i) {
        const int kx_index =
            i <= config.nx / 2 ? static_cast<int>(i)
                               : static_cast<int>(i) - static_cast<int>(config.nx);
        const Real kx =
            Real{2} * std::numbers::pi_v<Real> * static_cast<Real>(kx_index) /
            static_cast<Real>(config.nx);
        for (std::size_t kz_index = 0; kz_index < complex_z; ++kz_index) {
            const Real kz =
                Real{2} * std::numbers::pi_v<Real> *
                static_cast<Real>(kz_index) / static_cast<Real>(config.nz);
            const Real kh = std::sqrt(kx * kx + kz * kz);
            const std::size_t index = i * complex_z + kz_index;
            velocity_2d_file
                << kx_index << ',' << kz_index
                << ',' << std::format("{:.17g}", static_cast<double>(kx))
                << ',' << std::format("{:.17g}", static_cast<double>(kz))
                << ',' << std::format("{:.17g}", static_cast<double>(kh))
                << ',' << std::format("{:.17g}", static_cast<double>(
                       static_cast<Real>(weighted_spectrum_ux[index])))
                << ',' << std::format("{:.17g}", static_cast<double>(
                       static_cast<Real>(weighted_spectrum_uy[index])))
                << ',' << std::format("{:.17g}", static_cast<double>(
                       static_cast<Real>(weighted_spectrum_uz[index])))
                << ',' << std::format("{:.17g}", static_cast<double>(
                       static_cast<Real>(weighted_spectrum_u[index])))
                << '\n';
        }
    }
    velocity_2d_file.flush();
    velocity_2d_file.close();

    std::ofstream velocity_kx_file{output_dir / "spectrum_U_kx.csv"};
    if (!velocity_kx_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "spectrum_U_kx.csv").string());
    }
    velocity_kx_file << "kx_index,kx,kx_etaK,kx_etaB,E_ux,E_uy,E_uz,E_u\n";
    for (std::size_t i = 0; i < config.nx; ++i) {
        const int kx_index =
            i <= config.nx / 2 ? static_cast<int>(i)
                               : static_cast<int>(i) - static_cast<int>(config.nx);
        const Real kx =
            Real{2} * std::numbers::pi_v<Real> * static_cast<Real>(kx_index) /
            static_cast<Real>(config.nx);
        velocity_kx_file
            << kx_index
            << ',' << std::format("{:.17g}", static_cast<double>(kx))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kx, resolution.eta_k)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kx, resolution.eta_b)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_ux_kx[i])))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_uy_kx[i])))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_uz_kx[i])))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_u_kx[i])))
            << '\n';
    }
    velocity_kx_file.flush();
    velocity_kx_file.close();

    std::ofstream velocity_kz_file{output_dir / "spectrum_U_kz.csv"};
    if (!velocity_kz_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "spectrum_U_kz.csv").string());
    }
    velocity_kz_file << "kz_index,kz,kz_etaK,kz_etaB,E_ux,E_uy,E_uz,E_u\n";
    for (std::size_t kz_index = 0; kz_index < complex_z; ++kz_index) {
        const Real kz =
            Real{2} * std::numbers::pi_v<Real> *
            static_cast<Real>(kz_index) / static_cast<Real>(config.nz);
        velocity_kz_file
            << kz_index
            << ',' << std::format("{:.17g}", static_cast<double>(kz))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kz, resolution.eta_k)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kz, resolution.eta_b)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_ux_kz[kz_index])))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_uy_kz[kz_index])))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_uz_kz[kz_index])))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_u_kz[kz_index])))
            << '\n';
    }
    velocity_kz_file.flush();
    velocity_kz_file.close();

    std::ofstream velocity_kh_file{output_dir / "spectrum_U_kh.csv"};
    if (!velocity_kh_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "spectrum_U_kh.csv").string());
    }
    velocity_kh_file
        << "shell_index,kh_center,kh_etaK,kh_etaB,mode_count,E_ux,E_uy,E_uz,E_u\n";
    for (std::size_t shell = 0; shell < spectrum_u_kh.size(); ++shell) {
        const Real kh_center = static_cast<Real>(shell) * radial_dk;
        velocity_kh_file
            << shell
            << ',' << std::format("{:.17g}", static_cast<double>(kh_center))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kh_center, resolution.eta_k)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kh_center, resolution.eta_b)))
            << ',' << radial_mode_count[shell]
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_ux_kh[shell])))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_uy_kh[shell])))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_uz_kh[shell])))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(spectrum_u_kh[shell])))
            << '\n';
    }
    velocity_kh_file.flush();
    velocity_kh_file.close();

    std::ofstream cospectrum_2d_file{output_dir / "cospectrum_AB_2D.csv"};
    if (!cospectrum_2d_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "cospectrum_AB_2D.csv").string());
    }
    cospectrum_2d_file << "kx_index,kz_index,kx,kz,kh,C_AB\n";
    for (std::size_t i = 0; i < config.nx; ++i) {
        const int kx_index =
            i <= config.nx / 2 ? static_cast<int>(i)
                               : static_cast<int>(i) - static_cast<int>(config.nx);
        const Real kx =
            Real{2} * std::numbers::pi_v<Real> * static_cast<Real>(kx_index) /
            static_cast<Real>(config.nx);
        for (std::size_t kz_index = 0; kz_index < complex_z; ++kz_index) {
            const Real kz =
                Real{2} * std::numbers::pi_v<Real> *
                static_cast<Real>(kz_index) / static_cast<Real>(config.nz);
            const Real kh = std::sqrt(kx * kx + kz * kz);
            const std::size_t index = i * complex_z + kz_index;
            cospectrum_2d_file
                << kx_index << ',' << kz_index
                << ',' << std::format("{:.17g}", static_cast<double>(kx))
                << ',' << std::format("{:.17g}", static_cast<double>(kz))
                << ',' << std::format("{:.17g}", static_cast<double>(kh))
                << ',' << std::format("{:.17g}", static_cast<double>(
                       static_cast<Real>(weighted_cospectrum_ab[index])))
                << '\n';
        }
    }
    cospectrum_2d_file.flush();
    cospectrum_2d_file.close();

    std::ofstream cospectrum_kx_file{output_dir / "cospectrum_AB_kx.csv"};
    if (!cospectrum_kx_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "cospectrum_AB_kx.csv").string());
    }
    cospectrum_kx_file << "kx_index,kx,kx_etaK,kx_etaB,C_AB\n";
    for (std::size_t i = 0; i < config.nx; ++i) {
        const int kx_index =
            i <= config.nx / 2 ? static_cast<int>(i)
                               : static_cast<int>(i) - static_cast<int>(config.nx);
        const Real kx =
            Real{2} * std::numbers::pi_v<Real> * static_cast<Real>(kx_index) /
            static_cast<Real>(config.nx);
        cospectrum_kx_file
            << kx_index
            << ',' << std::format("{:.17g}", static_cast<double>(kx))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kx, resolution.eta_k)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kx, resolution.eta_b)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(cospectrum_ab_kx[i])))
            << '\n';
    }
    cospectrum_kx_file.flush();
    cospectrum_kx_file.close();

    std::ofstream cospectrum_kz_file{output_dir / "cospectrum_AB_kz.csv"};
    if (!cospectrum_kz_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "cospectrum_AB_kz.csv").string());
    }
    cospectrum_kz_file << "kz_index,kz,kz_etaK,kz_etaB,C_AB\n";
    for (std::size_t kz_index = 0; kz_index < complex_z; ++kz_index) {
        const Real kz =
            Real{2} * std::numbers::pi_v<Real> *
            static_cast<Real>(kz_index) / static_cast<Real>(config.nz);
        cospectrum_kz_file
            << kz_index
            << ',' << std::format("{:.17g}", static_cast<double>(kz))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kz, resolution.eta_k)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kz, resolution.eta_b)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(cospectrum_ab_kz[kz_index])))
            << '\n';
    }
    cospectrum_kz_file.flush();
    cospectrum_kz_file.close();

    std::ofstream cospectrum_kh_file{output_dir / "cospectrum_AB_kh.csv"};
    if (!cospectrum_kh_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "cospectrum_AB_kh.csv").string());
    }
    cospectrum_kh_file
        << "shell_index,kh_center,kh_etaK,kh_etaB,mode_count,C_AB\n";
    for (std::size_t shell = 0; shell < cospectrum_ab_kh.size(); ++shell) {
        const Real kh_center = static_cast<Real>(shell) * radial_dk;
        cospectrum_kh_file
            << shell
            << ',' << std::format("{:.17g}", static_cast<double>(kh_center))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kh_center, resolution.eta_k)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(kh_center, resolution.eta_b)))
            << ',' << radial_mode_count[shell]
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(cospectrum_ab_kh[shell])))
            << '\n';
    }
    cospectrum_kh_file.flush();
    cospectrum_kh_file.close();

    std::ofstream cumulative_cospectrum_kh_file{
        output_dir / "cumulative_cospectrum_AB_kh.csv"};
    if (!cumulative_cospectrum_kh_file) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error(
            "failed to open " +
            (output_dir / "cumulative_cospectrum_AB_kh.csv").string());
    }
    cumulative_cospectrum_kh_file
        << "shell_index,k_c,k_c_etaK,k_c_etaB,lambda_c,lambda_c_over_delta0,"
           "C_AB_resolved,C_AB_unresolved,resolved_covariance_fraction,"
           "unresolved_covariance_fraction,"
           "unresolved_covariance_magnitude_fraction\n";
    const bool has_safe_covariance_total =
        std::abs(sum_cospectrum_ab_kh) > covariance_fraction_threshold;
    for (std::size_t shell = 0; shell < cospectrum_ab_kh.size(); ++shell) {
        const Real k_c = static_cast<Real>(shell) * radial_dk;
        const Real lambda_c =
            k_c > Real{} ? Real{2} * std::numbers::pi_v<Real> / k_c : Real{};
        const long double resolved = cumulative_cospectrum_ab_resolved[shell];
        const long double unresolved = cumulative_cospectrum_ab_unresolved[shell];
        const Real resolved_fraction =
            has_safe_covariance_total
                ? static_cast<Real>(resolved / sum_cospectrum_ab_kh)
                : Real{};
        const Real unresolved_fraction =
            has_safe_covariance_total
                ? static_cast<Real>(unresolved / sum_cospectrum_ab_kh)
                : Real{};
        const Real unresolved_magnitude_fraction =
            has_safe_covariance_total
                ? static_cast<Real>(
                      std::abs(unresolved) / std::abs(sum_cospectrum_ab_kh))
                : Real{};

        cumulative_cospectrum_kh_file
            << shell
            << ',' << std::format("{:.17g}", static_cast<double>(k_c))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(k_c, resolution.eta_k)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   multiply_if_finite(k_c, resolution.eta_b)))
            << ',' << std::format("{:.17g}", static_cast<double>(lambda_c))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   config.delta0 > Real{} ? lambda_c / config.delta0 : Real{}))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(resolved)))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   static_cast<Real>(unresolved)))
            << ',' << std::format("{:.17g}", static_cast<double>(resolved_fraction))
            << ',' << std::format("{:.17g}", static_cast<double>(unresolved_fraction))
            << ',' << std::format("{:.17g}", static_cast<double>(
                   unresolved_magnitude_fraction))
            << '\n';
    }
    cumulative_cospectrum_kh_file.flush();
    cumulative_cospectrum_kh_file.close();

    const auto spectrum_stop = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> spectrum_elapsed =
        spectrum_stop - spectrum_start;

    std::ofstream metadata{output_dir / "spectrum_metadata.json"};
    if (!metadata) {
        fftw_destroy_plan(plan);
        fftw_free(output);
        throw std::runtime_error("failed to open " +
                                 (output_dir / "spectrum_metadata.json").string());
    }
    metadata
        << "{\n"
        << "  \"step\": " << step << ",\n"
        << "  \"time\": " << json_number(static_cast<Real>(step)) << ",\n"
        << "  \"Nx\": " << config.nx << ",\n"
        << "  \"Ny\": " << config.ny << ",\n"
        << "  \"Nz\": " << config.nz << ",\n"
        << "  \"fft_library\": \"FFTW\",\n"
        << "  \"fft_version\": \"" << fftw_version << "\",\n"
        << "  \"fft_transform_type\": \"2D real-to-complex x-z plane transform\",\n"
        << "  \"fft_planning_mode\": \"FFTW_ESTIMATE\",\n"
        << "  \"fft_normalization\": \"FFTW forward transform is unnormalized; E_plane = 0.5 * hermitian_multiplicity * |Zhat|^2 / (Nx*Nz)^2\",\n"
        << "  \"hermitian_multiplicity\": \"2 for stored kz modes with distinct negative-kz partners; 1 for kz=0 and Nyquist kz when Nz is even\",\n"
        << "  \"mixing_layer_weight\": \"w(y) = 4 * Zbar(y) * (1 - Zbar(y))\",\n"
        << "  \"sum_y_w\": " << json_number(static_cast<Real>(sum_weights)) << ",\n"
        << "  \"weighted_scalar_variance\": "
        << json_number(weighted_variance_real) << ",\n"
        << "  \"spectral_scalar_variance\": "
        << json_number(spectral_variance_real) << ",\n"
        << "  \"parseval_relative_error\": "
        << json_number(parseval_relative_error) << ",\n"
        << "  \"sum_E_Z_2D\": " << json_number(spectral_variance_real) << ",\n"
        << "  \"sum_E_Z_kx\": "
        << json_number(static_cast<Real>(sum_spectrum_kx)) << ",\n"
        << "  \"sum_E_Z_kz\": "
        << json_number(static_cast<Real>(sum_spectrum_kz)) << ",\n"
        << "  \"sum_E_Z_kh\": "
        << json_number(static_cast<Real>(sum_spectrum_kh)) << ",\n"
        << "  \"kx_sum_relative_difference\": "
        << json_number(kx_sum_relative_difference) << ",\n"
        << "  \"kz_sum_relative_difference\": "
        << json_number(kz_sum_relative_difference) << ",\n"
        << "  \"kh_sum_relative_difference\": "
        << json_number(kh_sum_relative_difference) << ",\n"
        << "  \"radial_shell_binning\": \"shell = floor(kh / (2*pi/Nx) + 0.5); intended for cubic domains with Nx=Nz\",\n"
        << "  \"eta_K\": " << json_number(resolution.eta_k) << ",\n"
        << "  \"eta_B\": " << json_number(resolution.eta_b) << ",\n"
        << "  \"weighted_velocity_fluctuation_energy\": "
        << json_number(static_cast<Real>(weighted_velocity_energy)) << ",\n"
        << "  \"spectral_velocity_fluctuation_energy\": "
        << json_number(static_cast<Real>(spectral_velocity_energy)) << ",\n"
        << "  \"velocity_parseval_relative_error\": "
        << json_number(velocity_parseval_relative_error) << ",\n"
        << "  \"weighted_Eux\": "
        << json_number(static_cast<Real>(weighted_velocity_energy_x)) << ",\n"
        << "  \"weighted_Euy\": "
        << json_number(static_cast<Real>(weighted_velocity_energy_y)) << ",\n"
        << "  \"weighted_Euz\": "
        << json_number(static_cast<Real>(weighted_velocity_energy_z)) << ",\n"
        << "  \"spectral_Eux\": "
        << json_number(static_cast<Real>(spectral_velocity_energy_x)) << ",\n"
        << "  \"spectral_Euy\": "
        << json_number(static_cast<Real>(spectral_velocity_energy_y)) << ",\n"
        << "  \"spectral_Euz\": "
        << json_number(static_cast<Real>(spectral_velocity_energy_z)) << ",\n"
        << "  \"velocity_ux_parseval_relative_error\": "
        << json_number(velocity_ux_parseval_relative_error) << ",\n"
        << "  \"velocity_uy_parseval_relative_error\": "
        << json_number(velocity_uy_parseval_relative_error) << ",\n"
        << "  \"velocity_uz_parseval_relative_error\": "
        << json_number(velocity_uz_parseval_relative_error) << ",\n"
        << "  \"sum_E_u_2D\": "
        << json_number(static_cast<Real>(spectral_velocity_energy)) << ",\n"
        << "  \"sum_E_u_kx\": "
        << json_number(static_cast<Real>(sum_velocity_spectrum_kx)) << ",\n"
        << "  \"sum_E_u_kz\": "
        << json_number(static_cast<Real>(sum_velocity_spectrum_kz)) << ",\n"
        << "  \"sum_E_u_kh\": "
        << json_number(static_cast<Real>(sum_velocity_spectrum_kh)) << ",\n"
        << "  \"velocity_kx_sum_relative_difference\": "
        << json_number(velocity_kx_sum_relative_difference) << ",\n"
        << "  \"velocity_kz_sum_relative_difference\": "
        << json_number(velocity_kz_sum_relative_difference) << ",\n"
        << "  \"velocity_kh_sum_relative_difference\": "
        << json_number(velocity_kh_sum_relative_difference) << ",\n"
        << "  \"weighted_cov_AB\": "
        << json_number(static_cast<Real>(weighted_cov_ab)) << ",\n"
        << "  \"spectral_cov_AB\": "
        << json_number(static_cast<Real>(spectral_cov_ab)) << ",\n"
        << "  \"covariance_parseval_absolute_error\": "
        << json_number(static_cast<Real>(covariance_parseval_absolute_error)) << ",\n"
        << "  \"covariance_parseval_normalized_error\": "
        << json_number(covariance_parseval_normalized_error) << ",\n"
        << "  \"sum_C_AB_kx\": "
        << json_number(static_cast<Real>(sum_cospectrum_ab_kx)) << ",\n"
        << "  \"sum_C_AB_kz\": "
        << json_number(static_cast<Real>(sum_cospectrum_ab_kz)) << ",\n"
        << "  \"sum_C_AB_kh\": "
        << json_number(static_cast<Real>(sum_cospectrum_ab_kh)) << ",\n"
        << "  \"C_AB_total\": "
        << json_number(static_cast<Real>(sum_cospectrum_ab_kh)) << ",\n"
        << "  \"cumulative_covariance_max_conservation_error\": "
        << json_number(static_cast<Real>(
               cumulative_covariance_max_conservation_error)) << ",\n"
        << "  \"kx_covariance_conservation_error\": "
        << json_number(kx_covariance_conservation_error) << ",\n"
        << "  \"kz_covariance_conservation_error\": "
        << json_number(kz_covariance_conservation_error) << ",\n"
        << "  \"kh_covariance_conservation_error\": "
        << json_number(kh_covariance_conservation_error) << ",\n"
        << "  \"spectrum_wall_time_seconds\": "
        << json_number(static_cast<Real>(spectrum_elapsed.count())) << "\n"
        << "}\n";
    metadata.flush();
    metadata.close();

    fftw_destroy_plan(plan);
    fftw_free(output);
#endif
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

void run_simulation(const Config& config) {
    lbm::LatticeMemory<FluidLattice, Real> fluid{config.nx, config.ny, config.nz};
    lbm::LatticeMemory<ScalarLattice, Real> species_a{config.nx, config.ny, config.nz};
    lbm::LatticeMemory<ScalarLattice, Real> species_b{config.nx, config.ny, config.nz};

    PerturbationDefinition perturbation{};
    RestartState restart_state{};
    if (config.restart_from.empty()) {
        perturbation = prepare_perturbation(config);
        initialize_fields(config, perturbation, fluid, species_a, species_b);
    } else {
        restart_state = read_checkpoint(
            config,
            perturbation,
            fluid,
            species_a,
            species_b);
        if (config.steps <= restart_state.step) {
            std::cout << "Restart checkpoint is already at step "
                      << restart_state.step
                      << "; requested final --steps is " << config.steps
                      << ". Nothing to do.\n"
                      << std::flush;
            return;
        }
    }
    print_recap(config, perturbation);
    write_metadata_json(config, perturbation, restart_state);

    std::ofstream statistics{
        "statistics_double_shear_3d.csv",
        restart_state.enabled ? std::ios::app : std::ios::trunc};
    if (!statistics) {
        throw std::runtime_error("failed to open statistics_double_shear_3d.csv");
    }
    if (!restart_state.enabled ||
        !std::filesystem::exists("statistics_double_shear_3d.csv") ||
        std::filesystem::file_size("statistics_double_shear_3d.csv") == 0) {
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
    }

    std::ofstream filter_statistics;
    if (filtering_enabled(config)) {
        filter_statistics.open(
            "filter_statistics_double_shear_3d.csv",
            restart_state.enabled ? std::ios::app : std::ios::trunc);
        if (!filter_statistics) {
            throw std::runtime_error(
                "failed to open filter_statistics_double_shear_3d.csv");
        }
        if (!restart_state.enabled ||
            !std::filesystem::exists("filter_statistics_double_shear_3d.csv") ||
            std::filesystem::file_size("filter_statistics_double_shear_3d.csv") == 0) {
            write_filter_statistics_header(filter_statistics);
        }
    }

    const Real omega_f = Real{1} / config.tau_f;
    const Real omega_s = Real{1} / config.tau_s;
    const std::filesystem::path vtk_dir{"vtk_double_shear_3d"};
    const std::filesystem::path profile_dir{"profiles_double_shear_3d"};
    const std::filesystem::path pdf_dir{"pdfs_double_shear_3d"};
    const std::filesystem::path spectrum_dir{"spectra_double_shear_3d"};
    const std::filesystem::path filter_conditional_dir{
        "filter_conditionals_double_shear_3d"};
    const std::filesystem::path filter_pdf_dir{"filter_pdfs_double_shear_3d"};
    const auto start = std::chrono::high_resolution_clock::now();

    FlowDiagnostics flow = compute_flow_diagnostics(config, fluid);
    Real kinetic_energy_previous =
        restart_state.enabled ? restart_state.kinetic_energy_previous
                              : flow.mean_kinetic_energy;
    int previous_kinetic_energy_step =
        restart_state.enabled ? restart_state.previous_kinetic_energy_step : 0;
    Real latest_kinetic_energy_decay_rate =
        restart_state.enabled ? restart_state.latest_kinetic_energy_decay_rate : Real{};
    ScalarBudgetDiagnostics scalar_budget{};

    ScalarDiagnostics scalar =
        compute_scalar_diagnostics(config, species_a, species_b);
    Real previous_var_z =
        restart_state.enabled ? restart_state.previous_var_z : scalar.var_z;
    int previous_statistics_step =
        restart_state.enabled ? restart_state.previous_statistics_step : 0;
    const Real initial_mean_z =
        restart_state.enabled ? restart_state.initial_mean_z : scalar.mean_z;
    const Real initial_mean_rho =
        restart_state.enabled ? restart_state.initial_mean_rho : flow.mean_rho;

    if (restart_state.enabled) {
        std::cout << "Restarted from " << restart_state.source_path.string()
                  << " at completed step " << restart_state.step
                  << "; continuing to absolute final step " << config.steps
                  << ".\n"
                  << std::flush;
    } else {
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
        if (config.vtk_freq > 0) {
            write_binary_vtk(config, vtk_dir, 0, fluid, species_a, species_b);
        }
        if (config.profile_freq > 0) {
            write_y_profile_csv(config, profile_dir, 0, fluid, species_a, species_b);
        }
        if (config.pdf_freq > 0) {
            write_pdf_outputs(config, pdf_dir, 0, species_a, species_b);
        }
        if (filtering_enabled(config)) {
            write_filter_statistics(
                filter_statistics,
                filter_conditional_dir,
                filter_pdf_dir,
                config,
                0,
                compute_resolution_diagnostics(config, flow),
                species_a,
                species_b);
        }
        if (config.spectrum_freq > 0) {
            write_scalar_spectrum_outputs(
                config,
                spectrum_dir,
                0,
                compute_resolution_diagnostics(config, flow),
                fluid,
                species_a,
                species_b);
        }
    }

    const int first_step = restart_state.enabled ? restart_state.step + 1 : 1;
    auto next_walltime_checkpoint = start +
        std::chrono::duration<double>(config.checkpoint_walltime_hours * 3600.0);
    int last_checkpoint_step = restart_state.enabled ? restart_state.step : -1;
    constexpr double checkpoint_write_estimate_floor_seconds = 1.0;
    double checkpoint_write_estimate_seconds = checkpoint_write_estimate_floor_seconds;
    bool graceful_walltime_exit = false;

    for (int step = first_step; step <= config.steps; ++step) {
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
            const int kinetic_delta_steps = step - previous_kinetic_energy_step;
            latest_kinetic_energy_decay_rate =
                kinetic_delta_steps > 0
                    ? (kinetic_energy_previous - flow.mean_kinetic_energy) /
                          static_cast<Real>(kinetic_delta_steps)
                    : Real{};
            kinetic_energy_previous = flow.mean_kinetic_energy;
            previous_kinetic_energy_step = step;

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
                "Step [{} / {}] Umax: {:.6g} Ek: {:.6g} Eperp: {:.6g} Efluc: {:.6g} Enst: {:.6g} eps: {:.6g} chiZ: {:.6g} budget: {:.6g} tauMix*: {:.6g} DaMix: {:.6g} dx/etaB: {:.6g} theta: {:.6g} dZ: {:.6g} ReTheta: {:.6g} Mach: {:.6g} dE/dt: {:.6g} MLUPS: {:.6g}\n",
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
                mlups)
                      << std::flush;
        }

        if (config.vtk_freq > 0 && step % config.vtk_freq == 0) {
            write_binary_vtk(config, vtk_dir, step, fluid, species_a, species_b);
        }
        if (config.profile_freq > 0 && step % config.profile_freq == 0) {
            write_y_profile_csv(config, profile_dir, step, fluid, species_a, species_b);
        }
        if (config.pdf_freq > 0 && step % config.pdf_freq == 0) {
            write_pdf_outputs(config, pdf_dir, step, species_a, species_b);
        }
        if (filtering_enabled(config) && step % config.filter_freq == 0) {
            const FlowDiagnostics filter_flow =
                step % config.stat_freq == 0 ? flow : compute_flow_diagnostics(config, fluid);
            write_filter_statistics(
                filter_statistics,
                filter_conditional_dir,
                filter_pdf_dir,
                config,
                step,
                compute_resolution_diagnostics(config, filter_flow),
                species_a,
                species_b);
        }
        if (config.spectrum_freq > 0 && step % config.spectrum_freq == 0) {
            const FlowDiagnostics spectrum_flow =
                step % config.stat_freq == 0 ? flow : compute_flow_diagnostics(config, fluid);
            write_scalar_spectrum_outputs(
                config,
                spectrum_dir,
                step,
                compute_resolution_diagnostics(config, spectrum_flow),
                fluid,
                species_a,
                species_b);
        }

        const auto write_checkpoint_for_step = [&](int checkpoint_step) {
            const CheckpointWriteResult result = write_checkpoint(
                config,
                perturbation,
                checkpoint_step,
                kinetic_energy_previous,
                previous_kinetic_energy_step,
                latest_kinetic_energy_decay_rate,
                previous_var_z,
                previous_statistics_step,
                initial_mean_z,
                initial_mean_rho,
                fluid,
                species_a,
                species_b);
            checkpoint_write_estimate_seconds = std::max(
                checkpoint_write_estimate_seconds,
                std::max(result.elapsed_seconds, checkpoint_write_estimate_floor_seconds));
            last_checkpoint_step = checkpoint_step;
            return result;
        };

        bool checkpoint_due = false;
        if (config.checkpoint_freq > 0 && step % config.checkpoint_freq == 0) {
            checkpoint_due = true;
        }
        if (config.checkpoint_walltime_hours > Real{}) {
            const auto now = std::chrono::high_resolution_clock::now();
            if (now >= next_walltime_checkpoint) {
                checkpoint_due = true;
                do {
                    next_walltime_checkpoint +=
                        std::chrono::duration<double>(
                            config.checkpoint_walltime_hours * 3600.0);
                } while (now >= next_walltime_checkpoint);
            }
        }
        if (checkpoint_due && step != last_checkpoint_step) {
            write_checkpoint_for_step(step);
        }

        if (config.max_walltime_hours > Real{} && step < config.steps) {
            const auto now = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> elapsed = now - start;
            const double max_walltime_seconds =
                static_cast<double>(config.max_walltime_hours) * 3600.0;
            const double safety_margin_seconds =
                static_cast<double>(config.walltime_safety_margin_hours) * 3600.0;
            if (elapsed.count() + checkpoint_write_estimate_seconds +
                    safety_margin_seconds >=
                max_walltime_seconds) {
                CheckpointWriteResult walltime_checkpoint{
                    .path = checkpoint_path_for_step(step),
                    .elapsed_seconds = 0.0};
                if (step != last_checkpoint_step) {
                    walltime_checkpoint = write_checkpoint_for_step(step);
                }
                const auto exit_now = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double> exit_elapsed = exit_now - start;
                std::cout << "Graceful walltime exit after completed step "
                          << step
                          << ": elapsed=" << exit_elapsed.count()
                          << " s, estimated_checkpoint_time="
                          << checkpoint_write_estimate_seconds
                          << " s, checkpoint="
                          << walltime_checkpoint.path.string()
                          << '\n'
                          << std::flush;
                graceful_walltime_exit = true;
                break;
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
    statistics.flush();
    if (filter_statistics.is_open()) {
        filter_statistics.flush();
        filter_statistics.close();
    }
    statistics.close();

    std::cout << (graceful_walltime_exit ? "Simulation stopped gracefully in "
                                         : "Simulation complete in ")
              << elapsed.count()
              << " s.\nStatistics: statistics_double_shear_3d.csv"
              << "\nMetadata: metadata_double_shear_3d.json"
              << "\nVTK directory: "
              << (config.vtk_freq > 0 ? vtk_dir.string() : "disabled") << '\n'
              << "Profile directory: "
              << (config.profile_freq > 0 ? profile_dir.string() : "disabled") << '\n'
              << "PDF directory: "
              << (config.pdf_freq > 0 ? pdf_dir.string() : "disabled") << '\n'
              << "Spectrum directory: "
              << (config.spectrum_freq > 0 ? spectrum_dir.string() : "disabled") << '\n'
              << "Filter statistics: "
              << (filtering_enabled(config)
                      ? "filter_statistics_double_shear_3d.csv"
                      : "disabled")
              << "\nFilter conditionals: "
              << (filtering_enabled(config)
                      ? filter_conditional_dir.string()
                      : "disabled")
              << "\nFilter PDFs: "
              << (filtering_enabled(config)
                      ? filter_pdf_dir.string()
                      : "disabled")
              << '\n'
              << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_arguments(argc, argv);
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

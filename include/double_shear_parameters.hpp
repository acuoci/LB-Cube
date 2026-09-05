#pragma once
/**
 * @file double_shear_parameters.hpp
 * @brief Parameter conversion utilities for the 3D reactive double-shear driver.
 *
 * The production double-shear executable supports either direct lattice
 * parameters or nondimensional physical inputs. This header keeps that
 * conversion logic independently testable without pulling in solver memory or
 * collision code.
 */

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace lbm::double_shear {

/**
 * @brief User-facing parameterization mode selected from explicit CLI options.
 */
enum class ParameterizationMode {
    /** @brief Direct lattice parameters: tau_f, tau_s, and k_react. */
    DirectLbm,
    /** @brief Nondimensional inputs: Re_delta, Sc, and Da_delta. */
    Physical
};

/**
 * @brief Convert a mode enum to stable metadata text.
 *
 * @param mode Resolved parameterization mode.
 * @return Human-readable mode name for console and metadata output.
 */
[[nodiscard]] constexpr const char* to_string(ParameterizationMode mode) noexcept {
    switch (mode) {
    case ParameterizationMode::DirectLbm:
        return "DIRECT LBM";
    case ParameterizationMode::Physical:
        return "PHYSICAL";
    }

    return "UNKNOWN";
}

/**
 * @brief Track which mutually exclusive physics options were explicitly supplied.
 *
 * Internal defaults for `tau_f`, `tau_s`, and `k_react` must not make physical
 * mode look mixed. These flags capture only command-line intent.
 */
struct ExplicitParameterFlags {
    bool tau_f{false};
    bool tau_s{false};
    bool k_react{false};
    bool re_delta{false};
    bool sc{false};
    bool da_delta{false};
};

/**
 * @brief Raw dimensional and nondimensional inputs before mode resolution.
 *
 * @tparam Real Floating-point precision used for parameter conversion.
 */
template <typename Real>
struct ParameterInputs {
    std::size_t nx{128};
    std::size_t ny{128};
    std::size_t nz{128};
    Real tau_f{0.505};
    Real tau_s{0.5005};
    Real k_react{0.1};
    Real u0{0.05};
    Real c0{1};
    Real delta_ratio{0.0625};
    Real re_delta{};
    Real sc{};
    Real da_delta{};
};

/**
 * @brief Complete numerical and physical representation used by the solver.
 *
 * @tparam Real Floating-point precision used for derived parameters.
 */
template <typename Real>
struct ResolvedParameters {
    ParameterizationMode mode{ParameterizationMode::DirectLbm};
    std::size_t nx{128};
    std::size_t ny{128};
    std::size_t nz{128};
    Real tau_f{0.505};
    Real tau_s{0.5005};
    Real k_react{0.1};
    Real u0{0.05};
    Real c0{1};
    Real delta_ratio{0.0625};
    Real delta0{8};
    Real delta_u{0.1};
    Real nu{};
    Real scalar_diffusivity{};
    Real re_delta{};
    Real sc{};
    Real da_delta{};
    Real tau_delta{};
    Real tau_chem{std::numeric_limits<Real>::infinity()};
};

/**
 * @brief Resolve direct or physical double-shear parameters into LBM values.
 *
 * Direct mode keeps `tau_f`, `tau_s`, and `k_react` unchanged and reconstructs
 * the nondimensional groups. Physical mode requires the complete
 * `(Re_delta, Sc, Da_delta)` triplet and computes the corresponding lattice
 * relaxation times and reaction rate for the selected grid and interface width.
 *
 * @tparam Real Floating-point precision used for parameter conversion.
 * @param input Raw input values, including defaults.
 * @param explicit_flags Flags marking which mutually exclusive mode options were
 * explicitly provided by the user.
 * @return Fully resolved physical and numerical parameter set.
 */
template <typename Real>
[[nodiscard]] ResolvedParameters<Real> resolve_parameters(
    const ParameterInputs<Real>& input,
    const ExplicitParameterFlags& explicit_flags) {
    if (input.nx == 0 || input.ny == 0 || input.nz == 0) {
        throw std::invalid_argument("Nx, Ny, and Nz must be positive");
    }
    if (input.u0 <= Real{}) {
        throw std::invalid_argument("U0 must be positive");
    }
    if (input.c0 <= Real{}) {
        throw std::invalid_argument("C0 must be positive");
    }
    if (input.delta_ratio <= Real{} || input.delta_ratio >= Real{0.25}) {
        throw std::invalid_argument("delta_ratio must satisfy 0 < delta_ratio < 0.25");
    }

    const bool physical_any =
        explicit_flags.re_delta || explicit_flags.sc || explicit_flags.da_delta;
    const bool physical_all =
        explicit_flags.re_delta && explicit_flags.sc && explicit_flags.da_delta;
    const bool direct_any =
        explicit_flags.tau_f || explicit_flags.tau_s || explicit_flags.k_react;

    if (physical_any && !physical_all) {
        throw std::invalid_argument(
            "physical parameter mode requires all three options: "
            "--Re_delta, --Sc, and --Da_delta");
    }
    if (physical_any && direct_any) {
        throw std::invalid_argument(
            "choose either physical parameter mode "
            "(--Re_delta, --Sc, --Da_delta) or direct LBM parameter mode "
            "(--tau_f, --tau_s, --k_react), but do not explicitly specify both");
    }

    ResolvedParameters<Real> resolved{};
    resolved.nx = input.nx;
    resolved.ny = input.ny;
    resolved.nz = input.nz;
    resolved.u0 = input.u0;
    resolved.c0 = input.c0;
    resolved.delta_ratio = input.delta_ratio;
    resolved.delta0 = input.delta_ratio * static_cast<Real>(input.ny);
    resolved.delta_u = Real{2} * input.u0;

    if (physical_any) {
        if (input.re_delta <= Real{}) {
            throw std::invalid_argument("Re_delta must be positive");
        }
        if (input.sc <= Real{}) {
            throw std::invalid_argument("Sc must be positive");
        }
        if (input.da_delta < Real{}) {
            throw std::invalid_argument("Da_delta must be non-negative");
        }

        resolved.mode = ParameterizationMode::Physical;
        resolved.re_delta = input.re_delta;
        resolved.sc = input.sc;
        resolved.da_delta = input.da_delta;
        resolved.nu = resolved.delta_u * resolved.delta0 / resolved.re_delta;
        resolved.scalar_diffusivity = resolved.nu / resolved.sc;
        resolved.k_react =
            resolved.da_delta * resolved.delta_u / (resolved.c0 * resolved.delta0);
        resolved.tau_f = Real{0.5} + Real{3} * resolved.nu;
        resolved.tau_s = Real{0.5} + Real{4} * resolved.scalar_diffusivity;
    } else {
        if (input.tau_f <= Real{0.5}) {
            throw std::invalid_argument("tau_f must be greater than 0.5");
        }
        if (input.tau_s <= Real{0.5}) {
            throw std::invalid_argument("tau_s must be greater than 0.5");
        }
        if (input.k_react < Real{}) {
            throw std::invalid_argument("k_react must be non-negative");
        }

        resolved.mode = ParameterizationMode::DirectLbm;
        resolved.tau_f = input.tau_f;
        resolved.tau_s = input.tau_s;
        resolved.k_react = input.k_react;
        resolved.nu = (resolved.tau_f - Real{0.5}) / Real{3};
        resolved.scalar_diffusivity = (resolved.tau_s - Real{0.5}) / Real{4};
        resolved.re_delta = resolved.delta_u * resolved.delta0 / resolved.nu;
        resolved.sc = resolved.nu / resolved.scalar_diffusivity;
        resolved.da_delta =
            resolved.k_react * resolved.c0 * resolved.delta0 / resolved.delta_u;
    }

    if (resolved.tau_f <= Real{0.5}) {
        throw std::invalid_argument("resolved tau_f must be greater than 0.5");
    }
    if (resolved.tau_s <= Real{0.5}) {
        throw std::invalid_argument("resolved tau_s must be greater than 0.5");
    }

    resolved.tau_delta = resolved.delta0 / resolved.delta_u;
    resolved.tau_chem =
        resolved.k_react > Real{} ?
        Real{1} / (resolved.k_react * resolved.c0) :
        std::numeric_limits<Real>::infinity();

    return resolved;
}

} // namespace lbm::double_shear

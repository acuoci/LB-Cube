#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "double_shear_parameters.hpp"

namespace {

using lbm::double_shear::ExplicitParameterFlags;
using lbm::double_shear::ParameterInputs;
using lbm::double_shear::ParameterizationMode;
using lbm::double_shear::resolve_parameters;

constexpr double tolerance = 1.0e-12;

[[nodiscard]] ParameterInputs<double> base_inputs(std::size_t ny) {
    ParameterInputs<double> input{};
    input.nx = ny;
    input.ny = ny;
    input.nz = ny;
    input.u0 = 0.05;
    input.c0 = 1.0;
    input.delta_ratio = 0.0625;
    return input;
}

[[nodiscard]] ExplicitParameterFlags physical_flags() {
    ExplicitParameterFlags flags{};
    flags.re_delta = true;
    flags.sc = true;
    flags.da_delta = true;
    return flags;
}

} // namespace

TEST(DoubleShearParameters, PhysicalModeNy256Conversion) {
    auto input = base_inputs(256);
    input.re_delta = 400.0;
    input.sc = 10.0;
    input.da_delta = 10.0;

    const auto resolved = resolve_parameters<double>(input, physical_flags());

    EXPECT_EQ(resolved.mode, ParameterizationMode::Physical);
    EXPECT_NEAR(resolved.delta0, 16.0, tolerance);
    EXPECT_NEAR(resolved.delta_u, 0.1, tolerance);
    EXPECT_NEAR(resolved.nu, 0.004, tolerance);
    EXPECT_NEAR(resolved.tau_f, 0.512, tolerance);
    EXPECT_NEAR(resolved.scalar_diffusivity, 0.0004, tolerance);
    EXPECT_NEAR(resolved.tau_s, 0.5016, tolerance);
    EXPECT_NEAR(resolved.k_react, 0.0625, tolerance);
}

TEST(DoubleShearParameters, PhysicalModeGridRefinementPreservesGroups) {
    for (const std::size_t ny : {std::size_t{128}, std::size_t{256}, std::size_t{512}}) {
        auto input = base_inputs(ny);
        input.re_delta = 400.0;
        input.sc = 10.0;
        input.da_delta = 10.0;

        const auto resolved = resolve_parameters<double>(input, physical_flags());
        const double reconstructed_re =
            resolved.delta_u * resolved.delta0 / resolved.nu;
        const double reconstructed_sc =
            resolved.nu / resolved.scalar_diffusivity;
        const double reconstructed_da =
            resolved.k_react * resolved.c0 * resolved.delta0 / resolved.delta_u;

        EXPECT_NEAR(reconstructed_re, 400.0, tolerance) << "Ny=" << ny;
        EXPECT_NEAR(reconstructed_sc, 10.0, tolerance) << "Ny=" << ny;
        EXPECT_NEAR(reconstructed_da, 10.0, tolerance) << "Ny=" << ny;
    }
}

TEST(DoubleShearParameters, DirectModeReconstructsPhysicalGroups) {
    auto input = base_inputs(256);
    input.tau_f = 0.512;
    input.tau_s = 0.5016;
    input.k_react = 0.0625;

    ExplicitParameterFlags flags{};
    flags.tau_f = true;
    flags.tau_s = true;
    flags.k_react = true;

    const auto resolved = resolve_parameters<double>(input, flags);

    EXPECT_EQ(resolved.mode, ParameterizationMode::DirectLbm);
    EXPECT_NEAR(resolved.re_delta, 400.0, tolerance);
    EXPECT_NEAR(resolved.sc, 10.0, tolerance);
    EXPECT_NEAR(resolved.da_delta, 10.0, tolerance);
}

TEST(DoubleShearParameters, IncompletePhysicalModeThrows) {
    auto input = base_inputs(256);
    input.re_delta = 400.0;
    input.sc = 10.0;

    ExplicitParameterFlags flags{};
    flags.re_delta = true;
    flags.sc = true;

    EXPECT_THROW(
        static_cast<void>(resolve_parameters<double>(input, flags)),
        std::invalid_argument);
}

TEST(DoubleShearParameters, MixedExplicitModesThrow) {
    auto input = base_inputs(256);
    input.re_delta = 400.0;
    input.sc = 10.0;
    input.da_delta = 10.0;
    input.tau_f = 0.512;

    auto flags = physical_flags();
    flags.tau_f = true;

    EXPECT_THROW(
        static_cast<void>(resolve_parameters<double>(input, flags)),
        std::invalid_argument);
}

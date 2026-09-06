#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "lattice_filter.hpp"

namespace {

using Real = double;

struct Field3D {
    std::size_t nx{};
    std::size_t ny{};
    std::size_t nz{};
    std::vector<Real> data;

    Field3D(std::size_t x, std::size_t y, std::size_t z)
        : nx{x},
          ny{y},
          nz{z},
          data(nx * ny * nz) {}

    [[nodiscard]] lbm::ScalarField3DView<Real> view() {
        return lbm::ScalarField3DView<Real>{data.data(), nz, ny, nx};
    }

    [[nodiscard]] lbm::ConstScalarField3DView<Real> view() const {
        return lbm::ConstScalarField3DView<Real>{data.data(), nz, ny, nx};
    }

    [[nodiscard]] Real& operator()(std::size_t z, std::size_t y, std::size_t x) {
        return data[(z * ny + y) * nx + x];
    }

    [[nodiscard]] const Real& operator()(
        std::size_t z,
        std::size_t y,
        std::size_t x) const {
        return data[(z * ny + y) * nx + x];
    }
};

[[nodiscard]] Real mean(const Field3D& field) {
    return std::reduce(field.data.begin(), field.data.end(), Real{}) /
           static_cast<Real>(field.data.size());
}

[[nodiscard]] Real max_abs_difference(const Field3D& a, const Field3D& b) {
    Real max_error{};
    for (std::size_t i = 0; i < a.data.size(); ++i) {
        max_error = std::max(max_error, std::abs(a.data[i] - b.data[i]));
    }
    return max_error;
}

[[nodiscard]] Real discrete_box_transfer_function(
    int wave_number,
    std::size_t extent,
    std::size_t n_filter) {
    const auto half_width = static_cast<int>((n_filter - 1) / 2);
    Real sum{};
    for (int offset = -half_width; offset <= half_width; ++offset) {
        const Real angle =
            Real{2} * std::numbers::pi_v<Real> *
            static_cast<Real>(wave_number * offset) /
            static_cast<Real>(extent);
        sum += std::cos(angle);
    }
    return sum / static_cast<Real>(n_filter);
}

[[nodiscard]] Field3D make_fourier_mode(
    std::size_t nx,
    std::size_t ny,
    std::size_t nz,
    int kx,
    int ky,
    int kz) {
    Field3D field{nx, ny, nz};
    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                const Real ax =
                    Real{2} * std::numbers::pi_v<Real> *
                    static_cast<Real>(kx * static_cast<int>(x)) /
                    static_cast<Real>(nx);
                const Real ay =
                    Real{2} * std::numbers::pi_v<Real> *
                    static_cast<Real>(ky * static_cast<int>(y)) /
                    static_cast<Real>(ny);
                const Real az =
                    Real{2} * std::numbers::pi_v<Real> *
                    static_cast<Real>(kz * static_cast<int>(z)) /
                    static_cast<Real>(nz);
                field(z, y, x) = std::cos(ax) * std::cos(ay) * std::cos(az);
            }
        }
    }
    return field;
}

} // namespace

TEST(BoxFilter3D, RejectsInvalidWidths) {
    Field3D input{4, 4, 4};
    Field3D output{4, 4, 4};

    EXPECT_THROW(
        lbm::box_filter_3d<Real>(input.view(), output.view(), 0),
        std::invalid_argument);
    EXPECT_THROW(
        lbm::box_filter_3d<Real>(input.view(), output.view(), 2),
        std::invalid_argument);
    EXPECT_THROW(
        lbm::box_filter_3d<Real>(input.view(), output.view(), 8),
        std::invalid_argument);
}

TEST(BoxFilter3D, ConstantFieldIsPreservedPointwise) {
    Field3D input{8, 7, 6};
    Field3D output{8, 7, 6};
    std::ranges::fill(input.data, 3.25);

    for (const std::size_t width : {std::size_t{1}, std::size_t{3}, std::size_t{5}, std::size_t{9}}) {
        lbm::box_filter_3d<Real>(input.view(), output.view(), width);
        for (const Real value : output.data) {
            EXPECT_DOUBLE_EQ(value, 3.25) << "width=" << width;
        }
    }
}

TEST(BoxFilter3D, IdentityWidthOneCopiesInputExactly) {
    Field3D input{9, 8, 7};
    Field3D output{9, 8, 7};
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] =
            std::sin(static_cast<Real>(i) * Real{0.17}) +
            Real{0.01} * static_cast<Real>(i % 11);
    }

    lbm::box_filter_3d<Real>(input.view(), output.view(), 1);

    for (std::size_t i = 0; i < input.data.size(); ++i) {
        EXPECT_DOUBLE_EQ(output.data[i], input.data[i]);
    }
}

TEST(BoxFilter3D, PreservesGlobalMeanForNonuniformPeriodicField) {
    Field3D input{13, 11, 9};
    Field3D output{13, 11, 9};
    for (std::size_t z = 0; z < input.nz; ++z) {
        for (std::size_t y = 0; y < input.ny; ++y) {
            for (std::size_t x = 0; x < input.nx; ++x) {
                input(z, y, x) =
                    std::sin(Real{2} * std::numbers::pi_v<Real> *
                             static_cast<Real>(x) / static_cast<Real>(input.nx)) +
                    Real{0.25} *
                        std::cos(Real{4} * std::numbers::pi_v<Real> *
                                 static_cast<Real>(y) / static_cast<Real>(input.ny)) +
                    Real{0.1} * static_cast<Real>((x + 2 * y + 3 * z) % 5);
            }
        }
    }

    for (const std::size_t width : {std::size_t{3}, std::size_t{5}, std::size_t{9}}) {
        lbm::box_filter_3d<Real>(input.view(), output.view(), width);
        const Real input_mean = mean(input);
        const Real filtered_mean = mean(output);
        const Real abs_error = std::abs(filtered_mean - input_mean);
        const Real rel_error =
            abs_error / std::max(std::abs(input_mean), Real{1.0e-30});

        EXPECT_NEAR(filtered_mean, input_mean, 1.0e-15) << "width=" << width;
        EXPECT_LT(abs_error, 1.0e-15) << "width=" << width;
        EXPECT_LT(rel_error, 1.0e-14) << "width=" << width;
    }
}

TEST(BoxFilter3D, WrapsAcrossAllPeriodicBoundaries) {
    Field3D input{4, 5, 6};
    Field3D output{4, 5, 6};
    input(0, 0, input.nx - 1) = 3.0;
    input(0, input.ny - 1, 0) = 5.0;
    input(input.nz - 1, 0, 0) = 7.0;
    input(input.nz - 1, input.ny - 1, input.nx - 1) = 11.0;

    lbm::box_filter_3d<Real>(input.view(), output.view(), 3);

    EXPECT_DOUBLE_EQ(output(0, 0, 0), (3.0 + 5.0 + 7.0 + 11.0) / 27.0);
    EXPECT_DOUBLE_EQ(
        output(input.nz - 1, input.ny - 1, input.nx - 1),
        (3.0 + 5.0 + 7.0 + 11.0) / 27.0);
}

TEST(BoxFilter3D, FourierModeMatchesDiscreteTransferFunction) {
    constexpr std::size_t nx = 32;
    constexpr std::size_t ny = 30;
    constexpr std::size_t nz = 28;
    constexpr int kx = 3;
    constexpr int ky = 2;
    constexpr int kz = 4;

    const Field3D input = make_fourier_mode(nx, ny, nz, kx, ky, kz);
    Field3D output{nx, ny, nz};
    Field3D expected{nx, ny, nz};

    for (const std::size_t width : {std::size_t{1}, std::size_t{3}, std::size_t{5}, std::size_t{9}}) {
        lbm::box_filter_3d<Real>(input.view(), output.view(), width);
        const Real expected_ratio =
            discrete_box_transfer_function(kx, nx, width) *
            discrete_box_transfer_function(ky, ny, width) *
            discrete_box_transfer_function(kz, nz, width);

        Real numerator{};
        Real denominator{};
        for (std::size_t i = 0; i < input.data.size(); ++i) {
            expected.data[i] = expected_ratio * input.data[i];
            numerator += output.data[i] * input.data[i];
            denominator += input.data[i] * input.data[i];
        }
        const Real measured_ratio = numerator / denominator;
        const Real ratio_rel_error =
            std::abs(measured_ratio - expected_ratio) /
            std::max(std::abs(expected_ratio), Real{1.0e-30});
        const Real max_pointwise_error = max_abs_difference(output, expected);

        EXPECT_NEAR(measured_ratio, expected_ratio, 1.0e-14)
            << "width=" << width;
        EXPECT_LT(ratio_rel_error, 1.0e-13) << "width=" << width;
        EXPECT_LT(max_pointwise_error, 2.0e-15) << "width=" << width;
    }
}

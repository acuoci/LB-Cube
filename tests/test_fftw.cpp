/**
 * @file test_fftw.cpp
 * @brief Minimal FFTW integration smoke test for optional spectral dependencies.
 */

#include <cmath>
#include <cstdlib>
#include <fftw3.h>
#include <iostream>
#include <numbers>

int main() {
#ifndef LB_CUBE_HAS_FFTW
    std::cerr << "LB_CUBE_HAS_FFTW was not defined for the FFTW test target\n";
    return EXIT_FAILURE;
#endif

    constexpr int n = 8;
    double* input = static_cast<double*>(fftw_malloc(sizeof(double) * n));
    fftw_complex* output =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (n / 2 + 1)));

    if (input == nullptr || output == nullptr) {
        fftw_free(input);
        fftw_free(output);
        std::cerr << "FFTW allocation failed\n";
        return EXIT_FAILURE;
    }

    for (int i = 0; i < n; ++i) {
        input[i] =
            std::sin(2.0 * std::numbers::pi * static_cast<double>(i) /
                     static_cast<double>(n));
    }

    fftw_plan plan = fftw_plan_dft_r2c_1d(n, input, output, FFTW_ESTIMATE);
    if (plan == nullptr) {
        fftw_free(input);
        fftw_free(output);
        std::cerr << "FFTW plan creation failed\n";
        return EXIT_FAILURE;
    }

    fftw_execute(plan);
    const double mode1_magnitude = std::hypot(output[1][0], output[1][1]);

    fftw_destroy_plan(plan);
    fftw_free(input);
    fftw_free(output);

    if (std::abs(mode1_magnitude - 4.0) > 1.0e-10) {
        std::cerr << "Unexpected FFT mode magnitude: " << mode1_magnitude << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "LB-Cube FFTW smoke test passed; |F[1]|=" << mode1_magnitude << '\n';
    return EXIT_SUCCESS;
}

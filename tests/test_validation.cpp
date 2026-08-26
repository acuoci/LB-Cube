/**
 * @file test_validation.cpp
 * @brief GoogleTest validation suite for mass conservation and vortex decay.
 *
 * These tests exercise the host-side LBM loop using equilibrium initialization,
 * periodic boundaries, and macroscopic reconstruction from populations. The goal
 * is to validate numerical invariants before comparing CPU and CUDA backends.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numbers>
#include <random>

#include "lattice_core.hpp"
#include "lattice_memory.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

namespace {

/**
 * @brief Initialize one cell with equilibrium populations.
 *
 * @tparam Lattice Lattice traits type satisfying `lbm::IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param view Mutable population view for the current buffer.
 * @param x Cell x coordinate.
 * @param y Cell y coordinate.
 * @param z Cell z coordinate, ignored for 2D lattices.
 * @param macro Density and velocity used to evaluate the equilibrium.
 */
template <lbm::IsLatticeModel Lattice, std::floating_point Real>
void initialize_equilibrium_cell(
    typename lbm::LatticeMemory<Lattice, Real>::View view,
    std::size_t x,
    std::size_t y,
    std::size_t z,
    const lbm::MacroState<Lattice, Real>& macro) {
    for (int i = 0; i < Lattice::Q; ++i) {
        const auto q = static_cast<std::size_t>(i);
        if constexpr (Lattice::D == 2) {
            view[q, y, x] = lbm::compute_equilibrium<Lattice, Real>(i, macro);
        } else {
            view[q, z, y, x] = lbm::compute_equilibrium<Lattice, Real>(i, macro);
        }
    }
}

/**
 * @brief Sum density over the whole domain by summing all populations.
 *
 * @tparam Lattice Lattice traits type satisfying `lbm::IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param mem Host population memory whose current buffer is inspected.
 * @return Total mass accumulated in extended precision.
 */
template <lbm::IsLatticeModel Lattice, std::floating_point Real>
long double total_mass(const lbm::LatticeMemory<Lattice, Real>& mem) {
    const auto view = mem.get_current_view();
    long double mass = 0.0L;

    if constexpr (Lattice::D == 2) {
        const std::size_t y_extent = view.extent(1);
        const std::size_t x_extent = view.extent(2);

        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                for (int i = 0; i < Lattice::Q; ++i) {
                    mass += static_cast<long double>(view[static_cast<std::size_t>(i), y, x]);
                }
            }
        }
    } else {
        const std::size_t z_extent = view.extent(1);
        const std::size_t y_extent = view.extent(2);
        const std::size_t x_extent = view.extent(3);

        for (std::size_t z = 0; z < z_extent; ++z) {
            for (std::size_t y = 0; y < y_extent; ++y) {
                for (std::size_t x = 0; x < x_extent; ++x) {
                    for (int i = 0; i < Lattice::Q; ++i) {
                        mass += static_cast<long double>(view[static_cast<std::size_t>(i), z, y, x]);
                    }
                }
            }
        }
    }

    return mass;
}

/**
 * @brief Initialize a D3Q19 domain with random low-Mach equilibrium states.
 *
 * @param mem D3Q19 double-precision memory to initialize.
 */
void initialize_random_d3q19(lbm::LatticeMemory<lbm::D3Q19, double>& mem) {
    std::mt19937_64 rng{0x5eed1234ULL};
    std::uniform_real_distribution<double> density_dist{0.95, 1.05};
    std::uniform_real_distribution<double> velocity_dist{-0.01, 0.01};

    auto view = mem.get_current_view();
    const std::size_t z_extent = view.extent(1);
    const std::size_t y_extent = view.extent(2);
    const std::size_t x_extent = view.extent(3);

    for (std::size_t z = 0; z < z_extent; ++z) {
        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                lbm::MacroState<lbm::D3Q19, double> macro{};
                macro.density = density_dist(rng);
                macro.velocity << velocity_dist(rng), velocity_dist(rng), velocity_dist(rng);

                initialize_equilibrium_cell<lbm::D3Q19, double>(view, x, y, z, macro);
            }
        }
    }
}

/**
 * @brief Initialize a D2Q9 Taylor-Green vortex for analytical decay validation.
 *
 * The density perturbation and velocity field match the weakly compressible
 * initialization used by the kinetic-energy decay test.
 *
 * @param mem D2Q9 double-precision memory to initialize.
 * @param wave_number Fundamental wave number `2 pi / L`.
 * @param initial_velocity Velocity amplitude.
 * @param sound_speed_squared Isothermal lattice sound speed squared.
 */
void initialize_taylor_green_d2q9(
    lbm::LatticeMemory<lbm::D2Q9, double>& mem,
    double wave_number,
    double initial_velocity,
    double sound_speed_squared) {
    const auto view = mem.get_current_view();
    const std::size_t y_extent = view.extent(1);
    const std::size_t x_extent = view.extent(2);

    for (std::size_t y = 0; y < y_extent; ++y) {
        const double phase_y = wave_number * static_cast<double>(y);
        for (std::size_t x = 0; x < x_extent; ++x) {
            const double phase_x = wave_number * static_cast<double>(x);
            const double density_perturbation =
                (initial_velocity * initial_velocity) / (4.0 * sound_speed_squared) *
                (std::cos(2.0 * phase_x) + std::cos(2.0 * phase_y));

            lbm::MacroState<lbm::D2Q9, double> macro{};
            macro.density = 1.0 - density_perturbation;
            macro.velocity << -initial_velocity * std::cos(phase_x) * std::sin(phase_y),
                initial_velocity * std::sin(phase_x) * std::cos(phase_y);

            initialize_equilibrium_cell<lbm::D2Q9, double>(view, x, y, 0, macro);
        }
    }
}

/**
 * @brief Compute total kinetic energy from reconstructed D2Q9 macroscopic fields.
 *
 * @param mem D2Q9 memory whose current populations are inspected.
 * @return `0.5 * sum rho |u|^2` over the domain.
 */
double kinetic_energy_d2q9(const lbm::LatticeMemory<lbm::D2Q9, double>& mem) {
    const auto view = mem.get_current_view();
    const std::size_t y_extent = view.extent(1);
    const std::size_t x_extent = view.extent(2);

    long double energy = 0.0L;
    for (std::size_t y = 0; y < y_extent; ++y) {
        for (std::size_t x = 0; x < x_extent; ++x) {
            std::array<double, static_cast<std::size_t>(lbm::D2Q9::Q)> local_pops{};

            for (int i = 0; i < lbm::D2Q9::Q; ++i) {
                local_pops[static_cast<std::size_t>(i)] = view[static_cast<std::size_t>(i), y, x];
            }

            const lbm::MacroState<lbm::D2Q9, double> macro =
                lbm::compute_macro_state<lbm::D2Q9, double>(local_pops);
            energy += static_cast<long double>(0.5 * macro.density * macro.velocity.squaredNorm());
        }
    }

    return static_cast<double>(energy);
}

} // namespace

/**
 * @brief Verify that periodic BGK stepping conserves total mass in 3D.
 */
TEST(LBMValidation, MassConservationD3Q19Cpu) {
    constexpr std::size_t x_extent = 32;
    constexpr std::size_t y_extent = 32;
    constexpr std::size_t z_extent = 32;
    constexpr int iterations = 100;
    constexpr double omega = 1.0;

    lbm::LatticeMemory<lbm::D3Q19, double> mem{x_extent, y_extent, z_extent};
    initialize_random_d3q19(mem);

    const double initial_mass = static_cast<double>(total_mass(mem));

    for (int step = 0; step < iterations; ++step) {
        lbm::step_cpu<lbm::D3Q19, double>(mem, omega);
    }

    const double final_mass = static_cast<double>(total_mass(mem));
    const double mass_tolerance =
        std::numeric_limits<double>::epsilon() * initial_mass * 1024.0;
    EXPECT_NEAR(initial_mass, final_mass, mass_tolerance);
}

/**
 * @brief Validate D2Q9 Taylor-Green kinetic-energy decay against theory.
 *
 * The analytical decay is `K(t) = K(0) exp(-4 nu k^2 t)` with
 * `nu = c_s^2 (tau - 0.5)`.
 */
TEST(LBMValidation, TaylorGreenVortexD2Q9Decay) {
    constexpr std::size_t domain_size = 64;
    constexpr int iterations = 1000;
    constexpr double sound_speed_squared = 1.0 / 3.0;
    constexpr double initial_velocity = 0.01;
    constexpr double relaxation_time = 0.8;
    constexpr double omega = 1.0 / relaxation_time;
    constexpr double viscosity = sound_speed_squared * (relaxation_time - 0.5);
    constexpr double wave_number =
        2.0 * std::numbers::pi_v<double> / static_cast<double>(domain_size);

    lbm::LatticeMemory<lbm::D2Q9, double> mem{domain_size, domain_size};
    initialize_taylor_green_d2q9(mem, wave_number, initial_velocity, sound_speed_squared);

    const double initial_energy = kinetic_energy_d2q9(mem);

    for (int step = 0; step < iterations; ++step) {
        lbm::step_cpu<lbm::D2Q9, double>(mem, omega);
    }

    const double final_energy = kinetic_energy_d2q9(mem);
    const double analytical_energy =
        initial_energy * std::exp(-4.0 * viscosity * wave_number * wave_number * iterations);

    EXPECT_NEAR(final_energy, analytical_energy, 1.0e-5);
}

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
#include "lattice_mrt.hpp"
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

/**
 * @brief Initialize a periodic D2Q9 shear wave varying only in y.
 *
 * The velocity field is `u_x = u_0 sin(k y)` and `u_y = 0` at uniform density.
 * This is the canonical transverse viscous diffusion benchmark for LBM.
 *
 * @param mem D2Q9 double-precision memory to initialize.
 * @param wave_number Fundamental wave number `2 pi / L_y`.
 * @param initial_velocity Initial transverse velocity amplitude.
 */
void initialize_shear_wave_d2q9(
    lbm::LatticeMemory<lbm::D2Q9, double>& mem,
    double wave_number,
    double initial_velocity) {
    auto view = mem.get_current_view();
    const std::size_t y_extent = view.extent(1);
    const std::size_t x_extent = view.extent(2);

    for (std::size_t y = 0; y < y_extent; ++y) {
        const double phase_y = wave_number * static_cast<double>(y);
        for (std::size_t x = 0; x < x_extent; ++x) {
            lbm::MacroState<lbm::D2Q9, double> macro{};
            macro.density = 1.0;
            macro.velocity << initial_velocity * std::sin(phase_y), 0.0;

            initialize_equilibrium_cell<lbm::D2Q9, double>(view, x, y, 0, macro);
        }
    }
}

/**
 * @brief Initialize a uniform D2Q9 fluid field used to advect a passive scalar.
 *
 * @param mem Fluid population memory to initialize.
 * @param velocity_x Uniform x velocity.
 * @param velocity_y Uniform y velocity.
 */
void initialize_uniform_fluid_d2q9(
    lbm::LatticeMemory<lbm::D2Q9, double>& mem,
    double velocity_x,
    double velocity_y) {
    auto view = mem.get_current_view();
    const std::size_t y_extent = view.extent(1);
    const std::size_t x_extent = view.extent(2);

    for (std::size_t y = 0; y < y_extent; ++y) {
        for (std::size_t x = 0; x < x_extent; ++x) {
            lbm::MacroState<lbm::D2Q9, double> macro{};
            macro.density = 1.0;
            macro.velocity << velocity_x, velocity_y;

            initialize_equilibrium_cell<lbm::D2Q9, double>(view, x, y, 0, macro);
        }
    }
}

/**
 * @brief Initialize D2Q5 scalar populations from a sinusoidal concentration field.
 *
 * @param mem Scalar population memory to initialize.
 * @param wave_number Fundamental x wave number.
 * @param base_concentration Mean scalar concentration.
 * @param amplitude Scalar-wave amplitude.
 * @param fluid_velocity Uniform advecting velocity used in scalar equilibrium.
 */
void initialize_scalar_sine_d2q5(
    lbm::LatticeMemory<lbm::D2Q5, double>& mem,
    double wave_number,
    double base_concentration,
    double amplitude,
    const Eigen::Matrix<double, lbm::D2Q5::D, 1>& fluid_velocity) {
    auto view = mem.get_current_view();
    const std::size_t y_extent = view.extent(1);
    const std::size_t x_extent = view.extent(2);

    for (std::size_t y = 0; y < y_extent; ++y) {
        for (std::size_t x = 0; x < x_extent; ++x) {
            const double concentration =
                base_concentration + amplitude *
                std::sin(wave_number * static_cast<double>(x));

            for (int i = 0; i < lbm::D2Q5::Q; ++i) {
                view[static_cast<std::size_t>(i), y, x] =
                    lbm::compute_scalar_equilibrium<lbm::D2Q5, double>(
                        i,
                        concentration,
                        fluid_velocity);
            }
        }
    }
}

/**
 * @brief Reconstruct concentration from the current D2Q5 scalar populations.
 *
 * @param view Read-only scalar population view.
 * @param x Cell x coordinate.
 * @param y Cell y coordinate.
 * @return Local scalar concentration.
 */
double concentration_d2q5(
    typename lbm::LatticeMemory<lbm::D2Q5, double>::ConstView view,
    std::size_t x,
    std::size_t y) {
    std::array<double, static_cast<std::size_t>(lbm::D2Q5::Q)> scalar_pops{};
    for (int i = 0; i < lbm::D2Q5::Q; ++i) {
        scalar_pops[static_cast<std::size_t>(i)] =
            view[static_cast<std::size_t>(i), y, x];
    }

    return lbm::compute_concentration<lbm::D2Q5, double>(scalar_pops);
}

/**
 * @brief Initialize D2Q5 scalar populations from a uniform concentration.
 *
 * @param mem Scalar population memory to initialize.
 * @param concentration Uniform concentration over the whole domain.
 * @param fluid_velocity Uniform advecting velocity used in scalar equilibrium.
 */
void initialize_uniform_scalar_d2q5(
    lbm::LatticeMemory<lbm::D2Q5, double>& mem,
    double concentration,
    const Eigen::Matrix<double, lbm::D2Q5::D, 1>& fluid_velocity) {
    auto view = mem.get_current_view();
    const std::size_t y_extent = view.extent(1);
    const std::size_t x_extent = view.extent(2);

    for (std::size_t y = 0; y < y_extent; ++y) {
        for (std::size_t x = 0; x < x_extent; ++x) {
            for (int i = 0; i < lbm::D2Q5::Q; ++i) {
                view[static_cast<std::size_t>(i), y, x] =
                    lbm::compute_scalar_equilibrium<lbm::D2Q5, double>(
                        i,
                        concentration,
                        fluid_velocity);
            }
        }
    }
}

/**
 * @brief Initialize D2Q5 scalar populations from a left/right concentration step.
 *
 * @param mem Scalar population memory to initialize.
 * @param left_concentration Concentration assigned for `x < split_x`.
 * @param right_concentration Concentration assigned for `x >= split_x`.
 * @param split_x Interface location separating the two plateaus.
 * @param fluid_velocity Uniform advecting velocity used in scalar equilibrium.
 */
void initialize_split_scalar_d2q5(
    lbm::LatticeMemory<lbm::D2Q5, double>& mem,
    double left_concentration,
    double right_concentration,
    std::size_t split_x,
    const Eigen::Matrix<double, lbm::D2Q5::D, 1>& fluid_velocity) {
    auto view = mem.get_current_view();
    const std::size_t y_extent = view.extent(1);
    const std::size_t x_extent = view.extent(2);

    for (std::size_t y = 0; y < y_extent; ++y) {
        for (std::size_t x = 0; x < x_extent; ++x) {
            const double concentration = x < split_x ? left_concentration : right_concentration;
            for (int i = 0; i < lbm::D2Q5::Q; ++i) {
                view[static_cast<std::size_t>(i), y, x] =
                    lbm::compute_scalar_equilibrium<lbm::D2Q5, double>(
                        i,
                        concentration,
                        fluid_velocity);
            }
        }
    }
}

/**
 * @brief Sum D2Q5 concentration over the current scalar field.
 *
 * @param mem Scalar memory whose current buffer is inspected.
 * @return Total concentration over the domain.
 */
double total_concentration_d2q5(const lbm::LatticeMemory<lbm::D2Q5, double>& mem) {
    const auto view = mem.get_current_view();
    const std::size_t y_extent = view.extent(1);
    const std::size_t x_extent = view.extent(2);

    long double total = 0.0L;
    for (std::size_t y = 0; y < y_extent; ++y) {
        for (std::size_t x = 0; x < x_extent; ++x) {
            total += static_cast<long double>(concentration_d2q5(view, x, y));
        }
    }

    return static_cast<double>(total);
}

/**
 * @brief Run the D2Q9 Taylor-Green kinetic-energy decay validation.
 *
 * @tparam CT Compile-time collision operator under test.
 */
template <lbm::CollisionType CT>
void run_taylor_green_test() {
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
        lbm::step_cpu<lbm::D2Q9, double, CT>(mem, omega);
    }

    const double final_energy = kinetic_energy_d2q9(mem);
    const double analytical_energy =
        initial_energy * std::exp(-4.0 * viscosity * wave_number * wave_number * iterations);

    EXPECT_NEAR(final_energy, analytical_energy, 1.0e-5);
}

/**
 * @brief Run the transverse shear-wave decay validation.
 *
 * @tparam CT Compile-time collision operator under test.
 */
template <lbm::CollisionType CT>
void run_shear_wave_decay_test() {
    constexpr std::size_t x_extent = 8;
    constexpr std::size_t y_extent = 64;
    constexpr int iterations = 500;
    constexpr double sound_speed_squared = 1.0 / 3.0;
    constexpr double initial_velocity = 0.05;
    constexpr double relaxation_time = 0.8;
    constexpr double omega = 1.0 / relaxation_time;
    constexpr double viscosity = sound_speed_squared * (relaxation_time - 0.5);
    constexpr double wave_number =
        2.0 * std::numbers::pi_v<double> / static_cast<double>(y_extent);

    lbm::LatticeMemory<lbm::D2Q9, double> mem{x_extent, y_extent};
    initialize_shear_wave_d2q9(mem, wave_number, initial_velocity);

    for (int step = 0; step < iterations; ++step) {
        lbm::step_cpu<lbm::D2Q9, double, CT>(mem, omega);
    }

    const auto view = mem.get_current_view();
    const double decay = std::exp(-viscosity * wave_number * wave_number * iterations);
    constexpr double density_tolerance =
        CT == lbm::CollisionType::MRT ? 1.0e-6 : 1.0e-12;

    for (std::size_t y = 0; y < y_extent; ++y) {
        const double analytical_ux =
            initial_velocity * std::sin(wave_number * static_cast<double>(y)) * decay;

        for (std::size_t x = 0; x < x_extent; ++x) {
            std::array<double, static_cast<std::size_t>(lbm::D2Q9::Q)> local_pops{};
            for (int i = 0; i < lbm::D2Q9::Q; ++i) {
                local_pops[static_cast<std::size_t>(i)] = view[static_cast<std::size_t>(i), y, x];
            }

            const lbm::MacroState<lbm::D2Q9, double> macro =
                lbm::compute_macro_state<lbm::D2Q9, double>(local_pops);

            EXPECT_NEAR(macro.velocity[0], analytical_ux, 1.0e-4);
            EXPECT_LT(std::abs(macro.velocity[1]), 1.0e-7);
            EXPECT_NEAR(macro.density, 1.0, density_tolerance);
        }
    }
}

/**
 * @brief Run a diffusive-scaling Taylor-Green vortex and return relative L2 error.
 *
 * The physical state is kept fixed while the lattice resolution changes. The
 * lattice velocity, viscosity, relaxation time, and number of time steps are
 * scaled so the comparison measures spatial convergence rather than a different
 * physical problem.
 *
 * @param n Number of nodes in both x and y directions.
 * @return Relative L2 error of velocity magnitude at the final physical time.
 */
template <lbm::CollisionType CT = lbm::CollisionType::BGK>
double run_taylor_green_error(int n) {
    constexpr int base_grid = 16;
    constexpr double physical_time = 1.0;
    constexpr double base_velocity = 0.05;
    constexpr double base_relaxation_time = 0.8;
    constexpr double sound_speed_squared = 1.0 / 3.0;

    const double q = static_cast<double>(n) / static_cast<double>(base_grid);
    const double lattice_velocity = base_velocity / q;
    const double lattice_viscosity =
        sound_speed_squared * (base_relaxation_time - 0.5) / q;
    const double relaxation_time = lattice_viscosity / sound_speed_squared + 0.5;
    const double omega = 1.0 / relaxation_time;
    const int time_steps = static_cast<int>(std::llround(
        physical_time * static_cast<double>(base_grid * base_grid) * q * q));
    const double wave_number =
        2.0 * std::numbers::pi_v<double> / static_cast<double>(n);

    lbm::LatticeMemory<lbm::D2Q9, double> mem{
        static_cast<std::size_t>(n),
        static_cast<std::size_t>(n)};
    initialize_taylor_green_d2q9(mem, wave_number, lattice_velocity, sound_speed_squared);

    for (int step = 0; step < time_steps; ++step) {
        lbm::step_cpu<lbm::D2Q9, double, CT>(mem, omega);
    }

    const auto view = mem.get_current_view();
    const double decay = std::exp(
        -2.0 * lattice_viscosity * wave_number * wave_number *
        static_cast<double>(time_steps));
    long double numerator = 0.0L;
    long double denominator = 0.0L;

    for (int y = 0; y < n; ++y) {
        const double phase_y = wave_number * static_cast<double>(y);
        for (int x = 0; x < n; ++x) {
            const double phase_x = wave_number * static_cast<double>(x);
            const double analytical_ux =
                -lattice_velocity * std::cos(phase_x) * std::sin(phase_y) * decay;
            const double analytical_uy =
                lattice_velocity * std::sin(phase_x) * std::cos(phase_y) * decay;
            const double analytical_speed =
                std::hypot(analytical_ux, analytical_uy);

            std::array<double, static_cast<std::size_t>(lbm::D2Q9::Q)> local_pops{};
            for (int i = 0; i < lbm::D2Q9::Q; ++i) {
                local_pops[static_cast<std::size_t>(i)] =
                    view[static_cast<std::size_t>(i), static_cast<std::size_t>(y), static_cast<std::size_t>(x)];
            }

            const lbm::MacroState<lbm::D2Q9, double> macro =
                lbm::compute_macro_state<lbm::D2Q9, double>(local_pops);
            const double simulated_speed = macro.velocity.norm();
            const double error = simulated_speed - analytical_speed;

            numerator += static_cast<long double>(error * error);
            denominator += static_cast<long double>(analytical_speed * analytical_speed);
        }
    }

    return std::sqrt(static_cast<double>(numerator / denominator));
}

/**
 * @brief Run the diffusive-scaling convergence check for a collision operator.
 *
 * @tparam CT Compile-time collision operator under test.
 */
template <lbm::CollisionType CT>
void run_spatial_convergence_test(double eoc_tolerance = 0.15) {
    const double error_16 = run_taylor_green_error<CT>(16);
    const double error_32 = run_taylor_green_error<CT>(32);
    const double error_64 = run_taylor_green_error<CT>(64);

    const double eoc_16_to_32 = std::log2(error_16 / error_32);
    const double eoc_32_to_64 = std::log2(error_32 / error_64);

    EXPECT_GT(eoc_16_to_32, 1.5);
    EXPECT_NEAR(eoc_32_to_64, 2.0, eoc_tolerance);
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
 * @brief Validate BGK viscous decay of a transverse D2Q9 shear wave.
 */
TEST(Validation, ShearWaveDecay_BGK) {
    run_shear_wave_decay_test<lbm::CollisionType::BGK>();
}

/**
 * @brief Validate TRT viscous decay of a transverse D2Q9 shear wave.
 */
TEST(Validation, ShearWaveDecay_TRT) {
    run_shear_wave_decay_test<lbm::CollisionType::TRT>();
}

/**
 * @brief Validate MRT viscous decay of a transverse D2Q9 shear wave.
 */
TEST(Validation, ShearWaveDecay_MRT) {
    run_shear_wave_decay_test<lbm::CollisionType::MRT>();
}

/**
 * @brief Validate BGK D2Q9 Taylor-Green kinetic-energy decay against theory.
 */
TEST(Validation, TaylorGreenVortex_BGK) {
    run_taylor_green_test<lbm::CollisionType::BGK>();
}

/**
 * @brief Validate TRT D2Q9 Taylor-Green kinetic-energy decay against theory.
 */
TEST(Validation, TaylorGreenVortex_TRT) {
    run_taylor_green_test<lbm::CollisionType::TRT>();
}

/**
 * @brief Validate MRT D2Q9 Taylor-Green kinetic-energy decay against theory.
 */
TEST(Validation, TaylorGreenVortex_MRT) {
    run_taylor_green_test<lbm::CollisionType::MRT>();
}

/**
 * @brief Verify that the hardcoded D2Q9 MRT transform and inverse are consistent.
 */
TEST(Validation, MRT_D2Q9_Transformation_Identity) {
    constexpr std::array<double, 9> f_original{
        1.125, 2.25, 3.375, 4.5, 5.625, 6.75, 7.875, 9.0, 10.125
    };

    const std::array<double, 9> moments =
        lbm::mrt::compute_moments_d2q9(f_original);
    const std::array<double, 9> f_restored =
        lbm::mrt::compute_populations_d2q9(moments);

    for (std::size_t i = 0; i < f_original.size(); ++i) {
        EXPECT_DOUBLE_EQ(f_original[i], f_restored[i]);
    }
}

/**
 * @brief Verify that the hardcoded D3Q19 MRT transform and inverse are consistent.
 */
TEST(Validation, MRT_D3Q19_Transformation_Identity) {
    constexpr std::array<double, 19> f_original{
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19
    };

    const std::array<double, 19> moments =
        lbm::mrt::compute_moments_d3q19(f_original);
    const std::array<double, 19> f_restored =
        lbm::mrt::compute_populations_d3q19(moments);

    for (std::size_t i = 0; i < f_original.size(); ++i) {
        EXPECT_NEAR(f_original[i], f_restored[i], 1.0e-12);
    }
}

/**
 * @brief Validate passive scalar advection and diffusion by a uniform D2Q9 flow.
 *
 * A sinusoidal D2Q5 scalar wave is transported by a uniform x velocity and
 * damped by scalar diffusivity `D = c_s^2 (tau_c - 0.5)`.
 */
TEST(LBMValidation, PassiveScalarAdvectionDiffusion) {
    constexpr std::size_t x_extent = 64;
    constexpr std::size_t y_extent = 8;
    constexpr int iterations = 200;
    constexpr double fluid_velocity_x = 0.05;
    constexpr double fluid_velocity_y = 0.0;
    constexpr double scalar_relaxation_time = 0.8;
    constexpr double omega_c = 1.0 / scalar_relaxation_time;
    constexpr double scalar_diffusivity =
        lbm::D2Q5::cs2 * (scalar_relaxation_time - 0.5);
    constexpr double base_concentration = 1.0;
    constexpr double concentration_amplitude = 0.1;
    constexpr double wave_number =
        2.0 * std::numbers::pi_v<double> / static_cast<double>(x_extent);

    lbm::LatticeMemory<lbm::D2Q9, double> fluid_mem{x_extent, y_extent};
    lbm::LatticeMemory<lbm::D2Q5, double> scalar_mem{x_extent, y_extent};

    initialize_uniform_fluid_d2q9(fluid_mem, fluid_velocity_x, fluid_velocity_y);
    const Eigen::Matrix<double, lbm::D2Q5::D, 1> fluid_velocity{
        fluid_velocity_x,
        fluid_velocity_y};
    initialize_scalar_sine_d2q5(
        scalar_mem,
        wave_number,
        base_concentration,
        concentration_amplitude,
        fluid_velocity);

    constexpr double fluid_omega = 1.0;
    for (int step = 0; step < iterations; ++step) {
        lbm::step_cpu<lbm::D2Q9, double>(fluid_mem, fluid_omega);
        lbm::step_scalar_cpu<lbm::D2Q9, lbm::D2Q5, double>(
            scalar_mem,
            fluid_mem,
            omega_c,
            0.0);
    }

    const auto scalar_view = scalar_mem.get_current_view();
    const double decay =
        std::exp(-scalar_diffusivity * wave_number * wave_number * iterations);
    const double displacement = fluid_velocity_x * static_cast<double>(iterations);

    for (std::size_t y = 0; y < y_extent; ++y) {
        for (std::size_t x = 0; x < x_extent; ++x) {
            const double concentration = concentration_d2q5(scalar_view, x, y);
            const double analytical_concentration =
                base_concentration +
                concentration_amplitude *
                    std::sin(wave_number * (static_cast<double>(x) - displacement)) *
                    decay;

            EXPECT_NEAR(concentration, analytical_concentration, 1.0e-4);
        }
    }
}

/**
 * @brief Validate fused A+B batch kinetics for initially equal reactants.
 *
 * With uniform concentrations and a resting fluid, streaming and diffusion
 * preserve spatial uniformity, so the fused ADR loop reduces to the analytical
 * second-order batch reactor `C(t) = C0 / (1 + C0 k t)`.
 */
TEST(LBMValidation, ReactionKineticsBatch) {
    constexpr std::size_t x_extent = 16;
    constexpr std::size_t y_extent = 16;
    constexpr int iterations = 200;
    constexpr double initial_concentration = 1.0;
    constexpr double reaction_rate = 0.005;
    constexpr double scalar_relaxation_time = 0.8;
    constexpr double omega_c = 1.0 / scalar_relaxation_time;

    lbm::LatticeMemory<lbm::D2Q9, double> fluid_mem{x_extent, y_extent};
    lbm::LatticeMemory<lbm::D2Q5, double> species_a_mem{x_extent, y_extent};
    lbm::LatticeMemory<lbm::D2Q5, double> species_b_mem{x_extent, y_extent};

    initialize_uniform_fluid_d2q9(fluid_mem, 0.0, 0.0);
    const Eigen::Matrix<double, lbm::D2Q5::D, 1> fluid_velocity{0.0, 0.0};
    initialize_uniform_scalar_d2q5(
        species_a_mem,
        initial_concentration,
        fluid_velocity);
    initialize_uniform_scalar_d2q5(
        species_b_mem,
        initial_concentration,
        fluid_velocity);

    for (int step = 0; step < iterations; ++step) {
        lbm::step_reaction_AB<lbm::D2Q9, lbm::D2Q5, double>(
            fluid_mem,
            species_a_mem,
            species_b_mem,
            omega_c,
            reaction_rate);
    }

    const auto species_a_view = species_a_mem.get_current_view();
    const auto species_b_view = species_b_mem.get_current_view();
    const double analytical_concentration =
        initial_concentration /
        (1.0 + initial_concentration * reaction_rate * static_cast<double>(iterations));

    EXPECT_NEAR(
        concentration_d2q5(species_a_view, 0, 0),
        analytical_concentration,
        1.0e-4);
    EXPECT_NEAR(
        concentration_d2q5(species_b_view, 0, 0),
        analytical_concentration,
        1.0e-4);
}

/**
 * @brief Verify segregated reactants diffuse into an interface and consume symmetrically.
 *
 * This is a compact smoke test for the practical reactive-mixing setup: A starts
 * on the left half, B starts on the right half, and the fused ADR loop should
 * consume both species by the same amount while keeping concentrations finite
 * and non-negative.
 */
TEST(LBMValidation, SegregatedReactiveMixingD2Q5Cpu) {
    constexpr std::size_t x_extent = 32;
    constexpr std::size_t y_extent = 32;
    constexpr int iterations = 20;
    constexpr double reaction_rate = 0.05;
    constexpr double scalar_relaxation_time = 0.8;
    constexpr double omega_c = 1.0 / scalar_relaxation_time;

    lbm::LatticeMemory<lbm::D2Q9, double> fluid_mem{x_extent, y_extent};
    lbm::LatticeMemory<lbm::D2Q5, double> species_a_mem{x_extent, y_extent};
    lbm::LatticeMemory<lbm::D2Q5, double> species_b_mem{x_extent, y_extent};

    initialize_uniform_fluid_d2q9(fluid_mem, 0.0, 0.0);
    const Eigen::Matrix<double, lbm::D2Q5::D, 1> fluid_velocity{0.0, 0.0};
    initialize_split_scalar_d2q5(species_a_mem, 1.0, 0.0, x_extent / 2, fluid_velocity);
    initialize_split_scalar_d2q5(species_b_mem, 0.0, 1.0, x_extent / 2, fluid_velocity);

    const double initial_mass_a = total_concentration_d2q5(species_a_mem);
    const double initial_mass_b = total_concentration_d2q5(species_b_mem);

    for (int step = 0; step < iterations; ++step) {
        lbm::step_reaction_AB<lbm::D2Q9, lbm::D2Q5, double>(
            fluid_mem,
            species_a_mem,
            species_b_mem,
            omega_c,
            reaction_rate);
    }

    const double final_mass_a = total_concentration_d2q5(species_a_mem);
    const double final_mass_b = total_concentration_d2q5(species_b_mem);
    EXPECT_LT(final_mass_a, initial_mass_a);
    EXPECT_LT(final_mass_b, initial_mass_b);
    EXPECT_NEAR(final_mass_a, final_mass_b, 1.0e-10);

    const auto species_a_view = species_a_mem.get_current_view();
    const auto species_b_view = species_b_mem.get_current_view();
    for (std::size_t y = 0; y < y_extent; ++y) {
        for (std::size_t x = 0; x < x_extent; ++x) {
            const double concentration_a = concentration_d2q5(species_a_view, x, y);
            const double concentration_b = concentration_d2q5(species_b_view, x, y);

            EXPECT_TRUE(std::isfinite(concentration_a));
            EXPECT_TRUE(std::isfinite(concentration_b));
            EXPECT_GE(concentration_a, -1.0e-12);
            EXPECT_GE(concentration_b, -1.0e-12);
        }
    }
}

/**
 * @brief Verify approximately second-order spatial convergence for D2Q9.
 */
TEST(Validation, SpatialConvergence) {
    run_spatial_convergence_test<lbm::CollisionType::BGK>();
}

/**
 * @brief Verify MRT keeps second-order spatial convergence for D2Q9.
 */
TEST(Validation, SpatialConvergence_MRT) {
    run_spatial_convergence_test<lbm::CollisionType::MRT>(0.35);
}

/**
 * @file simulate_reaction.cpp
 * @brief Practical 2D segregated reactive-mixing executable for the ADR solver.
 *
 * The program initializes a D2Q9 Taylor-Green vortex and two D2Q5 reactants
 * separated by a vertical interface. The fluid is advanced first, then both
 * scalar species are advanced and reacted with the fused `A + B -> C` ADR loop.
 * VTK output contains the fluid velocity, species concentrations, and local
 * reaction rate for direct inspection in ParaView.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <format>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "lattice_core.hpp"
#include "lattice_io.hpp"
#include "lattice_memory.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

#ifndef LB_CUBE_ENABLE_CUDA
#define LB_CUBE_ENABLE_CUDA 0
#endif

#if LB_CUBE_ENABLE_CUDA
#include "lattice_cuda.cuh"
#endif

namespace {

using FluidLattice = lbm::D2Q9;
using ScalarLattice = lbm::D2Q5;
using Real = double;

/**
 * @brief Runtime configuration for the segregated reactive-mixing case.
 */
struct Config {
    std::size_t nx{256};
    std::size_t ny{256};
    int steps{5000};
    int output_frequency{250};
    Real fluid_relaxation_time{0.6};
    Real scalar_relaxation_time{0.8};
    Real initial_velocity{0.04};
    Real reaction_rate{0.05};
#if LB_CUBE_ENABLE_CUDA
    bool use_gpu{true};
#else
    bool use_gpu{false};
#endif
};

/**
 * @brief Parse a positive size-valued command-line option.
 */
[[nodiscard]] std::size_t parse_size_arg(const std::vector<std::string>& args, std::size_t& i) {
    if (i + 1 >= args.size()) {
        throw std::invalid_argument("missing value for " + args[i]);
    }

    const std::size_t value = std::stoull(args[++i]);
    if (value == 0) {
        throw std::invalid_argument(args[i - 1] + " must be greater than zero");
    }

    return value;
}

/**
 * @brief Parse an integer-valued command-line option.
 */
[[nodiscard]] int parse_int_arg(const std::vector<std::string>& args, std::size_t& i) {
    if (i + 1 >= args.size()) {
        throw std::invalid_argument("missing value for " + args[i]);
    }

    return std::stoi(args[++i]);
}

/**
 * @brief Parse a floating-point command-line option.
 */
[[nodiscard]] Real parse_real_arg(const std::vector<std::string>& args, std::size_t& i) {
    if (i + 1 >= args.size()) {
        throw std::invalid_argument("missing value for " + args[i]);
    }

    return std::stod(args[++i]);
}

/**
 * @brief Parse command-line options for production or quick exploratory runs.
 */
[[nodiscard]] Config parse_arguments(int argc, char** argv) {
    Config config{};
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc));

    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];

        if (arg == "-nx") {
            config.nx = parse_size_arg(args, i);
        } else if (arg == "-ny") {
            config.ny = parse_size_arg(args, i);
        } else if (arg == "-steps") {
            config.steps = parse_int_arg(args, i);
        } else if (arg == "-out_freq") {
            config.output_frequency = parse_int_arg(args, i);
        } else if (arg == "-u0") {
            config.initial_velocity = parse_real_arg(args, i);
        } else if (arg == "-tau") {
            config.fluid_relaxation_time = parse_real_arg(args, i);
        } else if (arg == "-tau_c") {
            config.scalar_relaxation_time = parse_real_arg(args, i);
        } else if (arg == "-k_react") {
            config.reaction_rate = parse_real_arg(args, i);
        } else if (arg == "-cpu") {
            config.use_gpu = false;
        } else if (arg == "-gpu") {
#if LB_CUBE_ENABLE_CUDA
            config.use_gpu = true;
#else
            throw std::invalid_argument("GPU backend requested, but this executable was built without CUDA");
#endif
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (config.steps < 0) {
        throw std::invalid_argument("-steps must be non-negative");
    }

    if (config.output_frequency <= 0) {
        throw std::invalid_argument("-out_freq must be greater than zero");
    }

    if (config.fluid_relaxation_time <= Real{0.5} ||
        config.scalar_relaxation_time <= Real{0.5}) {
        throw std::invalid_argument("relaxation times must be greater than 0.5");
    }

    if (config.reaction_rate < Real{}) {
        throw std::invalid_argument("-k_react must be non-negative");
    }

    return config;
}

/**
 * @brief Compute the Taylor-Green macroscopic state used for fluid initialization.
 */
[[nodiscard]] lbm::MacroState<FluidLattice, Real> taylor_green_macro(
    std::size_t x,
    std::size_t y,
    std::size_t nx,
    std::size_t ny,
    Real initial_velocity) {
    const Real kx = Real{2} * std::numbers::pi_v<Real> / static_cast<Real>(nx);
    const Real ky = Real{2} * std::numbers::pi_v<Real> / static_cast<Real>(ny);
    const Real phase_x = kx * static_cast<Real>(x);
    const Real phase_y = ky * static_cast<Real>(y);
    const Real density_perturbation =
        (initial_velocity * initial_velocity) / (Real{4} * static_cast<Real>(FluidLattice::cs2)) *
        (std::cos(Real{2} * phase_x) + std::cos(Real{2} * phase_y));

    lbm::MacroState<FluidLattice, Real> macro{};
    macro.density = Real{1} - density_perturbation;
    macro.velocity << -initial_velocity * std::cos(phase_x) * std::sin(phase_y),
        initial_velocity * std::sin(phase_x) * std::cos(phase_y);
    return macro;
}

/**
 * @brief Initialize fluid and split scalar populations from equilibrium.
 */
void initialize_fields(
    lbm::LatticeMemory<FluidLattice, Real>& fluid,
    lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    lbm::LatticeMemory<ScalarLattice, Real>& species_b,
    Real initial_velocity) {
    auto fluid_view = fluid.get_current_view();
    auto a_view = species_a.get_current_view();
    auto b_view = species_b.get_current_view();

    for (std::size_t y = 0; y < fluid.y_extent(); ++y) {
        for (std::size_t x = 0; x < fluid.x_extent(); ++x) {
            const lbm::MacroState<FluidLattice, Real> macro =
                taylor_green_macro(x, y, fluid.x_extent(), fluid.y_extent(), initial_velocity);
            for (int i = 0; i < FluidLattice::Q; ++i) {
                fluid_view[static_cast<std::size_t>(i), y, x] =
                    lbm::compute_equilibrium<FluidLattice, Real>(i, macro);
            }

            const Real concentration_a = x < fluid.x_extent() / 2 ? Real{1} : Real{0};
            const Real concentration_b = x < fluid.x_extent() / 2 ? Real{0} : Real{1};
            for (int i = 0; i < ScalarLattice::Q; ++i) {
                const auto q = static_cast<std::size_t>(i);
                a_view[q, y, x] =
                    lbm::compute_scalar_equilibrium<ScalarLattice, Real>(
                        i,
                        concentration_a,
                        macro.velocity);
                b_view[q, y, x] =
                    lbm::compute_scalar_equilibrium<ScalarLattice, Real>(
                        i,
                        concentration_b,
                        macro.velocity);
            }
        }
    }
}

/**
 * @brief Gather a D2Q9 cell and reconstruct its macroscopic state.
 */
[[nodiscard]] lbm::MacroState<FluidLattice, Real> fluid_macro_at(
    typename lbm::LatticeMemory<FluidLattice, Real>::ConstView view,
    std::size_t x,
    std::size_t y) {
    std::array<Real, static_cast<std::size_t>(FluidLattice::Q)> populations{};
    for (int i = 0; i < FluidLattice::Q; ++i) {
        populations[static_cast<std::size_t>(i)] = view[static_cast<std::size_t>(i), y, x];
    }

    return lbm::compute_macro_state<FluidLattice, Real>(populations);
}

/**
 * @brief Gather a D2Q5 cell and reconstruct concentration.
 */
[[nodiscard]] Real concentration_at(
    typename lbm::LatticeMemory<ScalarLattice, Real>::ConstView view,
    std::size_t x,
    std::size_t y) {
    std::array<Real, static_cast<std::size_t>(ScalarLattice::Q)> populations{};
    for (int i = 0; i < ScalarLattice::Q; ++i) {
        populations[static_cast<std::size_t>(i)] = view[static_cast<std::size_t>(i), y, x];
    }

    return lbm::compute_concentration<ScalarLattice, Real>(populations);
}

/**
 * @brief Write fluid velocity, reactant concentrations, and reaction rate to VTK.
 */
void write_reaction_vtk(
    const lbm::LatticeMemory<FluidLattice, Real>& fluid,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_a,
    const lbm::LatticeMemory<ScalarLattice, Real>& species_b,
    std::size_t time_step,
    Real reaction_rate) {
    const std::string filename = std::format("reaction_{:06}.vtk", time_step);
    std::ofstream vtk{filename};
    if (!vtk) {
        throw std::runtime_error("failed to open " + filename);
    }

    const auto fluid_view = fluid.get_current_view();
    const auto a_view = species_a.get_current_view();
    const auto b_view = species_b.get_current_view();
    const std::size_t nx = fluid.x_extent();
    const std::size_t ny = fluid.y_extent();
    const std::size_t point_count = nx * ny;

    vtk << "# vtk DataFile Version 3.0\n";
    vtk << std::format("LB-Cube reactive mixing step {}\n", time_step);
    vtk << "ASCII\n";
    vtk << "DATASET STRUCTURED_POINTS\n";
    vtk << std::format("DIMENSIONS {} {} 1\n", nx, ny);
    vtk << "ORIGIN 0 0 0\n";
    vtk << "SPACING 1 1 1\n";
    vtk << std::format("POINT_DATA {}\n", point_count);

    vtk << "SCALARS C_A double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            vtk << std::format("{:.17g}\n", static_cast<double>(concentration_at(a_view, x, y)));
        }
    }

    vtk << "SCALARS C_B double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            vtk << std::format("{:.17g}\n", static_cast<double>(concentration_at(b_view, x, y)));
        }
    }

    vtk << "SCALARS reaction_rate double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            const Real concentration_a = concentration_at(a_view, x, y);
            const Real concentration_b = concentration_at(b_view, x, y);
            vtk << std::format(
                "{:.17g}\n",
                static_cast<double>(reaction_rate * concentration_a * concentration_b));
        }
    }

    vtk << "VECTORS velocity double\n";
    for (std::size_t y = 0; y < ny; ++y) {
        for (std::size_t x = 0; x < nx; ++x) {
            const lbm::MacroState<FluidLattice, Real> macro = fluid_macro_at(fluid_view, x, y);
            vtk << std::format(
                "{:.17g} {:.17g} 0\n",
                static_cast<double>(macro.velocity[0]),
                static_cast<double>(macro.velocity[1]));
        }
    }
}

/**
 * @brief Print a concise, flushed run summary.
 */
void print_summary(const Config& config, Real omega, Real omega_c) {
    std::cout << "LB-Cube D2Q9/D2Q5 reactive mixing\n"
              << "Grid: " << config.nx << " x " << config.ny << '\n'
              << "Steps: " << config.steps << '\n'
              << "Output frequency: " << config.output_frequency << '\n'
              << "Backend: " << (config.use_gpu ? "GPU" : "CPU") << '\n'
              << "Fluid omega: " << omega << '\n'
              << "Scalar omega: " << omega_c << '\n'
              << "Reaction rate: " << config.reaction_rate << '\n'
              << std::flush;
}

/**
 * @brief Flatten a 2D SoA current buffer for CUDA transfer.
 */
template <lbm::IsLatticeModel Lattice>
[[nodiscard]] std::vector<Real> flatten_current_populations(
    const lbm::LatticeMemory<Lattice, Real>& mem) {
    const auto view = mem.get_current_view();
    std::vector<Real> flat(mem.population_count());

    for (int i = 0; i < Lattice::Q; ++i) {
        const auto q = static_cast<std::size_t>(i);
        for (std::size_t y = 0; y < mem.y_extent(); ++y) {
            for (std::size_t x = 0; x < mem.x_extent(); ++x) {
                flat[(q * mem.y_extent() + y) * mem.x_extent() + x] = view[q, y, x];
            }
        }
    }

    return flat;
}

#if LB_CUBE_ENABLE_CUDA
/**
 * @brief Throw a contextual exception for failed CUDA runtime calls.
 */
void check_cuda(cudaError_t error, const std::string& context) {
    if (error != cudaSuccess) {
        throw std::runtime_error(context + ": " + cudaGetErrorString(error));
    }
}

/**
 * @brief RAII owner for ping-pong device population buffers.
 */
template <typename T>
class DevicePopulationBuffers {
public:
    explicit DevicePopulationBuffers(std::size_t count)
        : count_(count),
          bytes_(count * sizeof(T)) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&current), bytes_), "cudaMalloc current");
        try {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&next), bytes_), "cudaMalloc next");
        } catch (...) {
            cudaFree(current);
            current = nullptr;
            throw;
        }
    }

    DevicePopulationBuffers(const DevicePopulationBuffers&) = delete;
    DevicePopulationBuffers& operator=(const DevicePopulationBuffers&) = delete;

    ~DevicePopulationBuffers() {
        cudaFree(current);
        cudaFree(next);
    }

    [[nodiscard]] std::size_t count() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    T* current{};
    T* next{};

private:
    std::size_t count_{};
    std::size_t bytes_{};
};

/**
 * @brief Copy a flattened device buffer back into a 2D host mdspan.
 */
template <lbm::IsLatticeModel Lattice>
void overwrite_current_populations(
    lbm::LatticeMemory<Lattice, Real>& mem,
    const std::vector<Real>& flat) {
    auto view = mem.get_current_view();

    for (int i = 0; i < Lattice::Q; ++i) {
        const auto q = static_cast<std::size_t>(i);
        for (std::size_t y = 0; y < mem.y_extent(); ++y) {
            for (std::size_t x = 0; x < mem.x_extent(); ++x) {
                view[q, y, x] = flat[(q * mem.y_extent() + y) * mem.x_extent() + x];
            }
        }
    }
}

/**
 * @brief Refresh a host memory object from the active device read buffer.
 */
template <lbm::IsLatticeModel Lattice>
void copy_device_current_to_host(
    DevicePopulationBuffers<Real>& device_buffers,
    lbm::LatticeMemory<Lattice, Real>& mem,
    std::vector<Real>& staging) {
    staging.resize(device_buffers.count());
    check_cuda(
        cudaMemcpy(staging.data(), device_buffers.current, device_buffers.bytes(), cudaMemcpyDeviceToHost),
        "cudaMemcpy device to host");
    overwrite_current_populations<Lattice>(mem, staging);
}
#endif

} // namespace

/**
 * @brief Run the segregated reactive-mixing simulation.
 */
int main(int argc, char** argv) {
    try {
        const Config config = parse_arguments(argc, argv);
        const Real omega = Real{1} / config.fluid_relaxation_time;
        const Real omega_c = Real{1} / config.scalar_relaxation_time;

        lbm::LatticeMemory<FluidLattice, Real> fluid{config.nx, config.ny};
        lbm::LatticeMemory<ScalarLattice, Real> species_a{config.nx, config.ny};
        lbm::LatticeMemory<ScalarLattice, Real> species_b{config.nx, config.ny};
        initialize_fields(fluid, species_a, species_b, config.initial_velocity);

        std::ofstream diagnostics{"reaction_diagnostics.csv"};
        if (!diagnostics) {
            throw std::runtime_error("failed to open reaction_diagnostics.csv");
        }
        diagnostics
            << "step,mean_A,mean_B,var_A,var_B,covariance,"
            << "segregation_intensity,true_reaction_rate,mixed_reaction_rate\n";

        print_summary(config, omega, omega_c);

#if LB_CUBE_ENABLE_CUDA
        std::vector<Real> fluid_staging = flatten_current_populations<FluidLattice>(fluid);
        std::vector<Real> a_staging = flatten_current_populations<ScalarLattice>(species_a);
        std::vector<Real> b_staging = flatten_current_populations<ScalarLattice>(species_b);
        std::unique_ptr<DevicePopulationBuffers<Real>> fluid_device;
        std::unique_ptr<DevicePopulationBuffers<Real>> a_device;
        std::unique_ptr<DevicePopulationBuffers<Real>> b_device;

        if (config.use_gpu) {
            fluid_device = std::make_unique<DevicePopulationBuffers<Real>>(fluid.population_count());
            a_device = std::make_unique<DevicePopulationBuffers<Real>>(species_a.population_count());
            b_device = std::make_unique<DevicePopulationBuffers<Real>>(species_b.population_count());

            check_cuda(
                cudaMemcpy(fluid_device->current, fluid_staging.data(), fluid_device->bytes(), cudaMemcpyHostToDevice),
                "cudaMemcpy fluid host to device");
            check_cuda(
                cudaMemcpy(a_device->current, a_staging.data(), a_device->bytes(), cudaMemcpyHostToDevice),
                "cudaMemcpy species A host to device");
            check_cuda(
                cudaMemcpy(b_device->current, b_staging.data(), b_device->bytes(), cudaMemcpyHostToDevice),
                "cudaMemcpy species B host to device");
        }
#endif

        write_reaction_vtk(fluid, species_a, species_b, 0, config.reaction_rate);
        lbm::log_reactive_diagnostics(
            diagnostics,
            0,
            lbm::compute_reactive_stats<ScalarLattice, Real>(
                species_a,
                species_b,
                config.reaction_rate));

        const auto start = std::chrono::high_resolution_clock::now();
        for (int step = 1; step <= config.steps; ++step) {
            if (config.use_gpu) {
#if LB_CUBE_ENABLE_CUDA
                check_cuda(
                    lbm::launch_step_gpu<FluidLattice, Real>(
                        fluid_device->current,
                        fluid_device->next,
                        config.nx,
                        config.ny,
                        1,
                        omega),
                    "launch fluid step");
                std::swap(fluid_device->current, fluid_device->next);

                check_cuda(
                    lbm::launch_reaction_AB_gpu<FluidLattice, ScalarLattice, Real>(
                        a_device->current,
                        a_device->next,
                        b_device->current,
                        b_device->next,
                        fluid_device->current,
                        config.nx,
                        config.ny,
                        1,
                        omega_c,
                        config.reaction_rate),
                    "launch reaction step");
                std::swap(a_device->current, a_device->next);
                std::swap(b_device->current, b_device->next);
#else
                throw std::runtime_error("GPU backend requested, but this executable was built without CUDA");
#endif
            } else {
                lbm::step_cpu<FluidLattice, Real>(fluid, omega);
                lbm::step_reaction_AB<FluidLattice, ScalarLattice, Real>(
                    fluid,
                    species_a,
                    species_b,
                    omega_c,
                    config.reaction_rate);
            }

            if (step % config.output_frequency == 0) {
#if LB_CUBE_ENABLE_CUDA
                if (config.use_gpu) {
                    copy_device_current_to_host<FluidLattice>(*fluid_device, fluid, fluid_staging);
                    copy_device_current_to_host<ScalarLattice>(*a_device, species_a, a_staging);
                    copy_device_current_to_host<ScalarLattice>(*b_device, species_b, b_staging);
                }
#endif

                write_reaction_vtk(
                    fluid,
                    species_a,
                    species_b,
                    static_cast<std::size_t>(step),
                    config.reaction_rate);
                lbm::log_reactive_diagnostics(
                    diagnostics,
                    static_cast<std::size_t>(step),
                    lbm::compute_reactive_stats<ScalarLattice, Real>(
                        species_a,
                        species_b,
                        config.reaction_rate));
                std::cout << "Step " << step << " / " << config.steps << " complete\n" << std::flush;
            }
        }

#if LB_CUBE_ENABLE_CUDA
        if (config.use_gpu) {
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
#endif

        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = stop - start;
        const double updates =
            static_cast<double>(config.nx) * static_cast<double>(config.ny) *
            static_cast<double>(config.steps);
        const double mlups = updates / elapsed.count() / 1.0e6;

        std::cout << "Elapsed time: " << elapsed.count() << " s\n"
                  << "Fluid lattice updates: " << mlups << " MLUPS\n"
                  << std::flush;

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
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

using Lattice = lbm::D3Q19;
using Real = double;

enum class InitialCondition {
    Rest,
    TaylorGreen
};

struct Config {
    std::size_t nx{64};
    std::size_t ny{64};
    std::size_t nz{64};
    int steps{1000};
    int output_frequency{100};
    InitialCondition initial_condition{InitialCondition::TaylorGreen};
#if LB_CUBE_ENABLE_CUDA
    bool use_gpu{true};
#else
    bool use_gpu{false};
#endif
};

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

[[nodiscard]] int parse_int_arg(const std::vector<std::string>& args, std::size_t& i) {
    if (i + 1 >= args.size()) {
        throw std::invalid_argument("missing value for " + args[i]);
    }

    return std::stoi(args[++i]);
}

[[nodiscard]] InitialCondition parse_initial_condition(const std::vector<std::string>& args, std::size_t& i) {
    if (i + 1 >= args.size()) {
        throw std::invalid_argument("missing value for " + args[i]);
    }

    const std::string value = args[++i];
    if (value == "rest") {
        return InitialCondition::Rest;
    }

    if (value == "tgv") {
        return InitialCondition::TaylorGreen;
    }

    throw std::invalid_argument("-init must be either 'rest' or 'tgv'");
}

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
        } else if (arg == "-nz") {
            config.nz = parse_size_arg(args, i);
        } else if (arg == "-steps") {
            config.steps = parse_int_arg(args, i);
        } else if (arg == "-out_freq") {
            config.output_frequency = parse_int_arg(args, i);
        } else if (arg == "-init") {
            config.initial_condition = parse_initial_condition(args, i);
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

    return config;
}

[[nodiscard]] std::string initial_condition_name(InitialCondition initial_condition) {
    switch (initial_condition) {
    case InitialCondition::Rest:
        return "rest";
    case InitialCondition::TaylorGreen:
        return "tgv";
    }

    return "unknown";
}

#if LB_CUBE_ENABLE_CUDA
void check_cuda(cudaError_t error, const std::string& context) {
    if (error != cudaSuccess) {
        throw std::runtime_error(context + ": " + cudaGetErrorString(error));
    }
}

template <typename T>
class DevicePopulationBuffers {
public:
    explicit DevicePopulationBuffers(std::size_t count)
        : count_(count),
          bytes_(count * sizeof(T)) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&current), bytes_), "cudaMalloc current populations");
        try {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&next), bytes_), "cudaMalloc next populations");
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
#endif

void initialize_rest_state(lbm::LatticeMemory<Lattice, Real>& mem) {
    auto view = mem.get_current_view();

    lbm::MacroState<Lattice, Real> macro{};
    macro.density = Real{1};
    macro.velocity.setZero();

    for (std::size_t z = 0; z < mem.z_extent(); ++z) {
        for (std::size_t y = 0; y < mem.y_extent(); ++y) {
            for (std::size_t x = 0; x < mem.x_extent(); ++x) {
                for (int i = 0; i < Lattice::Q; ++i) {
                    view[static_cast<std::size_t>(i), z, y, x] =
                        lbm::compute_equilibrium<Lattice, Real>(i, macro);
                }
            }
        }
    }
}

void initialize_taylor_green_state_2d(lbm::LatticeMemory<Lattice, Real>& mem) {
    constexpr Real sound_speed_squared = Real{1} / Real{3};
    constexpr Real initial_velocity = Real{0.01};

    const Real lx = static_cast<Real>(mem.x_extent());
    const Real ly = static_cast<Real>(mem.y_extent());
    const Real kx = Real{2} * std::numbers::pi_v<Real> / lx;
    const Real ky = Real{2} * std::numbers::pi_v<Real> / ly;
    auto view = mem.get_current_view();

    for (std::size_t z = 0; z < mem.z_extent(); ++z) {
        for (std::size_t y = 0; y < mem.y_extent(); ++y) {
            const Real phase_y = ky * static_cast<Real>(y);

            for (std::size_t x = 0; x < mem.x_extent(); ++x) {
                const Real phase_x = kx * static_cast<Real>(x);
                const Real density_perturbation =
                    (initial_velocity * initial_velocity) / (Real{4} * sound_speed_squared) *
                    (std::cos(Real{2} * phase_x) + std::cos(Real{2} * phase_y));

                lbm::MacroState<Lattice, Real> macro{};
                macro.density = Real{1} - density_perturbation;
                macro.velocity << -initial_velocity * std::cos(phase_x) * std::sin(phase_y),
                    initial_velocity * std::sin(phase_x) * std::cos(phase_y),
                    Real{0};

                for (int i = 0; i < Lattice::Q; ++i) {
                    view[static_cast<std::size_t>(i), z, y, x] =
                        lbm::compute_equilibrium<Lattice, Real>(i, macro);
                }
            }
        }
    }
}

void initialize_taylor_green_state_3d(lbm::LatticeMemory<Lattice, Real>& mem) {
    constexpr Real sound_speed_squared = Real{1} / Real{3};
    constexpr Real initial_velocity = Real{0.01};

    const Real lx = static_cast<Real>(mem.x_extent());
    const Real ly = static_cast<Real>(mem.y_extent());
    const Real lz = static_cast<Real>(mem.z_extent());

    const Real kx = Real{2} * std::numbers::pi_v<Real> / lx;
    const Real ky = Real{2} * std::numbers::pi_v<Real> / ly;
    const Real kz = Real{2} * std::numbers::pi_v<Real> / lz;
    
    auto view = mem.get_current_view();

    for (std::size_t z = 0; z < mem.z_extent(); ++z) {
        const Real phase_z = kz * static_cast<Real>(z);

        for (std::size_t y = 0; y < mem.y_extent(); ++y) {
            const Real phase_y = ky * static_cast<Real>(y);

            for (std::size_t x = 0; x < mem.x_extent(); ++x) {
                const Real phase_x = kx * static_cast<Real>(x);
                
                // 3D density perturbation
                const Real density_perturbation =
                    (initial_velocity * initial_velocity) / (Real{16} * sound_speed_squared) *
                    (std::cos(Real{2} * phase_x) + std::cos(Real{2} * phase_y)) * 
                    (std::cos(Real{2} * phase_z) + Real{2});

                lbm::MacroState<Lattice, Real> macro{};
                macro.density = Real{1} + density_perturbation; 
                
                // 3D velocity field
                macro.velocity << 
                    initial_velocity * std::sin(phase_x) * std::cos(phase_y) * std::cos(phase_z),
                    -initial_velocity * std::cos(phase_x) * std::sin(phase_y) * std::cos(phase_z),
                    Real{0};

                for (int i = 0; i < Lattice::Q; ++i) {
                    view[static_cast<std::size_t>(i), z, y, x] =
                        lbm::compute_equilibrium<Lattice, Real>(i, macro);
                }
            }
        }
    }
}

void initialize_state(lbm::LatticeMemory<Lattice, Real>& mem, InitialCondition initial_condition) {
    switch (initial_condition) {
    case InitialCondition::Rest:
        initialize_rest_state(mem);
        break;
    case InitialCondition::TaylorGreen:
        initialize_taylor_green_state_3d(mem);
        break;
    }
}

[[nodiscard]] std::vector<Real> flatten_current_populations(
    const lbm::LatticeMemory<Lattice, Real>& mem) {
    const auto view = mem.get_current_view();
    std::vector<Real> flat(mem.population_count());

    for (int i = 0; i < Lattice::Q; ++i) {
        const std::size_t q = static_cast<std::size_t>(i);
        for (std::size_t z = 0; z < mem.z_extent(); ++z) {
            for (std::size_t y = 0; y < mem.y_extent(); ++y) {
                for (std::size_t x = 0; x < mem.x_extent(); ++x) {
                    const std::size_t index =
                        (((q * mem.z_extent() + z) * mem.y_extent() + y) * mem.x_extent() + x);
                    flat[index] = view[q, z, y, x];
                }
            }
        }
    }

    return flat;
}

#if LB_CUBE_ENABLE_CUDA
void overwrite_current_populations(
    lbm::LatticeMemory<Lattice, Real>& mem,
    const std::vector<Real>& flat) {
    auto view = mem.get_current_view();

    for (int i = 0; i < Lattice::Q; ++i) {
        const std::size_t q = static_cast<std::size_t>(i);
        for (std::size_t z = 0; z < mem.z_extent(); ++z) {
            for (std::size_t y = 0; y < mem.y_extent(); ++y) {
                for (std::size_t x = 0; x < mem.x_extent(); ++x) {
                    const std::size_t index =
                        (((q * mem.z_extent() + z) * mem.y_extent() + y) * mem.x_extent() + x);
                    view[q, z, y, x] = flat[index];
                }
            }
        }
    }
}

void copy_device_current_to_host(
    DevicePopulationBuffers<Real>& device_buffers,
    lbm::LatticeMemory<Lattice, Real>& mem,
    std::vector<Real>& staging) {
    staging.resize(device_buffers.count());
    check_cuda(
        cudaMemcpy(staging.data(), device_buffers.current, device_buffers.bytes(), cudaMemcpyDeviceToHost),
        "cudaMemcpy device to host");
    overwrite_current_populations(mem, staging);
}
#endif

void print_summary(const Config& config, Real omega) {
    std::cout << "LB-Cube D3Q19 simulation\n"
              << "Grid: " << config.nx << " x " << config.ny << " x " << config.nz << '\n'
              << "Steps: " << config.steps << '\n'
              << "Output frequency: " << config.output_frequency << '\n'
              << "Initial condition: " << initial_condition_name(config.initial_condition) << '\n'
              << "Backend: " << (config.use_gpu ? "GPU" : "CPU") << '\n'
              << "Omega: " << omega << '\n'
              << std::flush;
}

} // namespace

int main(int argc, char** argv) {
    try {
        constexpr Real relaxation_time = Real{0.8};
        constexpr Real omega = Real{1} / relaxation_time;

        const Config config = parse_arguments(argc, argv);

        lbm::LatticeMemory<Lattice, Real> memory{config.nx, config.ny, config.nz};
        initialize_state(memory, config.initial_condition);

        std::ofstream diagnostics{"diagnostics.csv"};
        if (!diagnostics) {
            throw std::runtime_error("failed to open diagnostics.csv");
        }
        diagnostics << "step,total_mass,total_kinetic_energy,max_velocity_magnitude\n";

        print_summary(config, omega);

#if LB_CUBE_ENABLE_CUDA
        std::vector<Real> staging = flatten_current_populations(memory);
        std::unique_ptr<DevicePopulationBuffers<Real>> device_buffers;

        if (config.use_gpu) {
            device_buffers = std::make_unique<DevicePopulationBuffers<Real>>(memory.population_count());
            check_cuda(
                cudaMemcpy(device_buffers->current, staging.data(), device_buffers->bytes(), cudaMemcpyHostToDevice),
                "cudaMemcpy host to device");
        }
#endif

        const auto start = std::chrono::high_resolution_clock::now();

        lbm::write_vtk<Lattice, Real>(memory, config.nx, config.ny, config.nz, 0);
        lbm::log_diagnostics<Lattice, Real>(memory.get_current_view(), 0, diagnostics);

        for (int step = 1; step <= config.steps; ++step) {
            if (config.use_gpu) {
#if LB_CUBE_ENABLE_CUDA
                check_cuda(
                    lbm::launch_step_gpu<Lattice, Real>(
                        device_buffers->current,
                        device_buffers->next,
                        config.nx,
                        config.ny,
                        config.nz,
                        omega),
                    "launch_step_gpu");
                std::swap(device_buffers->current, device_buffers->next);
#else
                throw std::runtime_error("GPU backend requested, but this executable was built without CUDA");
#endif
            } else {
                lbm::step_cpu<Lattice, Real>(memory, omega);
            }

            if (step % config.output_frequency == 0) {
#if LB_CUBE_ENABLE_CUDA
                if (config.use_gpu) {
                    copy_device_current_to_host(*device_buffers, memory, staging);
                }
#endif

                lbm::write_vtk<Lattice, Real>(memory, config.nx, config.ny, config.nz, static_cast<std::size_t>(step));
                lbm::log_diagnostics<Lattice, Real>(
                    memory.get_current_view(),
                    static_cast<std::size_t>(step),
                    diagnostics);

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
        const double lattice_updates =
            static_cast<double>(config.nx) *
            static_cast<double>(config.ny) *
            static_cast<double>(config.nz) *
            static_cast<double>(config.steps);
        const double mlups = lattice_updates / elapsed.count() / 1.0e6;

        std::cout << "Elapsed time: " << elapsed.count() << " s\n"
                  << "Performance: " << mlups << " MLUPS\n"
                  << std::flush;

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

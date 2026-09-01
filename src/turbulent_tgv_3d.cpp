/**
 * @file turbulent_tgv_3d.cpp
 * @brief High-Reynolds-number 3D Taylor-Green vortex stability benchmark.
 *
 * This executable runs a fixed 3D Taylor-Green vortex initial condition through
 * several lattice/collision-operator combinations. The case is intentionally
 * close to the high-Re stability limit so BGK, MRT, and RLBM behavior can be
 * compared visually through VTK output and empirically through the maximum
 * velocity monitor.
 */

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <format>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

#include "lattice_core.hpp"
#include "lattice_io.hpp"
#include "lattice_memory.hpp"
#include "lattice_mrt.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

namespace {

using Real = double;
using lbm::CollisionType;

constexpr std::size_t nx = 128;
constexpr std::size_t ny = 128;
constexpr std::size_t nz = 128;
constexpr int total_steps = 5000;
constexpr int stability_check_frequency = 50;
constexpr int vtk_frequency = 1000;
constexpr Real initial_velocity = Real{0.05};
constexpr Real relaxation_time = Real{0.5005};
constexpr Real crash_velocity_threshold = Real{1.0};

/**
 * @brief Construct the analytical 3D TGV state at a lattice node.
 *
 * @tparam Lattice 3D lattice traits type.
 * @param x_index X-grid coordinate.
 * @param y_index Y-grid coordinate.
 * @param z_index Z-grid coordinate.
 * @return Macroscopic density and velocity used for equilibrium initialization.
 */
template <lbm::IsLatticeModel Lattice>
[[nodiscard]] lbm::MacroState<Lattice, Real> tgv_macro_state(
    std::size_t x_index,
    std::size_t y_index,
    std::size_t z_index) {
    static_assert(Lattice::D == 3);

    const Real x_phase =
        Real{2} * std::numbers::pi_v<Real> *
        static_cast<Real>(x_index) / static_cast<Real>(nx);
    const Real y_phase =
        Real{2} * std::numbers::pi_v<Real> *
        static_cast<Real>(y_index) / static_cast<Real>(ny);
    const Real z_phase =
        Real{2} * std::numbers::pi_v<Real> *
        static_cast<Real>(z_index) / static_cast<Real>(nz);

    lbm::MacroState<Lattice, Real> macro{};
    macro.density = Real{1};
    macro.velocity <<
        initial_velocity * std::sin(x_phase) * std::cos(y_phase) * std::cos(z_phase),
        -initial_velocity * std::cos(x_phase) * std::sin(y_phase) * std::cos(z_phase),
        Real{0};

    return macro;
}

/**
 * @brief Initialize all populations from the 3D TGV equilibrium field.
 *
 * @tparam Lattice 3D lattice traits type.
 * @param mem Population memory to initialize.
 */
template <lbm::IsLatticeModel Lattice>
void initialize_tgv(lbm::LatticeMemory<Lattice, Real>& mem) {
    static_assert(Lattice::D == 3);
    auto view = mem.get_current_view();

    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                const lbm::MacroState<Lattice, Real> macro =
                    tgv_macro_state<Lattice>(x, y, z);

                for (int i = 0; i < Lattice::Q; ++i) {
                    view[static_cast<std::size_t>(i), z, y, x] =
                        lbm::compute_equilibrium<Lattice, Real>(i, macro);
                }
            }
        }
    }
}

/**
 * @brief Reconstruct one local macroscopic state from a 3D population view.
 *
 * @tparam Lattice 3D lattice traits type.
 * @param view Current population view.
 * @param x X-grid coordinate.
 * @param y Y-grid coordinate.
 * @param z Z-grid coordinate.
 * @return Macroscopic state at the requested cell.
 */
template <lbm::IsLatticeModel Lattice>
[[nodiscard]] lbm::MacroState<Lattice, Real> macro_at(
    typename lbm::LatticeMemory<Lattice, Real>::ConstView view,
    std::size_t x,
    std::size_t y,
    std::size_t z) {
    static_assert(Lattice::D == 3);
    std::array<Real, static_cast<std::size_t>(Lattice::Q)> populations{};

    for (int i = 0; i < Lattice::Q; ++i) {
        populations[static_cast<std::size_t>(i)] =
            view[static_cast<std::size_t>(i), z, y, x];
    }

    return lbm::compute_macro_state<Lattice, Real>(populations);
}

/**
 * @brief Compute the maximum velocity magnitude in the current domain.
 *
 * Non-finite values are reported as infinity so the stability guard can stop
 * the benchmark gracefully.
 *
 * @tparam Lattice 3D lattice traits type.
 * @param mem Current population memory.
 * @return Maximum velocity magnitude.
 */
template <lbm::IsLatticeModel Lattice>
[[nodiscard]] Real compute_max_velocity(
    const lbm::LatticeMemory<Lattice, Real>& mem) {
    static_assert(Lattice::D == 3);
    const auto view = mem.get_current_view();
    Real max_velocity{};

    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                const lbm::MacroState<Lattice, Real> macro =
                    macro_at<Lattice>(view, x, y, z);
                const Real speed = std::sqrt(macro.velocity.squaredNorm());

                if (!std::isfinite(speed)) {
                    return std::numeric_limits<Real>::infinity();
                }

                max_velocity = std::max(max_velocity, speed);
            }
        }
    }

    return max_velocity;
}

/**
 * @brief Write density, velocity magnitude, and velocity vector to VTK.
 *
 * @tparam Lattice 3D lattice traits type.
 * @param output_dir Directory dedicated to one lattice/operator configuration.
 * @param name Configuration name used in the file name and VTK title.
 * @param mem Current population memory.
 * @param step Simulation step number.
 */
template <lbm::IsLatticeModel Lattice>
void write_tgv_vtk(
    const std::filesystem::path& output_dir,
    const std::string& name,
    const lbm::LatticeMemory<Lattice, Real>& mem,
    int step) {
    static_assert(Lattice::D == 3);

    const std::filesystem::path filename =
        output_dir / std::format("tgv3d_{}_{:06}.vtk", name, step);
    std::ofstream vtk{filename};
    if (!vtk) {
        throw std::runtime_error("failed to open " + filename.string());
    }

    const auto view = mem.get_current_view();
    constexpr std::size_t point_count = nx * ny * nz;

    vtk << "# vtk DataFile Version 3.0\n";
    vtk << std::format("LB-Cube 3D turbulent TGV {} step {}\n", name, step);
    vtk << "ASCII\n";
    vtk << "DATASET STRUCTURED_POINTS\n";
    vtk << std::format("DIMENSIONS {} {} {}\n", nx, ny, nz);
    vtk << "ORIGIN 0 0 0\n";
    vtk << "SPACING 1 1 1\n";
    vtk << std::format("POINT_DATA {}\n", point_count);

    vtk << "SCALARS density double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                const lbm::MacroState<Lattice, Real> macro =
                    macro_at<Lattice>(view, x, y, z);
                vtk << std::format("{:.17g}\n", static_cast<double>(macro.density));
            }
        }
    }

    vtk << "SCALARS velocity_magnitude double 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                const lbm::MacroState<Lattice, Real> macro =
                    macro_at<Lattice>(view, x, y, z);
                vtk << std::format(
                    "{:.17g}\n",
                    static_cast<double>(std::sqrt(macro.velocity.squaredNorm())));
            }
        }
    }

    vtk << "VECTORS velocity double\n";
    for (std::size_t z = 0; z < nz; ++z) {
        for (std::size_t y = 0; y < ny; ++y) {
            for (std::size_t x = 0; x < nx; ++x) {
                const lbm::MacroState<Lattice, Real> macro =
                    macro_at<Lattice>(view, x, y, z);
                vtk << std::format(
                    "{:.17g} {:.17g} {:.17g}\n",
                    static_cast<double>(macro.velocity[0]),
                    static_cast<double>(macro.velocity[1]),
                    static_cast<double>(macro.velocity[2]));
            }
        }
    }
}

/**
 * @brief Run one lattice/operator configuration of the TGV benchmark.
 *
 * @tparam Lattice 3D lattice traits type.
 * @tparam CT Compile-time collision operator.
 * @param name Configuration label used in logs and output directory names.
 */
template <lbm::IsLatticeModel Lattice, CollisionType CT>
void run_tgv_3d(const std::string& name) {
    static_assert(Lattice::D == 3);

    const std::filesystem::path output_dir =
        std::format("output_tgv3d_{}", name);
    std::filesystem::create_directories(output_dir);

    const std::filesystem::path stability_filename =
        output_dir / std::format("stability_{}.csv", name);
    std::ofstream stability_log{stability_filename};
    if (!stability_log) {
        throw std::runtime_error("failed to open " + stability_filename.string());
    }
    stability_log << "step,max_velocity\n";

    lbm::LatticeMemory<Lattice, Real> mem{nx, ny, nz};
    initialize_tgv<Lattice>(mem);

    constexpr Real omega = Real{1} / relaxation_time;
    constexpr Real viscosity = Lattice::cs2 * (relaxation_time - Real{0.5});
    constexpr std::size_t total_cells = nx * ny * nz;

    std::cout << "Running " << name << " 3D turbulent TGV benchmark\n"
              << "Grid: " << nx << " x " << ny << " x " << nz << '\n'
              << "tau=" << relaxation_time
              << ", omega=" << omega
              << ", nu=" << viscosity << '\n'
              << "Output: " << output_dir.string() << '\n'
              << std::flush;

    write_tgv_vtk<Lattice>(output_dir, name, mem, 0);
    stability_log << "0," << std::format("{:.17g}", compute_max_velocity<Lattice>(mem)) << '\n';

    const auto start = std::chrono::high_resolution_clock::now();
    int completed_steps = 0;
    bool crashed = false;

    for (int step = 1; step <= total_steps; ++step) {
        lbm::step_cpu<Lattice, Real, CT>(mem, omega);
        completed_steps = step;

        if (step % stability_check_frequency == 0) {
            const Real max_velocity = compute_max_velocity<Lattice>(mem);
            stability_log << step << ',' << std::format("{:.17g}", max_velocity) << '\n';

            if (!std::isfinite(max_velocity) ||
                max_velocity > crash_velocity_threshold) {
                std::cout << '[' << name
                          << "] Solver Crashed due to instability! "
                          << "step=" << step
                          << ", Umax=" << max_velocity << '\n'
                          << std::flush;
                crashed = true;
                break;
            }
        }

        if (step % vtk_frequency == 0) {
            write_tgv_vtk<Lattice>(output_dir, name, mem, step);
            std::cout << '[' << name << "] Step " << step << " / "
                      << total_steps << " complete\n"
                      << std::flush;
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double mlups =
        static_cast<double>(total_cells) *
        static_cast<double>(completed_steps) /
        elapsed.count() /
        1.0e6;

    std::cout << '[' << name << "] elapsed=" << elapsed.count()
              << " s, MLUPS=" << mlups
              << (crashed ? " (terminated early)" : "")
              << "\n\n"
              << std::flush;
}

} // namespace

/**
 * @brief Run the full operator gauntlet for the 3D turbulent TGV benchmark.
 *
 * @return Zero on success, nonzero if a setup or output error occurs.
 */
int main() {
    try {
    //    run_tgv_3d<lbm::D3Q19, CollisionType::BGK>("D3Q19_BGK");
        run_tgv_3d<lbm::D3Q19, CollisionType::MRT>("D3Q19_MRT");
    //    run_tgv_3d<lbm::D3Q19, CollisionType::RLBM>("D3Q19_RLBM");
    //    run_tgv_3d<lbm::D3Q27, CollisionType::MRT>("D3Q27_MRT");
    //    run_tgv_3d<lbm::D3Q27, CollisionType::RLBM>("D3Q27_RLBM");
    } catch (const std::exception& error) {
        std::cerr << "Fatal benchmark error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}

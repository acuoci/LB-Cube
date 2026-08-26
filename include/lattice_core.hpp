#pragma once

#include <array>
#include <cstddef>

#include "lattice_memory.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

namespace lbm {

namespace detail {

[[nodiscard]] constexpr std::size_t periodic_pull_index(
    std::size_t coordinate,
    int velocity,
    std::size_t extent) noexcept {
    const auto signed_coordinate = static_cast<std::ptrdiff_t>(coordinate);
    const auto signed_extent = static_cast<std::ptrdiff_t>(extent);
    const auto signed_index = signed_coordinate - static_cast<std::ptrdiff_t>(velocity) + signed_extent;
    return static_cast<std::size_t>(signed_index % signed_extent);
}

} // namespace detail

template <IsLatticeModel Lattice, std::floating_point Real>
inline void step_cpu(LatticeMemory<Lattice, Real>& mem, Real omega) {
    auto current_view = mem.get_current_view();
    auto next_view = mem.get_next_view();

    if constexpr (Lattice::D == 2) {
        const std::size_t y_extent = current_view.extent(1);
        const std::size_t x_extent = current_view.extent(2);

        for (std::size_t y = 0; y < y_extent; ++y) {
            for (std::size_t x = 0; x < x_extent; ++x) {
                std::array<Real, static_cast<std::size_t>(Lattice::Q)> local_pops{};

                for (int i = 0; i < Lattice::Q; ++i) {
                    const auto direction_offset = static_cast<std::size_t>(i * Lattice::D);
                    const int cx = Lattice::directions[direction_offset];
                    const int cy = Lattice::directions[direction_offset + 1];
                    const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
                    const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);

                    local_pops[static_cast<std::size_t>(i)] =
                        current_view[static_cast<std::size_t>(i), ny, nx];
                }

                collide_bgk<Lattice, Real>(local_pops, omega);

                for (int i = 0; i < Lattice::Q; ++i) {
                    next_view[static_cast<std::size_t>(i), y, x] =
                        local_pops[static_cast<std::size_t>(i)];
                }
            }
        }
    } else {
        const std::size_t z_extent = current_view.extent(1);
        const std::size_t y_extent = current_view.extent(2);
        const std::size_t x_extent = current_view.extent(3);

        for (std::size_t z = 0; z < z_extent; ++z) {
            for (std::size_t y = 0; y < y_extent; ++y) {
                for (std::size_t x = 0; x < x_extent; ++x) {
                    std::array<Real, static_cast<std::size_t>(Lattice::Q)> local_pops{};

                    for (int i = 0; i < Lattice::Q; ++i) {
                        const auto direction_offset = static_cast<std::size_t>(i * Lattice::D);
                        const int cx = Lattice::directions[direction_offset];
                        const int cy = Lattice::directions[direction_offset + 1];
                        const int cz = Lattice::directions[direction_offset + 2];
                        const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
                        const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);
                        const std::size_t nz = detail::periodic_pull_index(z, cz, z_extent);

                        local_pops[static_cast<std::size_t>(i)] =
                            current_view[static_cast<std::size_t>(i), nz, ny, nx];
                    }

                    collide_bgk<Lattice, Real>(local_pops, omega);

                    for (int i = 0; i < Lattice::Q; ++i) {
                        next_view[static_cast<std::size_t>(i), z, y, x] =
                            local_pops[static_cast<std::size_t>(i)];
                    }
                }
            }
        }
    }

    mem.swap_buffers();
}

} // namespace lbm

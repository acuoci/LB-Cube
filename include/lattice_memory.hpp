#pragma once
/**
 * @file lattice_memory.hpp
 * @brief Host-side macroscopic state and Structure-of-Arrays population storage.
 *
 * This header defines the precision-templated state and memory container used by
 * the CPU backend and host-side diagnostics. Population buffers are stored as
 * flat arrays and exposed through `std::mdspan` views to keep the memory layout
 * explicit.
 */

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <mdspan>
#include <type_traits>
#include <vector>

#include "lattice_traits.hpp"

namespace lbm {

/**
 * @brief Macroscopic density and velocity reconstructed at one lattice node.
 *
 * The velocity vector dimension follows the compile-time lattice dimension, and
 * its scalar type follows the selected precision. This state is intentionally a
 * lightweight value type used by stateless physics functions rather than a
 * persistent field allocation.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`. Only 2D and
 * 3D lattices are supported by the velocity alias.
 * @tparam Real Floating-point precision used for density, velocity, and
 * populations, typically `float` or `double`.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
struct MacroState {
    static_assert(Lattice::D == 2 || Lattice::D == 3);

    using Velocity = std::conditional_t<
        Lattice::D == 2,
        Eigen::Matrix<Real, 2, 1>,
        Eigen::Matrix<Real, 3, 1>>;

    /** @brief Local density, equal to the sum of populations at one cell. */
    Real density{};
    /** @brief Local macroscopic velocity with dimension determined by `Lattice::D`. */
    Velocity velocity = Velocity::Zero();
};

/**
 * @brief Host-side double-buffered population storage in Structure-of-Arrays form.
 *
 * The container owns exactly two flat `std::vector<Real>` buffers, `pop_0` and
 * `pop_1`, used as current and next populations in the ping-pong time-stepping
 * scheme. The public views expose the layout as `[Q, Y, X]` for 2D lattices and
 * `[Q, Z, Y, X]` for 3D lattices with `std::layout_right`; therefore `X` is the
 * fastest-varying spatial index and each population field is spatially
 * contiguous. This SoA layout mirrors the intended CUDA coalescing strategy while
 * remaining convenient for host validation and IO.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision, typically `float` or
 * `double`.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
class LatticeMemory {
    static_assert(Lattice::D == 2 || Lattice::D == 3);

public:
    /** @brief Unsigned size type used for extents and flat allocation counts. */
    using size_type = std::size_t;

    /** @brief Mutable 2D SoA view with extents `[Q, Y, X]`. */
    using View2D = std::mdspan<
        Real,
        std::extents<size_type, static_cast<size_type>(Lattice::Q), std::dynamic_extent, std::dynamic_extent>,
        std::layout_right>;

    /** @brief Read-only 2D SoA view with extents `[Q, Y, X]`. */
    using ConstView2D = std::mdspan<
        const Real,
        std::extents<size_type, static_cast<size_type>(Lattice::Q), std::dynamic_extent, std::dynamic_extent>,
        std::layout_right>;

    /** @brief Mutable 3D SoA view with extents `[Q, Z, Y, X]`. */
    using View3D = std::mdspan<
        Real,
        std::extents<size_type, static_cast<size_type>(Lattice::Q), std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>,
        std::layout_right>;

    /** @brief Read-only 3D SoA view with extents `[Q, Z, Y, X]`. */
    using ConstView3D = std::mdspan<
        const Real,
        std::extents<size_type, static_cast<size_type>(Lattice::Q), std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>,
        std::layout_right>;

    /** @brief Dimension-dependent mutable mdspan type used by kernels and IO. */
    using View = std::conditional_t<Lattice::D == 2, View2D, View3D>;
    /** @brief Dimension-dependent read-only mdspan type used by diagnostics. */
    using ConstView = std::conditional_t<Lattice::D == 2, ConstView2D, ConstView3D>;

    /**
     * @brief Allocate the two ping-pong buffers for a 2D lattice.
     *
     * @param x Number of nodes in the x direction.
     * @param y Number of nodes in the y direction.
     */
    LatticeMemory(size_type x, size_type y)
        requires (Lattice::D == 2)
        : x_(x),
          y_(y),
          z_(1),
          pop_0(population_count(x, y, 1)),
          pop_1(population_count(x, y, 1)) {}

    /**
     * @brief Allocate the two ping-pong buffers for a 3D lattice.
     *
     * @param x Number of nodes in the x direction.
     * @param y Number of nodes in the y direction.
     * @param z Number of nodes in the z direction.
     */
    LatticeMemory(size_type x, size_type y, size_type z)
        requires (Lattice::D == 3)
        : x_(x),
          y_(y),
          z_(z),
          pop_0(population_count(x, y, z)),
          pop_1(population_count(x, y, z)) {}

    /** @return Number of nodes in the x direction, used as the fastest spatial stride. */
    [[nodiscard]] size_type x_extent() const noexcept { return x_; }
    /** @return Number of nodes in the y direction. */
    [[nodiscard]] size_type y_extent() const noexcept { return y_; }
    /** @return Number of nodes in the z direction, equal to 1 for 2D allocations. */
    [[nodiscard]] size_type z_extent() const noexcept { return z_; }
    /** @return Number of scalar population values in one ping-pong buffer. */
    [[nodiscard]] size_type population_count() const noexcept { return pop_0.size(); }

    /**
     * @brief View the read-side buffer for the current time level.
     *
     * Time-stepping routines pull populations from this view and write updated
     * post-collision populations into `get_next_view()`. The logical current
     * buffer changes only when `swap_buffers()` is called.
     *
     * @return Mutable mdspan over the active read buffer.
     */
    [[nodiscard]] View get_current_view() noexcept {
        return make_view(current_buffer().data());
    }

    /**
     * @brief View the read-side buffer for diagnostics or output.
     *
     * This overload allows reconstruction of macroscopic quantities without
     * granting write access to the population storage.
     *
     * @return Read-only mdspan over the active read buffer.
     */
    [[nodiscard]] ConstView get_current_view() const noexcept {
        return make_const_view(current_buffer().data());
    }

    /**
     * @brief View the write-side buffer for the next time level.
     *
     * Fused collision-streaming writes into this buffer to avoid write races and
     * preserve the previous time level until the full step has completed.
     *
     * @return Mutable mdspan over the inactive write buffer.
     */
    [[nodiscard]] View get_next_view() noexcept {
        return make_view(next_buffer().data());
    }

    /**
     * @brief Read-only view of the write-side buffer.
     *
     * This is mainly useful for validation or debugging of the ping-pong state
     * before a buffer swap.
     *
     * @return Read-only mdspan over the inactive write buffer.
     */
    [[nodiscard]] ConstView get_next_view() const noexcept {
        return make_const_view(next_buffer().data());
    }

    /**
     * @brief Advance the ping-pong scheme by exchanging current and next roles.
     *
     * The underlying vectors are not moved or copied; only the active-buffer
     * index flips. This makes each time step an O(1) ownership transition after
     * the full domain update has completed.
     */
    void swap_buffers() noexcept {
        current_buffer_index_ = 1 - current_buffer_index_;
    }

private:
    /**
     * @brief Compute the number of population scalars required by one buffer.
     *
     * @param x Number of nodes in x.
     * @param y Number of nodes in y.
     * @param z Number of nodes in z, or 1 for 2D.
     * @return `Q * x * y * z`.
     */
    [[nodiscard]] static constexpr size_type population_count(
        size_type x,
        size_type y,
        size_type z) noexcept {
        return static_cast<size_type>(Lattice::Q) * x * y * z;
    }

    /**
     * @brief Create a dimension-correct mutable mdspan over a flat buffer.
     *
     * @param data Pointer to the first scalar population in the selected buffer.
     * @return mdspan using the solver's SoA extents and row-major mapping.
     */
    [[nodiscard]] View make_view(Real* data) noexcept {
        if constexpr (Lattice::D == 2) {
            return View{data, y_, x_};
        } else {
            return View{data, z_, y_, x_};
        }
    }

    /**
     * @brief Create a dimension-correct const mdspan over a flat buffer.
     *
     * @param data Pointer to the first scalar population in the selected buffer.
     * @return read-only mdspan using the solver's SoA extents.
     */
    [[nodiscard]] ConstView make_const_view(const Real* data) const noexcept {
        if constexpr (Lattice::D == 2) {
            return ConstView{data, y_, x_};
        } else {
            return ConstView{data, z_, y_, x_};
        }
    }

    /** @return The vector currently used as the read source for time stepping. */
    [[nodiscard]] std::vector<Real>& current_buffer() noexcept {
        return current_buffer_index_ == 0 ? pop_0 : pop_1;
    }

    /** @return The vector currently used as the read source for const operations. */
    [[nodiscard]] const std::vector<Real>& current_buffer() const noexcept {
        return current_buffer_index_ == 0 ? pop_0 : pop_1;
    }

    /** @return The vector currently used as the write target for the next step. */
    [[nodiscard]] std::vector<Real>& next_buffer() noexcept {
        return current_buffer_index_ == 0 ? pop_1 : pop_0;
    }

    /** @return The vector currently used as the write target for const inspection. */
    [[nodiscard]] const std::vector<Real>& next_buffer() const noexcept {
        return current_buffer_index_ == 0 ? pop_1 : pop_0;
    }

    size_type x_{};
    size_type y_{};
    size_type z_{};
    std::vector<Real> pop_0;
    std::vector<Real> pop_1;
    int current_buffer_index_{0};
};

} // namespace lbm

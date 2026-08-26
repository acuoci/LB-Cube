#pragma once

#include <Eigen/Dense>

#include <concepts>
#include <cstddef>
#include <mdspan>
#include <type_traits>
#include <vector>

#include "lattice_traits.hpp"

namespace lbm {

template <IsLatticeModel Lattice, std::floating_point Real>
struct MacroState {
    static_assert(Lattice::D == 2 || Lattice::D == 3);

    using Velocity = std::conditional_t<
        Lattice::D == 2,
        Eigen::Matrix<Real, 2, 1>,
        Eigen::Matrix<Real, 3, 1>>;

    Real density{};
    Velocity velocity = Velocity::Zero();
};

template <IsLatticeModel Lattice, std::floating_point Real>
class LatticeMemory {
    static_assert(Lattice::D == 2 || Lattice::D == 3);

public:
    using size_type = std::size_t;

    using View2D = std::mdspan<
        Real,
        std::extents<size_type, static_cast<size_type>(Lattice::Q), std::dynamic_extent, std::dynamic_extent>,
        std::layout_right>;

    using ConstView2D = std::mdspan<
        const Real,
        std::extents<size_type, static_cast<size_type>(Lattice::Q), std::dynamic_extent, std::dynamic_extent>,
        std::layout_right>;

    using View3D = std::mdspan<
        Real,
        std::extents<size_type, static_cast<size_type>(Lattice::Q), std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>,
        std::layout_right>;

    using ConstView3D = std::mdspan<
        const Real,
        std::extents<size_type, static_cast<size_type>(Lattice::Q), std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>,
        std::layout_right>;

    using View = std::conditional_t<Lattice::D == 2, View2D, View3D>;
    using ConstView = std::conditional_t<Lattice::D == 2, ConstView2D, ConstView3D>;

    LatticeMemory(size_type x, size_type y)
        requires (Lattice::D == 2)
        : x_(x),
          y_(y),
          z_(1),
          pop_0(population_count(x, y, 1)),
          pop_1(population_count(x, y, 1)) {}

    LatticeMemory(size_type x, size_type y, size_type z)
        requires (Lattice::D == 3)
        : x_(x),
          y_(y),
          z_(z),
          pop_0(population_count(x, y, z)),
          pop_1(population_count(x, y, z)) {}

    [[nodiscard]] size_type x_extent() const noexcept { return x_; }
    [[nodiscard]] size_type y_extent() const noexcept { return y_; }
    [[nodiscard]] size_type z_extent() const noexcept { return z_; }
    [[nodiscard]] size_type population_count() const noexcept { return pop_0.size(); }

    [[nodiscard]] View get_current_view() noexcept {
        return make_view(current_buffer().data());
    }

    [[nodiscard]] ConstView get_current_view() const noexcept {
        return make_const_view(current_buffer().data());
    }

    [[nodiscard]] View get_next_view() noexcept {
        return make_view(next_buffer().data());
    }

    [[nodiscard]] ConstView get_next_view() const noexcept {
        return make_const_view(next_buffer().data());
    }

    void swap_buffers() noexcept {
        current_buffer_index_ = 1 - current_buffer_index_;
    }

private:
    [[nodiscard]] static constexpr size_type population_count(
        size_type x,
        size_type y,
        size_type z) noexcept {
        return static_cast<size_type>(Lattice::Q) * x * y * z;
    }

    [[nodiscard]] View make_view(Real* data) noexcept {
        if constexpr (Lattice::D == 2) {
            return View{data, y_, x_};
        } else {
            return View{data, z_, y_, x_};
        }
    }

    [[nodiscard]] ConstView make_const_view(const Real* data) const noexcept {
        if constexpr (Lattice::D == 2) {
            return ConstView{data, y_, x_};
        } else {
            return ConstView{data, z_, y_, x_};
        }
    }

    [[nodiscard]] std::vector<Real>& current_buffer() noexcept {
        return current_buffer_index_ == 0 ? pop_0 : pop_1;
    }

    [[nodiscard]] const std::vector<Real>& current_buffer() const noexcept {
        return current_buffer_index_ == 0 ? pop_0 : pop_1;
    }

    [[nodiscard]] std::vector<Real>& next_buffer() noexcept {
        return current_buffer_index_ == 0 ? pop_1 : pop_0;
    }

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

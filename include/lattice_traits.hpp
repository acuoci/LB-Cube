#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace lbm {

template <typename Lattice>
concept IsLatticeModel =
    requires {
        requires std::same_as<decltype(Lattice::D), const int>;
        requires std::same_as<decltype(Lattice::Q), const int>;
        requires (Lattice::D > 0);
        requires (Lattice::Q > 0);
        requires std::same_as<
            std::remove_cv_t<decltype(Lattice::weights)>,
            std::array<double, static_cast<std::size_t>(Lattice::Q)>>;
        requires std::same_as<
            std::remove_cv_t<decltype(Lattice::directions)>,
            std::array<int, static_cast<std::size_t>(Lattice::Q * Lattice::D)>>;
    };

struct D2Q9 {
    static constexpr int D = 2;
    static constexpr int Q = 9;

    static constexpr std::array<double, Q> weights{
        4.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };

    static constexpr std::array<int, Q * D> directions{
        0,  0,
        1,  0,
        0,  1,
       -1,  0,
        0, -1,
        1,  1,
       -1,  1,
       -1, -1,
        1, -1
    };
};

struct D3Q19 {
    static constexpr int D = 3;
    static constexpr int Q = 19;

    static constexpr std::array<double, Q> weights{
        1.0 / 3.0,
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };

    static constexpr std::array<int, Q * D> directions{
        0,  0,  0,
        1,  0,  0,
       -1,  0,  0,
        0,  1,  0,
        0, -1,  0,
        0,  0,  1,
        0,  0, -1,
        1,  1,  0,
       -1,  1,  0,
        1, -1,  0,
       -1, -1,  0,
        1,  0,  1,
       -1,  0,  1,
        1,  0, -1,
       -1,  0, -1,
        0,  1,  1,
        0, -1,  1,
        0,  1, -1,
        0, -1, -1
    };
};

struct D3Q27 {
    static constexpr int D = 3;
    static constexpr int Q = 27;

    static constexpr std::array<double, Q> weights{
        8.0 / 27.0,
        2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0,
        2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0,
        1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
        1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
        1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
        1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
        1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0,
        1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0
    };

    static constexpr std::array<int, Q * D> directions{
        0,  0,  0,
        1,  0,  0,
       -1,  0,  0,
        0,  1,  0,
        0, -1,  0,
        0,  0,  1,
        0,  0, -1,
        1,  1,  0,
       -1,  1,  0,
        1, -1,  0,
       -1, -1,  0,
        1,  0,  1,
       -1,  0,  1,
        1,  0, -1,
       -1,  0, -1,
        0,  1,  1,
        0, -1,  1,
        0,  1, -1,
        0, -1, -1,
        1,  1,  1,
       -1,  1,  1,
        1, -1,  1,
       -1, -1,  1,
        1,  1, -1,
       -1,  1, -1,
        1, -1, -1,
       -1, -1, -1
    };
};

static_assert(IsLatticeModel<D2Q9>);
static_assert(IsLatticeModel<D3Q19>);
static_assert(IsLatticeModel<D3Q27>);

} // namespace lbm

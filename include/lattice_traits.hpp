#pragma once
/**
 * @file lattice_traits.hpp
 * @brief Compile-time lattice model definitions used to specialize LBM kernels.
 *
 * The solver represents each lattice as a stateless traits type. This keeps the
 * discrete velocity set and quadrature weights available at compile time, avoids
 * runtime virtual dispatch, and allows CPU and CUDA kernels to be specialized for
 * a fixed value of Q and D.
 */

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace lbm {

/**
 * @brief Concept satisfied by compile-time LBM lattice model traits.
 *
 * A lattice model must expose the spatial dimension count `D`, the number of
 * discrete populations `Q`, a quadrature weight array, and a flattened velocity
 * table. The flattened velocity layout is `[q * D + d]`, which is trivial to
 * index in device code and does not rely on dynamically allocated state.
 *
 * @tparam Lattice Candidate traits type. It is accepted when it provides static
 * constexpr `int D`, static constexpr `int Q`, static constexpr `double cs2`,
 * `std::array<double, Q> weights`, and flattened direction arrays named both
 * `directions` and `c`.
 */
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
        requires std::convertible_to<decltype(Lattice::cs2), double>;
        requires std::same_as<
            std::remove_cv_t<decltype(Lattice::directions)>,
            std::array<int, static_cast<std::size_t>(Lattice::Q * Lattice::D)>>;
        requires std::same_as<
            std::remove_cv_t<decltype(Lattice::c)>,
            std::array<int, static_cast<std::size_t>(Lattice::Q * Lattice::D)>>;
    };

/**
 * @brief Two-dimensional nine-velocity lattice model.
 *
 * D2Q9 is commonly used for 2D weakly compressible isothermal Navier-Stokes
 * benchmarks. Directions are ordered with the rest population first, followed by
 * axial and diagonal links.
 */
struct D2Q9 {
    /** @brief Number of spatial dimensions. */
    static constexpr int D = 2;
    /** @brief Number of discrete populations per lattice node. */
    static constexpr int Q = 9;
    /** @brief Squared lattice sound speed. */
    static constexpr double cs2 = 1.0 / 3.0;

    /** @brief Standard D2Q9 lattice weights. */
    static constexpr std::array<double, Q> weights{
        4.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };

    /** @brief Flattened discrete velocity vectors stored as `(cx, cy)` pairs. */
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
    /** @brief Alias for the flattened discrete velocity table. */
    static constexpr auto c = directions;
};

/**
 * @brief Three-dimensional nineteen-velocity lattice model.
 *
 * D3Q19 omits the cube-corner links of D3Q27 and is a common performance-oriented
 * choice for 3D incompressible-flow LBM simulations.
 */
struct D3Q19 {
    /** @brief Number of spatial dimensions. */
    static constexpr int D = 3;
    /** @brief Number of discrete populations per lattice node. */
    static constexpr int Q = 19;
    /** @brief Squared lattice sound speed. */
    static constexpr double cs2 = 1.0 / 3.0;

    /** @brief Standard D3Q19 lattice weights. */
    static constexpr std::array<double, Q> weights{
        1.0 / 3.0,
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };

    /** @brief Flattened discrete velocity vectors stored as `(cx, cy, cz)` triples. */
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
    /** @brief Alias for the flattened discrete velocity table. */
    static constexpr auto c = directions;
};

/**
 * @brief Three-dimensional twenty-seven-velocity lattice model.
 *
 * D3Q27 includes rest, axial, face-diagonal, and cube-corner links. It provides
 * the full tensor-product lattice for applications that benefit from the
 * additional isotropy relative to D3Q19.
 */
struct D3Q27 {
    /** @brief Number of spatial dimensions. */
    static constexpr int D = 3;
    /** @brief Number of discrete populations per lattice node. */
    static constexpr int Q = 27;
    /** @brief Squared lattice sound speed. */
    static constexpr double cs2 = 1.0 / 3.0;

    /** @brief Standard D3Q27 lattice weights. */
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

    /** @brief Flattened discrete velocity vectors stored as `(cx, cy, cz)` triples. */
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
    /** @brief Alias for the flattened discrete velocity table. */
    static constexpr auto c = directions;
};

/**
 * @brief Two-dimensional five-velocity lattice for passive scalar transport.
 *
 * D2Q5 contains the rest population and four axial links. The weights give
 * `c_s^2 = 1/3`, which is suitable for a compact advection-diffusion scalar
 * model coupled to a D2 fluid lattice.
 */
struct D2Q5 {
    /** @brief Number of spatial dimensions. */
    static constexpr int D = 2;
    /** @brief Number of scalar populations per lattice node. */
    static constexpr int Q = 5;
    /** @brief Squared lattice sound speed for scalar transport. */
    static constexpr double cs2 = 1.0 / 3.0;

    /** @brief Standard D2Q5 scalar lattice weights. */
    static constexpr std::array<double, Q> weights{
        1.0 / 3.0,
        1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0
    };

    /** @brief Flattened discrete velocity vectors stored as `(cx, cy)` pairs. */
    static constexpr std::array<int, Q * D> directions{
        0,  0,
        1,  0,
        0,  1,
       -1,  0,
        0, -1
    };
    /** @brief Alias for the flattened discrete velocity table. */
    static constexpr auto c = directions;
};

/**
 * @brief Three-dimensional seven-velocity lattice for passive scalar transport.
 *
 * D3Q7 contains the rest population and six axial links. The axis weights are
 * chosen so the second moment gives `c_s^2 = 1/4`.
 */
struct D3Q7 {
    /** @brief Number of spatial dimensions. */
    static constexpr int D = 3;
    /** @brief Number of scalar populations per lattice node. */
    static constexpr int Q = 7;
    /** @brief Squared lattice sound speed for scalar transport. */
    static constexpr double cs2 = 1.0 / 4.0;

    /** @brief Standard D3Q7 scalar lattice weights. */
    static constexpr std::array<double, Q> weights{
        1.0 / 4.0,
        1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0,
        1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0
    };

    /** @brief Flattened discrete velocity vectors stored as `(cx, cy, cz)` triples. */
    static constexpr std::array<int, Q * D> directions{
        0,  0,  0,
        1,  0,  0,
       -1,  0,  0,
        0,  1,  0,
        0, -1,  0,
        0,  0,  1,
        0,  0, -1
    };
    /** @brief Alias for the flattened discrete velocity table. */
    static constexpr auto c = directions;
};

static_assert(IsLatticeModel<D2Q9>);
static_assert(IsLatticeModel<D3Q19>);
static_assert(IsLatticeModel<D3Q27>);
static_assert(IsLatticeModel<D2Q5>);
static_assert(IsLatticeModel<D3Q7>);

} // namespace lbm

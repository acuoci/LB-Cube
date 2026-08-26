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
 * constexpr `int D`, static constexpr `int Q`, `std::array<double, Q> weights`,
 * and `std::array<int, Q * D> directions`.
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
        requires std::same_as<
            std::remove_cv_t<decltype(Lattice::directions)>,
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
};

static_assert(IsLatticeModel<D2Q9>);
static_assert(IsLatticeModel<D3Q19>);
static_assert(IsLatticeModel<D3Q27>);

} // namespace lbm

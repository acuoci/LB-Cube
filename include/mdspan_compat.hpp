#pragma once
/**
 * @file mdspan_compat.hpp
 * @brief Compatibility aliases for native C++23 mdspan and Kokkos standalone mdspan.
 *
 * GCC versions supported by recent CUDA toolchains may not provide the final
 * C++23 `<mdspan>` header. This shim lets the rest of LB-Cube name mdspan types
 * through `lbm::...` aliases while CMake selects either the standard library
 * implementation or Kokkos' `std::experimental` implementation.
 */

#ifdef USE_KOKKOS_MDSPAN
    #include <experimental/mdspan>
    namespace lbm {
        using std::experimental::mdspan;
        using std::experimental::extents;
        using std::experimental::dextents;
        using std::experimental::layout_right;
        using std::experimental::layout_left;
        using std::experimental::layout_stride;
        using std::experimental::default_accessor;
        inline constexpr auto dynamic_extent = std::experimental::dynamic_extent;
    }
#else
    #include <mdspan>
    namespace lbm {
        using std::mdspan;
        using std::extents;
        using std::dextents;
        using std::layout_right;
        using std::layout_left;
        using std::layout_stride;
        using std::default_accessor;
        inline constexpr auto dynamic_extent = std::dynamic_extent;
    }
#endif

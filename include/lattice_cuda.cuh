#pragma once

#include <cuda_runtime.h>

#include <concepts>
#include <cstddef>

#include "lattice_memory.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

namespace lbm {

inline dim3 default_cuda_block_size() noexcept {
    return dim3{16, 8, 1};
}

#ifdef __CUDACC__
template <IsLatticeModel Lattice, std::floating_point Real>
__global__ void kernel_step(
    const Real* current_populations,
    Real* next_populations,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega);
#endif

template <IsLatticeModel Lattice, std::floating_point Real>
cudaError_t launch_step_gpu(
    const Real* current_populations,
    Real* next_populations,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega,
    dim3 block = default_cuda_block_size());

template <IsLatticeModel Lattice, std::floating_point Real>
cudaError_t launch_step_gpu(
    const LatticeMemory<Lattice, Real>& mem,
    const Real* current_populations,
    Real* next_populations,
    Real omega,
    dim3 block = default_cuda_block_size());

} // namespace lbm

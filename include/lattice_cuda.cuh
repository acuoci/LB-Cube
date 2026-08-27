#pragma once
/**
 * @file lattice_cuda.cuh
 * @brief CUDA kernel and launcher declarations for GPU LBM time stepping.
 *
 * The CUDA backend operates on raw device pointers laid out in the same SoA order
 * as the host mdspan views: `[Q, Z, Y, X]` with `X` contiguous. The declarations
 * are separated from the implementation so host targets can link against the CUDA
 * backend without compiling device code in ordinary C++ translation units.
 */

#include <cuda_runtime.h>

#include <concepts>
#include <cstddef>

#include "lattice_memory.hpp"
#include "lattice_physics.hpp"
#include "lattice_traits.hpp"

namespace lbm {

/**
 * @brief Default CUDA block dimensions used by the host launcher.
 *
 * The shape favors contiguous `x` traversal while leaving the value easy to
 * override for architecture-specific tuning.
 *
 * @return `dim3{16, 8, 1}`.
 */
inline dim3 default_cuda_block_size() noexcept {
    return dim3{16, 8, 1};
}

#ifdef __CUDACC__
/**
 * @brief CUDA kernel implementing one fused pull-streaming BGK update.
 *
 * Each CUDA thread maps to one spatial lattice node. It gathers all incoming
 * populations from periodic neighbors in the current buffer, calls the stateless
 * BGK collision function, and writes post-collision populations to the next
 * buffer using the same flat SoA layout as the CPU path.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_populations Device pointer to the read-side population buffer.
 * @param next_populations Device pointer to the write-side population buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z, or 1 for 2D lattices.
 * @param omega BGK relaxation frequency.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
__global__ void kernel_step(
    const Real* current_populations,
    Real* next_populations,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega);

/**
 * @brief CUDA kernel implementing one passive-scalar pull-streaming update.
 *
 * Each thread reconstructs the local fluid velocity from the fluid population
 * buffer at its cell, pulls scalar populations from neighboring scalar cells,
 * applies scalar BGK relaxation and source forcing, and writes to the scalar
 * next buffer.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_scalar Device pointer to the read-side scalar population buffer.
 * @param next_scalar Device pointer to the write-side scalar population buffer.
 * @param current_fluid Device pointer to the read-side fluid population buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z, or 1 for 2D lattices.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param source_term Local scalar source contribution applied uniformly here.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
__global__ void kernel_scalar_step(
    const Real* current_scalar,
    Real* next_scalar,
    const Real* current_fluid,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega_c,
    Real source_term);

/**
 * @brief CUDA kernel for fused second-order `A + B -> C` scalar consumption.
 *
 * The kernel reconstructs fluid velocity once per destination cell, pulls both
 * reactant species from their upstream scalar neighbors, computes the exact
 * one-step batch consumption source, and applies it during scalar collision.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_a Device pointer to species A read buffer.
 * @param next_a Device pointer to species A write buffer.
 * @param current_b Device pointer to species B read buffer.
 * @param next_b Device pointer to species B write buffer.
 * @param current_fluid Device pointer to fluid read buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z, or 1 for 2D lattices.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param k_react Second-order reaction-rate constant.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
__global__ void kernel_reaction_AB(
    const Real* current_a,
    Real* next_a,
    const Real* current_b,
    Real* next_b,
    const Real* current_fluid,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega_c,
    Real k_react);
#endif

/**
 * @brief Launch the CUDA fused collision-streaming kernel over a domain.
 *
 * The function computes a grid from the supplied block shape, normalizes the z
 * extent for 2D lattices, enqueues `kernel_step`, and returns the immediate CUDA
 * launch status without synchronizing. This lets benchmarks control
 * synchronization explicitly.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_populations Device pointer to the read-side population buffer.
 * @param next_populations Device pointer to the write-side population buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z, or 1 for 2D lattices.
 * @param omega BGK relaxation frequency.
 * @param block CUDA block dimensions used for launch tuning.
 * @return Immediate CUDA error status from validation or kernel launch.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
cudaError_t launch_step_gpu(
    const Real* current_populations,
    Real* next_populations,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega,
    dim3 block = default_cuda_block_size());

/**
 * @brief Launch the CUDA passive-scalar advection-diffusion kernel.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_scalar Device pointer to the scalar read buffer.
 * @param next_scalar Device pointer to the scalar write buffer.
 * @param current_fluid Device pointer to the fluid read buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z, or 1 for 2D lattices.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param source_term Local scalar source contribution, defaulting to zero.
 * @param block CUDA block dimensions used for launch tuning.
 * @return Immediate CUDA error status from validation or kernel launch.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
cudaError_t launch_scalar_step_gpu(
    const Real* current_scalar,
    Real* next_scalar,
    const Real* current_fluid,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega_c,
    Real source_term = Real{},
    dim3 block = default_cuda_block_size());

/**
 * @brief Launch the CUDA scalar step using extents stored by host memory.
 *
 * This overload is a metadata convenience only; all population pointers are
 * expected to point to device memory.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param scalar_mem Host scalar memory object used only for domain extents.
 * @param fluid_mem Host fluid memory object used only for dimension consistency.
 * @param current_scalar Device pointer to the scalar read buffer.
 * @param next_scalar Device pointer to the scalar write buffer.
 * @param current_fluid Device pointer to the fluid read buffer.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param source_term Local scalar source contribution, defaulting to zero.
 * @param block CUDA block dimensions used for launch tuning.
 * @return Immediate CUDA error status from validation or kernel launch.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
cudaError_t launch_scalar_step_gpu(
    const LatticeMemory<ScalarLattice, Real>& scalar_mem,
    const LatticeMemory<FluidLattice, Real>& fluid_mem,
    const Real* current_scalar,
    Real* next_scalar,
    const Real* current_fluid,
    Real omega_c,
    Real source_term = Real{},
    dim3 block = default_cuda_block_size());

/**
 * @brief Launch the CUDA fused `A + B -> C` reaction step.
 *
 * @tparam FluidLattice Fluid lattice traits type satisfying `IsLatticeModel`.
 * @tparam ScalarLattice Scalar lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param current_a Device pointer to species A read buffer.
 * @param next_a Device pointer to species A write buffer.
 * @param current_b Device pointer to species B read buffer.
 * @param next_b Device pointer to species B write buffer.
 * @param current_fluid Device pointer to fluid read buffer.
 * @param x_extent Number of nodes in x.
 * @param y_extent Number of nodes in y.
 * @param z_extent Number of nodes in z, or 1 for 2D lattices.
 * @param omega_c Scalar BGK relaxation frequency.
 * @param k_react Second-order reaction-rate constant.
 * @param block CUDA block dimensions used for launch tuning.
 * @return Immediate CUDA error status from validation or kernel launch.
 */
template <
    IsLatticeModel FluidLattice,
    IsLatticeModel ScalarLattice,
    std::floating_point Real>
cudaError_t launch_reaction_AB_gpu(
    const Real* current_a,
    Real* next_a,
    const Real* current_b,
    Real* next_b,
    const Real* current_fluid,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega_c,
    Real k_react,
    dim3 block = default_cuda_block_size());

/**
 * @brief Launch the CUDA step using extents stored by a host `LatticeMemory`.
 *
 * This overload is a convenience bridge while device memory is managed
 * separately. It does not copy data; callers still provide the raw device
 * buffers corresponding to current and next populations.
 *
 * @tparam Lattice Lattice traits type satisfying `IsLatticeModel`.
 * @tparam Real Floating-point population precision.
 * @param mem Host memory object used only for domain extents.
 * @param current_populations Device pointer to the read-side population buffer.
 * @param next_populations Device pointer to the write-side population buffer.
 * @param omega BGK relaxation frequency.
 * @param block CUDA block dimensions used for launch tuning.
 * @return Immediate CUDA error status from validation or kernel launch.
 */
template <IsLatticeModel Lattice, std::floating_point Real>
cudaError_t launch_step_gpu(
    const LatticeMemory<Lattice, Real>& mem,
    const Real* current_populations,
    Real* next_populations,
    Real omega,
    dim3 block = default_cuda_block_size());

} // namespace lbm

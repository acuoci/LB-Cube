#include "lattice_cuda.cuh"

#include <array>

namespace lbm {

namespace detail {

__device__ inline std::size_t periodic_pull_index(
    std::size_t coordinate,
    int velocity,
    std::size_t extent) {
    const auto signed_coordinate = static_cast<std::ptrdiff_t>(coordinate);
    const auto signed_extent = static_cast<std::ptrdiff_t>(extent);
    const auto signed_index = signed_coordinate - static_cast<std::ptrdiff_t>(velocity) + signed_extent;
    return static_cast<std::size_t>(signed_index % signed_extent);
}

__host__ __device__ inline std::size_t population_index(
    int i,
    std::size_t x,
    std::size_t y,
    std::size_t z,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent) {
    return (((static_cast<std::size_t>(i) * z_extent + z) * y_extent + y) * x_extent + x);
}

} // namespace detail

template <IsLatticeModel Lattice, std::floating_point Real>
__global__ void kernel_step(
    const Real* current_populations,
    Real* next_populations,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega) {
    const std::size_t x = static_cast<std::size_t>(blockIdx.x * blockDim.x + threadIdx.x);
    const std::size_t y = static_cast<std::size_t>(blockIdx.y * blockDim.y + threadIdx.y);
    const std::size_t z = static_cast<std::size_t>(blockIdx.z * blockDim.z + threadIdx.z);

    if (x >= x_extent || y >= y_extent || z >= z_extent) {
        return;
    }

    std::array<Real, static_cast<std::size_t>(Lattice::Q)> local_pops{};

    if constexpr (Lattice::D == 2) {
        for (int i = 0; i < Lattice::Q; ++i) {
            const auto direction_offset = static_cast<std::size_t>(i * Lattice::D);
            const int cx = Lattice::directions[direction_offset];
            const int cy = Lattice::directions[direction_offset + 1];
            const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
            const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);

            local_pops[static_cast<std::size_t>(i)] =
                current_populations[detail::population_index(
                    i, nx, ny, 0, x_extent, y_extent, z_extent)];
        }

        collide_bgk<Lattice, Real>(local_pops, omega);

        for (int i = 0; i < Lattice::Q; ++i) {
            next_populations[detail::population_index(
                i, x, y, 0, x_extent, y_extent, z_extent)] =
                local_pops[static_cast<std::size_t>(i)];
        }
    } else {
        for (int i = 0; i < Lattice::Q; ++i) {
            const auto direction_offset = static_cast<std::size_t>(i * Lattice::D);
            const int cx = Lattice::directions[direction_offset];
            const int cy = Lattice::directions[direction_offset + 1];
            const int cz = Lattice::directions[direction_offset + 2];
            const std::size_t nx = detail::periodic_pull_index(x, cx, x_extent);
            const std::size_t ny = detail::periodic_pull_index(y, cy, y_extent);
            const std::size_t nz = detail::periodic_pull_index(z, cz, z_extent);

            local_pops[static_cast<std::size_t>(i)] =
                current_populations[detail::population_index(
                    i, nx, ny, nz, x_extent, y_extent, z_extent)];
        }

        collide_bgk<Lattice, Real>(local_pops, omega);

        for (int i = 0; i < Lattice::Q; ++i) {
            next_populations[detail::population_index(
                i, x, y, z, x_extent, y_extent, z_extent)] =
                local_pops[static_cast<std::size_t>(i)];
        }
    }
}

template <IsLatticeModel Lattice, std::floating_point Real>
cudaError_t launch_step_gpu(
    const Real* current_populations,
    Real* next_populations,
    std::size_t x_extent,
    std::size_t y_extent,
    std::size_t z_extent,
    Real omega,
    dim3 block) {
    if (current_populations == nullptr || next_populations == nullptr) {
        return cudaErrorInvalidDevicePointer;
    }

    if (block.x == 0 || block.y == 0 || block.z == 0) {
        return cudaErrorInvalidConfiguration;
    }

    if constexpr (Lattice::D == 2) {
        z_extent = 1;
        block.z = 1;
    }

    if (x_extent == 0 || y_extent == 0 || z_extent == 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 grid{
        static_cast<unsigned int>((x_extent + block.x - 1) / block.x),
        static_cast<unsigned int>((y_extent + block.y - 1) / block.y),
        static_cast<unsigned int>((z_extent + block.z - 1) / block.z)
    };

    kernel_step<Lattice, Real><<<grid, block>>>(
        current_populations,
        next_populations,
        x_extent,
        y_extent,
        z_extent,
        omega);

    return cudaGetLastError();
}

template <IsLatticeModel Lattice, std::floating_point Real>
cudaError_t launch_step_gpu(
    const LatticeMemory<Lattice, Real>& mem,
    const Real* current_populations,
    Real* next_populations,
    Real omega,
    dim3 block) {
    return launch_step_gpu<Lattice, Real>(
        current_populations,
        next_populations,
        mem.x_extent(),
        mem.y_extent(),
        mem.z_extent(),
        omega,
        block);
}

template __global__ void kernel_step<D2Q9, float>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D2Q9, double>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);
template __global__ void kernel_step<D3Q19, float>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D3Q19, double>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);
template __global__ void kernel_step<D3Q27, float>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float);
template __global__ void kernel_step<D3Q27, double>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double);

template cudaError_t launch_step_gpu<D2Q9, float>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D2Q9, double>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);
template cudaError_t launch_step_gpu<D3Q19, float>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D3Q19, double>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);
template cudaError_t launch_step_gpu<D3Q27, float>(
    const float*, float*, std::size_t, std::size_t, std::size_t, float, dim3);
template cudaError_t launch_step_gpu<D3Q27, double>(
    const double*, double*, std::size_t, std::size_t, std::size_t, double, dim3);

template cudaError_t launch_step_gpu<D2Q9, float>(
    const LatticeMemory<D2Q9, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D2Q9, double>(
    const LatticeMemory<D2Q9, double>&, const double*, double*, double, dim3);
template cudaError_t launch_step_gpu<D3Q19, float>(
    const LatticeMemory<D3Q19, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D3Q19, double>(
    const LatticeMemory<D3Q19, double>&, const double*, double*, double, dim3);
template cudaError_t launch_step_gpu<D3Q27, float>(
    const LatticeMemory<D3Q27, float>&, const float*, float*, float, dim3);
template cudaError_t launch_step_gpu<D3Q27, double>(
    const LatticeMemory<D3Q27, double>&, const double*, double*, double, dim3);

} // namespace lbm

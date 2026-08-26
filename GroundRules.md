# Codex / AI Coding Ground Rules: C++/CUDA LBM Solver

## 1. Project Philosophy & Scope
This project is a Lattice Boltzmann Method (LBM) solver designed strictly for academic research in cubic/rectangular domains with periodic boundary conditions. 
- **Goal:** High performance, readability, and modern C++ design.
- **Avoid Over-engineering:** Do not implement deep inheritance hierarchies, virtual dispatch in performance-critical loops, or abstractions for complex geometries/AMR. 
- **Future Extensibility:** Keep boundaries and macroscopic transport modular to allow for future additions like passive scalar transport or localized boundary variations.

## 2. Core Architecture & Physics
- **Lattice Models:** Support multiple lattices (D2Q9, D3Q19, D3Q27) via compile-time template traits (Traits/Policies), avoiding object-oriented runtime polymorphism (`virtual` functions).
- **Collision Operator:** BGK (Bhatnagar-Gross-Krook).
- **Memory Layout:** Structure of Arrays (SoA). This is strictly required for CUDA memory coalescing. Do not use Array of Structures (AoS) for the particle populations.
- **Algorithm Scheme:** Double-Population (Ping-Pong). Allocate two SoA grids (`pop_current` and `pop_next`) to fuse collision and streaming without race conditions.

## 3. Technology Stack
- **Language Standard:** Modern C++ (C++23). Exploit features like C++23 Concepts (to constrain lattice templates), `std::mdspan` for multi-dimensional array viewing, improved `<ranges>`, and `std::expected`.
- **Linear Algebra / Vectors:** Eigen C++ (`Eigen::Vector3d` and `Eigen::Vector2d` depending on the lattice dimension for macroscopic variables).
- **GPU Backend:** Native CUDA. Ensure the host compiler (e.g., GCC 13+) and `nvcc` are compatible with C++23.
- **Build System:** Modern CMake (`CMakeLists.txt`), utilizing target-based properties (`cxx_std_23`), `enable_language(CUDA)`, and fetching external dependencies (Eigen, GTest).
- **Testing:** GoogleTest (GTest) for unit tests (e.g., verifying exact mass conservation, local equilibrium correctness).

## 4. Coding Standards
- **Stateless Physics:** The core LBM math (e.g., computing equilibrium distribution, relaxation) must be implemented as stateless, `inline`, `__host__ __device__` templated functions. This ensures exact code reuse between the CPU reference implementation and the CUDA kernels for any lattice trait.
- **Memory Management:** No raw `new`/`delete`. Use `std::vector` or smart pointers for host memory, and encapsulate CUDA memory management (`cudaMalloc`/`cudaFree`) safely within RAII wrapper classes or managed memory pointers.
- **Interfaces First:** Always define and review header files (`.hpp`) and data structures before writing implementation (`.cpp` or `.cu`).

## 5. Implementation Workflow (For Codex)
When instructed to write code, follow this sequence:
1. **Lattice Traits:** Define the `constexpr` structs for D2Q9, D3Q19, and D3Q27, and constrain them with a C++23 Concept.
2. **Data Structures:** Define the templated SoA memory classes and macroscopic struct definitions.
3. **Local Physics:** Implement the isolated, stateless BGK math functions templated on the lattice model.
4. **CPU Backend:** Implement the fused collision-streaming loop and periodic boundaries for the host.
5. **GPU Backend:** Port the grid traversal to a CUDA kernel and write the launch configuration.
6. **Verification & Build:** Setup CMake and write benchmark/validation tests.
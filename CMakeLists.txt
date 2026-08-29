cmake_minimum_required(VERSION 3.24)

project(
    LBCube
    VERSION 0.1.0
    DESCRIPTION "C++23/CUDA Lattice Boltzmann Method solver"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LB_CUBE_ENABLE_CUDA "Build the CUDA backend" ON)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)

add_library(lbm_core INTERFACE)
add_library(lbm::core ALIAS lbm_core)

target_include_directories(
    lbm_core
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(
    lbm_core
    INTERFACE
        cxx_std_23
)

target_link_libraries(
    lbm_core
    INTERFACE
        Eigen3::Eigen
)

if(LB_CUBE_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(lbm_cuda STATIC src/lattice_cuda.cu)
    add_library(lbm::cuda ALIAS lbm_cuda)

    target_link_libraries(
        lbm_cuda
        PUBLIC
            lbm::core
            CUDA::cudart
    )

    target_compile_features(
        lbm_cuda
        PUBLIC
            cxx_std_23
    )

    set_target_properties(
        lbm_cuda
        PROPERTIES
            CXX_STANDARD 23
            CXX_STANDARD_REQUIRED YES
            CXX_EXTENSIONS NO
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED YES
            CUDA_EXTENSIONS NO
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            CUDA_ARCHITECTURES native
            POSITION_INDEPENDENT_CODE ON
    )
endif()

add_executable(lbm_sim src/main.cpp)

target_link_libraries(
    lbm_sim
    PRIVATE
        lbm::core
)

if(LB_CUBE_ENABLE_CUDA)
    target_link_libraries(lbm_sim PRIVATE lbm::cuda)
    target_compile_definitions(lbm_sim PRIVATE LB_CUBE_ENABLE_CUDA=1)
else()
    target_compile_definitions(lbm_sim PRIVATE LB_CUBE_ENABLE_CUDA=0)
endif()

target_compile_features(
    lbm_sim
    PRIVATE
        cxx_std_23
)

set_target_properties(
    lbm_sim
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

add_executable(lbm_reaction_sim src/simulate_reaction.cpp)

target_link_libraries(
    lbm_reaction_sim
    PRIVATE
        lbm::core
)

if(LB_CUBE_ENABLE_CUDA)
    target_link_libraries(lbm_reaction_sim PRIVATE lbm::cuda)
    target_compile_definitions(lbm_reaction_sim PRIVATE LB_CUBE_ENABLE_CUDA=1)
else()
    target_compile_definitions(lbm_reaction_sim PRIVATE LB_CUBE_ENABLE_CUDA=0)
endif()

target_compile_features(
    lbm_reaction_sim
    PRIVATE
        cxx_std_23
)

set_target_properties(
    lbm_reaction_sim
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

add_executable(lbm_compare_operators src/compare_operators.cpp)

target_link_libraries(
    lbm_compare_operators
    PRIVATE
        lbm::core
)

target_compile_features(
    lbm_compare_operators
    PRIVATE
        cxx_std_23
)

set_target_properties(
    lbm_compare_operators
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

enable_testing()

add_executable(lbm_validation_tests tests/test_validation.cpp)

target_link_libraries(
    lbm_validation_tests
    PRIVATE
        lbm::core
        GTest::gtest_main
)

target_compile_features(
    lbm_validation_tests
    PRIVATE
        cxx_std_23
)

set_target_properties(
    lbm_validation_tests
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

include(GoogleTest)
gtest_discover_tests(lbm_validation_tests)

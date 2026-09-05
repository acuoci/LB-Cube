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
option(LB_CUBE_ENABLE_OPENMP "Enable OpenMP parallel CPU loops when available" ON)
option(LB_CUBE_ENABLE_FFTW "Enable optional FFTW support for future spectral modules" OFF)
option(USE_KOKKOS_MDSPAN "Use Kokkos standalone mdspan instead of native C++23 std::mdspan" OFF)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
set(LB_CUBE_OPENMP_CXX_FLAGS "" CACHE STRING "Manual OpenMP CXX compile flags, e.g. -fopenmp")
set(LB_CUBE_OPENMP_LINK_FLAGS "" CACHE STRING "Manual OpenMP linker flags, e.g. -fopenmp")
set(LB_CUBE_OPENMP_LIBRARIES "" CACHE STRING "Manual OpenMP libraries, e.g. /path/to/libgomp.so")
set(LB_CUBE_OPENMP_INCLUDE_DIRS "" CACHE STRING "Manual OpenMP include directories, if omp.h is not on the default include path")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)

if(LB_CUBE_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
endif()

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

if(LB_CUBE_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: found and activated for LB-Cube CPU kernels")
    if(DEFINED OpenMP_CXX_VERSION)
        message(STATUS "OpenMP: CXX version ${OpenMP_CXX_VERSION}")
    endif()
    message(STATUS "OpenMP: CXX flags '${OpenMP_CXX_FLAGS}'")
    message(STATUS "OpenMP: CXX libraries '${OpenMP_CXX_LIBRARIES}'")
    target_link_libraries(
        lbm_core
        INTERFACE
            OpenMP::OpenMP_CXX
    )
    target_compile_definitions(
        lbm_core
        INTERFACE
            LB_CUBE_OPENMP_ENABLED=1
    )
    if(OpenMP_CXX_FLAGS)
        separate_arguments(OpenMP_CXX_FLAGS_LIST NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
        set(LB_CUBE_OPENMP_CXX_OPTIONS)
        foreach(openmp_cxx_flag IN LISTS OpenMP_CXX_FLAGS_LIST)
            list(APPEND LB_CUBE_OPENMP_CXX_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${openmp_cxx_flag}>")
        endforeach()
        target_compile_options(
            lbm_core
            INTERFACE
                ${LB_CUBE_OPENMP_CXX_OPTIONS}
        )
    endif()
elseif(LB_CUBE_ENABLE_OPENMP AND (LB_CUBE_OPENMP_CXX_FLAGS OR LB_CUBE_OPENMP_LINK_FLAGS OR LB_CUBE_OPENMP_LIBRARIES OR LB_CUBE_OPENMP_INCLUDE_DIRS))
    message(WARNING "OpenMP: CMake auto-detection failed, using user-provided LB_CUBE_OPENMP_* settings")
    message(STATUS "OpenMP: manual CXX flags '${LB_CUBE_OPENMP_CXX_FLAGS}'")
    message(STATUS "OpenMP: manual link flags '${LB_CUBE_OPENMP_LINK_FLAGS}'")
    message(STATUS "OpenMP: manual libraries '${LB_CUBE_OPENMP_LIBRARIES}'")
    message(STATUS "OpenMP: manual include dirs '${LB_CUBE_OPENMP_INCLUDE_DIRS}'")

    target_compile_definitions(
        lbm_core
        INTERFACE
            LB_CUBE_OPENMP_ENABLED=1
    )

    if(LB_CUBE_OPENMP_INCLUDE_DIRS)
        separate_arguments(LB_CUBE_OPENMP_INCLUDE_DIRS_LIST NATIVE_COMMAND "${LB_CUBE_OPENMP_INCLUDE_DIRS}")
        target_include_directories(
            lbm_core
            INTERFACE
                ${LB_CUBE_OPENMP_INCLUDE_DIRS_LIST}
        )
    endif()

    if(LB_CUBE_OPENMP_CXX_FLAGS)
        separate_arguments(LB_CUBE_OPENMP_CXX_FLAGS_LIST NATIVE_COMMAND "${LB_CUBE_OPENMP_CXX_FLAGS}")
        set(LB_CUBE_MANUAL_OPENMP_CXX_OPTIONS)
        foreach(openmp_cxx_flag IN LISTS LB_CUBE_OPENMP_CXX_FLAGS_LIST)
            list(APPEND LB_CUBE_MANUAL_OPENMP_CXX_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${openmp_cxx_flag}>")
        endforeach()
        target_compile_options(
            lbm_core
            INTERFACE
                ${LB_CUBE_MANUAL_OPENMP_CXX_OPTIONS}
        )
    endif()

    if(LB_CUBE_OPENMP_LINK_FLAGS)
        separate_arguments(LB_CUBE_OPENMP_LINK_FLAGS_LIST NATIVE_COMMAND "${LB_CUBE_OPENMP_LINK_FLAGS}")
        target_link_options(
            lbm_core
            INTERFACE
                ${LB_CUBE_OPENMP_LINK_FLAGS_LIST}
        )
    endif()

    if(LB_CUBE_OPENMP_LIBRARIES)
        separate_arguments(LB_CUBE_OPENMP_LIBRARIES_LIST NATIVE_COMMAND "${LB_CUBE_OPENMP_LIBRARIES}")
        target_link_libraries(
            lbm_core
            INTERFACE
                ${LB_CUBE_OPENMP_LIBRARIES_LIST}
        )
    endif()
elseif(LB_CUBE_ENABLE_OPENMP)
    message(WARNING "OpenMP: not found; LB-Cube CPU kernels will build in serial mode. Provide LB_CUBE_OPENMP_CXX_FLAGS/LINK_FLAGS/LIBRARIES/INCLUDE_DIRS to force a manual OpenMP configuration.")
else()
    message(STATUS "OpenMP: disabled by LB_CUBE_ENABLE_OPENMP=OFF")
endif()

if(USE_KOKKOS_MDSPAN)
    find_package(mdspan REQUIRED)
    target_compile_definitions(
        lbm_core
        INTERFACE
            USE_KOKKOS_MDSPAN
    )
    target_link_libraries(
        lbm_core
        INTERFACE
            std::mdspan
    )
endif()

if(LB_CUBE_ENABLE_FFTW)
    add_library(lbm_fftw INTERFACE)
    add_library(lbm::fftw ALIAS lbm_fftw)

    set(LB_CUBE_FFTW_DETECTION_METHOD "")
    set(LB_CUBE_FFTW_INCLUDE_DIRS "")
    set(LB_CUBE_FFTW_LIBRARY_DIR_HINTS "")

    find_package(FFTW3 QUIET CONFIG)
    if(FFTW3_FOUND)
        set(LB_CUBE_FFTW_DETECTION_METHOD "CMake config (${FFTW3_CONFIG})")
        set(LB_CUBE_FFTW_INCLUDE_DIRS ${FFTW3_INCLUDE_DIRS})
        set(LB_CUBE_FFTW_LIBRARY_DIR_HINTS ${FFTW3_LIBRARY_DIRS})
    else()
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(PC_FFTW3 QUIET fftw3)
            if(PC_FFTW3_FOUND)
                set(LB_CUBE_FFTW_DETECTION_METHOD "pkg-config (fftw3)")
                set(LB_CUBE_FFTW_INCLUDE_DIRS ${PC_FFTW3_INCLUDE_DIRS})
                set(LB_CUBE_FFTW_LIBRARY_DIR_HINTS ${PC_FFTW3_LIBRARY_DIRS})
            endif()
        endif()
    endif()

    find_path(
        LB_CUBE_FFTW_INCLUDE_DIR
        NAMES fftw3.h
        HINTS ${LB_CUBE_FFTW_INCLUDE_DIRS}
    )
    find_library(
        LB_CUBE_FFTW_LIBRARY
        NAMES fftw3
        HINTS ${LB_CUBE_FFTW_LIBRARY_DIR_HINTS}
    )

    if(NOT LB_CUBE_FFTW_INCLUDE_DIR OR NOT LB_CUBE_FFTW_LIBRARY)
        find_path(LB_CUBE_FFTW_INCLUDE_DIR NAMES fftw3.h)
        find_library(LB_CUBE_FFTW_LIBRARY NAMES fftw3)
        if(LB_CUBE_FFTW_INCLUDE_DIR AND LB_CUBE_FFTW_LIBRARY AND NOT LB_CUBE_FFTW_DETECTION_METHOD)
            set(LB_CUBE_FFTW_DETECTION_METHOD "find_path/find_library fallback")
        endif()
    endif()

    if(NOT LB_CUBE_FFTW_INCLUDE_DIR OR NOT LB_CUBE_FFTW_LIBRARY)
        message(FATAL_ERROR "LB_CUBE_ENABLE_FFTW=ON, but FFTW double-precision headers/libraries were not found")
    endif()
    if(NOT LB_CUBE_FFTW_DETECTION_METHOD)
        set(LB_CUBE_FFTW_DETECTION_METHOD "find_path/find_library fallback")
    endif()

    if(NOT TARGET FFTW3::fftw3)
        add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
        set_target_properties(
            FFTW3::fftw3
            PROPERTIES
                IMPORTED_LOCATION "${LB_CUBE_FFTW_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${LB_CUBE_FFTW_INCLUDE_DIR}"
        )
    endif()

    target_link_libraries(
        lbm_fftw
        INTERFACE
            FFTW3::fftw3
    )
    target_compile_definitions(
        lbm_fftw
        INTERFACE
            LB_CUBE_HAS_FFTW=1
    )

    get_filename_component(LB_CUBE_FFTW_LIBRARY_DIR "${LB_CUBE_FFTW_LIBRARY}" DIRECTORY)
    find_library(LB_CUBE_FFTW_FLOAT_LIBRARY NAMES fftw3f HINTS "${LB_CUBE_FFTW_LIBRARY_DIR}" ${LB_CUBE_FFTW_LIBRARY_DIR_HINTS})
    find_library(LB_CUBE_FFTW_LONG_DOUBLE_LIBRARY NAMES fftw3l HINTS "${LB_CUBE_FFTW_LIBRARY_DIR}" ${LB_CUBE_FFTW_LIBRARY_DIR_HINTS})
    find_library(LB_CUBE_FFTW_THREADS_LIBRARY NAMES fftw3_threads HINTS "${LB_CUBE_FFTW_LIBRARY_DIR}" ${LB_CUBE_FFTW_LIBRARY_DIR_HINTS})
    find_library(LB_CUBE_FFTW_FLOAT_THREADS_LIBRARY NAMES fftw3f_threads HINTS "${LB_CUBE_FFTW_LIBRARY_DIR}" ${LB_CUBE_FFTW_LIBRARY_DIR_HINTS})
    find_library(LB_CUBE_FFTW_OMP_LIBRARY NAMES fftw3_omp HINTS "${LB_CUBE_FFTW_LIBRARY_DIR}" ${LB_CUBE_FFTW_LIBRARY_DIR_HINTS})
    find_library(LB_CUBE_FFTW_MPI_LIBRARY NAMES fftw3_mpi HINTS "${LB_CUBE_FFTW_LIBRARY_DIR}" ${LB_CUBE_FFTW_LIBRARY_DIR_HINTS})

    message(STATUS "FFTW: enabled")
    message(STATUS "FFTW: detection method ${LB_CUBE_FFTW_DETECTION_METHOD}")
    message(STATUS "FFTW: include directory ${LB_CUBE_FFTW_INCLUDE_DIR}")
    message(STATUS "FFTW: double library ${LB_CUBE_FFTW_LIBRARY}")
    message(STATUS "FFTW: float library ${LB_CUBE_FFTW_FLOAT_LIBRARY}")
    message(STATUS "FFTW: long double library ${LB_CUBE_FFTW_LONG_DOUBLE_LIBRARY}")
    message(STATUS "FFTW: threads library ${LB_CUBE_FFTW_THREADS_LIBRARY}")
    message(STATUS "FFTW: float threads library ${LB_CUBE_FFTW_FLOAT_THREADS_LIBRARY}")
    message(STATUS "FFTW: OpenMP library ${LB_CUBE_FFTW_OMP_LIBRARY}")
    message(STATUS "FFTW: MPI library ${LB_CUBE_FFTW_MPI_LIBRARY}")

    if(LB_CUBE_FFTW_THREADS_LIBRARY)
        add_library(lbm_fftw_threads INTERFACE)
        add_library(lbm::fftw_threads ALIAS lbm_fftw_threads)
        target_link_libraries(
            lbm_fftw_threads
            INTERFACE
                "${LB_CUBE_FFTW_THREADS_LIBRARY}"
                lbm::fftw
        )
        target_compile_definitions(
            lbm_fftw_threads
            INTERFACE
                LB_CUBE_HAS_FFTW_THREADS=1
        )
    endif()
else()
    message(STATUS "FFTW: disabled by LB_CUBE_ENABLE_FFTW=OFF")
endif()

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

add_executable(lbm_turbulent_shear_layer src/turbulent_shear_layer.cpp)

target_link_libraries(
    lbm_turbulent_shear_layer
    PRIVATE
        lbm::core
)

target_compile_features(
    lbm_turbulent_shear_layer
    PRIVATE
        cxx_std_23
)

set_target_properties(
    lbm_turbulent_shear_layer
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

add_executable(lbm_turbulent_tgv_3d src/turbulent_tgv_3d.cpp)

target_link_libraries(
    lbm_turbulent_tgv_3d
    PRIVATE
        lbm::core
)

target_compile_features(
    lbm_turbulent_tgv_3d
    PRIVATE
        cxx_std_23
)

set_target_properties(
    lbm_turbulent_tgv_3d
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

add_executable(lbm_turbulent_reactive_shear_3d src/turbulent_reactive_shear_3d.cpp)

target_link_libraries(
    lbm_turbulent_reactive_shear_3d
    PRIVATE
        lbm::core
)

target_compile_features(
    lbm_turbulent_reactive_shear_3d
    PRIVATE
        cxx_std_23
)

set_target_properties(
    lbm_turbulent_reactive_shear_3d
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

add_executable(lbm_turbulent_reactive_double_shear_3d src/turbulent_reactive_double_shear_3d.cpp)

target_link_libraries(
    lbm_turbulent_reactive_double_shear_3d
    PRIVATE
        lbm::core
)

target_compile_features(
    lbm_turbulent_reactive_double_shear_3d
    PRIVATE
        cxx_std_23
)

set_target_properties(
    lbm_turbulent_reactive_double_shear_3d
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

enable_testing()

if(LB_CUBE_ENABLE_FFTW)
    add_executable(lbm_fftw_test tests/test_fftw.cpp)

    target_link_libraries(
        lbm_fftw_test
        PRIVATE
            lbm::fftw
    )

    target_compile_features(
        lbm_fftw_test
        PRIVATE
            cxx_std_23
    )

    set_target_properties(
        lbm_fftw_test
        PROPERTIES
            CXX_STANDARD 23
            CXX_STANDARD_REQUIRED YES
            CXX_EXTENSIONS NO
    )

    add_test(NAME lbm_fftw_test COMMAND lbm_fftw_test)
endif()

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

add_executable(lbm_double_shear_parameter_tests tests/test_double_shear_parameters.cpp)

target_link_libraries(
    lbm_double_shear_parameter_tests
    PRIVATE
        lbm::core
        GTest::gtest_main
)

target_compile_features(
    lbm_double_shear_parameter_tests
    PRIVATE
        cxx_std_23
)

set_target_properties(
    lbm_double_shear_parameter_tests
    PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

gtest_discover_tests(lbm_double_shear_parameter_tests)

cmake_minimum_required(VERSION 3.24)
project(nbody_louxinyu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nbody_core
    src/config.cpp
    src/io.cpp
    src/cpu_solver.cpp)
target_include_directories(nbody_core PUBLIC include)
target_compile_options(nbody_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(nbody src/main.cpp)
target_link_libraries(nbody PRIVATE nbody_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(nbody_tests tests/test_cpu.cpp)
    target_link_libraries(nbody_tests PRIVATE nbody_core)
    add_test(NAME cpu_core COMMAND nbody_tests)

    add_executable(nbody_io_tests tests/test_io.cpp)
    target_link_libraries(nbody_io_tests PRIVATE nbody_core)
    add_test(NAME io_parsing COMMAND nbody_io_tests)

    add_executable(nbody_trajectory_tests tests/test_trajectory.cpp)
    target_link_libraries(nbody_trajectory_tests PRIVATE nbody_core)
    add_test(NAME trajectory_roundtrip COMMAND nbody_trajectory_tests)

    add_executable(nbody_convergence_tests tests/test_convergence.cpp)
    target_link_libraries(nbody_convergence_tests PRIVATE nbody_core)
    add_test(NAME dt_convergence COMMAND nbody_convergence_tests)
endif()

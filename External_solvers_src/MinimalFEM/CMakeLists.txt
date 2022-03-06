cmake_minimum_required(VERSION 3.11)

project(MinimalFEM LANGUAGES CXX VERSION 1.0 DESCRIPTION "2D Minimal FEM for topology optimization")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -ftree-vectorize")

find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(MinimalFEM ${MINIMAL_FEM_DIR}/main.cpp)

target_link_libraries(MinimalFEM Eigen3::Eigen)

install(TARGETS MinimalFEM DESTINATION HydrOpTop/Solvers)

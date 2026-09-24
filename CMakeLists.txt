cmake_minimum_required(VERSION 3.16...3.22)

include(CMakeFindDependencyMacro)
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

project(
    pRRTC
    VERSION 0.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

find_package(Eigen3 REQUIRED NO_MODULE)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

add_executable(evaluate_mr
    scripts/evaluate_mr.cu
    src/planning/pRRTC.cu
)

# Must come AFTER the target exists
set_target_properties(evaluate_mr PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

target_compile_options(evaluate_mr PRIVATE
    -w
    -mavx2 -mfma
    --std=c++17
)

target_include_directories(evaluate_mr
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(evaluate_mr
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)

# Exporter for the asao -> cuRobo pipeline: runs the planner and writes per-problem
# seed files (final + intermediate improving paths) for curobo/seed_from_path.py.
add_executable(export_seeds
    scripts/export_seeds.cu
    src/planning/pRRTC.cu
)

set_target_properties(export_seeds PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

target_compile_options(export_seeds PRIVATE
    -w
    -mavx2 -mfma
    --std=c++17
)

target_include_directories(export_seeds
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(export_seeds
    PRIVATE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)
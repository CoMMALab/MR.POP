cmake_minimum_required(VERSION 3.16...3.22)

project(MR_POP VERSION 0.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
find_package(Eigen3 REQUIRED NO_MODULE)

set(POP_PLANNER_SOURCES
    src/planning/pop_planner.cu
    src/planning/path_processing.cu
    src/planning/path_simplification.cu
)

foreach(planner_target IN ITEMS evaluate_mr export_seeds)
    add_executable(${planner_target}
        scripts/${planner_target}.cu
        ${POP_PLANNER_SOURCES}
    )

    set_target_properties(${planner_target} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    target_compile_options(${planner_target} PRIVATE -w -mavx2 -mfma --std=c++17)
    target_include_directories(${planner_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${planner_target} PRIVATE nlohmann_json::nlohmann_json Eigen3::Eigen)
endforeach()

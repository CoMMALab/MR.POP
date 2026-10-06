#pragma once

#include "pop_settings.hh"

#include <cuda_runtime.h>

namespace dRRT {

    constexpr int MAX_PATH_SIZE = 50000;
    constexpr float UNWRITTEN_VAL = -9999.0f;
    constexpr int MAX_GRANULARITY = 64;
    constexpr int MAX_THREADS_PER_BLOCK = 4 * MAX_GRANULARITY;

    // Shared device state is defined once in pop_planner.cu.
    extern __device__ volatile int all_block_linkCC[300][8000];
    extern __device__ float path[2][MAX_PATH_SIZE];
    extern __device__ volatile float path_out[MAX_PATH_SIZE];
    extern __device__ volatile int path_size[2];
    extern __device__ int reached_goal_idx;
    extern __device__ int solved_iters;
    extern __device__ volatile float best_cost;
    extern __device__ volatile float cost;
    extern __constant__ pop_settings d_settings;

} // namespace dRRT

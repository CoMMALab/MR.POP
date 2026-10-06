#pragma once

#include "Robots.hh"
#include "src/collision/environment.hh"

#include <cuda_runtime.h>

namespace dRRT {

    // Launch only; synchronization and error reporting remain with the planner.
    template <typename Robot>
    cudaError_t launch_path_simplification(int num_blocks, int threads_per_block,
                                           ppln::collision::Environment<float> *env,
                                           float **sphere_pos_all);

} // namespace dRRT

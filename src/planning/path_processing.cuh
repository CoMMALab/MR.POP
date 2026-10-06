#pragma once

#include "Planners.hh"

#include <cuda_runtime.h>

namespace dRRT {

    // Called by thread zero of the block that wins the solved-state claim.
    template <typename Robot>
    __device__ void record_connected_path(int t_tree_id, int o_tree_id, const float *t_nodes,
                                          const float *o_nodes, const int *t_parents,
                                          const int *o_parents, int tree_node_index,
                                          int other_tree_node_index, int iter);

    // Copy the selected path and retain it in the improving-solution history.
    template <typename Robot>
    void copy_path_to_result(PlannerResult<Robot> &result, bool path_simplify, float solution_cost,
                             int solved_status, int solution_iters);

} // namespace dRRT

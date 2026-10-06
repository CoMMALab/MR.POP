#include "path_processing.cuh"
#include "pop_planner_state.cuh"
#include "utils.cuh"

#include <algorithm>

namespace dRRT {
    using namespace ppln;

    template <typename Robot>
    __device__ void record_connected_path(int t_tree_id, int o_tree_id, const float *t_nodes,
                                          const float *o_nodes, const int *t_parents,
                                          const int *o_parents, int tree_node_index,
                                          int other_tree_node_index, int iter) {
        static constexpr auto dim = Robot::dimension;
        // trace back to the start and goal.
        cost = 0.0f;
        int current = tree_node_index;
        int parent;
        int t_path_size = 0;
        int o_path_size = 0;
        while (t_parents[current] != current) {
            parent = t_parents[current];
            cost += device_utils::l2_dist((float *)&t_nodes[current * dim],
                                          (float *)&t_nodes[parent * dim], dim);
            for (int i = 0; i < dim; i++)
                path[t_tree_id][t_path_size * dim + i] = t_nodes[current * dim + i];
            t_path_size++;
            current = parent;
        }
        for (int i = 0; i < dim; i++)
            path[t_tree_id][t_path_size * dim + i] = t_nodes[i];
        t_path_size++;

        if (t_tree_id == 1)
            reached_goal_idx = current;
        current = other_tree_node_index;
        while (o_parents[current] != current) {
            parent = o_parents[current];
            cost += device_utils::l2_dist((float *)&o_nodes[current * dim],
                                          (float *)&o_nodes[parent * dim], dim);
            for (int i = 0; i < dim; i++)
                path[o_tree_id][o_path_size * dim + i] = o_nodes[current * dim + i];
            o_path_size++;
            current = parent;
        }
        for (int i = 0; i < dim; i++)
            path[o_tree_id][o_path_size * dim + i] = o_nodes[i];
        o_path_size++;

        if (t_tree_id == 0)
            reached_goal_idx = current;
        path_size[t_tree_id] = t_path_size;
        path_size[o_tree_id] = o_path_size;
        solved_iters = iter;
        best_cost = min(best_cost, cost);
        int path_out_ind = 0;
        for (int i = path_size[1] - 1; i >= 0; i--) {
            for (int d = 0; d < dim; d++)
                path_out[path_out_ind * dim + d] = path[1][i * dim + d];

            path_out_ind++;
        }
        for (int i = 0; i < path_size[0]; i++) {
            for (int d = 0; d < dim; d++)
                path_out[path_out_ind * dim + d] = path[0][i * dim + d];

            path_out_ind++;
        }
    }

    template <typename Robot>
    void copy_path_to_result(PlannerResult<Robot> &result, bool path_simplify, float solution_cost,
                             int solved_status, int solution_iters) {
        static constexpr auto dim = Robot::dimension;
        int h_path_size[2];
        float h_paths[2][MAX_PATH_SIZE];
        float h_simplified_path[MAX_PATH_SIZE];
        int h_reached_goal_idx;
        cudaMemcpyFromSymbol(h_path_size, path_size, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(h_paths, path, sizeof(float) * 2 * MAX_PATH_SIZE, 0,
                             cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(h_simplified_path, path_out, sizeof(float) * MAX_PATH_SIZE, 0,
                             cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&h_reached_goal_idx, reached_goal_idx, sizeof(int), 0,
                             cudaMemcpyDeviceToHost);
        cudaCheckError(cudaGetLastError());

        result.path.clear();
        typename Robot::Configuration config;

        if (!path_simplify) {
            for (int i = h_path_size[1] - 1; i >= 0; i--) {
                std::copy_n(h_paths[1] + i * dim, dim, config.begin());
                result.path.emplace_back(config);
            }
            for (int i = 0; i < h_path_size[0]; i++) {
                std::copy_n(h_paths[0] + i * dim, dim, config.begin());
                result.path.emplace_back(config);
            }
        }

        if (path_simplify) {
            for (int i = 0; i < h_path_size[0] + h_path_size[1]; i++) {
                std::copy_n(h_simplified_path + i * dim, dim, config.begin());
                if (config[0] == UNWRITTEN_VAL) {
                    continue;
                }
                result.path.emplace_back(config);
            }
        }

        result.cost = solution_cost;
        result.path_length = result.path.size();
        result.solved = solved_status > 0;
        result.iters = solution_iters;
        // Snapshot this improving solution so downstream consumers can use
        // the planner's whole optimization history as candidate seeds, not
        // just the final best. rrtc_iter=0 gives the initial solution;
        // each later improving rrtc_iter appends a shorter path.
        result.seed_paths.emplace_back(solution_cost, result.path);
    }

    template __device__ void
    record_connected_path<ppln::robots::Panda_four>(int, int, const float *, const float *,
                                                    const int *, const int *, int, int, int);
    template __device__ void
    record_connected_path<ppln::robots::Panda_five>(int, int, const float *, const float *,
                                                    const int *, const int *, int, int, int);

    template void
    copy_path_to_result<ppln::robots::Panda_four>(PlannerResult<ppln::robots::Panda_four> &, bool,
                                                  float, int, int);
    template void
    copy_path_to_result<ppln::robots::Panda_five>(PlannerResult<ppln::robots::Panda_five> &, bool,
                                                  float, int, int);

} // namespace dRRT

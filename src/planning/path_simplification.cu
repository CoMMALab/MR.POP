#include "path_simplification.cuh"
#include "pop_planner_state.cuh"
#include "src/collision/panda_collision.cuh"

#include <cooperative_groups.h>
#include <float.h>
#include <math.h>

namespace cg = cooperative_groups;

namespace dRRT {
    using namespace ppln;

    __device__ volatile float simplify_path_cost[MAX_PATH_SIZE] = {FLT_MAX};
    __device__ volatile int simplify_path_parent[MAX_PATH_SIZE] = {-1};

    template <typename Robot>
    __global__ void multiBlock_simplify(int numBlock, int threadsPerBlock,
                                        ppln::collision::Environment<float> *env,
                                        float **sphere_pos_all) {
        const int dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        __shared__ volatile unsigned int local_cc_result[1];
        __shared__ volatile unsigned int any_approx_env_collision[1];
        __shared__ volatile unsigned int any_approx_self_collision[1];
        __align__(16) __shared__ float T[64 * 2 * 16];
        __shared__ float delta[dim];
        __shared__ float middle_point[dim];
        __shared__ volatile int env_sphere_to_check[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int self_sphere_to_check[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int env_thread_dist[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int self_thread_dist[MAX_THREADS_PER_BLOCK];
        volatile float *sphere_pos = sphere_pos_all[bid];
        volatile int *link_CC = all_block_linkCC[bid];

        cg::grid_group grid = cg::this_grid();
        const int path_total_size = path_size[0] + path_size[1];
        const float granularity_tolerance = 0.001;

        for (int simplify_ind = 0; simplify_ind < d_settings.simplify_round; simplify_ind++) {
            if (tid == 0 && bid == 0) {
                simplify_path_cost[0] = 0;
                simplify_path_parent[0] = -1;
            }

            for (int t = bid * (threadsPerBlock) + tid + 1; t < path_total_size;
                 t += threadsPerBlock * numBlock) {
                simplify_path_cost[t] = FLT_MAX;
                simplify_path_parent[t] = -1;
            }

            grid.sync();

            // B-spline smoothing
            for (int step = 0; step < d_settings.bspline_step; step++) {
                for (int smooth_round = 0; smooth_round < 1; smooth_round++) {
                    for (int r = 0; r < ceil((path_total_size - 3) / (2.0f * numBlock)); r++) {

                        int i = 2 * bid + smooth_round + 1 + r * 2 * numBlock;
                        grid.sync();
                        if (i >= path_total_size - 1)
                            continue;
                        volatile float *cfg1;
                        cfg1 = path_out + (i - 1) * dim;
                        volatile float *cfg2;
                        cfg2 = path_out + (i)*dim;
                        volatile float *cfg3;
                        cfg3 = path_out + (i + 1) * dim;
                        volatile float *cc_cfgs[2] = {cfg1, cfg3};
                        if (tid < dim) {
                            float b_spline_config1 = (cfg1[tid] + cfg2[tid]) / 2.0f;
                            float b_spline_config2 = (cfg2[tid] + cfg3[tid]) / 2.0f;
                            middle_point[tid] = (b_spline_config1 + b_spline_config2) / 2.0f;
                        }
                        __syncthreads();

                        if (device_utils::l2_dist((float *)cfg2, (float *)middle_point, dim) <
                            d_settings.bspline_min_change)
                            continue;

                        bool both_edge_good = true;

                        // collision check between index-1 and midpoint & midpoint and index+1
                        for (int edge_cnt = 0; edge_cnt < 2; edge_cnt++) {
                            const int cc_iter1 =
                                ceil(device_utils::l2_dist((float *)cc_cfgs[edge_cnt],
                                                           (float *)middle_point, dim) /
                                     (float)d_settings.range);
                            if (tid < dim) {
                                delta[tid] =
                                    (middle_point[tid] - cc_cfgs[edge_cnt][tid]) / cc_iter1;
                            }
                            float interp_cfg[dim];
                            for (int d = 0; d < dim; d++) {
                                interp_cfg[d] = cc_cfgs[edge_cnt][d];
                            }
                            __syncthreads();

                            for (int d = 0; d < dim; d++) {
                                interp_cfg[d] +=
                                    (int(tid / 4 + 1) * (delta[d] / (float)d_settings.granularity));
                            }
                            for (int cc_round = 0; cc_round < cc_iter1; cc_round++) {

                                if (tid == 0) {
                                    any_approx_env_collision[0] = 0;
                                    any_approx_self_collision[0] = 0;
                                    local_cc_result[0] = 0;
                                }
                                __syncthreads();

                                ppln::device_utils::fkcc_single_buffer<Robot>(
                                    interp_cfg, env, tid, sphere_pos, link_CC, T, local_cc_result,
                                    any_approx_env_collision, any_approx_self_collision,
                                    env_sphere_to_check, self_sphere_to_check, env_thread_dist,
                                    self_thread_dist, d_settings.granularity);

                                bool edge_good = (local_cc_result[0] == 0);
                                __syncthreads();
                                if (!edge_good) {
                                    both_edge_good = false;
                                    break;
                                }
                                for (int d = 0; d < dim; d++) {
                                    interp_cfg[d] += delta[d];
                                }
                                __syncthreads();
                            }
                            if (!both_edge_good) {
                                break;
                            }
                        }
                        if (both_edge_good) {
                            if (tid < dim) {
                                path_out[i * dim + tid] = middle_point[tid];
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < path_total_size - 1; i++) {
                volatile float *cfg1;
                cfg1 = path_out + i * dim;

                grid.sync();
                const int bid_path_ind = bid + i + 1;
                if (bid_path_ind >= path_total_size)
                    continue;

                volatile float *cfg2;
                cfg2 = path_out + bid_path_ind * dim;
                __syncthreads();

                if (bid > 0) {
                    bool edge_good = true;
                    float interp_cfg[dim];
                    const int cc_iter =
                        ceil((device_utils::l2_dist((float *)cfg1, (float *)cfg2, dim) -
                              granularity_tolerance) /
                             (float)d_settings.range);
                    if (tid < dim) {
                        delta[tid] = (cfg2[tid] - cfg1[tid]) / cc_iter;
                    }
                    for (int d = 0; d < dim; d++) {
                        interp_cfg[d] = cfg1[d];
                    }
                    __syncthreads();

                    for (int d = 0; d < dim; d++) {
                        interp_cfg[d] +=
                            (int(tid / 4 + 1) * (delta[d] / (float)d_settings.granularity));
                    }

                    for (int cc_round = 0; cc_round < cc_iter; cc_round++) {

                        if (tid == 0) {
                            any_approx_env_collision[0] = 0;
                            any_approx_self_collision[0] = 0;
                            local_cc_result[0] = 0;
                        }
                        __syncthreads();

                        ppln::device_utils::fkcc_single_buffer<Robot>(
                            interp_cfg, env, tid, sphere_pos, link_CC, T, local_cc_result,
                            any_approx_env_collision, any_approx_self_collision,
                            env_sphere_to_check, self_sphere_to_check, env_thread_dist,
                            self_thread_dist, d_settings.granularity);

                        edge_good = (local_cc_result[0] == 0);
                        __syncthreads();
                        if (!edge_good)
                            break;

                        for (int d = 0; d < dim; d++) {
                            interp_cfg[d] += delta[d];
                        }
                        __syncthreads();
                    }

                    if (!edge_good) {
                        continue;
                    }
                }

                if (tid == 0) {
                    if (simplify_path_cost[bid_path_ind] >
                        device_utils::l2_dist((float *)cfg1, (float *)cfg2, dim) +
                            simplify_path_cost[i]) {
                        simplify_path_cost[bid_path_ind] =
                            device_utils::l2_dist((float *)cfg1, (float *)cfg2, dim) +
                            simplify_path_cost[i];
                        simplify_path_parent[bid_path_ind] = i;
                    }
                }
            }
            grid.sync();

            // reconstruct path
            if (simplify_path_cost[path_total_size - 1] < best_cost) {
                int children = path_total_size - 1;
                int parent = simplify_path_parent[children];
                while (parent < path_total_size && parent != -1) {
                    const int bid_path_ind = bid + parent + 1;
                    if (bid_path_ind < children) {
                        if (tid < dim)
                            path_out[bid_path_ind * dim + tid] = path_out[parent * dim + tid];
                    }
                    children = parent;
                    parent = simplify_path_parent[children];
                }
            }

            if (tid == 0 && bid == 0) {
                cost = simplify_path_cost[path_total_size - 1];
                best_cost = min(best_cost, cost);
            }

            grid.sync();
        }
    }

    template <typename Robot>
    cudaError_t launch_path_simplification(int num_blocks, int threads_per_block,
                                           ppln::collision::Environment<float> *env,
                                           float **sphere_pos_all) {
        void *kernel_args[] = {(void *)&num_blocks, (void *)&threads_per_block, (void *)&env,
                               (void *)&sphere_pos_all};

        return cudaLaunchCooperativeKernel((void *)multiBlock_simplify<Robot>, dim3(num_blocks),
                                           dim3(threads_per_block), kernel_args,
                                           0, // sharedMem
                                           0  // stream
        );
    }

    template cudaError_t launch_path_simplification<ppln::robots::Panda_four>(
        int, int, ppln::collision::Environment<float> *, float **);
    template cudaError_t launch_path_simplification<ppln::robots::Panda_five>(
        int, int, ppln::collision::Environment<float> *, float **);

} // namespace dRRT

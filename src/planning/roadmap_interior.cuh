#include "Planners.hh"
#include "Robots.hh"
#include "utils.cuh"
#include "pop_settings.hh"
#include "src/collision/environment.hh"

#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <float.h>
#include <math.h>

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>

namespace cg = cooperative_groups;
#define ROADMAP_BUILD_ITER 230 // how many iterations till component id propagation
#define MAX_ROADMAP_SIZE 100000
constexpr float UNWRITTEN_VAL = -9999.0f;
constexpr int UNWRITTEN_VAL_INT = -999;
constexpr int BIASED_ITER = 10000;

__device__ __forceinline__ bool check_unwritten(float *config, int dim) {

    for (int i = 0; i < dim; i++) {
        if (config[i] == UNWRITTEN_VAL) {
            return true;
        }
    }
    return false;
}

template <typename Robot>
__device__ void build_roadmap_interior(
    ppln::collision::Environment<float> *env, float *start, float *goal, const int num_goals,
    volatile float *edge_nn, curandStateXORWOW_t *states, const int block_per_robot,
    const int num_robot, const int granularity, volatile float *config,
    volatile unsigned int *local_cc_result, volatile unsigned int *any_approx_env_collision,
    volatile unsigned int *any_approx_self_collision, volatile int *node_id,
    volatile int *current_nn, volatile int *link_CC, float *T, float *delta,
    volatile float **roadmaps, volatile int **roadmap_component_id, volatile int *roadmap_size,
    volatile uint32_t(***roadmap_edges), volatile bool *start_goal_connected,
    volatile float **sphere_pos_all, volatile int *env_sphere_to_check,
    volatile int *self_sphere_to_check, volatile int *env_thread_dist,
    volatile int *self_thread_dist, volatile float *sdata, volatile int *sindex,
    pop_settings *d_settings, int roadmap_launch_cnt) {

    static constexpr auto dim = Robot::dimension;
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int robot_id = bid / block_per_robot;
    volatile float *working_on_roadmap = roadmaps[robot_id];
    volatile float *sphere_pos = sphere_pos_all[bid];
    volatile float *nearest_node;
    int iter = 0;
    int after_connect_iter = 0;

    if (tid == 0) {
        any_approx_env_collision[0] = 0;
        any_approx_self_collision[0] = 0;
        local_cc_result[0] = 0;
    }
    __syncthreads();

    if (roadmap_launch_cnt == 0) {
        if (bid % block_per_robot == 0) {
            if (tid < dim) {
                working_on_roadmap[tid] = start[robot_id * dim + tid];
                for (int t = 0; t < num_goals; t++) {
                    working_on_roadmap[dim + t * dim + tid] = goal[robot_id * dim + t * dim + tid];
                }
            }
            if (tid == 0) {
                roadmap_size[robot_id] = num_goals + 1;
                roadmap_component_id[robot_id][0] = 0;
                roadmap_edges[robot_id][0][0] = 1;
                // component id: 0 for start, 1 for goal
                for (int i = 0; i < num_goals; i++) {
                    roadmap_component_id[robot_id][i + 1] = 1;
                    roadmap_edges[robot_id][i + 1][(i + 1) >> 5] |= 1u << ((i + 1) & 31);
                }
            }
        }
    }
    if (bid % block_per_robot == 0)
        start_goal_connected[robot_id] = false;

    grid.sync();

    while (true) {

        if (start_goal_connected[robot_id] == false) {
            for (int xx = 0; xx < ROADMAP_BUILD_ITER; xx++) {

                if (tid == 0) {
                    any_approx_env_collision[0] = 0;
                    any_approx_self_collision[0] = 0;
                    local_cc_result[0] = 0;
                }

                if (tid < dim) {
                    // sample random config
                    curandStateXORWOW_t local = states[bid * dim + tid];
                    float rnd = curand_uniform(&local);
                    config[tid] = rnd;
                    states[bid * dim + tid] = local;
                    Robot::scale_cfg((float *)config);
                }
                __syncthreads();

                iter++;

                if (iter % BIASED_ITER == 0) {
                    // parallelized nearest neighbor search
                    float local_min_dist = FLT_MAX;
                    int local_near_idx = 0;
                    float dist;
                    int size = roadmap_size[robot_id];
                    for (int i = tid; i < size; i += blockDim.x) {
                        if (check_unwritten((float *)&working_on_roadmap[i * dim], dim))
                            continue;
                        dist = ppln::device_utils::sq_l2_dist((float *)&working_on_roadmap[i * dim],
                                                              (float *)config, dim);
                        if (dist < local_min_dist) {
                            local_min_dist = dist;
                            local_near_idx = i;
                        }
                    }
                    sdata[tid] = local_min_dist;
                    sindex[tid] = local_near_idx;
                    __syncthreads();

                    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                        float sdata_tid;
                        float sdata_tid_s;
                        if (tid < s) {
                            sdata_tid = sdata[tid];
                            sdata_tid_s = sdata[tid + s];
                        }
                        __syncthreads();
                        if (tid < s) {
                            if (sdata_tid_s < sdata_tid) {
                                sdata[tid] = sdata[tid + s];
                                sindex[tid] = sindex[tid + s];
                            }
                        }
                        __syncthreads();
                    }
                    nearest_node = &working_on_roadmap[sindex[0] * dim];
                    // nn index is in sindex[0], distance in sdata[0]
                    if (tid == 0) {
                        sdata[0] = sqrt(sdata[0]);
                    }
                    __syncthreads();

                    if (tid < dim) {
                        float scale = min(1.0f, d_settings->range / sdata[0]);
                        config[tid] = (config[tid] - nearest_node[tid]) * scale + nearest_node[tid];
                    }
                    __syncthreads();
                }

                bool edge_good = true;
                ppln::device_utils::fkcc_single_buffer_mr<Robot>(
                    (float *)config, env, tid, sphere_pos, link_CC, T, local_cc_result,
                    any_approx_env_collision, any_approx_self_collision, env_sphere_to_check,
                    self_sphere_to_check, env_thread_dist, self_thread_dist, granularity, num_robot,
                    robot_id);
                edge_good = (local_cc_result[0] == 0);
                __syncthreads();
                if (!edge_good)
                    continue;

                if (tid == 0) {
                    node_id[0] = atomicAdd((int *)&roadmap_size[robot_id], 1);
                }
                __syncthreads();

                // NN search for radius R
                int nn_id[200];
                int nn_count = 0;
                for (int i = tid; i < roadmap_size[robot_id]; i += blockDim.x) {
                    if (check_unwritten((float *)&working_on_roadmap[i * dim], dim))
                        continue;
                    float dist = ppln::device_utils::l2_dist((float *)&working_on_roadmap[i * dim],
                                                             (float *)config, dim);
                    if (dist < d_settings->range) {
                        nn_id[nn_count] = i;
                        nn_count += 1;
                    }
                    if (nn_count >= 200)
                        break;
                }
                __syncthreads();

                // collision check and add edges
                int processed_nn;
                for (int thread = 0; thread < blockDim.x; thread++) {
                    processed_nn = 0;
                    while (true) {
                        if (tid == thread) {
                            if (processed_nn < nn_count)
                                current_nn[0] = nn_id[processed_nn];
                            else
                                current_nn[0] = UNWRITTEN_VAL_INT;
                        }
                        __syncthreads();
                        if (current_nn[0] == UNWRITTEN_VAL_INT)
                            break;
                        if (tid < dim) {
                            edge_nn[tid] = working_on_roadmap[current_nn[0] * dim + tid];
                        }
                        __syncthreads();

                        if (tid < dim) {
                            delta[tid] = (config[tid] - edge_nn[tid]) / (float)granularity;
                        }
                        __syncthreads();

                        if (tid == 0) {
                            any_approx_env_collision[0] = 0;
                            any_approx_self_collision[0] = 0;
                            local_cc_result[0] = 0;
                        }

                        // validate edge
                        float interp_cfg[dim];
                        for (int i = 0; i < dim; i++) {
                            interp_cfg[i] = edge_nn[i] + (int(tid / 4 + 1) * delta[i]);
                        }
                        __syncthreads();

                        ppln::device_utils::fkcc_single_buffer_mr<Robot>(
                            interp_cfg, env, tid, sphere_pos, link_CC, T, local_cc_result,
                            any_approx_env_collision, any_approx_self_collision,
                            env_sphere_to_check, self_sphere_to_check, env_thread_dist,
                            self_thread_dist, granularity, num_robot, robot_id);
                        edge_good = local_cc_result[0] == 0;
                        __syncthreads();

                        if (edge_good && tid == 0) {
                            // roadmap_edges[robot_id][node_id[0]][current_nn[0]] = 1;
                            // roadmap_edges[robot_id][current_nn[0]][node_id[0]] = 1;
                            // roadmap_edges[robot_id][node_id[0]][node_id[0]] = 1;
                            atomicOr((uint32_t *)&roadmap_edges[robot_id][node_id[0]]
                                                               [current_nn[0] >> 5],
                                     1u << (current_nn[0] & 31));
                            atomicOr((uint32_t *)&roadmap_edges[robot_id][current_nn[0]]
                                                               [node_id[0] >> 5],
                                     1u << (node_id[0] & 31));
                            atomicOr(
                                (uint32_t *)&roadmap_edges[robot_id][node_id[0]][node_id[0] >> 5],
                                1u << (node_id[0] & 31));
                        }

                        processed_nn++;
                        __syncthreads();
                    }
                }
                __syncthreads();

                if (tid < dim)
                    working_on_roadmap[node_id[0] * dim + tid] = config[tid];
                if (tid == 0)
                    roadmap_component_id[robot_id][node_id[0]] = -1;
            }
        }

        grid.sync();

        // merge components (at the very end, only one block since it is lightweight)
        if (bid % block_per_robot == 0 && start_goal_connected[robot_id] == false) {

            __shared__ volatile bool bfs_change;

            if (tid == 0)
                bfs_change = true;
            __syncthreads();

            while (bfs_change) {
                if (tid == 0)
                    bfs_change = false;
                __syncthreads();

                for (int t = tid; t < roadmap_size[robot_id]; t += blockDim.x) {
                    if (roadmap_component_id[robot_id][t] == 0) {
                        for (int nn = 0; nn < roadmap_size[robot_id]; nn++) {
                            if (((roadmap_edges[robot_id][t][nn >> 5] >> (nn & 31)) & 1u) &&
                                roadmap_component_id[robot_id][nn] != 0) {
                                roadmap_component_id[robot_id][nn] = 0;
                                bfs_change = true;
                            }
                        }
                    }
                }
                __syncthreads();
            }

            __syncthreads();
            if (tid == 0) {
                for (int j = 0; j < num_goals; j++) {
                    if (roadmap_component_id[robot_id][j + 1] ==
                        0) { // start and goal are connected
                        start_goal_connected[robot_id] = true;
                        printf("robot %d start, goal connected\n", robot_id);
                        after_connect_iter++;
                    }
                }
            }
        }
        grid.sync();

        bool finished = true;
        if (roadmap_launch_cnt == 0) {
            for (int i = 0; i < num_robot; i++) {
                if (start_goal_connected[i] == false) {
                    finished = false;
                    if (start_goal_connected[robot_id] == true && d_settings->roadmap_reassign) {
                        robot_id = i;
                        working_on_roadmap = roadmaps[robot_id];
                    }
                    break;
                }
            }
        }

        if (bid == 0 && tid == 0)
            printf("roadmap size %d %d %d %d %d\n", roadmap_size[0], roadmap_size[1],
                   roadmap_size[2], roadmap_size[3], roadmap_size[4]);
        if (finished) {
            return;
        }
        if (roadmap_size[robot_id] >= MAX_ROADMAP_SIZE)
            return;
    }
    return;
}

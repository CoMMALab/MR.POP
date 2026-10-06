#include "Planners.hh"
#include "Robots.hh"
#include "utils.cuh"
#include "pop_settings.hh"
#include "path_simplification.cuh"
#include "path_processing.cuh"
#include "pop_planner_state.cuh"
#include "src/collision/environment.hh"
#include "roadmap_interior.cuh"
#include "src/collision/panda_collision.cuh"
#include "src/robots/panda.cuh"
#include "src/robots/panda_multi_geometry.cuh"
#include "src/robots/panda_four.cuh"
#include "src/robots/panda_five.cuh"

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

// Each block expands the start or goal tree according to the balance strategy.

namespace dRRT {
    using namespace ppln;
    __device__ volatile int solved = 0;
    __device__ volatile int roadmap_launch_cnt = 0;
    __device__ volatile int optimize = 0;
    __device__ volatile float best_cost = FLT_MAX;
    __device__ volatile int all_block_linkCC[300][8000];
    __device__ volatile int atomic_free_index[2]; // separate for tree_a and tree_b
    __device__ volatile int nodes_size[2];
    __device__ volatile int completed_nodes[2]; // track completed nodes for each tree
    __device__ float path[2][MAX_PATH_SIZE];    // solution path segments for tree_a, and tree_b
    __device__ volatile float path_out[MAX_PATH_SIZE]; // Combined solution path.
    __device__ volatile int path_size[2] = {0, 0};
    __device__ volatile float cost = 0.0;
    __device__ int reached_goal_idx = 0;
    __device__ int solved_iters = 0; // value of iters in the block that solves the problem
    __device__ volatile int roadmap_size[8];
    __device__ volatile bool start_goal_connected[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    __constant__ pop_settings d_settings;

    __global__ void init_xorwow(curandStateXORWOW_t *states, unsigned long long seed) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Each thread gets a deterministic, independent subsequence
        curand_init(seed, // base seed
                    tid,  // sequence number (gives each thread its own stream)
                    0,    // offset
                    &states[tid]);
    }

    __global__ void init_rng(curandState *states, unsigned long seed, int num_rng_states) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_rng_states)
            return;
        curand_init(seed + idx, idx, 0, &states[idx]);
    }

    inline void setup_environment_on_device(ppln::collision::Environment<float> *&d_env,
                                            const ppln::collision::Environment<float> &h_env) {
        // allocate the environment struct
        cudaMalloc(&d_env, sizeof(ppln::collision::Environment<float>));

        // Initialize struct to zeros first
        cudaMemset(d_env, 0, sizeof(ppln::collision::Environment<float>));

        // Handle each primitive type separately
        if (h_env.num_spheres > 0) {
            // Allocate and copy spheres array
            ppln::collision::Sphere<float> *d_spheres;
            cudaMalloc(&d_spheres, sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres);
            cudaMemcpy(d_spheres, h_env.spheres,
                       sizeof(ppln::collision::Sphere<float>) * h_env.num_spheres,
                       cudaMemcpyHostToDevice);

            // Update the struct fields directly
            cudaMemcpy(&(d_env->spheres), &d_spheres, sizeof(ppln::collision::Sphere<float> *),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_spheres), &h_env.num_spheres, sizeof(unsigned int),
                       cudaMemcpyHostToDevice);
        }

        if (h_env.num_capsules > 0) {
            ppln::collision::Capsule<float> *d_capsules;
            cudaMalloc(&d_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules);
            cudaMemcpy(d_capsules, h_env.capsules,
                       sizeof(ppln::collision::Capsule<float>) * h_env.num_capsules,
                       cudaMemcpyHostToDevice);

            cudaMemcpy(&(d_env->capsules), &d_capsules, sizeof(ppln::collision::Capsule<float> *),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_capsules), &h_env.num_capsules, sizeof(unsigned int),
                       cudaMemcpyHostToDevice);
        }

        // Repeat for each primitive type...
        if (h_env.num_z_aligned_capsules > 0) {
            ppln::collision::Capsule<float> *d_z_capsules;
            cudaMalloc(&d_z_capsules,
                       sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules);
            cudaMemcpy(d_z_capsules, h_env.z_aligned_capsules,
                       sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules,
                       cudaMemcpyHostToDevice);

            cudaMemcpy(&(d_env->z_aligned_capsules), &d_z_capsules,
                       sizeof(ppln::collision::Capsule<float> *), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_capsules), &h_env.num_z_aligned_capsules,
                       sizeof(unsigned int), cudaMemcpyHostToDevice);
        }

        if (h_env.num_cylinders > 0) {
            ppln::collision::Cylinder<float> *d_cylinders;
            cudaMalloc(&d_cylinders,
                       sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders);
            cudaMemcpy(d_cylinders, h_env.cylinders,
                       sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders,
                       cudaMemcpyHostToDevice);

            cudaMemcpy(&(d_env->cylinders), &d_cylinders,
                       sizeof(ppln::collision::Cylinder<float> *), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cylinders), &h_env.num_cylinders, sizeof(unsigned int),
                       cudaMemcpyHostToDevice);
        }

        if (h_env.num_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_cuboids;
            cudaMalloc(&d_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids);
            cudaMemcpy(d_cuboids, h_env.cuboids,
                       sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids,
                       cudaMemcpyHostToDevice);

            cudaMemcpy(&(d_env->cuboids), &d_cuboids, sizeof(ppln::collision::Cuboid<float> *),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cuboids), &h_env.num_cuboids, sizeof(unsigned int),
                       cudaMemcpyHostToDevice);
        }

        if (h_env.num_z_aligned_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_z_cuboids;
            cudaMalloc(&d_z_cuboids,
                       sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids);
            cudaMemcpy(d_z_cuboids, h_env.z_aligned_cuboids,
                       sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids,
                       cudaMemcpyHostToDevice);

            cudaMemcpy(&(d_env->z_aligned_cuboids), &d_z_cuboids,
                       sizeof(ppln::collision::Cuboid<float> *), cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_cuboids), &h_env.num_z_aligned_cuboids,
                       sizeof(unsigned int), cudaMemcpyHostToDevice);
        }
    }

    __device__ __forceinline__ bool check_unwritten(float *config, int dim) {

        for (int i = 0; i < dim; i++) {
            if (config[i] == UNWRITTEN_VAL) {
                return true;
            }
        }
        return false;
    }

    template <typename Robot>
    __global__ void
    build_roadmap(ppln::collision::Environment<float> *env, float *start, float *goal,
                  float **sphere_pos_all, float **roadmaps, int **roadmap_component_id,
                  uint32_t ***roadmap_edges, const int num_goals, curandStateXORWOW_t *states,
                  const int block_per_robot, const int num_robot, const int granularity) {

        static constexpr auto dim = Robot::dimension;
        cg::grid_group grid = cg::this_grid();
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int robot_id = bid / block_per_robot;
        __shared__ float config[dim];
        __shared__ volatile float edge_nn[dim];
        __shared__ volatile unsigned int local_cc_result[1];
        __shared__ volatile unsigned int any_approx_env_collision[1];
        __shared__ volatile unsigned int any_approx_self_collision[1];
        __shared__ volatile int node_id[1];
        __shared__ volatile int current_nn[1];
        __shared__ volatile int env_sphere_to_check[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int self_sphere_to_check[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int env_thread_dist[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int self_thread_dist[MAX_THREADS_PER_BLOCK];
        __shared__ volatile float sdata[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int sindex[MAX_THREADS_PER_BLOCK];

        __align__(16) __shared__ float T[64 * 2 * 16];
        __align__(16) __shared__ int link_CC[1500];
        __shared__ float delta[dim];

        build_roadmap_interior<Robot>(
            env, start, goal, num_goals, edge_nn, states, block_per_robot, num_robot, granularity,
            config, local_cc_result, any_approx_env_collision, any_approx_self_collision, node_id,
            current_nn, link_CC, T, delta, (volatile float **)roadmaps,
            (volatile int **)roadmap_component_id, (volatile int *)roadmap_size,
            (volatile uint32_t ***)roadmap_edges, (volatile bool *)start_goal_connected,
            (volatile float **)sphere_pos_all, (volatile int *)env_sphere_to_check,
            (volatile int *)self_sphere_to_check, (volatile int *)env_thread_dist,
            (volatile int *)self_thread_dist, (volatile float *)sdata, (volatile int *)sindex,
            &d_settings, roadmap_launch_cnt);

        if (bid == 0 && tid == 0)
            roadmap_launch_cnt++;

        return;
    }

    template <typename Robot>
    __global__ void rrtc(float **nodes, int **parents, float **node_cost,
                         volatile int ***node_roadmap_id, float **sphere_pos_all, float **roadmaps,
                         uint32_t ***roadmap_edges, float **radii, curandStateXORWOW_t *states,
                         curandState *rng_states, ppln::collision::Environment<float> *env) {
        static constexpr auto dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x; // 0 ... NUM_NEW_CONFIGS
        int num_robot = 1;
        if (Robot::name == "panda_four")
            num_robot = 4;
        if (Robot::name == "panda_five")
            num_robot = 5;
        const int single_robot_dim = (int)(dim / num_robot);
        __shared__ volatile int t_tree_id; // this tree
        __shared__ volatile int o_tree_id; // the other tree
        __shared__ float config[dim];
        __shared__ volatile float nn_mr_new_config[dim];
        __shared__ volatile int nn_mr_new_config_roadmap_id[10];
        __shared__ float rot_world_from_ellipse[dim][dim];
        __shared__ float tf_world_from_ellipse[dim][dim];
        __shared__ float sdata[MAX_THREADS_PER_BLOCK];
        __shared__ int sindex[MAX_THREADS_PER_BLOCK];
        __shared__ float sdata_nn[MAX_THREADS_PER_BLOCK];
        __shared__ int sindex_nn[MAX_THREADS_PER_BLOCK];
        __shared__ float *focus1;
        __shared__ float *focus2;
        __shared__ float center[dim];
        __shared__ float transverse_diameter;
        __shared__ float min_transverse_diameter;
        __shared__ volatile unsigned int local_cc_result[1];
        __shared__ volatile unsigned int any_approx_env_collision[1];
        __shared__ volatile unsigned int any_approx_self_collision[1];
        __shared__ volatile float c_rand;
        __shared__ float *t_nodes;
        __shared__ float *o_nodes;
        __shared__ int *t_parents;
        __shared__ int *o_parents;
        __shared__ float scale;
        __shared__ float *nearest_node;
        __shared__ volatile float delta[dim];
        __shared__ volatile int index;
        __shared__ volatile float vec[dim];
        __shared__ volatile unsigned int n_extensions;
        __shared__ volatile bool should_skip;
        __shared__ volatile int self_sphere_to_check[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int self_thread_dist[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int env_sphere_to_check[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int env_thread_dist[MAX_THREADS_PER_BLOCK];
        __align__(16) __shared__ float T[64 * 2 * 16];
        volatile float *sphere_pos = sphere_pos_all[bid];
        volatile int *link_CC = all_block_linkCC[bid];

        if (tid == 0 && bid == 0) {
            solved = 0;
            node_cost[0][0] = 0.0f;
            int num_goals = atomic_free_index[1];
            for (int i = 0; i < num_goals; i++) {
                node_cost[1][i] = 0.0f;
            }
        }
        if (bid == 1 && tid < num_robot) {
            node_roadmap_id[1][tid][0] = 1;
        }

        if (tid == 0) {
            c_rand = FLT_MAX;
        }

        // phs setup - update rotation & transformation
        if (optimize > 0 && d_settings.phs) {
            if (tid == 0) {
                focus1 = nodes[0];
                focus2 = nodes[1];
                min_transverse_diameter = device_utils::l2_dist(focus1, focus2, dim);
                transverse_diameter = best_cost;
            }
            __syncthreads();
            if (tid < dim) {
                center[tid] = (focus1[tid] + focus2[tid]) / 2.0f;
                static constexpr float circle_tolerance = 1e-6;
                if (min_transverse_diameter < circle_tolerance) {
                    // treat this as a circle
                    for (int c = 0; c < dim; c++) {
                        rot_world_from_ellipse[tid][c] = (tid == c) ? 1.0f : 0.0f;
                    }
                } else {
                    float tranverse_axis[dim]; //reuse config as temp storage
                    for (int d = 0; d < dim; d++) {
                        tranverse_axis[d] = (focus2[d] - focus1[d]) / min_transverse_diameter;
                    }
                    float v0 = 1.0f - tranverse_axis[0];
                    // Compute v^T v
                    float vTv = v0 * v0;
                    for (int i = 1; i < dim; ++i) {
                        float vi = -tranverse_axis[i];
                        vTv += vi * vi;
                    }
                    float beta = 2.0f / vTv;

                    // Precompute v vector on the fly: v[0]=1-a0, v[i]=-ai
                    // Fill H = I - beta * v v^T
                    float vc = (tid == 0) ? (1.0f - tranverse_axis[0]) : (-tranverse_axis[tid]);
                    for (int r = 0; r < dim; ++r) {
                        float vr = (r == 0) ? (1.0f - tranverse_axis[0]) : (-tranverse_axis[r]);
                        float val = -beta * vr * vc;
                        if (r == tid)
                            val += 1.0f;
                        rot_world_from_ellipse[r][tid] = val;
                    }
                }
            }
            const float conjugate_diamater =
                sqrt(transverse_diameter * transverse_diameter -
                     min_transverse_diameter * min_transverse_diameter);
            float *diag = config; //reuse config as temp storage
            if (tid == 0)
                diag[tid] = transverse_diameter / 2;
            if (tid >= 1 && tid < dim)
                diag[tid] = conjugate_diamater / 2;
            __syncthreads();
            if (tid < dim) {
                for (int c = 0; c < dim; c++) {
                    tf_world_from_ellipse[tid][c] = rot_world_from_ellipse[tid][c] * diag[c];
                }
            }
        }
        __syncthreads();

        int iter = 0;
        while (true) {

            __syncthreads();
            if (tid == 1)
                delta[0] = solved;
            __syncthreads();
            if (delta[0] != 0)
                return;

            if (tid == 0) {
                iter++;
                if (iter > d_settings.max_iters ||
                    (optimize == 1 && iter > d_settings.optimize_iters)) {
                    atomicAdd((int *)&solved, -1);
                }

                if (iter == 1) {
                    t_tree_id = (bid < (d_settings.num_new_configs / 2)) ? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                } else if (d_settings.balance == 2) { // vamp balance
                    float ratio = abs(atomic_free_index[t_tree_id] - atomic_free_index[o_tree_id]) /
                                  (float)atomic_free_index[t_tree_id];
                    if (ratio < d_settings.tree_ratio) {
                        t_tree_id = 1 - t_tree_id;
                        o_tree_id = 1 - t_tree_id;
                    }
                }

                t_nodes = nodes[t_tree_id];
                o_nodes = nodes[o_tree_id];
                t_parents = parents[t_tree_id];
                o_parents = parents[o_tree_id];

                local_cc_result[0] = 0;
                any_approx_env_collision[0] = 0;
                any_approx_self_collision[0] = 0;
            }

            __syncthreads();

            if (!d_settings.phs || optimize == 0) {
                if (tid < dim) {
                    // sample random config
                    curandStateXORWOW_t local = states[bid * dim + tid];
                    float rnd = curand_uniform(&local);
                    config[tid] = rnd;
                    states[bid * dim + tid] = local;
                }
            } else {
                // PHS sampling
                // logit
                if (tid < dim) {
                    curandStateXORWOW_t local = states[bid * dim + tid];
                    float rnd = curand_uniform(&local);
                    states[bid * dim + tid] = local;
                    config[tid] = logf(rnd * (__frcp_rn(1.0f - rnd))) * sqrtf(M_PI / 8.0f);
                }
                __syncthreads();
                // uniform on ball
                if (tid == 0) {
                    delta[0] = 0.0f;
                    for (int d = 0; d < dim; d++) {
                        delta[0] += config[d] * config[d];
                    }
                    delta[0] = sqrtf(delta[0]);
                }
                __syncthreads();
                if (tid < dim) {
                    config[tid] = config[tid] / delta[0];
                }
                //uniform in ball
                if (tid == 0) {
                    curandStateXORWOW_t local = states[bid * dim + tid];
                    float rnd = curand_uniform(&local);
                    states[bid * dim + tid] = local;
                    delta[1] = powf(rnd, 1.0f / (float)dim);
                }
                __syncthreads();
                if (tid < dim) {
                    config[tid] = config[tid] * delta[1];
                }
                __syncthreads();
                // transform
                if (tid < dim) {
                    float val = 0.0f;
                    for (int c = 0; c < dim; c++) {
                        val += tf_world_from_ellipse[tid][c] * config[c];
                    }
                    config[tid] = val + center[tid];
                    Robot::descale_cfg((float *)config);
                    config[tid] = max(min(1.0f, config[tid]), 0.0f);
                }
            }
            __syncthreads();
            if (tid < dim)
                Robot::scale_cfg((float *)config);

            __syncthreads();

            if (tid == 0) {
                should_skip = (device_utils::l2_dist((float *)nodes[0], config, dim) +
                                   device_utils::l2_dist((float *)nodes[1], config, dim) >=
                               best_cost);
            }
            __syncthreads();
            if (should_skip)
                continue;

            // sample cost bound
            if (tid == 0 && optimize == 1) {
                const float g_hat = device_utils::l2_dist((float *)nodes[t_tree_id], config, dim);
                const float h_hat = device_utils::l2_dist((float *)nodes[o_tree_id], config, dim);
                const float min_possible_cost = g_hat + h_hat;
                __threadfence();
                float c_range = max(best_cost - min_possible_cost, 0.0f);
                curandStateXORWOW_t local = states[bid * dim + tid];
                float rnd = curand_uniform(&local);
                states[bid * dim + tid] = local;
                c_rand = rnd * c_range + g_hat;
            }
            __syncthreads();

            // parallelized nearest neighbor search
            float local_min_dist = FLT_MAX;
            int local_near_idx = 0;
            float dist;
            int size = min(atomic_free_index[t_tree_id], completed_nodes[t_tree_id]);
            for (int i = tid; i < size; i += blockDim.x) {
                if (check_unwritten((float *)&t_nodes[i * dim], dim))
                    continue;
                dist = device_utils::sq_l2_dist((float *)&t_nodes[i * dim], (float *)config, dim);
                if (dist < local_min_dist && (optimize == 0 || node_cost[t_tree_id][i] < c_rand)) {
                    local_min_dist = dist;
                    local_near_idx = i;
                }
            }
            sdata[tid] = local_min_dist;
            sindex[tid] = local_near_idx;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                float sdata_tid = sdata[tid];
                float sdata_tid_s = sdata[tid + s];
                __syncthreads();
                if (tid < s) {
                    if (sdata_tid_s < sdata_tid) {
                        sdata[tid] = sdata[tid + s];
                        sindex[tid] = sindex[tid + s];
                    }
                }
                __syncthreads();
            }
            // nn index is in sindex[0], distance in sdata[0]
            if (tid == 0) {
                sdata[0] = sqrt(sdata[0]);
                scale = min(1.0f, d_settings.range / (sdata[0]));
                nearest_node = &t_nodes[sindex[0] * dim];
            }
            __syncthreads();

            if (should_skip)
                continue;
            __syncthreads();

            float tree_edge_cost_bound = c_rand - node_cost[t_tree_id][sindex[0]];
            if (optimize == 0)
                tree_edge_cost_bound = FLT_MAX;

            bool found_nn = ppln::device_utils::nn_angle_mr<Robot>(
                config, nearest_node, sindex[0], nn_mr_new_config, nn_mr_new_config_roadmap_id,
                node_roadmap_id[t_tree_id], roadmaps, roadmap_size, roadmap_edges,
                tree_edge_cost_bound, d_settings.range, sdata_nn, sindex_nn, num_robot, optimize);

            if (!found_nn)
                continue;

            // check if nn_mr_new_config is already in tree. If so, does the neighbor "nearest_node" provide better node cost for nn_mr_new_config?
            for (int n = tid; n < completed_nodes[t_tree_id]; n += blockDim.x) {
                if (device_utils::sq_l2_dist((float *)&t_nodes[n * dim], (float *)nn_mr_new_config,
                                             dim) <= 1e-5) {
                    sdata_nn[0] = -1;
                    sindex_nn[0] = n;
                }
            }
            should_skip =
                (d_settings.dynamic_domain &&
                 radii[t_tree_id][sindex[0]] <
                     device_utils::l2_dist((float *)nearest_node, (float *)nn_mr_new_config, dim));
            __syncthreads();

            bool node_already_exist = (sdata_nn[0] == -1);
            float new_node_cost = node_cost[t_tree_id][sindex[0]] + sdata[0];
            int parent_idx = sindex[0];
            if (node_already_exist &&
                (optimize == 0 || new_node_cost >= node_cost[t_tree_id][sindex_nn[0]]))
                continue;
            if (optimize == 1 && new_node_cost >= best_cost)
                continue;
            if (should_skip)
                continue;

            if (tid < dim) {
                config[tid] = nn_mr_new_config[tid];
                delta[tid] = (config[tid] - nearest_node[tid]) / (float)d_settings.granularity;
            }
            __syncthreads();

            // validate edge
            float interp_cfg[dim];
            for (int i = 0; i < dim; i++) {
                interp_cfg[i] = nearest_node[i] + (int(tid / 4 + 1) * delta[i]);
            }
            __syncthreads();

            ppln::device_utils::fkcc_single_buffer_drrt<Robot>(
                interp_cfg, env, tid, sphere_pos, link_CC, T, local_cc_result,
                any_approx_env_collision, any_approx_self_collision, self_sphere_to_check,
                self_thread_dist, d_settings.granularity);

            bool edge_good = local_cc_result[0] == 0;
            __syncthreads();

            if (edge_good) {

                // node already exists - update cost and parent, then continue to next iteration
                if (node_already_exist) {
                    if (tid == 0) {
                        t_parents[sindex_nn[0]] = parent_idx;
                        node_cost[t_tree_id][sindex_nn[0]] = new_node_cost;
                    }
                    continue;
                }

                float new_node_cost = node_cost[t_tree_id][sindex[0]] + sdata[0];
                int parent_idx = sindex[0];
                // grow tree
                if (tid == 0) {
                    index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                    if (index >= d_settings.max_samples)
                        solved = -1;

                    if (d_settings.dynamic_domain) {
                        radii[t_tree_id][index] = FLT_MAX;
                        volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                        float old_radius, new_radius;
                        int expected, desired;
                        do {
                            old_radius = *radius_ptr;
                            if (old_radius == FLT_MAX)
                                break;
                            new_radius = old_radius * (1 + d_settings.dd_alpha);
                            expected = __float_as_int(old_radius);
                            desired = __float_as_int(new_radius);
                        } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
                    }
                }
                __syncthreads();


                if (tid < dim) {
                    t_nodes[index * dim + tid] = config[tid];
                }
                if (tid < num_robot) {
                    node_roadmap_id[t_tree_id][tid][index] = nn_mr_new_config_roadmap_id[tid];
                }
                __syncthreads();
                if (tid == 0) {
                    t_parents[index] = parent_idx;
                    node_cost[t_tree_id][index] = new_node_cost;
                    atomicAdd((int *)&completed_nodes[t_tree_id], 1);
                    __threadfence();
                }
                __syncthreads();

                // connect
                local_min_dist = FLT_MAX;
                local_near_idx = -1;
                int size = min(atomic_free_index[o_tree_id], completed_nodes[o_tree_id]);
                float cost_upper_bound = best_cost - node_cost[t_tree_id][index];
                if (optimize == 0)
                    cost_upper_bound = FLT_MAX;
                for (unsigned int i = tid; i < size; i += blockDim.x) {
                    if (check_unwritten((float *)&o_nodes[i * dim], dim))
                        continue;

                    dist =
                        device_utils::sq_l2_dist((float *)&o_nodes[i * dim], (float *)config, dim);

                    if (dist < local_min_dist && (optimize == 0 || sqrt(dist) < cost_upper_bound)) {
                        local_min_dist = dist;
                        local_near_idx = i;
                    }
                }
                sdata[tid] = local_min_dist;
                sindex[tid] = local_near_idx;
                __syncthreads();
                __threadfence();

                for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                    float sdata_tid = sdata[tid];
                    float sdata_tid_s = sdata[tid + s];
                    __syncthreads();
                    if (tid < s) {
                        if (sdata_tid_s < sdata_tid) {
                            sdata[tid] = sdata[tid + s];
                            sindex[tid] = sindex[tid + s];
                        }
                    }
                    __syncthreads();
                }

                if (sindex[0] == -1) {
                    continue;
                }

                if (tid == 0) {
                    sdata[0] = sqrt(sdata[0]);
                    nearest_node = &o_nodes[sindex[0] * dim];
                    n_extensions = int(sdata[0] / d_settings.range) + 1;
                    local_cc_result[0] = 0;
                    any_approx_env_collision[0] = 0;
                    any_approx_self_collision[0] = 0;
                    __threadfence();
                }
                __syncthreads();

                if (tid < dim) {
                    vec[tid] = (nearest_node[tid] - config[tid]) / (float)n_extensions;
                }
                __syncthreads();

                // validate the edge to the nearest neighbor in opposite tree, go as far as we can
                int extension_parent_idx = index;
                bool ext_edge_good;

                for (int t = 0; t < n_extensions; t++) {
                    for (int i = 0; i < dim; i++) {
                        interp_cfg[i] = config[i] + (int(tid / 4 + 1) *
                                                     (vec[i] / (float)d_settings.granularity));
                    }
                    ppln::device_utils::fkcc_single_buffer<Robot>(
                        interp_cfg, env, tid, sphere_pos, link_CC, T, local_cc_result,
                        any_approx_env_collision, any_approx_self_collision, env_sphere_to_check,
                        self_sphere_to_check, env_thread_dist, self_thread_dist,
                        d_settings.granularity);
                    ext_edge_good = local_cc_result[0] == 0;
                    __syncthreads();
                    if (!ext_edge_good)
                        break;
                    if (tid < dim) {
                        config[tid] += vec[tid];
                    }
                    __syncthreads();
                }

                if (ext_edge_good) {
                    if (tid == 0) {
                        index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                        t_parents[index] = extension_parent_idx;
                        radii[t_tree_id][index] = FLT_MAX;
                        node_cost[t_tree_id][index] =
                            node_cost[t_tree_id][extension_parent_idx] + sdata[0] / n_extensions;
                        extension_parent_idx = index;
                        local_cc_result[0] = 0;
                        any_approx_env_collision[0] = 0;
                        any_approx_self_collision[0] = 0;
                    }
                    __syncthreads();
                    if (tid < dim) {
                        t_nodes[index * dim + tid] = config[tid];
                    }
                    if (tid == 0) {
                        atomicAdd((int *)&completed_nodes[t_tree_id], 1);
                        __threadfence();
                    }
                    __syncthreads();

                    // connected
                    if (tid == 0 && atomicCAS((int *)&solved, 0, bid + 1) == 0) {
                        record_connected_path<Robot>(t_tree_id, o_tree_id, t_nodes, o_nodes,
                                                     t_parents, o_parents, index, sindex[0], iter);
                    }
                    __syncthreads();
                }
            } 
            __syncthreads();
        }
    }

    template <typename Robot>
    PlannerResult<Robot>
    solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals,
          ppln::collision::Environment<float> &h_environment, pop_settings &settings,
          const int num_robots, const int block_per_robot) {
        auto start_time = std::chrono::steady_clock::now();

        bool init_sol = false;

        static constexpr auto dim = Robot::dimension;
        std::size_t start_index = 0;
        PlannerResult<Robot> res;

        curandStateXORWOW_t *d_states;
        cudaMalloc(&d_states, sizeof(curandStateXORWOW_t) * settings.num_new_configs * dim);
        init_xorwow<<<settings.num_new_configs, dim>>>(d_states, 1234ULL);

        // create a curandState for each thread
        curandState *rng_states;
        int num_rng_states = settings.num_new_configs * dim;
        cudaMalloc(&rng_states, num_rng_states * sizeof(curandState));
        init_rng<<<5, MAX_GRANULARITY>>>(rng_states, 1, num_rng_states);

        // copy data to GPU
        cudaMemcpyToSymbol(d_settings, &settings, sizeof(settings));

        int num_goals = goals.size();

        float *d_start;
        cudaMalloc(&d_start, start.size() * sizeof(float));
        cudaMemcpy(d_start, start.data(), start.size() * sizeof(float), cudaMemcpyHostToDevice);

        float *d_goals;
        cudaMalloc(&d_goals, dim * num_robots * goals.size() * sizeof(float));
        for (int i = 0; i < num_goals; i++)
            cudaMemcpy(d_goals + i * dim * num_robots, goals[i].data(),
                       num_robots * dim * sizeof(float), cudaMemcpyHostToDevice);

        float *nodes[2];
        int *parents[2];
        float *radii[2];
        float *node_cost[2];
        float **d_nodes;
        int **d_parents;
        float **d_radii;
        float **d_node_cost;
        cudaMalloc(&d_nodes, 2 * sizeof(float *));
        cudaMalloc(&d_parents, 2 * sizeof(int *));
        cudaMalloc(&d_radii, 2 * sizeof(float *));
        cudaMalloc(&d_node_cost, 2 * sizeof(float *));
        const std::size_t config_size = dim * sizeof(float);

        for (int i = 0; i < 2; i++) {
            cudaMalloc(&nodes[i], settings.max_samples * config_size);
            cudaMalloc(&parents[i], settings.max_samples * sizeof(int));
            cudaMalloc(&radii[i], settings.max_samples * sizeof(float));
            cudaMalloc(&node_cost[i], settings.max_samples * sizeof(int));
        }
        cudaMemcpy(d_nodes, nodes, 2 * sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(d_parents, parents, 2 * sizeof(int *), cudaMemcpyHostToDevice);
        cudaMemcpy(d_radii, radii, 2 * sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(d_node_cost, node_cost, 2 * sizeof(float *), cudaMemcpyHostToDevice);

        float *sphere_pos[300];
        float **d_sphere_pos;
        cudaMalloc(&d_sphere_pos, 300 * sizeof(float *));
        for (int i = 0; i < 300; i++) {
            cudaMalloc(&sphere_pos[i], 100000 * sizeof(float));
        }
        cudaMemcpy(d_sphere_pos, sphere_pos, 300 * sizeof(float *), cudaMemcpyHostToDevice);

        float *tmp_roadmap_init = new float[700000];
        std::fill(tmp_roadmap_init, tmp_roadmap_init + 700000, UNWRITTEN_VAL);
        float *roadmaps[8];
        float **d_roadmaps;
        cudaMalloc(&d_roadmaps, 8 * sizeof(float *));
        for (int i = 0; i < 8; i++) {
            cudaMalloc(&roadmaps[i], 700000 * sizeof(float));
            cudaMemcpy(roadmaps[i], tmp_roadmap_init, 700000 * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_roadmaps, roadmaps, 8 * sizeof(float *), cudaMemcpyHostToDevice);
        cudaCheckError(cudaGetLastError());

        int *roadmaps_id[8];
        int **d_roadmaps_id;
        cudaMalloc(&d_roadmaps_id, 8 * sizeof(int *));
        for (int i = 0; i < 8; i++) {
            cudaMalloc(&roadmaps_id[i], 100000 * sizeof(int));
        }
        cudaMemcpy(d_roadmaps_id, roadmaps_id, 8 * sizeof(int *), cudaMemcpyHostToDevice);

        const int D0 = 8, D1 = 100000, D2 = 3200;

        uint32_t ***d_roadmap_edges;
        cudaMalloc(&d_roadmap_edges, D0 * sizeof(uint32_t **));

        uint32_t **d_mid[D0];

        for (int i = 0; i < D0; i++) {
            uint32_t *block;
            cudaMalloc(&block, (size_t)D1 * D2 * sizeof(uint32_t));
            cudaMemset(block, 0, (size_t)D1 * D2 * sizeof(uint32_t)); // all -> 0u

            uint32_t **leaf = (uint32_t **)malloc(D1 * sizeof(uint32_t *));
            for (int j = 0; j < D1; j++)
                leaf[j] = block + (size_t)j * D2;

            cudaMalloc(&d_mid[i], D1 * sizeof(uint32_t *));
            cudaMemcpy(d_mid[i], leaf, D1 * sizeof(uint32_t *), cudaMemcpyHostToDevice);

            free(leaf);
        }

        cudaMemcpy(d_roadmap_edges, d_mid, D0 * sizeof(uint32_t **), cudaMemcpyHostToDevice);

        // allocate for obstacles
        ppln::collision::Environment<float> *env;
        setup_environment_on_device(env, h_environment);
        cudaCheckError(cudaGetLastError());

        // initialize radii
        std::vector<float> radii_init(num_goals, FLT_MAX);
        cudaMemcpy((void *)radii[0], radii_init.data(), sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)radii[1], radii_init.data(), sizeof(float) * num_goals,
                   cudaMemcpyHostToDevice);

        // set nodes to unitialized
        std::vector<float> nodes_init(settings.max_samples * dim, UNWRITTEN_VAL);

        cudaMemcpy((void *)parents[0], &start_index, sizeof(int), cudaMemcpyHostToDevice);
        std::vector<int> parents_b_init(num_goals);
        iota(parents_b_init.begin(), parents_b_init.end(),
             0); // consecutive integers from 0 ... num_goals - 1
        cudaMemcpy((void *)parents[1], parents_b_init.data(), sizeof(int) * num_goals,
                   cudaMemcpyHostToDevice);

        const int D00 = 2, D11 = 8, D22 = settings.max_samples;

        volatile int ***d_node_roadmap_id;
        cudaMalloc(&d_node_roadmap_id, D00 * sizeof(int **));

        volatile int **d_mid2[D00];

        for (int i = 0; i < D00; i++) {
            volatile int *block;
            cudaMalloc((void **)&block, (size_t)D11 * D22 * sizeof(int));
            cudaMemset((void *)block, 0, (size_t)D11 * D22 * sizeof(int)); // all -> 0

            volatile int **leaf = (volatile int **)malloc(D11 * sizeof(int *));
            for (int j = 0; j < D11; j++)
                leaf[j] = block + (size_t)j * D22;

            cudaMalloc((void **)&d_mid2[i], D11 * sizeof(int *));
            cudaMemcpy((void *)d_mid2[i], (void *)leaf, D11 * sizeof(int *),
                       cudaMemcpyHostToDevice);

            free((void *)leaf);
        }

        cudaMemcpy((void *)d_node_roadmap_id, (void *)d_mid2, D00 * sizeof(int **),
                   cudaMemcpyHostToDevice);

        // Setup pinned memory for signaling
        int *h_solved;
        int current_samples[2];
        int h_solved_iters = -1;
        cudaMallocHost(&h_solved, sizeof(int));
        *h_solved = -1;
        int h_free_index[2] = {1, num_goals};
        int h_completed_nodes[2] = {1, num_goals}; // start and goals are already written

        int cgNumBlocks_simplify = 16;
        int cgThreadsPerBlock_simplify = 4 * settings.granularity;

        int cgNumBlocks = num_robots * block_per_robot;
        int cgThreadsPerBlock = 4 * settings.granularity;
        void *cgKernelArgs[] = {(void *)&env, // device pointer
                                (void *)&d_start,
                                (void *)&d_goals,
                                (void *)&d_sphere_pos,
                                (void *)&d_roadmaps,
                                (void *)&d_roadmaps_id,
                                (void *)&d_roadmap_edges,
                                (void *)&num_goals,
                                (void *)&d_states,
                                (void *)&block_per_robot,
                                (void *)&num_robots,
                                (void *)&settings.granularity};

        auto kernel_start_time = std::chrono::steady_clock::now();

        for (int rrtc_iter = 0; rrtc_iter < settings.rrtc_iter; rrtc_iter++) {

            cudaMemcpy((void *)nodes[0], nodes_init.data(), config_size * settings.max_samples,
                       cudaMemcpyHostToDevice);
            cudaMemcpy((void *)nodes[1], nodes_init.data(), config_size * settings.max_samples,
                       cudaMemcpyHostToDevice);

            // free index for next available position in tree_a and tree_b

            cudaMemcpyToSymbol(atomic_free_index, &h_free_index, sizeof(int) * 2);
            cudaMemcpyToSymbol(nodes_size, &h_free_index, sizeof(int) * 2);

            // initialize completed_nodes counter

            cudaMemcpyToSymbol(completed_nodes, &h_completed_nodes, sizeof(int) * 2);
            auto copy_start_time = std::chrono::steady_clock::now();
            // add start to tree_a and goals to tree_b
            cudaMemcpy((void *)nodes[0], start.data(), config_size, cudaMemcpyHostToDevice);
            cudaMemcpy((void *)nodes[1], goals.data(), config_size * num_goals,
                       cudaMemcpyHostToDevice);
            res.copy_ns = get_elapsed_nanoseconds(copy_start_time);

            int h_optimize = rrtc_iter != 0 ? 1 : 0;
            cudaMemcpyToSymbol(optimize, &h_optimize, sizeof(int));
            cudaDeviceSynchronize();

            cudaError_t err;
            do {
                err =
                    cudaLaunchCooperativeKernel((void *)build_roadmap<typename Robot::SingleRobot>,
                                                dim3(cgNumBlocks),       // gridDim
                                                dim3(cgThreadsPerBlock), // blockDim
                                                cgKernelArgs,
                                                0, // sharedMem
                                                0  // stream
                    );
                if (err != cudaSuccess)
                    printf("build_roadmap launch failed: %s\n", cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    printf("build_roadmap sync failed: %s\n", cudaGetErrorString(err));

                rrtc<Robot><<<settings.num_new_configs, 4 * settings.granularity>>>(
                    d_nodes, d_parents, d_node_cost, d_node_roadmap_id, d_sphere_pos, d_roadmaps,
                    d_roadmap_edges, d_radii, d_states, rng_states, env);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                    printf("rrtc launch failed: %s\n", cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess)
                    printf("rrtc sync failed: %s\n", cudaGetErrorString(err));
                printf("rrtc complete\n");
                cudaMemcpyFromSymbol(h_solved, solved, sizeof(int), 0, cudaMemcpyDeviceToHost);
            } while (*h_solved <= 0 && h_optimize == 0);

            if (settings.path_simplify) {
                err = launch_path_simplification<Robot>(
                    cgNumBlocks_simplify, cgThreadsPerBlock_simplify, env, d_sphere_pos);
                if (err != cudaSuccess)
                    printf("multiBlock_simplify launch failed: %s\n", cudaGetErrorString(err));
            }
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
                printf("final sync failed: %s\n", cudaGetErrorString(err));
            printf("path simplify complete\n");

            // get data from device
            copy_start_time = std::chrono::steady_clock::now();

            float h_cost;
            cudaMemcpyFromSymbol(&h_cost, cost, sizeof(float), 0, cudaMemcpyDeviceToHost);

            cudaCheckError(cudaGetLastError());

            // add data to result struct
            res.start_tree_size = current_samples[0];
            res.goal_tree_size = current_samples[1];
            if (*h_solved <= 0)
                printf("aorrtc failed\n");
            if (*h_solved > 0 && (init_sol == false || res.cost > h_cost)) {
                printf("cost %f new cost %f\n", res.cost, h_cost);
                std::cout << "time kernel_ns " << get_elapsed_nanoseconds(kernel_start_time)
                          << std::endl;
                cudaMemcpyFromSymbol(current_samples, atomic_free_index, sizeof(int) * 2, 0,
                                     cudaMemcpyDeviceToHost);
                cudaMemcpyFromSymbol(&h_solved_iters, solved_iters, sizeof(int), 0,
                                     cudaMemcpyDeviceToHost);
                init_sol = true;
                copy_path_to_result<Robot>(res, settings.path_simplify, h_cost, *h_solved,
                                           h_solved_iters);
            }
            res.copy_ns += get_elapsed_nanoseconds(copy_start_time);
        }
        res.kernel_ns = get_elapsed_nanoseconds(kernel_start_time);

        cudaCheckError(cudaGetLastError());
        res.wall_ns = get_elapsed_nanoseconds(start_time);
        cudaDeviceReset();
        return res;
    }

    template PlannerResult<typename ppln::robots::Panda_four>
    solve<ppln::robots::Panda_four>(std::array<float, 28> &, std::vector<std::array<float, 28>> &,
                                    ppln::collision::Environment<float> &, pop_settings &,
                                    const int, const int);
    template PlannerResult<typename ppln::robots::Panda_five>
    solve<ppln::robots::Panda_five>(std::array<float, 35> &, std::vector<std::array<float, 35>> &,
                                    ppln::collision::Environment<float> &, pop_settings &,
                                    const int, const int);

} // namespace dRRT

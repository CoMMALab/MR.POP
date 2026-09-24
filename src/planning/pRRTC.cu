#include "Planners.hh"
#include "Robots.hh"
#include "utils.cuh"
#include "pRRTC_settings.hh"
#include "phs.hh"
#include "src/collision/environment.hh"
#include "roadmap_interior.cuh"
#include "src/robots/panda.cuh"
#include "src/robots/panda_two.cuh"
#include "src/robots/panda_four.cuh"
#include "src/robots/panda_five.cuh"
//#include "src/robots/fetch.cuh"
//#include "src/robots/baxter.cuh"

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

/*
Parallelized RRTC: Each block works to add a config to the tree (either start or goal depending on balance)
*/


namespace pRRTC {
    using namespace ppln;
    __device__ volatile int solved = 0;
    __device__ volatile int roadmap_launch_cnt = 0;
    __device__ volatile int optimize = 0;
    __device__ volatile float best_cost = FLT_MAX;
    //__device__ volatile float node_cost[2][500000] = {0.0f};
    //__device__ volatile int node_roadmap_id[2][8][500000];
    __device__ volatile int all_block_linkCC[300][8000];
    //__device__ volatile float sphere_pos_all[300][100000];
    __device__ volatile int atomic_free_index[2]; // separate for tree_a and tree_b
    __device__ volatile int nodes_size[2];
    __device__ volatile int completed_nodes[2]; // track completed nodes for each tree
    constexpr int MAX_PATH_SIZE = 50000;
    __device__ float path[2][MAX_PATH_SIZE]; // solution path segments for tree_a, and tree_b
    __device__ volatile float path_out[MAX_PATH_SIZE]; // solution path segments for tree_a, and tree_b
    __device__ volatile int path_size[2] = {0, 0};
    __device__ volatile float simplify_path_cost[MAX_PATH_SIZE] = {FLT_MAX};
    __device__ volatile int simplify_path_parent[MAX_PATH_SIZE] = {-1};
    __device__ volatile float cost = 0.0;
    __device__ int reached_goal_idx = 0;
    __device__ int solved_iters = 0; // value of iters in the block that solves the problem
    //__device__ volatile float roadmaps[8][70000];
    // __device__ volatile int roadmap_component_id[8][100000];
    __device__ volatile int roadmap_size[8];
    //__device__ volatile uint32_t roadmap_edges[8][100000][3200];
    __device__ volatile bool start_goal_connected[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    __constant__ pRRTC_settings d_settings;

    constexpr int MAX_GRANULARITY = 64;
    constexpr int MAX_THREADS_PER_BLOCK = 4*MAX_GRANULARITY;

    constexpr int BLOCK_SIZE = 64;
    constexpr float UNWRITTEN_VAL = -9999.0f;

    template<typename Robot>
    struct HaltonState {
        float b[Robot::dimension];   // bases
        float n[Robot::dimension];   // numerators
        float d[Robot::dimension];   // denominators
    };


    __global__
    void init_xorwow(curandStateXORWOW_t* states, unsigned long long seed)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Each thread gets a deterministic, independent subsequence
        curand_init(
            seed,       // base seed
            tid,        // sequence number (gives each thread its own stream)
            0,          // offset
            &states[tid]
        );
    }

    __global__ void init_rng(curandState* states, unsigned long seed, int num_rng_states) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_rng_states) return;
        curand_init(seed + idx, idx, 0, &states[idx]);
    }


    __device__ inline void print_config(volatile float *config, int dim) {
        for (int i = 0; i < dim; i++) {
            printf("%f ,", config[i]);
        }
        printf("\n");
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
            cudaMemcpy(&(d_env->spheres), &d_spheres, sizeof(ppln::collision::Sphere<float>*), 
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
            
            cudaMemcpy(&(d_env->capsules), &d_capsules, sizeof(ppln::collision::Capsule<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_capsules), &h_env.num_capsules, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        // Repeat for each primitive type...
        if (h_env.num_z_aligned_capsules > 0) {
            ppln::collision::Capsule<float> *d_z_capsules;
            cudaMalloc(&d_z_capsules, sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules);
            cudaMemcpy(d_z_capsules, h_env.z_aligned_capsules,
                    sizeof(ppln::collision::Capsule<float>) * h_env.num_z_aligned_capsules,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->z_aligned_capsules), &d_z_capsules, sizeof(ppln::collision::Capsule<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_capsules), &h_env.num_z_aligned_capsules, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_cylinders > 0) {
            ppln::collision::Cylinder<float> *d_cylinders;
            cudaMalloc(&d_cylinders, sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders);
            cudaMemcpy(d_cylinders, h_env.cylinders,
                    sizeof(ppln::collision::Cylinder<float>) * h_env.num_cylinders,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->cylinders), &d_cylinders, sizeof(ppln::collision::Cylinder<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cylinders), &h_env.num_cylinders, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_cuboids;
            cudaMalloc(&d_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids);
            cudaMemcpy(d_cuboids, h_env.cuboids,
                    sizeof(ppln::collision::Cuboid<float>) * h_env.num_cuboids,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->cuboids), &d_cuboids, sizeof(ppln::collision::Cuboid<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_cuboids), &h_env.num_cuboids, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }

        if (h_env.num_z_aligned_cuboids > 0) {
            ppln::collision::Cuboid<float> *d_z_cuboids;
            cudaMalloc(&d_z_cuboids, sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids);
            cudaMemcpy(d_z_cuboids, h_env.z_aligned_cuboids,
                    sizeof(ppln::collision::Cuboid<float>) * h_env.num_z_aligned_cuboids,
                    cudaMemcpyHostToDevice);
            
            cudaMemcpy(&(d_env->z_aligned_cuboids), &d_z_cuboids, sizeof(ppln::collision::Cuboid<float>*),
                    cudaMemcpyHostToDevice);
            cudaMemcpy(&(d_env->num_z_aligned_cuboids), &h_env.num_z_aligned_cuboids, sizeof(unsigned int),
                    cudaMemcpyHostToDevice);
        }
    }


    inline void cleanup_environment_on_device(ppln::collision::Environment<float> *d_env, 
                                        const ppln::collision::Environment<float> &h_env) {
        // Get the pointers from device struct before freeing
        ppln::collision::Sphere<float> *d_spheres = nullptr;
        ppln::collision::Capsule<float> *d_capsules = nullptr;
        ppln::collision::Capsule<float> *d_z_capsules = nullptr;
        ppln::collision::Cylinder<float> *d_cylinders = nullptr;
        ppln::collision::Cuboid<float> *d_cuboids = nullptr;
        ppln::collision::Cuboid<float> *d_z_cuboids = nullptr;

        // Copy each pointer from device memory
        if (h_env.num_spheres > 0) {
            cudaMemcpy(&d_spheres, &(d_env->spheres), sizeof(ppln::collision::Sphere<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_spheres);
        }
        
        if (h_env.num_capsules > 0) {
            cudaMemcpy(&d_capsules, &(d_env->capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_capsules);
        }
        
        if (h_env.num_z_aligned_capsules > 0) {
            cudaMemcpy(&d_z_capsules, &(d_env->z_aligned_capsules), sizeof(ppln::collision::Capsule<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_capsules);
        }
        
        if (h_env.num_cylinders > 0) {
            cudaMemcpy(&d_cylinders, &(d_env->cylinders), sizeof(ppln::collision::Cylinder<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cylinders);
        }
        
        if (h_env.num_cuboids > 0) {
            cudaMemcpy(&d_cuboids, &(d_env->cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_cuboids);
        }
        
        if (h_env.num_z_aligned_cuboids > 0) {
            cudaMemcpy(&d_z_cuboids, &(d_env->z_aligned_cuboids), sizeof(ppln::collision::Cuboid<float>*), cudaMemcpyDeviceToHost);
            cudaFree(d_z_cuboids);
        }

        // Finally free the environment struct itself
        cudaFree(d_env);
    }

    __global__ void reset_device_variables_kernel() {
        solved = 0;
        
        atomic_free_index[0] = 0;
        atomic_free_index[1] = 0;
        nodes_size[0] = 0;
        nodes_size[1] = 0;
        completed_nodes[0] = 0;
        completed_nodes[1] = 0;
        
        path_size[0] = 0;
        path_size[1] = 0;
        
        for (int tree = 0; tree < 2; tree++) {
            for (int i = 0; i < MAX_PATH_SIZE; i++) {
                path[tree][i] = 0.0f;
            }
        }
        
        cost = 0.0f;
        reached_goal_idx = 0;
    }

    void reset_device_variables() {
        reset_device_variables_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
    }

    __device__ __forceinline__ void reset_to_unwritten_state(volatile float *buffer, int size, int tid) {
        if (tid == 0) {
            for (int i = 0; i < size; i++) {
                buffer[i] = UNWRITTEN_VAL;
            }
        }
        __syncthreads();
    }

    __device__ __forceinline__ bool check_unwritten(float *config, int dim) {
        
        for (int i=0; i<dim; i++){
            if (config[i]==UNWRITTEN_VAL){
                return true;
            }
        }
        return false;
    }

    template <typename Robot>
    __global__ void build_roadmap(ppln::collision::Environment<float> *env, float* start, float* goal, float **sphere_pos_all, float ** roadmaps, int ** roadmap_component_id, uint32_t*** roadmap_edges, 
                                  const int num_goals,
                                  curandStateXORWOW_t* states, const int block_per_robot, const int num_robot, const int granularity){
        
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
        
        //__align__(16) __shared__ volatile float sphere_pos[6000];
        //volatile int * link_CC = all_block_linkCC[bid];
        __align__(16) __shared__ float T[64 * 2 * 16]; 
        __align__(16) __shared__ int link_CC[1500]; 
        __shared__ float delta[dim];


        build_roadmap_interior<Robot>(env, start, goal, num_goals, edge_nn, states, block_per_robot, num_robot, granularity,
             config, local_cc_result, any_approx_env_collision,
             any_approx_self_collision, node_id, current_nn, link_CC, T, delta,
             (volatile float **)roadmaps,
             (volatile int **)roadmap_component_id,
             (volatile int *)roadmap_size,
             (volatile uint32_t ***) roadmap_edges,
             (volatile bool *)start_goal_connected,
             (volatile float **)sphere_pos_all,
             (volatile int *) env_sphere_to_check,
             (volatile int *) self_sphere_to_check,
             (volatile int *) env_thread_dist,
             (volatile int *) self_thread_dist,
             (volatile float *) sdata,
             (volatile int *)sindex,
             &d_settings, roadmap_launch_cnt);

        if (bid==0 && tid==0) roadmap_launch_cnt++;

        return;

    }

    
    template <typename Robot>
    __global__ void
    // __launch_bounds__(128, 8)
    rrtc(
        float **nodes,
        int **parents,
        float **node_cost,
        volatile int *** node_roadmap_id,
        float **sphere_pos_all,
        float **roadmaps,
        uint32_t ***roadmap_edges,
        float **radii,
        curandStateXORWOW_t* states,
        curandState *rng_states,
        ppln::collision::Environment<float> *env
    )
    {
        static constexpr auto dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x; // 0 ... NUM_NEW_CONFIGS
        int num_robot=1;
        if (Robot::name == "panda_dual") num_robot=2;
        if (Robot::name == "panda_four") num_robot=4;
        if (Robot::name == "panda_five") num_robot=5;
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
        __shared__ float* focus1;
        __shared__ float* focus2;
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
        //__align__(16) __shared__ volatile float sphere_pos[6000];
        //__align__(16) __shared__ volatile int link_CC[640]; 
        __align__(16) __shared__ float T[64 * 2 * 16]; 
        volatile float * sphere_pos = sphere_pos_all[bid];
        volatile int * link_CC = all_block_linkCC[bid];

        if (tid==0 && bid==0){
            //atomicExch((int *)&solved, 0);
            solved=0;
            node_cost[0][0] = 0.0f;
            int num_goals = atomic_free_index[1];
            for (int i=0; i<num_goals; i++){
                node_cost[1][i] = 0.0f;
            }
        }
        if (bid==1 && tid<num_robot){
            node_roadmap_id[1][tid][0] = 1;
        }

        if (tid==0){
            c_rand = FLT_MAX;
        }



        // phs setup - update rotation & transformation
        if (optimize>0 && d_settings.phs){
            if (tid==0){
                focus1 = nodes[0];
                focus2 = nodes[1];
                min_transverse_diameter = device_utils::l2_dist(focus1, focus2, dim);
                transverse_diameter = best_cost;
            }
            __syncthreads();
            if (tid<dim){
                center[tid] = (focus1[tid] + focus2[tid]) / 2.0f;
                static constexpr float circle_tolerance = 1e-6;
                if (min_transverse_diameter<circle_tolerance){
                    // treat this as a circle
                    for (int c=0; c<dim; c++){
                        rot_world_from_ellipse[tid][c] = (tid==c) ? 1.0f : 0.0f;
                    }
                }
                else{
                    float tranverse_axis[dim]; //reuse config as temp storage
                    for (int d=0; d<dim; d++){
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
                        if (r == tid) val += 1.0f;
                        rot_world_from_ellipse[r][tid] = val;
                    }
                }
            }
            const float conjugate_diamater = sqrt(transverse_diameter * transverse_diameter - min_transverse_diameter * min_transverse_diameter);
            float* diag = config; //reuse config as temp storage
            if (tid==0) diag[tid] = transverse_diameter/2;
            if (tid>=1 && tid<dim) diag[tid] = conjugate_diamater/2;
            __syncthreads();
            if (tid<dim){
                for (int c=0; c<dim; c++){
                    tf_world_from_ellipse[tid][c] = rot_world_from_ellipse[tid][c] * diag[c];
                }
            }
        }
        __syncthreads();
        
        int iter = 0;
        while (true) {

            __syncthreads();
            if (tid==1) delta[0] = solved;
            __syncthreads();
            if (delta[0]!=0) return;

            if (tid == 0) {
                iter++;
                //printf("bid %d iter %d max iter %d solved %d\n", bid, iter, d_settings.max_iters, atomicAdd((int*)&solved, 0));
                if (iter > d_settings.max_iters || (optimize==1 && iter>d_settings.optimize_iters)) {
                    atomicAdd((int *)&solved, -1);
                }

                if (d_settings.balance == 0 || iter == 1) {
                    t_tree_id = (bid < (d_settings.num_new_configs / 2))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1 && abs(atomic_free_index[0]-atomic_free_index[1]) < 1.5 * d_settings.num_new_configs) { // dynamic balance
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0]+atomic_free_index[1]);
                    float balance_factor = 1 - ratio;
                    t_tree_id = (bid < (d_settings.num_new_configs * balance_factor))? 0 : 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 1) {
                    float ratio = atomic_free_index[0] / (float)(atomic_free_index[0] + atomic_free_index[1]);
                    if (ratio < d_settings.tree_ratio) t_tree_id = 0;
                    else t_tree_id = 1;
                    o_tree_id = 1 - t_tree_id;
                }
                else if (d_settings.balance == 2) { // vamp balance
                    float ratio = abs(atomic_free_index[t_tree_id] - atomic_free_index[o_tree_id]) / (float) atomic_free_index[t_tree_id];
                    if (ratio < d_settings.tree_ratio)
                    {
                        t_tree_id = 1 - t_tree_id;
                        o_tree_id = 1 - t_tree_id;
                    }
                }

                // for testing dRRT - start
                // t_tree_id = 0;
                // o_tree_id = 1;
                // for testing dRRT - end

                t_nodes = nodes[t_tree_id];
                o_nodes = nodes[o_tree_id];
                t_parents = parents[t_tree_id];
                o_parents = parents[o_tree_id];
                
                local_cc_result[0] = 0;
                any_approx_env_collision[0] = 0;
                any_approx_self_collision[0] = 0;
                

            }

            __syncthreads();

            if (!d_settings.phs || optimize==0){
                if (tid<dim){
                    // sample random config
                    curandStateXORWOW_t local = states[bid*dim+tid];
                    float rnd = curand_uniform(&local);
                    config[tid] = rnd;
                    states[bid*dim+tid] = local;
                    //printf("tid %d config %f\n", tid, config[tid]);
                }
            }
            else{
                // PHS sampling **********************************************************************
                // logit
                if (tid<dim){
                    curandStateXORWOW_t local = states[bid*dim+tid];
                    float rnd = curand_uniform(&local);
                    states[bid*dim+tid] = local;
                    config[tid] = logf(rnd * (__frcp_rn(1.0f - rnd))) * sqrtf(M_PI / 8.0f);
                }
                __syncthreads();
                // uniform on ball
                if (tid==0){
                    delta[0] = 0.0f;
                    for (int d = 0; d < dim; d++) {
                        delta[0] += config[d] * config[d];
                    }
                    delta[0] = sqrtf(delta[0]);
                }
                __syncthreads();
                if (tid<dim){
                    config[tid] = config[tid]/delta[0];
                }
                //uniform in ball
                if (tid == 0) {
                    curandStateXORWOW_t local = states[bid*dim+tid];
                    float rnd = curand_uniform(&local);
                    states[bid*dim+tid] = local;
                    delta[1] = powf(rnd, 1.0f / (float)dim);
                }
                __syncthreads();
                if (tid<dim){
                    config[tid] = config[tid] * delta[1];
                }
                __syncthreads();
                // transform
                if (tid<dim){
                    float val = 0.0f;
                    for (int c=0; c<dim; c++){
                        val += tf_world_from_ellipse[tid][c] * config[c];
                    }
                    config[tid] = val + center[tid];
                    Robot::descale_cfg((float *)config);
                    config[tid] = max(min(1.0f, config[tid]), 0.0f);
                }
            }
            __syncthreads();
            if (tid<dim) Robot::scale_cfg((float *)config);
            
            __syncthreads();

            if (tid==0){
                should_skip = (device_utils::l2_dist((float *)nodes[0], config, dim) + device_utils::l2_dist((float *)nodes[1], config, dim) >= best_cost);
            }
            __syncthreads();
            if (should_skip) continue;

            // sample cost bound
            if (tid==0 && optimize==1){
                const float g_hat = device_utils::l2_dist((float *)nodes[t_tree_id], config, dim);
                const float h_hat = device_utils::l2_dist((float *)nodes[o_tree_id], config, dim);
                const float min_possible_cost = g_hat + h_hat;
                __threadfence();
                float c_range = max(best_cost-min_possible_cost, 0.0f);
                curandStateXORWOW_t local = states[bid*dim+tid];
                float rnd = curand_uniform(&local);
                states[bid*dim+tid] = local;
                c_rand = rnd * c_range + g_hat;
                //printf("sampled cost bound %f g_hat %f h_hat %f best cost %f\n", c_rand, g_hat, h_hat, best_cost);
            }
            __syncthreads();
            

            // parallelized nearest neighbor search
            float local_min_dist = FLT_MAX;
            int local_near_idx = 0;
            float dist;
            int size = min(atomic_free_index[t_tree_id], completed_nodes[t_tree_id]);
            for (int i = tid; i < size; i += blockDim.x) {
                if (check_unwritten((float *)&t_nodes[i * dim], dim)) continue;
                dist = device_utils::sq_l2_dist((float *)&t_nodes[i * dim], (float *) config, dim);
                //if (optimize==1) printf("node cost %f index %d bound %f best cost %f\n", node_cost[t_tree_id][i], i, c_rand, best_cost);
                if (dist < local_min_dist && (optimize==0 || node_cost[t_tree_id][i] < c_rand)) {
                //if (dist < local_min_dist) {
                    local_min_dist = dist;
                    local_near_idx = i;
                    //if (optimize==1) printf("node cost %f bound %f\n", node_cost[t_tree_id][i], c_rand);
                }
            }
            sdata[tid] = local_min_dist;
            sindex[tid] = local_near_idx;
            __syncthreads();

            for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                float sdata_tid = sdata[tid];
                float sdata_tid_s = sdata[tid + s];
                __syncthreads();
                if (tid < s){
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
            //if (optimize==1 && tid==0) printf("config %f %f %f %f sdata[0] %f\n", config[0], config[1], config[2], config[3], sdata[0]);
            //if (optimize==1 && tid==0) printf("should skip %d\n", should_skip);

            if (should_skip) continue;
            __syncthreads();

            float tree_edge_cost_bound = c_rand - node_cost[t_tree_id][sindex[0]];
            if (optimize==0) tree_edge_cost_bound = FLT_MAX;
            //if (optimize==1) printf("tree_edge_cost_bound %f\n", tree_edge_cost_bound);
            
            bool found_nn = ppln::device_utils::nn_angle_mr<Robot>(config, nearest_node, sindex[0], nn_mr_new_config, nn_mr_new_config_roadmap_id, 
                        node_roadmap_id[t_tree_id], roadmaps, roadmap_size, roadmap_edges, tree_edge_cost_bound,
                        d_settings.range, sdata_nn, sindex_nn, num_robot, optimize);

            
            
            if (!found_nn) continue;

            // check if nn_mr_new_config is already in tree. If so, does the neighbor "nearest_node" provide better node cost for nn_mr_new_config?
            for (int n=tid; n<completed_nodes[t_tree_id]; n+=blockDim.x){
                if (device_utils::sq_l2_dist((float *)&t_nodes[n * dim], (float *)nn_mr_new_config, dim)<=1e-5){
                    sdata_nn[0] = -1;
                    sindex_nn[0] = n;
                }
            }
            should_skip = (d_settings.dynamic_domain && radii[t_tree_id][sindex[0]] < device_utils::l2_dist((float *)nearest_node, (float *) nn_mr_new_config, dim));
            __syncthreads();

            //if (tid==0) printf("found nn %d dist %f radii %f\n", found_nn, sdata_nn[0], radii[t_tree_id][sindex[0]]);


            bool node_already_exist = (sdata_nn[0] == -1);
            float new_node_cost = node_cost[t_tree_id][sindex[0]] + sdata[0];
            //if (optimize==1) printf("new node cost %f\n", new_node_cost);
            int parent_idx = sindex[0];
            //if (tid==0) printf("node_already_exist %d optimize %d should_skip %d\n", node_already_exist, optimize, should_skip);
            if (node_already_exist && (optimize==0 || new_node_cost >= node_cost[t_tree_id][sindex_nn[0]])) continue;
            if (optimize==1 && new_node_cost >= best_cost) continue;
            if (should_skip) continue;


            //printf("nn for new sample found\n");

            if (tid < dim) {
                config[tid] = nn_mr_new_config[tid];
                delta[tid] = (config[tid] - nearest_node[tid]) / (float) d_settings.granularity;
            }
            __syncthreads();

            // validate edge
            float interp_cfg[dim];
            for (int i = 0; i < dim; i++) {
                interp_cfg[i] = nearest_node[i] + (int(tid/4 + 1) * delta[i]);
            }
            //if (tid%100==0) printf("bid %d tid %d interp %f %f %f %f %f\n", bid, tid, interp_cfg[0], interp_cfg[20], interp_cfg[30], interp_cfg[40], interp_cfg[52]);
            __syncthreads();
            
            ppln::device_utils::fkcc_single_buffer_drrt<Robot>(interp_cfg, env, tid, sphere_pos, link_CC, T, local_cc_result, any_approx_env_collision, any_approx_self_collision,
                                                               self_sphere_to_check, self_thread_dist, d_settings.granularity);

            bool edge_good = local_cc_result[0] == 0;
            __syncthreads();
            //printf("edge status %d\n", edge_good);

            if (edge_good) {

                // node already exists - update cost and parent, then continue to next iteration
                if (node_already_exist){
                    if (tid==0){
                        t_parents[sindex_nn[0]] = parent_idx;
                        node_cost[t_tree_id][sindex_nn[0]] = new_node_cost;
                    }
                    continue;
                }

                float new_node_cost = node_cost[t_tree_id][sindex[0]] + sdata[0];
                int parent_idx = sindex[0];
                // grow tree
                if (tid == 0) {
                    //printf("edge good\n");
                    index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                    //printf("added node %d to tree %d\n", index, t_tree_id);
                    if (index >= d_settings.max_samples) solved = -1;
                    
                    if (d_settings.dynamic_domain) {
                        radii[t_tree_id][index] = FLT_MAX;
                        volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                        float old_radius, new_radius;
                        int expected, desired;
                        do {
                            old_radius = *radius_ptr;
                            if (old_radius == FLT_MAX) break;
                            new_radius = old_radius * (1 + d_settings.dd_alpha);
                            expected = __float_as_int(old_radius);
                            desired = __float_as_int(new_radius);
                        } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
                    }
                }
                __syncthreads();

                // resample cost bound
                
                if (optimize==1 && d_settings.cost_bound_resample){
                    const float g_hat = device_utils::l2_dist((float *)t_nodes, (float *) config, dim);
                    for (int y=0; y<d_settings.resample_iter; y++){
                        if (tid==0){
                            float old_cost = c_rand;
                            const float c_range = max(new_node_cost - g_hat, 0.0f);
                            curandStateXORWOW_t local = states[bid*dim+tid];
                            float rnd = curand_uniform(&local);
                            states[bid*dim+tid] = local;
                            c_rand = rnd * c_range + g_hat;
                            if (c_range==0) should_skip = true;
                            //printf("resample %d old cost %f new cost %f c_range %f g_hat %f should_skip %d\n", y, old_cost, c_rand, c_range, g_hat, should_skip);
                        }
                        __syncthreads();
                        if (should_skip) break;

                        // parallelized nearest neighbor search
                        float local_min_dist = FLT_MAX;
                        int local_near_idx = 0;
                        float dist;
                        int size = min(atomic_free_index[t_tree_id], completed_nodes[t_tree_id]);
                        for (int i = tid; i < size; i += blockDim.x) {
                            if (check_unwritten((float *)&t_nodes[i * dim], dim)) continue;
                            dist = device_utils::sq_l2_dist((float *)&t_nodes[i * dim], (float *) config, dim);
                            //if (dist>d_settings.range * d_settings.range) continue;
                            if (dist < local_min_dist && node_cost[t_tree_id][i] + sqrt(dist) < c_rand) {
                            //if (dist < local_min_dist) {
                                local_min_dist = dist;
                                local_near_idx = i;
                            }
                        }
                        sdata[tid] = local_min_dist;
                        sindex[tid] = local_near_idx;
                        __syncthreads();

                        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                            float sdata_tid = sdata[tid];
                            float sdata_tid_s = sdata[tid + s];
                            __syncthreads();
                            if (tid < s){
                                if (sdata_tid_s < sdata_tid) {
                                    sdata[tid] = sdata[tid + s];
                                    sindex[tid] = sindex[tid + s];
                                }
                            }
                            __syncthreads();
                        }

                        if (tid == 0) {
                            float sdata_ori = sdata[0];
                            sdata[0] = sqrt(sdata[0]);
                            // break if we have connected to the same parent or worse cost
                            should_skip = (abs(sdata_ori-FLT_MAX)<0.1) || (sindex[0]==parent_idx) || (node_cost[t_tree_id][sindex[0]] + sdata[0] >= new_node_cost);
                            nearest_node = &t_nodes[sindex[0] * dim];
                        }
                        __syncthreads();
                        if (should_skip) break;
                        //if (tid==5) printf("nn %f %f %f %f %f %f %f parent %d sindex %d sdata %f\n", nearest_node[0], nearest_node[1], nearest_node[2], nearest_node[3], nearest_node[4], nearest_node[5], nearest_node[6], parent_idx, sindex[0], sdata[0]);

                        bool edge_good;
                        const int cc_iter = ceil(sdata[0] / (float) d_settings.range);
                        if (tid<dim){
                            delta[tid] = (config[tid] - nearest_node[tid]) / cc_iter;
                        }
                        for (int d=0; d<dim; d++){
                            interp_cfg[d] = nearest_node[d];
                        }
                        __syncthreads();
                        
                        for (int cc_round=0; cc_round<cc_iter; cc_round++){
                            
                            for (int d=0; d<dim; d++){
                                interp_cfg[d] += (int(tid/4 + 1) * (delta[d] / (float)d_settings.granularity));
                            }
                            
                            if (tid==0){
                                any_approx_env_collision[0] = 0;
                                any_approx_self_collision[0] = 0;
                                local_cc_result[0] = 0;
                            }
                            __syncthreads();
                            
                            ppln::device_utils::fkcc_single_buffer<Robot>(interp_cfg, env, tid, sphere_pos, link_CC, T, local_cc_result, any_approx_env_collision, any_approx_self_collision,
                                                                          env_sphere_to_check, self_sphere_to_check,
                                                                          env_thread_dist, self_thread_dist, 
                                                                          d_settings.granularity);

                            edge_good = (local_cc_result[0] == 0);
                            __syncthreads();
                            if (!edge_good) break;
                            
                            for (int d=0; d<dim; d++){
                                interp_cfg[d]+=delta[d];
                            }
                            __syncthreads();
                            
                        }
                        if (!edge_good) break;
                        if (tid==0){
                            //printf("old cost %f, new cost %f\n", new_node_cost, node_cost[t_tree_id][sindex[0]] + sdata[0]);
                            // update parent & node_cost
                            parent_idx = sindex[0];
                            new_node_cost = node_cost[t_tree_id][sindex[0]] + sdata[0];
                            //if (bid<3) printf("config %f %f %f %f %f %f %f nn %f %f %f %f %f %f %f cc_iter %d sdata %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6],
                                                                                //nearest_node[0], nearest_node[1], nearest_node[2], nearest_node[3], nearest_node[4], nearest_node[5], nearest_node[6], cc_iter, sdata[0]);
                        }
                        __syncthreads();
                    }
                }
                

                if (tid < dim) {
                    t_nodes[index * dim + tid] = config[tid];
                }
                if (tid < num_robot){
                    node_roadmap_id[t_tree_id][tid][index] = nn_mr_new_config_roadmap_id[tid];
                }
                __syncthreads();
                if (tid == 0) {
                    t_parents[index] = parent_idx;
                    node_cost[t_tree_id][index] = new_node_cost;
                    atomicAdd((int*)&completed_nodes[t_tree_id], 1);
                    __threadfence();
                    //printf("optimize %d cost bound resample %d c_rand %f\n", optimize, d_settings.cost_bound_resample, c_rand);
                }
                __syncthreads();

                // connect
                local_min_dist = FLT_MAX;
                local_near_idx = -1;
                int size = min(atomic_free_index[o_tree_id], completed_nodes[o_tree_id]);
                float cost_upper_bound = best_cost - node_cost[t_tree_id][index];
                if (optimize == 0) cost_upper_bound = FLT_MAX;
                for (unsigned int i = tid; i < size; i += blockDim.x) {
                    if (check_unwritten((float *)&o_nodes[i * dim], dim)) continue;
                    //if (device_utils::l2_dist((float *)&o_nodes[i * dim], (float *) config, dim) > d_settings.range) continue;
                    
                    dist = device_utils::sq_l2_dist((float *)&o_nodes[i * dim], (float *)config, dim);
                    /*
                    bool roadmap_connected=true;
                    for (int r=0; r<num_robot; r++){
                        if (!(roadmap_edges[r][nn_mr_new_config_roadmap_id[r]][node_roadmap_id[o_tree_id][r][i] >> 5] >> (node_roadmap_id[o_tree_id][r][i] & 31) & 1u)){
                            roadmap_connected=false;
                            break;
                        }
                    }
                    */
                    //printf("dist: %f cost upper bound: %f\n", sqrt(dist), cost_upper_bound);
                    if (dist < local_min_dist && (optimize==0 || sqrt(dist) < cost_upper_bound)) {
                    //if (dist < local_min_dist) {
                        local_min_dist = dist;
                        local_near_idx = i;
                    }
                }
                sdata[tid] = local_min_dist;
                sindex[tid] = local_near_idx;
                __syncthreads();
                __threadfence();
                
                for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                    float sdata_tid = sdata[tid];
                    float sdata_tid_s = sdata[tid + s];
                    __syncthreads();
                    if (tid < s){
                        if (sdata_tid_s < sdata_tid) {
                            sdata[tid] = sdata[tid + s];
                            sindex[tid] = sindex[tid + s];
                        }
                    }
                    __syncthreads();
                }

                if (sindex[0]==-1){
                    //if (tid==0) printf("extend - no NN found\n");
                    continue;
                } 
                //if (optimize==1 && (node_cost[t_tree_id][index] + node_cost[o_tree_id][sindex[0]] + sqrt(sdata[0])) >= best_cost) continue;
                
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
                    vec[tid] = (nearest_node[tid] - config[tid]) / (float) n_extensions;
                }
                __syncthreads();

                /*
                if (tid==0) printf("attempt connect: %f %f %f %f %f %f %f %f %f %f %f %f %f %f, %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
                            config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8], config[9], config[10], config[11], config[12], config[13],
                            nearest_node[0], nearest_node[1], nearest_node[2], nearest_node[3], nearest_node[4], nearest_node[5], nearest_node[6], nearest_node[7], nearest_node[8], nearest_node[9], nearest_node[10], nearest_node[11], nearest_node[12], nearest_node[13]);
                */

                // validate the edge to the nearest neighbor in opposite tree, go as far as we can
                int extension_parent_idx = index;
                bool ext_edge_good;

                for (int t=0; t<n_extensions; t++){
                    for (int i = 0; i < dim; i++) {
                        interp_cfg[i] = config[i] + (int(tid/4 + 1) * (vec[i] / (float) d_settings.granularity));
                    }
                    ppln::device_utils::fkcc_single_buffer<Robot>(interp_cfg, env, tid, sphere_pos, link_CC, T, local_cc_result, any_approx_env_collision, any_approx_self_collision,
                                                                  env_sphere_to_check, self_sphere_to_check,
                                                                  env_thread_dist, self_thread_dist, 
                                                                  d_settings.granularity);
                    ext_edge_good = local_cc_result[0] == 0;
                    __syncthreads();
                    if (!ext_edge_good) break;
                    if (tid < dim) {
                        config[tid] += vec[tid];
                    }
                    __syncthreads();
                }
                

                if (ext_edge_good){
                    if (tid == 0) {
                        index = atomicAdd((int *)&atomic_free_index[t_tree_id], 1);
                        t_parents[index] = extension_parent_idx;
                        radii[t_tree_id][index] = FLT_MAX;
                        node_cost[t_tree_id][index] = node_cost[t_tree_id][extension_parent_idx] + sdata[0]/n_extensions;
                        extension_parent_idx = index;
                        local_cc_result[0] = 0;
                        any_approx_env_collision[0] = 0;
                        any_approx_self_collision[0] = 0;
                    }
                    __syncthreads();
                    if (tid < dim) {
                        //config[tid] = config[tid] + vec[tid];
                        t_nodes[index * dim + tid] = config[tid];
                    }
                    if (tid == 0) {
                        atomicAdd((int*)&completed_nodes[t_tree_id], 1);
                        __threadfence();
                        //printf("added node %d to tree %d n_extensions %d i_extensions %d\n", index, t_tree_id, n_extensions, i_extensions);
                    }
                    __syncthreads();
                    
                
                    // connected
                    if (tid == 0 && atomicCAS((int *)&solved, 0, bid+1) == 0) {
                        // trace back to the start and goal.
                        //printf("connected!\n");
                        cost=0.0f;
                        int current = index;
                        int parent;
                        int t_path_size = 0;
                        int o_path_size = 0;
                        while (t_parents[current] != current) {
                            parent = t_parents[current];
                            cost += device_utils::l2_dist((float *)&t_nodes[current * dim], (float *)&t_nodes[parent * dim], dim);
                            for (int i = 0; i < dim; i++) path[t_tree_id][t_path_size * dim + i] = t_nodes[current * dim + i];
                            t_path_size++;
                            current = parent;
                        }
                        for (int i = 0; i < dim; i++) path[t_tree_id][t_path_size * dim + i] = t_nodes[i];
                        t_path_size++;
                        
                        if (t_tree_id == 1) reached_goal_idx = current;
                        current = sindex[0];
                        while(o_parents[current] != current) {
                            parent = o_parents[current];
                            cost += device_utils::l2_dist((float *)&o_nodes[current * dim], (float *)&o_nodes[parent * dim], dim);
                            for (int i = 0; i < dim; i++) path[o_tree_id][o_path_size * dim + i] = o_nodes[current * dim + i];
                            o_path_size++;
                            current = parent;
                        }
                        for (int i = 0; i < dim; i++) path[o_tree_id][o_path_size * dim + i] = o_nodes[i];
                        o_path_size++;

                        if (t_tree_id == 0) reached_goal_idx = current;
                        path_size[t_tree_id] = t_path_size;
                        path_size[o_tree_id] = o_path_size;
                        solved_iters = iter;
                        //printf("path cost %f best cost %f\n", cost, best_cost);
                        best_cost = min(best_cost, cost);
                        int path_out_ind=0;
                        for (int i=path_size[1]-1; i>=0; i--){
                            for (int d = 0; d < dim; d++) path_out[path_out_ind * dim + d] = path[1][i * dim + d];
                            
                            // debug
                            // for (int d = 0; d < dim; d++) printf("%f, ", path_out[path_out_ind * dim + d]);
                            // printf("\n"); // debug end

                            path_out_ind++;
                            
                        }
                        for (int i=0; i<path_size[0]; i++){
                            for (int d = 0; d < dim; d++) path_out[path_out_ind * dim + d] = path[0][i * dim + d];
                            
                            // debug
                            // for (int d = 0; d < dim; d++) printf("%f, ", path_out[path_out_ind * dim + d]);
                            // printf("\n"); // debug end

                            path_out_ind++;
                        }
                        
                    }
                    __syncthreads();
                }
            }
            else if (d_settings.dynamic_domain && tid == 0) {      
                // printf("no config added\n");
                volatile float *radius_ptr = &radii[t_tree_id][sindex[0]];
                float old_radius, new_radius;
                int expected, desired;
                do {
                    old_radius = *radius_ptr;
                    if (old_radius == FLT_MAX) {
                        new_radius = d_settings.dd_radius;
                    } else {
                        new_radius = fmaxf(old_radius * (1.f - d_settings.dd_alpha), d_settings.dd_min_radius);
                    }
                    expected = __float_as_int(old_radius);
                    desired = __float_as_int(new_radius);
                } while (atomicCAS((int *)radius_ptr, expected, desired) != expected);
            }
            __syncthreads();
            //if (atomicAdd((int *)&solved, 0) != 0) return;
        }
    }

    template <typename Robot>
    __global__ void
    // __launch_bounds__(128, 8)
    multiBlock_simplify(
        int numBlock, int threadsPerBlock,
        ppln::collision::Environment<float> *env,
        float** sphere_pos_all
    ){
        const int dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        __shared__ volatile unsigned int local_cc_result[1];
        __shared__ volatile unsigned int any_approx_env_collision[1];
        __shared__ volatile unsigned int any_approx_self_collision[1];
        __align__(16) __shared__ float T[64 * 2 * 16]; //
        __shared__ float delta[dim];
        __shared__ float middle_point[dim];
        __shared__ volatile int env_sphere_to_check[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int self_sphere_to_check[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int env_thread_dist[MAX_THREADS_PER_BLOCK];
        __shared__ volatile int self_thread_dist[MAX_THREADS_PER_BLOCK];
        volatile float * sphere_pos = sphere_pos_all[bid];
        volatile int * link_CC = all_block_linkCC[bid];
        

        
        cg::grid_group grid = cg::this_grid();
        const int path_total_size = path_size[0] + path_size[1];
        const float granularity_tolerance = 0.001;
        
        for (int simplify_ind=0; simplify_ind<d_settings.simplify_round; simplify_ind++){
            if (tid==0 && bid==0){
                simplify_path_cost[0] = 0;
                simplify_path_parent[0] = -1;
            } 

            for (int t=bid*(threadsPerBlock) + tid + 1; t<path_total_size; t+=threadsPerBlock * numBlock){
                simplify_path_cost[t] = FLT_MAX;
                simplify_path_parent[t] = -1;
            }

            grid.sync();

            // B-spline smoothing
            for (int step = 0; step < d_settings.bspline_step; step++){
                for (int smooth_round=0; smooth_round<1; smooth_round++){
                    for (int r=0; r<ceil((path_total_size-3)/(2.0f*numBlock)); r++){

                        int i = 2*bid+smooth_round+1 + r*2*numBlock;
                        grid.sync();
                        if (i>=path_total_size-1) continue;
                        volatile float* cfg1;
                        cfg1 = path_out + (i-1)*dim;
                        volatile float* cfg2;
                        cfg2 = path_out + (i)*dim;
                        volatile float* cfg3;
                        cfg3 = path_out + (i+1)*dim;
                        //volatile float ** cc_cfgs = new volatile float*[2]{cfg1, cfg3};
                        volatile float* cc_cfgs[2] = {cfg1, cfg3};
                        if (tid<dim){
                            float b_spline_config1 = (cfg1[tid] + cfg2[tid]) / 2.0f;
                            float b_spline_config2 = (cfg2[tid] + cfg3[tid]) / 2.0f;
                            middle_point[tid] = (b_spline_config1 + b_spline_config2) / 2.0f;
                        }
                        __syncthreads();

                        if (device_utils::l2_dist((float *)cfg2, (float *)middle_point, dim)<d_settings.bspline_min_change) continue;

                        bool both_edge_good = true;

                        // collision check between index-1 and midpoint & midpoint and index+1
                        for (int edge_cnt=0; edge_cnt<2; edge_cnt++){
                            const int cc_iter1 = ceil(device_utils::l2_dist((float *)cc_cfgs[edge_cnt], (float *)middle_point, dim) / (float) d_settings.range);
                            if (tid<dim){
                                delta[tid] = (middle_point[tid] - cc_cfgs[edge_cnt][tid]) / cc_iter1;
                            }
                            float interp_cfg[dim];
                            for (int d=0; d<dim; d++){
                                interp_cfg[d] = cc_cfgs[edge_cnt][d];
                            }
                            __syncthreads();

                            for (int d=0; d<dim; d++){
                                interp_cfg[d] += (int(tid/4 + 1) * (delta[d] / (float)d_settings.granularity));
                            }
                            for (int cc_round=0; cc_round<cc_iter1; cc_round++){
                                
                                if (tid==0){
                                    any_approx_env_collision[0] = 0;
                                    any_approx_self_collision[0] = 0;
                                    local_cc_result[0] = 0;
                                }
                                __syncthreads();
                                
                                ppln::device_utils::fkcc_single_buffer<Robot>(interp_cfg, env, tid, sphere_pos, link_CC, T, local_cc_result, any_approx_env_collision, any_approx_self_collision,
                                                                              env_sphere_to_check, self_sphere_to_check,
                                                                              env_thread_dist, self_thread_dist, 
                                                                              d_settings.granularity);

                                bool edge_good = (local_cc_result[0] == 0);
                                __syncthreads();
                                if (!edge_good) {
                                    both_edge_good = false;
                                    break;
                                }
                                for (int d=0; d<dim; d++){
                                    interp_cfg[d]+=delta[d];
                                }
                                __syncthreads();
                            }
                            if (!both_edge_good){
                                break;
                            }
                        }
                        if (both_edge_good){
                            //if (tid==0) printf("smoothed point, before %f %f %f %f %f; after %f %f %f %f %f\n", cfg2[0], cfg2[1], cfg2[2], cfg2[3], cfg2[4], middle_point[0], middle_point[1], middle_point[2], middle_point[3], middle_point[4]);
                            if (tid<dim){
                                path_out[i*dim + tid] = middle_point[tid];
                            }

                        }
                    }
                }
            }


            for (int i=0; i<path_total_size-1; i++){
                volatile float* cfg1;
                cfg1 = path_out + i*dim;

                grid.sync();
                const int bid_path_ind = bid + i + 1;
                if (bid_path_ind>=path_total_size) continue;

                volatile float* cfg2;
                cfg2 = path_out + bid_path_ind*dim;
                __syncthreads();

                if (bid>0){
                    bool edge_good = true;
                    float interp_cfg[dim];
                    const int cc_iter = ceil((device_utils::l2_dist((float *)cfg1, (float *)cfg2, dim) - granularity_tolerance) / (float) d_settings.range);
                    if (tid<dim){
                        delta[tid] = (cfg2[tid] - cfg1[tid]) / cc_iter;
                    }
                    for (int d=0; d<dim; d++){
                        interp_cfg[d] = cfg1[d];
                    }
                    __syncthreads();

                    for (int d=0; d<dim; d++){
                        interp_cfg[d] += (int(tid/4 + 1) * (delta[d] / (float)d_settings.granularity));
                    }
                    
                    for (int cc_round=0; cc_round<cc_iter; cc_round++){
                        
                        if (tid==0){
                            any_approx_env_collision[0] = 0;
                            any_approx_self_collision[0] = 0;
                            local_cc_result[0] = 0;
                        }
                        __syncthreads();
                        
                        ppln::device_utils::fkcc_single_buffer<Robot>(interp_cfg, env, tid, sphere_pos, link_CC, T, local_cc_result, any_approx_env_collision, any_approx_self_collision,
                                                                      env_sphere_to_check, self_sphere_to_check,
                                                                      env_thread_dist, self_thread_dist, 
                                                                      d_settings.granularity);

                        edge_good = (local_cc_result[0] == 0);
                        __syncthreads();
                        if (!edge_good) break;
                        
                        for (int d=0; d<dim; d++){
                            interp_cfg[d]+=delta[d];
                        }
                        __syncthreads();
                    }

                    //if (tid==0 && bid==0) printf("checking shortcut edge between %d and %d, dist %f, edge good %d\n", i, bid_path_ind, device_utils::l2_dist((float *)cfg1, (float *)cfg2, dim), edge_good);

                    if (!edge_good){
                        continue;
                    }
                }
                
                if (tid==0){
                    if (simplify_path_cost[bid_path_ind]>device_utils::l2_dist((float *)cfg1, (float *)cfg2, dim) + simplify_path_cost[i]){
                        simplify_path_cost[bid_path_ind] = device_utils::l2_dist((float *)cfg1, (float *)cfg2, dim) + simplify_path_cost[i];
                        simplify_path_parent[bid_path_ind] = i;
                        //printf("Updated simplified path cost %f at index %d with parent %d\n", simplify_path_cost[bid], bid, i);
                    }
                }
                // debug
                /*
                for (int i=0; i<path_total_size; i++){
                    printf("simplified path cost %f: ", simplify_path_cost[i]);
                    printf("\n");
                }
                */
            }
            grid.sync();

            // reconstruct path
            if (simplify_path_cost[path_total_size-1] < best_cost){
                int children = path_total_size-1;
                int parent = simplify_path_parent[children];
                while (parent < path_total_size && parent != -1){
                    const int bid_path_ind = bid + parent + 1;
                    if (bid_path_ind<children){
                        if (tid<dim) path_out[bid_path_ind*dim + tid] = path_out[parent*dim + tid];
                    }
                    children = parent;
                    parent = simplify_path_parent[children];
                }
            }

            //grid.sync();

            if (tid==0 && bid==0){
                cost = simplify_path_cost[path_total_size-1];
                //printf("simplify cost %f best %f\n", cost, best_cost);
                best_cost = min(best_cost, cost);
            }

            grid.sync();
            
        }
    }


    template <typename Robot>
    PlannerResult<Robot> solve(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &h_environment,
        pRRTC_settings &settings, const int num_robots, const int block_per_robot
    ) 
    {
        auto start_time = std::chrono::steady_clock::now();
        
        bool init_sol=false;

        static constexpr auto dim = Robot::dimension;
        std::size_t start_index = 0;
        PlannerResult<Robot> res;

        curandStateXORWOW_t* d_states;
        //int N = numBlocks * blockDim;
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

        float* d_start;
        cudaMalloc(&d_start, start.size() * sizeof(float));
        cudaMemcpy(d_start, start.data(), start.size() * sizeof(float), cudaMemcpyHostToDevice);

        float* d_goals;
        cudaMalloc(&d_goals, dim * num_robots * goals.size() * sizeof(float));
        for (int i=0; i<num_goals; i++)  cudaMemcpy(d_goals + i * dim * num_robots, goals[i].data(), num_robots * dim * sizeof(float), cudaMemcpyHostToDevice);

        float *nodes[2];
        int *parents[2];
        float *radii[2];
        float *node_cost[2];
        float **d_nodes;
        int **d_parents;
        float **d_radii;
        float **d_node_cost;
        cudaMalloc(&d_nodes, 2 * sizeof(float*));
        cudaMalloc(&d_parents, 2 * sizeof(int*));
        cudaMalloc(&d_radii, 2 * sizeof(float*));
        cudaMalloc(&d_node_cost, 2 * sizeof(float*));
        const std::size_t config_size = dim * sizeof(float);

        for (int i = 0; i < 2; i++) {
            cudaMalloc(&nodes[i], settings.max_samples * config_size);
            cudaMalloc(&parents[i], settings.max_samples * sizeof(int));
            cudaMalloc(&radii[i], settings.max_samples * sizeof(float));
            cudaMalloc(&node_cost[i], settings.max_samples * sizeof(int));
        }
        cudaMemcpy(d_nodes, nodes, 2 * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_parents, parents, 2 * sizeof(int*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_radii, radii, 2 * sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_node_cost, node_cost, 2 * sizeof(float*), cudaMemcpyHostToDevice);

        float *sphere_pos[300];
        float **d_sphere_pos;
        cudaMalloc(&d_sphere_pos, 300 * sizeof(float*));
        for (int i=0; i<300; i++){
            cudaMalloc(&sphere_pos[i], 100000 * sizeof(float));
        }
        cudaMemcpy(d_sphere_pos, sphere_pos, 300 * sizeof(float*), cudaMemcpyHostToDevice);

        float *tmp_roadmap_init = new float[700000];
        std::fill(tmp_roadmap_init, tmp_roadmap_init + 700000, UNWRITTEN_VAL);
        float *roadmaps[8];
        float **d_roadmaps;
        cudaMalloc(&d_roadmaps, 8 * sizeof(float*));
        for (int i=0; i<8; i++){
            cudaMalloc(&roadmaps[i], 700000 * sizeof(float));
            cudaMemcpy(roadmaps[i], tmp_roadmap_init, 700000 * sizeof(float), cudaMemcpyHostToDevice);
            
        }
        cudaMemcpy(d_roadmaps, roadmaps, 8 * sizeof(float*), cudaMemcpyHostToDevice);
        cudaCheckError(cudaGetLastError());

        int *roadmaps_id[8];
        int **d_roadmaps_id;
        cudaMalloc(&d_roadmaps_id, 8 * sizeof(int*));
        for (int i=0; i<8; i++){
            cudaMalloc(&roadmaps_id[i], 100000 * sizeof(int));
            
        }
        cudaMemcpy(d_roadmaps_id, roadmaps_id, 8 * sizeof(int*), cudaMemcpyHostToDevice);

        
        const int D0 = 8, D1 = 100000, D2 = 3200;

        uint32_t ***d_roadmap_edges;
        cudaMalloc(&d_roadmap_edges, D0 * sizeof(uint32_t**));

        uint32_t **d_mid[D0];
        uint32_t  *d_blocks[D0];   // save block bases for memset + cleanup

        for (int i = 0; i < D0; i++) {
            uint32_t *block;
            cudaMalloc(&block, (size_t)D1 * D2 * sizeof(uint32_t));
            cudaMemset(block, 0, (size_t)D1 * D2 * sizeof(uint32_t));  // all -> 0u
            d_blocks[i] = block;

            uint32_t **leaf = (uint32_t**)malloc(D1 * sizeof(uint32_t*));
            for (int j = 0; j < D1; j++)
                leaf[j] = block + (size_t)j * D2;

            cudaMalloc(&d_mid[i], D1 * sizeof(uint32_t*));
            cudaMemcpy(d_mid[i], leaf, D1 * sizeof(uint32_t*), cudaMemcpyHostToDevice);

            free(leaf);
        }

        cudaMemcpy(d_roadmap_edges, d_mid, D0 * sizeof(uint32_t**), cudaMemcpyHostToDevice);



        // allocate for obstacles
        ppln::collision::Environment<float> *env;
        setup_environment_on_device(env, h_environment);
        cudaCheckError(cudaGetLastError());

        // initialize radii
        std::vector<float> radii_init(num_goals, FLT_MAX);
        cudaMemcpy((void *)radii[0], radii_init.data(), sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *)radii[1], radii_init.data(), sizeof(float) * num_goals, cudaMemcpyHostToDevice);

        // set nodes to unitialized
        std::vector<float> nodes_init(settings.max_samples * dim, UNWRITTEN_VAL);

        cudaMemcpy((void *)parents[0], &start_index, sizeof(int), cudaMemcpyHostToDevice);
        std::vector<int> parents_b_init(num_goals);
        iota(parents_b_init.begin(), parents_b_init.end(), 0); // consecutive integers from 0 ... num_goals - 1
        cudaMemcpy((void *)parents[1], parents_b_init.data(), sizeof(int) * num_goals, cudaMemcpyHostToDevice);


        const int D00 = 2, D11 = 8, D22 = settings.max_samples;

        volatile int ***d_node_roadmap_id;
        cudaMalloc(&d_node_roadmap_id, D00 * sizeof(int**));

        volatile int **d_mid2[D00];
        volatile int  *d_blocks2[D00];   // save block bases for memset + cleanup

        for (int i = 0; i < D00; i++) {
            volatile int *block;
            cudaMalloc((void**)&block, (size_t)D11 * D22 * sizeof(int));
            cudaMemset((void*)block, 0, (size_t)D11 * D22 * sizeof(int));  // all -> 0
            d_blocks2[i] = block;

            volatile int **leaf = (volatile int**)malloc(D11 * sizeof(int*));
            for (int j = 0; j < D11; j++)
                leaf[j] = block + (size_t)j * D22;

            cudaMalloc((void**)&d_mid2[i], D11 * sizeof(int*));
            cudaMemcpy((void*)d_mid2[i], (void*)leaf, D11 * sizeof(int*), cudaMemcpyHostToDevice);

            free((void*)leaf);
        }

        cudaMemcpy((void*)d_node_roadmap_id, (void*)d_mid2, D00 * sizeof(int**), cudaMemcpyHostToDevice);

        
        // Setup pinned memory for signaling
        int *h_solved;
        int current_samples[2];
        int h_solved_iters = -1;
        cudaMallocHost(&h_solved, sizeof(int));
        *h_solved = -1;
        int h_free_index[2] = {1, num_goals};
        int h_completed_nodes[2] = {1, num_goals}; // start and goals are already written

        /*
        float *tmp_roadmap_init = new float[8 * 70000];
        std::fill(tmp_roadmap_init, tmp_roadmap_init + 8 * 70000, UNWRITTEN_VAL);

        
        float *d_roadmaps_ptr;
        cudaGetSymbolAddress((void**)&d_roadmaps_ptr, roadmaps);
        cudaMemcpy(d_roadmaps_ptr, tmp_roadmap_init, 8 * 70000 * sizeof(float), cudaMemcpyHostToDevice);
        */
        /*
        int* d_ptr_roadmap_component;
        cudaGetSymbolAddress((void**)&d_ptr_roadmap_component, roadmap_component_id);
        cudaMemset(d_ptr_roadmap_component, 0xFF, 8 * 10000 * sizeof(int));  // 0xFF bytes → -1 for ints
        */
        /*
        uint32_t* d_edges_ptr;
        cudaGetSymbolAddress((void**)&d_edges_ptr, roadmap_edges);
        cudaMemset(d_edges_ptr, 0, sizeof(roadmap_edges));
        */
        
        int cgNumBlocks_simplify = 16;
        int cgThreadsPerBlock_simplify = 4 * settings.granularity;
        void *cgKernelArgs_simplify[] = {
            (void *)&cgNumBlocks_simplify,
            (void *)&cgThreadsPerBlock_simplify,
            (void *)&env, // device pointer
            (void *)&d_sphere_pos
        };

        int cgNumBlocks = num_robots * block_per_robot;
        int cgThreadsPerBlock = 4 * settings.granularity;
        void *cgKernelArgs[] = {
            (void *)&env, // device pointer
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
            (void *)&settings.granularity
        };
        

        auto kernel_start_time = std::chrono::steady_clock::now();


        for (int rrtc_iter=0; rrtc_iter<settings.rrtc_iter; rrtc_iter++){

            cudaMemcpy((void *)nodes[0], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);
            cudaMemcpy((void *)nodes[1], nodes_init.data(), config_size * settings.max_samples, cudaMemcpyHostToDevice);

            // free index for next available position in tree_a and tree_b
            
            cudaMemcpyToSymbol(atomic_free_index, &h_free_index, sizeof(int) * 2);
            cudaMemcpyToSymbol(nodes_size, &h_free_index, sizeof(int) * 2);
            
            // initialize completed_nodes counter
            
            cudaMemcpyToSymbol(completed_nodes, &h_completed_nodes, sizeof(int) * 2);
            auto copy_start_time = std::chrono::steady_clock::now();
            // add start to tree_a and goals to tree_b
            cudaMemcpy((void *)nodes[0], start.data(), config_size, cudaMemcpyHostToDevice);
            cudaMemcpy((void *)nodes[1], goals.data(), config_size * num_goals, cudaMemcpyHostToDevice);
            res.copy_ns = get_elapsed_nanoseconds(copy_start_time);

            int h_optimize = rrtc_iter!=0 ? 1 : 0;
            //printf("RRTC iteration %d optimize %d\n", rrtc_iter, h_optimize);
            cudaMemcpyToSymbol(optimize, &h_optimize, sizeof(int));
            cudaDeviceSynchronize();
            

            cudaError_t err;
            do{
                err = cudaLaunchCooperativeKernel(
                    (void*)build_roadmap<typename Robot::SingleRobot>,
                    dim3(cgNumBlocks), // gridDim
                    dim3(cgThreadsPerBlock), // blockDim
                    cgKernelArgs,
                    0, // sharedMem
                    0 // stream
                );
                if (err != cudaSuccess) printf("build_roadmap launch failed: %s\n", cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) printf("build_roadmap sync failed: %s\n", cudaGetErrorString(err));
                
                
                rrtc<Robot><<<settings.num_new_configs, 4*settings.granularity>>> (
                    d_nodes,
                    d_parents,
                    d_node_cost,
                    d_node_roadmap_id,
                    d_sphere_pos,
                    d_roadmaps,
                    d_roadmap_edges,
                    d_radii,
                    d_states,
                    rng_states,
                    env
                );
                err = cudaGetLastError();
                if (err != cudaSuccess) printf("rrtc launch failed: %s\n", cudaGetErrorString(err));
                err = cudaDeviceSynchronize();
                if (err != cudaSuccess) printf("rrtc sync failed: %s\n", cudaGetErrorString(err));
                printf("rrtc complete\n");
                cudaMemcpyFromSymbol(h_solved, solved, sizeof(int), 0, cudaMemcpyDeviceToHost);
            }while (*h_solved<=0 && h_optimize==0);
            
            

            
            if (settings.path_simplify){
                // Launch cooperative kernel
                err = cudaLaunchCooperativeKernel(
                    (void*)multiBlock_simplify<Robot>,
                    dim3(cgNumBlocks_simplify), // gridDim
                    dim3(cgThreadsPerBlock_simplify), // blockDim
                    cgKernelArgs_simplify,
                    0, // sharedMem
                    0 // stream
                );
                if (err != cudaSuccess) printf("multiBlock_simplify launch failed: %s\n", cudaGetErrorString(err));
            }
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) printf("final sync failed: %s\n", cudaGetErrorString(err));
            printf("path simplify complete\n");
            

            //res.kernel_ns += get_elapsed_nanoseconds(kernel_start_time);
            
            // get data from device
            copy_start_time = std::chrono::steady_clock::now();
            
            float h_cost;
            cudaMemcpyFromSymbol(&h_cost, cost, sizeof(float), 0, cudaMemcpyDeviceToHost);
            
            cudaCheckError(cudaGetLastError());
            
            // add data to result struct
            //if (*h_solved!=1) *h_solved=0;
            res.start_tree_size = current_samples[0];
            res.goal_tree_size = current_samples[1];
            if (*h_solved<=0) printf("aorrtc failed\n");
            if (*h_solved >0 && (init_sol==false || res.cost > h_cost)) {
                printf("cost %f new cost %f\n", res.cost, h_cost);
                std::cout << "time kernel_ns " << get_elapsed_nanoseconds(kernel_start_time) << std::endl;
                //printf("Found solution with cost: %f\n", h_cost);
                cudaMemcpyFromSymbol(current_samples, atomic_free_index, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
                cudaMemcpyFromSymbol(&h_solved_iters, solved_iters, sizeof(int), 0, cudaMemcpyDeviceToHost);
                init_sol=true;
                int h_path_size[2];
                float h_paths[2][MAX_PATH_SIZE];
                float h_simplified_path[MAX_PATH_SIZE];
                int h_reached_goal_idx;
                cudaMemcpyFromSymbol(h_path_size, path_size, sizeof(int) * 2, 0, cudaMemcpyDeviceToHost);
                cudaMemcpyFromSymbol(h_paths, path, sizeof(float) * 2 * MAX_PATH_SIZE, 0, cudaMemcpyDeviceToHost);
                cudaMemcpyFromSymbol(h_simplified_path, path_out, sizeof(float) * MAX_PATH_SIZE, 0, cudaMemcpyDeviceToHost);
                cudaMemcpyFromSymbol(&h_reached_goal_idx, reached_goal_idx, sizeof(int), 0, cudaMemcpyDeviceToHost);
                cudaCheckError(cudaGetLastError());

                res.path.clear();
                //res.path.emplace_back(goals[h_reached_goal_idx]);
                typename Robot::Configuration config;
                
                if (!settings.path_simplify){
                    for (int i = h_path_size[1] - 1; i >= 0; i--) {
                        std::copy_n(h_paths[1] + i * dim, dim, config.begin());
                        res.path.emplace_back(config);
                    }
                    for (int i = 0; i < h_path_size[0]; i++) {
                        std::copy_n(h_paths[0] + i * dim, dim, config.begin());
                        res.path.emplace_back(config);
                    }
                }

                if (settings.path_simplify){
                    for (int i = 0; i < h_path_size[0] + h_path_size[1]; i++){
                        std::copy_n(h_simplified_path + i * dim, dim, config.begin());
                        if (config[0]==UNWRITTEN_VAL){
                            continue;
                        }
                        res.path.emplace_back(config);
                    }
                }
                
                //res.path.emplace_back(start);
                res.cost = h_cost;
                res.path_length = res.path.size();
                res.solved = (*h_solved) > 0;
                res.iters = h_solved_iters;
                // Snapshot this improving solution so downstream consumers can use
                // the planner's whole optimization history as candidate seeds, not
                // just the final best. rrtc_iter=0 gives the initial solution;
                // each later improving rrtc_iter appends a shorter path.
                res.seed_paths.emplace_back(h_cost, res.path);
                /*
                res.kernel_ns=0;
                for (int y=0; y<rrtc_iter+1; y++){
                    res.kernel_ns+=elapsed_times[y];
                }
                */
            }
            res.copy_ns += get_elapsed_nanoseconds(copy_start_time);
            //res.kernel_ns+=elapsed_times[rrtc_iter];
        }
        res.kernel_ns = get_elapsed_nanoseconds(kernel_start_time);
        /*
        for (int i = 0; i < 10; ++i) {
            printf("elapsed_times[%d] = %lld ns\n", i, elapsed_times[i]);
        }
        */

        /*
        cleanup_environment_on_device(env, h_environment);
        reset_device_variables();
        cudaFree((void *)nodes[0]);
        cudaFree((void *)nodes[1]);
        cudaFree((void *)parents[0]);
        cudaFree((void *)parents[1]);
        cudaFree((void *)radii[0]);
        cudaFree((void *)radii[1]);
        cudaFree(rng_states);
        cudaFree(d_nodes);
        cudaFree(d_parents);
        cudaFree(d_radii);
        cudaFreeHost(h_solved);
        */
        cudaCheckError(cudaGetLastError());
        res.wall_ns = get_elapsed_nanoseconds(start_time);
        cudaDeviceReset();
        return res;
    }

    //template PlannerResult<typename ppln::robots::Sphere> solve<ppln::robots::Sphere>(std::array<float, 3>&, std::vector<std::array<float, 3>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    //template PlannerResult<typename ppln::robots::Panda> solve<ppln::robots::Panda>(std::array<float, 7>&, std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    template PlannerResult<typename ppln::robots::Panda_dual> solve<ppln::robots::Panda_dual>(std::array<float, 14>&, std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, pRRTC_settings&, const int, const int);
    template PlannerResult<typename ppln::robots::Panda_four> solve<ppln::robots::Panda_four>(std::array<float, 28>&, std::vector<std::array<float, 28>>&, ppln::collision::Environment<float>&, pRRTC_settings&, const int, const int);
    template PlannerResult<typename ppln::robots::Panda_five> solve<ppln::robots::Panda_five>(std::array<float, 35>&, std::vector<std::array<float, 35>>&, ppln::collision::Environment<float>&, pRRTC_settings&, const int, const int);
    //template PlannerResult<typename ppln::robots::Fetch> solve<ppln::robots::Fetch>(std::array<float, 8>&, std::vector<std::array<float, 8>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    //template PlannerResult<typename ppln::robots::Baxter> solve<ppln::robots::Baxter>(std::array<float, 14>&, std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, pRRTC_settings&);

    

}
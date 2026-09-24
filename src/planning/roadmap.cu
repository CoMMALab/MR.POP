#include "Planners.hh"
#include "Robots.hh"
#include "utils.cuh"
#include "pRRTC_settings.hh"
#include "phs.hh"
#include "src/collision/environment.hh"
#include "src/robots/panda.cuh"
#include "src/robots/panda_two.cuh"
#include "src/robots/panda_four.cuh"
#include "src/robots/fetch.cuh"
#include "src/robots/baxter.cuh"
#include "roadmap_interior.cuh"

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
#define MAX_ROADMAP_SIZE 3000




namespace mr_roadmap {
    using namespace ppln;
    __device__ volatile int solved = 0;
    __device__ volatile int optimize = 0;
    __device__ volatile float best_cost = FLT_MAX;
    __device__ volatile float sphere_pos_all[100][15000];
    __device__ volatile float node_cost[2][500000] = {0.0f};
    __device__ volatile int atomic_free_index[2]; // separate for tree_a and tree_b
    __device__ volatile int nodes_size[2];
    __device__ volatile int completed_nodes[2]; // track completed nodes for each tree
    constexpr int MAX_PATH_SIZE = 10000;
    __device__ float path[2][MAX_PATH_SIZE]; // solution path segments for tree_a, and tree_b
    __device__ volatile float path_out[MAX_PATH_SIZE]; // solution path segments for tree_a, and tree_b
    __device__ volatile int path_size[2] = {0, 0};
    __device__ volatile float simplify_path_cost[MAX_PATH_SIZE] = {FLT_MAX};
    __device__ volatile int simplify_path_parent[MAX_PATH_SIZE] = {-1};
    __device__ volatile float cost = 0.0;
    __device__ int reached_goal_idx = 0;
    __device__ int solved_iters = 0; // value of iters in the block that solves the problem
    __device__ volatile float roadmaps[10][50000];
    __device__ volatile int roadmap_component_id[10][50000];
    __device__ volatile int roadmap_size[10];
    __device__ volatile int roadmap_edges[10][7000][7000];
    __device__ volatile bool start_goal_connected[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    __constant__ pRRTC_settings d_settings;

    constexpr int MAX_GRANULARITY = 64;
    constexpr int MAX_THREADS_PER_BLOCK = 4 * MAX_GRANULARITY;

    constexpr int BLOCK_SIZE = 64;
    constexpr float UNWRITTEN_VAL = -9999.0f;
    constexpr int UNWRITTEN_VAL_INT = -999;

    template<typename Robot>
    struct HaltonState {
        float b[Robot::dimension];   // bases
        float n[Robot::dimension];   // numerators
        float d[Robot::dimension];   // denominators
    };

    void __device__ shuffle_array(float *array, int n, curandState &state) {
        for (int i = n - 1; i > 0; i--) {
            int j = curand(&state) % (i + 1);
            float temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    template<typename Robot>
    __device__ void halton_initialize(HaltonState<Robot>& state, size_t skip_iterations, curandState& rng_state, int idx) {
        
        float primes[16] = {
            3.f, 5.f, 7.f, 11.f, 13.f, 17.f, 19.f, 23.f,
            29.f, 31.f, 37.f, 41.f, 43.f, 47.f, 53.f, 59.f
        };
        if (idx != 0) shuffle_array(primes, 16, rng_state);
        
        // Initialize bases from primes
        for (size_t i = 0; i < Robot::dimension; i++) {
            state.b[i] = primes[i];
            state.n[i] = 0.0f;
            state.d[i] = 1.0f;
        }
        
        // Skip iterations if requested
        volatile float temp_result[Robot::dimension];
        for (size_t i = 0; i < skip_iterations; i++) {
            halton_next(state, (float *)temp_result);
        }
    }

    template<typename Robot>
    __device__ void halton_next(HaltonState<Robot>& state, float* result) {
        for (size_t i = 0; i < Robot::dimension; i++) {
            float xf = state.d[i] - state.n[i];
            bool x_eq_1 = (xf == 1.0f);
            
            if (x_eq_1) {
                // x == 1 case
                state.d[i] = floorf(state.d[i] * state.b[i]);
                state.n[i] = 1.0f;
            } else {
                // x != 1 case
                float y = floorf(state.d[i] / state.b[i]);
                
                // Continue dividing by b until we find the right digit position
                while (xf <= y) {
                    y = floorf(y / state.b[i]);
                }
                
                state.n[i] = floorf((state.b[i] + 1.0f) * y) - xf;
            }
            
            result[i] = state.n[i] / state.d[i];
        }
    }

    template<typename Robot>
    __device__ void xorshift64(uint64_t *state, float* result) {
        
        for (size_t i = 0; i < Robot::dimension; i++) {
            /*
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            auto res = *state;
            float res2 = res>>40;
            res2 = res2 * (1.0f / 16777216.0f);
            result[i] = min(max(res2, 0.001f), 0.999f);
            */

            uint64_t x = state[0];
            uint64_t const y = state[1];
            state[0] = y;
            x ^= x << 23; // shift & xor
            x ^= x >> 17; // shift & xor
            x ^= y; // xor
            state[1] = x + y;
            auto res = state[1];
            float res2 = res>>40;
            res2 = res2 * (1.0f / 16777216.0f);
            result[i] = min(max(res2, 0.001f), 0.999f);
            
        }
        return;
        
    }

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

    template <typename Robot>
    __global__ void init_halton(HaltonState<Robot>* states, curandState* cr_states) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= d_settings.num_new_configs) return;
        // int skip = (curand_uniform(&cr_states[idx]) * 50000.0f);
        int skip = 0;
        if (idx == 0) skip = 0;
        halton_initialize(states[idx], skip, cr_states[idx], idx);
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
    __global__ void build_roadmap(ppln::collision::Environment<float> *env, float* start, float* goal, const int num_goals,
                                  curandStateXORWOW_t* states, const int block_per_robot, const int num_robot, const int granularity){
        
        static constexpr auto dim = Robot::dimension;
        cg::grid_group grid = cg::this_grid();
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int robot_id = bid / block_per_robot;
        __shared__ float config[dim];
        __shared__ float nearest_node[dim];
        __shared__ volatile unsigned int local_cc_result[1];
        __shared__ volatile unsigned int any_approx_env_collision[1];
        __shared__ volatile unsigned int any_approx_self_collision[1];
        __shared__ volatile int node_id[1];
        __shared__ volatile int current_nn[1];
        //__align__(16) __shared__ volatile float sphere_pos[6000];
        __align__(16) __shared__ volatile int link_CC[1500]; 
        __align__(16) __shared__ float T[64 * 2 * 16]; 
        __shared__ float delta[dim];

        build_roadmap_interior<Robot>(env, start, goal, num_goals, states, block_per_robot, num_robot, granularity,
             config, nearest_node, local_cc_result, any_approx_env_collision,
             any_approx_self_collision, node_id, current_nn, link_CC, T, delta,
             (volatile float (*)[50000])roadmaps,
             (volatile int (*)[50000])roadmap_component_id,
             (volatile int *)roadmap_size,
             (volatile int (*)[7000][7000])roadmap_edges,
             (volatile bool *)start_goal_connected,
             (volatile float (*)[15000])sphere_pos_all,
             &d_settings);

        return;

    }
        

    
    


    template <typename Robot>
    PlannerResult<Robot> roadmap_launch(
        typename Robot::Configuration &start,
        std::vector<typename Robot::Configuration> &goals,
        ppln::collision::Environment<float> &h_environment,
        pRRTC_settings &settings, const int num_robots, const int block_per_robot
    ) 
    {
        bool init_sol=false;
        
        static constexpr auto dim = Robot::dimension;
        std::size_t start_index = 0;
        PlannerResult<Robot> res;

        curandStateXORWOW_t* d_states;
        //int N = numBlocks * blockDim;
        cudaMalloc(&d_states, sizeof(curandStateXORWOW_t) * settings.num_new_configs*dim);
        init_xorwow<<<settings.num_new_configs, dim>>>(d_states, 1234ULL);

        // create a curandState for each thread
        curandState *rng_states;
        int num_rng_states = settings.num_new_configs * dim;
        cudaMalloc(&rng_states, num_rng_states * sizeof(curandState));
        int numBlocks = (num_rng_states + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_rng<<<numBlocks, BLOCK_SIZE>>>(rng_states, 1, num_rng_states);

        // copy data to GPU
        cudaMemcpyToSymbol(d_settings, &settings, sizeof(settings));
        int num_goals = goals.size();

        float* d_start;
        cudaMalloc(&d_start, start.size() * sizeof(float));
        cudaMemcpy(d_start, start.data(), start.size() * sizeof(float), cudaMemcpyHostToDevice);

        float* d_goals;
        cudaMalloc(&d_goals, dim * num_robots * goals.size() * sizeof(float));
        for (int i=0; i<num_goals; i++)  cudaMemcpy(d_goals + i * dim * num_robots, goals[i].data(), num_robots * dim * sizeof(float), cudaMemcpyHostToDevice);

        // allocate for obstacles
        ppln::collision::Environment<float> *env;
        setup_environment_on_device(env, h_environment);
        cudaCheckError(cudaGetLastError());

        int* d_ptr_roadmap_component;
        cudaGetSymbolAddress((void**)&d_ptr_roadmap_component, roadmap_component_id);
        cudaMemset(d_ptr_roadmap_component, 0xFF, 10 * 5000 * sizeof(int));  // 0xFF bytes → -1 for ints

        int* d_ptr_roadmap_edges;
        cudaGetSymbolAddress((void**)&d_ptr_roadmap_edges, roadmap_edges);
        cudaMemset(d_ptr_roadmap_edges, 0, 10 * MAX_ROADMAP_SIZE * MAX_ROADMAP_SIZE * sizeof(int));  

        
        int cgNumBlocks = num_robots * block_per_robot;
        int cgThreadsPerBlock = 4 * settings.granularity;
        void *cgKernelArgs[] = {
            (void *)&env, // device pointer
            (void *)&d_start,
            (void *)&d_goals,
            (void *)&num_goals,
            (void *)&d_states,
            (void *)&block_per_robot,
            (void *)&num_robots,
            (void *)&settings.granularity
        };

        auto start_time = std::chrono::steady_clock::now();
        
        cudaError_t err = cudaLaunchCooperativeKernel(
            (void*)build_roadmap<typename Robot::SingleRobot>,
            dim3(cgNumBlocks), // gridDim
            dim3(cgThreadsPerBlock), // blockDim
            cgKernelArgs,
            0, // sharedMem
            0 // stream
        );
        cudaDeviceSynchronize();
        
        
        res.kernel_ns = get_elapsed_nanoseconds(start_time);
        cudaCheckError(cudaGetLastError());

        res.wall_ns = get_elapsed_nanoseconds(start_time);
        cudaDeviceReset();
        return res;
    }

    //template PlannerResult<typename ppln::robots::Sphere> solve<ppln::robots::Sphere>(std::array<float, 3>&, std::vector<std::array<float, 3>>&, ppln::collision::Environment<float>&, pRRTC_settings&);
    //template PlannerResult<typename ppln::robots::Panda> roadmap_launch<ppln::robots::Panda>(std::array<float, 7>&, std::vector<std::array<float, 7>>&, ppln::collision::Environment<float>&, pRRTC_settings&, const int, const int);
    template PlannerResult<typename ppln::robots::Panda_dual> roadmap_launch<ppln::robots::Panda_dual>(std::array<float, 14>&, std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, pRRTC_settings&, const int, const int);
    template PlannerResult<typename ppln::robots::Panda_four> roadmap_launch<ppln::robots::Panda_four>(std::array<float, 28>&, std::vector<std::array<float, 28>>&, ppln::collision::Environment<float>&, pRRTC_settings&, const int, const int);
    //template PlannerResult<typename ppln::robots::Fetch> roadmap_launch<ppln::robots::Fetch>(std::array<float, 8>&, std::vector<std::array<float, 8>>&, ppln::collision::Environment<float>&, pRRTC_settings&, const int, const int);
    //template PlannerResult<typename ppln::robots::Baxter> roadmap_launch<ppln::robots::Baxter>(std::array<float, 14>&, std::vector<std::array<float, 14>>&, ppln::collision::Environment<float>&, pRRTC_settings&, const int, const int);

}
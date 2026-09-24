#pragma once

#include "Robots.hh"
#include "src/collision/environment.hh"
#include "src/collision/shapes.hh"
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cassert>

/* All device utils and collision functions */
#define M 4

#define FIXED -1
#define X_PRISM 0
#define Y_PRISM 1
#define Z_PRISM 2
#define X_ROT 3
#define Y_ROT 4
#define Z_ROT 5

namespace ppln::device_utils {
    using namespace collision;

    __device__ __forceinline__ bool warp_any_active_mask(bool pred) {
        // Active-lane mask: which threads are alive in this warp
        unsigned mask = __activemask();
        // Nonzero if any lane's pred is true
        return __any_sync(mask , pred);
    }

    __device__ __forceinline__ bool warp_any_full_mask(bool pred) {
        return __any_sync(0xffffffff , pred);
    }
    
    /* math utils */
    __device__ __forceinline__ constexpr float dot_2(const float &ax, const float &ay, const float &bx, const float &by) 
    {
        return (ax * bx) + (ay * by);
    }

     __device__ __forceinline__ constexpr float dot_3(
    const float ax,
    const float ay,
    const float az,
    const float bx,
    const float by,
    const float bz)
    {
        return ax * bx + ay * by + az * bz;
    }

    __device__ __forceinline__ constexpr float sql2_3(
        const float &ax,
        const float &ay,
        const float &az,
        const float &bx,
        const float &by,
        const float &bz)
    {
        const float xs = (ax - bx);
        const float ys = (ay - by);
        const float zs = (az - bz);

        return dot_3(xs, ys, zs, xs, ys, zs);
    }

    __device__ __forceinline__ constexpr float clamp(const float &v, const float &lower, const float &upper) 
    {
        return fmaxf(fminf(v, upper), lower);
    }

    // angle should be in shared memory
    __device__ __forceinline__ void compute_angle(float * config1, float * config2, float * config3,
                                                   volatile float * angle, const int dim, int bid, int tid){

        
        float v[20];
        float w[20];
        for (int t=0; t<dim; t++){
            v[t] = config2[t] - config1[t];
            w[t] = config3[t] - config1[t];
        }
        float dot_product=0;
        for (int t=0; t<dim; t++){
            dot_product += v[t] * w[t];
        }
        float v_norm = 0;
        float w_norm = 0;
        for (int t=0; t<dim; t++){
            v_norm += v[t] * v[t];
            w_norm += w[t] * w[t];
        }
        v_norm = sqrt(v_norm);
        w_norm = sqrt(w_norm);
        // guard against zero-length vectors (coincident configs)
        const float eps = 1e-8f;
        if (v_norm < eps || w_norm < eps) {
            *angle = 0.0f;          // or whatever sentinel your caller expects
            return;
        }

        float c = dot_product / (v_norm * w_norm);
        c = fminf(1.0f, fmaxf(-1.0f, c));   // clamp to acos domain
        *angle = acosf(c);
    
        //if (tid==5) printf("dim in angle func %d\n", dim);

    }

    /* end math utils */


    /* Sphere collision utils*/
    __device__ __forceinline__ constexpr float sphere_sphere_sql2(
        const float ax,
        const float ay,
        const float az,
        const float ar,
        const float bx,
        const float by,
        const float bz,
        const float br)
    {
        float sum = sql2_3(ax, ay, az, bx, by, bz);
        float rs = ar + br;
        return sum - rs * rs;
    }

    __device__ __forceinline__ constexpr float sphere_sphere_sql2(
        const Sphere<float> &a,
        const float &x,
        const float &y,
        const float &z,
        const float &r) 
    {
        return sphere_sphere_sql2(a.x, a.y, a.z, a.r, x, y, z, r);
    }

    __device__ __forceinline__ constexpr float sphere_sphere_self_collision(float ax, float ay, float az, float ar, float bx, float by, float bz, float br)
    {
        return (sphere_sphere_sql2(ax, ay, az, ar, bx, by, bz, br) < 0);
    }

    // returns l2 distance between two configs
    __host__ __device__ __forceinline__ float l2_dist(float *config_a, float *config_b, const int dim) {
        float ans = 0;
        float diff;
        #pragma unroll
        for (int i = 0; i < dim; i++) {
            diff = config_a[i] - config_b[i];
            ans += diff * diff;
        }
        return sqrt(ans);
    }

    __device__ __forceinline__ float sq_l2_dist(float *config_a, float *config_b, const int dim) {
        float ans = 0;
        float diff;
        #pragma unroll
        for (int i = 0; i < dim; i++) {
            diff = config_a[i] - config_b[i];
            ans += diff * diff;
        }
        return ans;
    }
    /* End Sphere collision utils*/

    /* Capsule collision utils */
    __device__ __forceinline__ constexpr float sphere_capsule(
        const Capsule<float> &c,
        const float &x,
        const float &y,
        const float &z,
        const float &r) noexcept
    {
        float dot = dot_3(x - c.x1, y - c.y1, z - c.z1, c.xv, c.yv, c.zv);
        float cdf = clamp((dot * c.rdv), 0.F, 1.F);

        float sum = sql2_3(x, y, z, c.x1 + c.xv * cdf, c.y1 + c.yv * cdf, c.z1 + c.zv * cdf);
        float rs = r + c.r;
        return sum - rs * rs;
    }

    
    __device__ __forceinline__ constexpr float sphere_capsule(const Capsule<float> &c, const Sphere<float> &s) noexcept 
    {
        return sphere_capsule(c, s.x, s.y, s.z, s.r);
    }

    
    __device__ __forceinline__ constexpr float sphere_z_aligned_capsule(
        const Capsule<float> &c,
        const float &x,
        const float &y,
        const float &z,
        const float &r) noexcept
    {
        float dot = (z - c.z1) * c.zv;
        float cdf = clamp((dot * c.rdv), 0.F, 1.F);

        float sum = sql2_3(x, y, z, c.x1, c.y1, c.z1 + c.zv * cdf);
        float rs = r + c.r;
        return sum - rs * rs;
    }

    __device__ __forceinline__ constexpr float sphere_z_aligned_capsule(const Capsule<float> &c, const Sphere<float> &s) noexcept
        
    {
        return sphere_z_aligned_capsule(c, s.x, s.y, s.z, s.r);
    }
    /* End Capsule collision utils*/

    /* Cuboid collision utils*/
    __device__ __forceinline__ constexpr float sphere_cuboid(
        const Cuboid<float> &c,
        const float &x,
        const float &y,
        const float &z,
        const float &rsq) noexcept
    {
        float xs = x - c.x;
        float ys = y - c.y;
        float zs = z - c.z;

        float a1 = fmaxf(0., abs(dot_3(c.axis_1_x, c.axis_1_y, c.axis_1_z, xs, ys, zs)) - c.axis_1_r);
        float a2 = fmaxf(0., abs(dot_3(c.axis_2_x, c.axis_2_y, c.axis_2_z, xs, ys, zs)) - c.axis_2_r);
        float a3 = fmaxf(0., abs(dot_3(c.axis_3_x, c.axis_3_y, c.axis_3_z, xs, ys, zs)) - c.axis_3_r);

        float sum = dot_3(a1, a2, a3, a1, a2, a3);
        return sum - rsq;
    }

    
    __device__ __forceinline__ constexpr float sphere_cuboid(const Cuboid<float> &c, const Sphere<float> &s) noexcept 
    {
        return sphere_cuboid(c, s.x, s.y, s.z, s.r * s.r);
    }

    
    __device__ __forceinline__ constexpr float sphere_z_aligned_cuboid(
        const Cuboid<float> &c,
        const float &x,
        const float &y,
        const float &z,
        const float &rsq) noexcept
    {
        float xs = x - c.x;
        float ys = y - c.y;
        float zs = z - c.z;

        float a1 = fmaxf(0., (abs(dot_2(c.axis_1_x, c.axis_1_y, xs, ys)) - c.axis_1_r));
        float a2 = fmaxf(0., (abs(dot_2(c.axis_2_x, c.axis_2_y, xs, ys)) - c.axis_2_r));
        float a3 = fmaxf(0, (abs(zs) - c.axis_3_r));

        float sum = dot_3(a1, a2, a3, a1, a2, a3);
        return sum - rsq;
    }

    
    __device__ __forceinline__ constexpr float sphere_z_aligned_cuboid(const Cuboid<float> &c, const Sphere<float> &s) noexcept
        
    {
        return sphere_z_aligned_cuboid(c, s.x, s.y, s.z, s.r * s.r);
    }
    /* End Cuboid collision util*/

    __device__ __forceinline__ bool sphere_environment_in_collision(ppln::collision::Environment<float> *env, float sx_, float sy_, float sz_, float sr_)
    {
        const float rsq = sr_ * sr_;
        bool in_collision = false;

        for (unsigned int i = 0; i < env->num_spheres && !in_collision; i++)
        {
            in_collision |= (sphere_sphere_sql2(env->spheres[i], sx_, sy_, sz_, sr_) < 0);
            
        }

        for (unsigned int i = 0; i < env->num_capsules && !in_collision; i++)
        {
            in_collision |= (sphere_capsule(env->capsules[i], sx_, sy_, sz_, sr_) < 0);
            
        }

        for (unsigned int i = 0; i < env->num_z_aligned_capsules && !in_collision; i++)
        {
            in_collision |= (sphere_z_aligned_capsule(env->z_aligned_capsules[i], sx_, sy_, sz_, sr_) < 0);
        }

        for (unsigned int i = 0; i < env->num_cuboids && !in_collision; i++)
        {
            in_collision |= (sphere_cuboid(env->cuboids[i], sx_, sy_, sz_, rsq) < 0);
            
        }

        for (unsigned int i = 0; i < env->num_z_aligned_cuboids && !in_collision; i++)
        {
            in_collision |= (sphere_z_aligned_cuboid(env->z_aligned_cuboids[i], sx_, sy_, sz_, rsq) < 0);
        }

        return in_collision;
    }

    __global__ void init_rng(curandState* states, unsigned long seed);
}


// Error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


/* Collision checking backend implementations for different robots */
namespace ppln::collision {
    using namespace device_utils;
    
    // fkcc -> checks if the config is "good"
    // cc returns false if the config does collide with an obstacle, returns true if the config does not collide

    template <typename Robot>
    __device__ __forceinline__ void fk(const float *config, volatile float* sphere_pos, float *T, const int tid);

    template <typename Robot>
    __device__ __forceinline__ void fk_mr(const float *config, volatile float* sphere_pos_approx, float *T, const int tid, int num_robot, int robot_id);

    template <typename Robot>
    __device__ __forceinline__ bool self_collision_check(volatile float* sphere_pos, volatile int* link_approx_CC, int self_batch_ind, int thread_interp_cnt, const int tid);

    template <typename Robot>
    __device__ __forceinline__ bool env_collision_check(volatile float* sphere_pos, volatile int* link_approx_CC, ppln::collision::Environment<float> *env, int env_batch_ind, int thread_interp_cnt, const int tid);

    template <typename Robot>
    __device__ __forceinline__ void fk_approx(const float *config, volatile float* sphere_pos_approx, float *T, const int tid);

    template <typename Robot>
    __device__ __forceinline__ void fk_approx_mr(const float *config, volatile float* sphere_pos_approx, float *T, const int tid, int num_robot, int robot_id);

    template <typename Robot>
    __device__ __forceinline__ bool self_collision_check_approx(volatile float* sphere_pos_approx, volatile int* link_approx_CC, volatile int * self_sphere_to_check, const int tid);

    template <typename Robot>
    __device__ __forceinline__ bool env_collision_check_approx(volatile float* sphere_pos_approx, volatile int* link_approx_CC, ppln::collision::Environment<float> *env, volatile int * env_sphere_to_check, const int tid);

    template <typename Robot>
    __device__ __forceinline__ bool fkcc(volatile float *config, ppln::collision::Environment<float> *env, int tid);

    /* adapted from https://github.com/NVlabs/curobo/blob/0a50de1ba72db304195d59d9d0b1ed269696047f/src/curobo/curobolib/cpp/kinematics_fused_kernel.cu */
    __device__ __forceinline__ void fixed_joint_fn(
        const float *fixed_transform,
        float *T_step_col
    )
    {
        T_step_col[0] = fixed_transform[0];
        T_step_col[1] = fixed_transform[M];
        T_step_col[2] = fixed_transform[M * 2];
        T_step_col[3] = fixed_transform[M * 3];
    }

    __device__ __forceinline__ void xrot_fn(
        const float *fixed_transforms,
        const float angle,
        const int col_idx,
        float *T_step_col
    )
    {
      // we found no change in convergence between fast approximate and IEEE sin,
      // cos functions using fast approximate method saves 5 registers per thread.
      float cos   = __cosf(angle);
      float sin   = __sinf(angle);
      float n_sin = -1 * sin;

      int bit1         = col_idx & 0x1;
      int bit2         = (col_idx & 0x2) >> 1;
      int _xor         = bit1 ^ bit2;  // 0 for threads 0 and 3, 1 for threads 1 and 2
      int col_idx_by_2 =
        col_idx / 2;                   // 0 for threads 0 and 1, 1 for threads 2 and 3

      float f1 = (1 - col_idx_by_2) * cos +
                 col_idx_by_2 * n_sin; // thread 1 get cos , thread 2 gets n_sin
      float f2 = (1 - col_idx_by_2) * sin +
                 col_idx_by_2 * cos;   // thread 1 get sin, thread 2 gets cos

      f1 = _xor * f1 + (1 - _xor) * 1; // threads 1 and 2 will get f1; the other
                                       // two threads will get 1
      f2 = _xor *
           f2;                         // threads 1 and 2 will get f2, the other two threads will
                                       // get 0.0
      float f3 = 1 - _xor;

      int addr_offset =
        _xor + (1 - _xor) *
        col_idx; // 1 for threads 1 and 2, col_idx for threads 0 and 3

      T_step_col[0] = fixed_transforms[0 + addr_offset] * f1 + f2 * fixed_transforms[2];
      T_step_col[1] = fixed_transforms[M + addr_offset] * f1 + f2 * fixed_transforms[M + 2];
      T_step_col[2] =
        fixed_transforms[M + M + addr_offset] * f1 + f2 * fixed_transforms[M + M + 2];
      T_step_col[3] = fixed_transforms[M + M + M + addr_offset] *
              f3; // threads 1 and 2 get 0.0, remaining two get fixed_transforms[3M];
    }

    // version with no control flow
    __device__ __forceinline__ void yrot_fn(
        const float *fixed_transforms,
        const float angle,
        const int col_idx,
        float *T_step_col
    )
    {
      float cos   = __cosf(angle);
      float sin   = __sinf(angle);
      float n_sin = -1 * sin;

      int col_idx_per_2 =
        col_idx % 2;                 // threads 0 and 2 will be 0 and threads 1 and 3 will be 1.
      int col_idx_by_2 =
        col_idx / 2;                 // threads 0 and 1 will be 0 and threads 2 and 3 will be 1.

      float f1 = (1 - col_idx_by_2) * cos +
                 col_idx_by_2 * sin; // thread 0 get cos , thread 2 gets sin
      float f2 = (1 - col_idx_by_2) * n_sin +
                 col_idx_by_2 * cos; // thread 0 get n_sin, thread 2 gets cos

      f1 = (1 - col_idx_per_2) * f1 +
           col_idx_per_2 * 1;        // threads 0 and 2 will get f1; the other two
                                     // threads will get 1
      f2 = (1 - col_idx_per_2) *
           f2;                       // threads 0 and 2 will get f2, the other two threads will get
                                     // 0.0
      float f3 =
        col_idx_per_2;               // threads 0 and 2 will be 0 and threads 1 and 3 will be 1.

      int addr_offset =
        col_idx_per_2 *
        col_idx; // threads 0 and 2 will get 0, the other two will get col_idx.

      T_step_col[0] = fixed_transforms[0 + addr_offset] * f1 + f2 * fixed_transforms[2];
      T_step_col[1] = fixed_transforms[M + addr_offset] * f1 + f2 * fixed_transforms[M + 2];
      T_step_col[2] =
        fixed_transforms[M + M + addr_offset] * f1 + f2 * fixed_transforms[M + M + 2];
      T_step_col[3] = fixed_transforms[M + M + M + addr_offset] *
              f3; // threads 0 and 2 threads get 0.0, remaining two get
                  // fixed_transforms[3M];
    }
    
    __device__ __forceinline__ void zrot_fn(
        const float *fixed_transforms,
        const float angle,
        const int col_idx,
        float *T_step_col
    ) {
        float cos = __cosf(angle);
        float sin = __sinf(angle);
        float n_sin = -1 * sin;

        int col_idx_by_2 =
            col_idx / 2; // first two threads will be 0 and the next two will be 1.
        int col_idx_per_2 =
            col_idx % 2; // first thread will be 0 and the second thread will be 1.
        float f1 = (1 - col_idx_per_2) * cos +
                   col_idx_per_2 * n_sin; // thread 0 get cos , thread 1 gets n_sin
        float f2 = (1 - col_idx_per_2) * sin +
                   col_idx_per_2 * cos; // thread 0 get sin, thread 1 gets cos

        f1 = (1 - col_idx_by_2) * f1 +
             col_idx_by_2 * 1; // first two threads get f1, other two threads get 1
        f2 = (1 - col_idx_by_2) *
             f2; // first two threads get f2, other two threads get 0.0

        int addr_offset =
            col_idx_by_2 *
            col_idx; // first 2 threads will get 0, the other two will get col_idx.

        T_step_col[0] = fixed_transforms[0 + addr_offset] * f1 + f2 * fixed_transforms[1];
        T_step_col[1] = fixed_transforms[M + addr_offset] * f1 + f2 * fixed_transforms[M + 1];
        T_step_col[2] = fixed_transforms[M + M + addr_offset] * f1 + f2 * fixed_transforms[M + M + 1];
        T_step_col[3] = fixed_transforms[M + M + M + addr_offset] * col_idx_by_2; // first two threads get 0.0, remaining two get fixed_transforms[3M];
    }

    // prism_fn withOUT control flow
    __device__ __forceinline__ void prism_fn(
        const float *fixed_transforms,
        const float angle,
        const int col_idx,
        float *T_step_col,
        const int xyz
    )
    {
        if (col_idx <= 2)
        {
            fixed_joint_fn(&fixed_transforms[col_idx], T_step_col);
        }
        else
        {
            T_step_col[0] = fixed_transforms[0 + xyz] * angle + fixed_transforms[3];     // FT_0[1];
            T_step_col[1] = fixed_transforms[M + xyz] * angle + fixed_transforms[M + 3]; // FT_1[1];
            T_step_col[2] = fixed_transforms[M + M + xyz] * angle +
                            fixed_transforms[M + M + 3];                                 // FT_2[1];
            T_step_col[3] = 1;
        }
    }
    
    __device__ __forceinline__ float dot4(float *a, float *b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    }

    __device__ __forceinline__ float dot4_col(float *a_col, float *b) {
        return a_col[0] * b[0] + a_col[M] * b[1] + a_col[M*2] * b[2] + a_col[M*3] * b[3];
    }

    template <typename Robot>
    __device__ __forceinline__ void fkcc_single_buffer(
        const float *config,
        ppln::collision::Environment<float> *env,
        const int tid,
        volatile float *sphere_pos,
        volatile int *link_CC,
        float *T,
        volatile unsigned int *cc_result,
        volatile unsigned int *any_approx_env_collision,
        volatile unsigned int *any_approx_self_collision,
        volatile int * env_sphere_to_check, volatile int * self_sphere_to_check,
        volatile int * env_thread_dist, volatile int * self_thread_dist, int granularity
    ) {
        // reset link_CC
        for (int i = tid; i < 8000; i += blockDim.x)
        {
            link_CC[i] = 0; // 00 = no detailed check needed, 01 = detailed env check needed, 10 = detailed self check needed, 11 = detailed env and self check needed
        }
    
    
        ppln::collision::fk_approx<Robot>(config, sphere_pos, T, tid);
    
        bool approx_env_collision =
            not ppln::collision::env_collision_check_approx<Robot>(sphere_pos, link_CC, env, env_sphere_to_check, tid);
        
        bool approx_self_collision =
            not ppln::collision::self_collision_check_approx<Robot>(sphere_pos, link_CC, self_sphere_to_check, tid);
        
        atomicOr((unsigned int *)any_approx_env_collision, approx_env_collision ? 1u : 0u);
        atomicOr((unsigned int *)any_approx_self_collision, approx_self_collision ? 1u : 0u);
        __syncthreads();

        // if any approx collision found, proceed to detailed FK and CC
        if (any_approx_env_collision[0] || any_approx_self_collision[0]) {

            // distribute threads for detailed CC
            if (tid==0){
                const int num_thread_total = 4 * granularity;
                int env_total_sphere=0;
                int self_total_sphere=0;
                for (int i=0; i<granularity; i++){
                    env_total_sphere += env_sphere_to_check[i];
                    self_total_sphere += self_sphere_to_check[i];
                }
                int thread_cnt_env=0;
                int thread_cnt_self=0;
                for (int i=0; i<granularity; i++){
                    env_thread_dist[i] = (int)((float)env_sphere_to_check[i] / (float)env_total_sphere * (float)num_thread_total);
                    thread_cnt_env+=env_thread_dist[i];
                    self_thread_dist[i] = (int)((float)self_sphere_to_check[i] / (float)self_total_sphere * (float)num_thread_total);
                    thread_cnt_self+=self_thread_dist[i];
                }
                // in case there are extras due to numericals
                env_thread_dist[granularity-1]+=(num_thread_total-thread_cnt_env);
                self_thread_dist[granularity-1]+=(num_thread_total-thread_cnt_self);
            }
            __syncthreads();
            int env_batch_ind = 0;
            int self_batch_ind = 0;
            int thread_cnt_temp = 0;
            for (int t=0; t<granularity; t++){
                thread_cnt_temp+=env_thread_dist[t];
                if (thread_cnt_temp-1>=tid){
                    env_batch_ind = t;
                    break;
                }
            }
            thread_cnt_temp = 0;
            for (int t=0; t<granularity; t++){
                thread_cnt_temp+=self_thread_dist[t];
                if (thread_cnt_temp-1>=tid){
                    self_batch_ind = t;
                    break;
                }
            }

            // if (tid == 0) printf("any_approx_env_collision: %d, any_approx_self_collision: %d\n", any_approx_env_collision, any_approx_self_collision);
            ppln::collision::fk<Robot>(config, sphere_pos, T, tid);
            __syncthreads();

            if (any_approx_env_collision[0]) {
                bool detailed_env_collision =
                    not ppln::collision::env_collision_check<Robot>(sphere_pos, link_CC, env, env_batch_ind, env_thread_dist[env_batch_ind], tid);
                atomicOr((unsigned int *)cc_result, detailed_env_collision ? 1u : 0u);
                // if (tid == 0) printf("detailed_env_collision: %d\n", detailed_env_collision);
            }
            //__syncthreads();
            if (!cc_result[0] && any_approx_self_collision[0]) {
                bool detailed_self_collision =
                    not ppln::collision::self_collision_check<Robot>(sphere_pos, link_CC, self_batch_ind, self_thread_dist[self_batch_ind], tid);
                atomicOr((unsigned int *)cc_result, detailed_self_collision ? 1u : 0u);
                // if (tid == 0) printf("detailed_self_collision: %d\n", detailed_self_collision);
            }
        }
        __syncthreads();
    }

    template <typename Robot>
    __device__ __forceinline__ void fkcc_single_buffer_mr(
        const float *config,
        ppln::collision::Environment<float> *env,
        const int tid,
        volatile float *sphere_pos,
        volatile int *link_CC,
        float *T,
        volatile unsigned int *cc_result,
        volatile unsigned int *any_approx_env_collision,
        volatile unsigned int *any_approx_self_collision,
        volatile int * env_sphere_to_check, volatile int * self_sphere_to_check,
        volatile int * env_thread_dist, volatile int * self_thread_dist, 
        int granularity, int num_robot, int robot_id
    ) {
        // reset link_CC
        for (int i = tid; i < 1500; i += blockDim.x)
        {
            link_CC[i] = 0; // 00 = no detailed check needed, 01 = detailed env check needed, 10 = detailed self check needed, 11 = detailed env and self check needed
        }
    
    
        ppln::collision::fk_approx_mr<Robot>(config, sphere_pos, T, tid, num_robot, robot_id);
        __syncthreads();
    
        bool approx_env_collision =
            not ppln::collision::env_collision_check_approx<Robot>(sphere_pos, link_CC, env, env_sphere_to_check, tid);
        
        bool approx_self_collision =
            not ppln::collision::self_collision_check_approx<Robot>(sphere_pos, link_CC, self_sphere_to_check, tid);
        
        atomicOr((unsigned int *)any_approx_env_collision, approx_env_collision ? 1u : 0u);
        atomicOr((unsigned int *)any_approx_self_collision, approx_self_collision ? 1u : 0u);
        __syncthreads();

        // if any approx collision found, proceed to detailed FK and CC
        if (any_approx_env_collision[0] || any_approx_self_collision[0]) {

            // distribute threads for detailed CC
            if (tid==0){
                const int num_thread_total = 4 * granularity;
                int env_total_sphere=0;
                int self_total_sphere=0;
                for (int i=0; i<granularity; i++){
                    env_total_sphere += env_sphere_to_check[i];
                    self_total_sphere += self_sphere_to_check[i];
                }
                int thread_cnt_env=0;
                int thread_cnt_self=0;
                for (int i=0; i<granularity; i++){
                    env_thread_dist[i] = (int)((float)env_sphere_to_check[i] / (float)env_total_sphere * (float)num_thread_total);
                    thread_cnt_env+=env_thread_dist[i];
                    self_thread_dist[i] = (int)((float)self_sphere_to_check[i] / (float)self_total_sphere * (float)num_thread_total);
                    thread_cnt_self+=self_thread_dist[i];
                }
                // in case there are extras due to numericals
                env_thread_dist[granularity-1]+=(num_thread_total-thread_cnt_env);
                self_thread_dist[granularity-1]+=(num_thread_total-thread_cnt_self);
            }
            __syncthreads();
            int env_batch_ind = 0;
            int self_batch_ind = 0;
            int thread_cnt_temp = 0;
            for (int t=0; t<granularity; t++){
                thread_cnt_temp+=env_thread_dist[t];
                if (thread_cnt_temp-1>=tid){
                    env_batch_ind = t;
                    break;
                }
            }
            thread_cnt_temp = 0;
            for (int t=0; t<granularity; t++){
                thread_cnt_temp+=self_thread_dist[t];
                if (thread_cnt_temp-1>=tid){
                    self_batch_ind = t;
                    break;
                }
            }

            // if (tid == 0) printf("any_approx_env_collision: %d, any_approx_self_collision: %d\n", any_approx_env_collision, any_approx_self_collision);
            ppln::collision::fk_mr<Robot>(config, sphere_pos, T, tid, num_robot, robot_id);
            __syncthreads();

            if (any_approx_env_collision[0]) {
                bool detailed_env_collision =
                    not ppln::collision::env_collision_check<Robot>(sphere_pos, link_CC, env, env_batch_ind, env_thread_dist[env_batch_ind], tid);
                atomicOr((unsigned int *)cc_result, detailed_env_collision ? 1u : 0u);
                // if (tid == 0) printf("detailed_env_collision: %d\n", detailed_env_collision);
            }
            //__syncthreads();
            if (!cc_result[0] && any_approx_self_collision[0]) {
                bool detailed_self_collision =
                    not ppln::collision::self_collision_check<Robot>(sphere_pos, link_CC, self_batch_ind, self_thread_dist[self_batch_ind], tid);
                atomicOr((unsigned int *)cc_result, detailed_self_collision ? 1u : 0u);
                // if (tid == 0) printf("detailed_self_collision: %d\n", detailed_self_collision);
            }
        }
        __syncthreads();
    }

    template <typename Robot>
    __device__ __forceinline__ void fkcc_single_buffer_drrt(
        const float *config,
        ppln::collision::Environment<float> *env,
        const int tid,
        volatile float *sphere_pos,
        volatile int *link_CC,
        float *T,
        volatile unsigned int *cc_result,
        volatile unsigned int *any_approx_env_collision,
        volatile unsigned int *any_approx_self_collision,
        volatile int * self_sphere_to_check,
        volatile int * self_thread_dist, int granularity
    ) {
        // reset link_CC
        for (int i = tid; i < 8000; i += blockDim.x)
        {
            link_CC[i] = 0; // 00 = no detailed check needed, 01 = detailed env check needed, 10 = detailed self check needed, 11 = detailed env and self check needed
        }
        

        ppln::collision::fk_approx<Robot>(config, sphere_pos, T, tid);
        __syncthreads();

        bool approx_self_collision =
            not ppln::collision::self_collision_check_approx<Robot>(sphere_pos, link_CC, self_sphere_to_check, tid);
        
        atomicOr((unsigned int *)any_approx_self_collision, approx_self_collision ? 1u : 0u);
        __syncthreads();

        // if any approx collision found, proceed to detailed FK and CC
        if (any_approx_self_collision[0]) {

            // distribute threads for detailed CC
            if (tid==0){
                int num_thread_total = 4 * granularity;
                int self_total_sphere=0;
                for (int i=0; i<granularity; i++){
                    self_total_sphere += self_sphere_to_check[i];
                }
                int thread_cnt_self=0;
                for (int i=0; i<granularity; i++){
                    self_thread_dist[i] = (int)((float)self_sphere_to_check[i] / (float)self_total_sphere * (float)num_thread_total);
                    thread_cnt_self+=self_thread_dist[i];
                    //printf("bid %d batch %d self_total_sphere %d thread dist %d\n", blockIdx.x, i, self_total_sphere, self_thread_dist[i]);
                }
                // in case there are extras due to numericals
                self_thread_dist[granularity-1]+=(num_thread_total-thread_cnt_self);
            }
            __syncthreads();
            int self_batch_ind = 0;
            int thread_cnt_temp = 0;
            for (int t=0; t<granularity; t++){
                thread_cnt_temp+=self_thread_dist[t];
                if (thread_cnt_temp-1>=tid){
                    self_batch_ind = t;
                    break;
                }
            }
            //printf("bid %d tid %d self_batch_ind %d\n", blockIdx.x, tid, self_batch_ind);

            // if (tid == 0) printf("any_approx_env_collision: %d, any_approx_self_collision: %d\n", any_approx_env_collision, any_approx_self_collision);
            ppln::collision::fk<Robot>(config, sphere_pos, T, tid);
            __syncthreads();

            bool detailed_self_collision =
                not ppln::collision::self_collision_check<Robot>(sphere_pos, link_CC, self_batch_ind, self_thread_dist[self_batch_ind], tid);
            atomicOr((unsigned int *)cc_result, detailed_self_collision ? 1u : 0u);
            // if (tid == 0) printf("detailed_self_collision: %d\n", detailed_self_collision);
            
        }
        __syncthreads();

    }


    template <typename Robot>
    __device__ __forceinline__ bool nn_angle_mr(
        float * config, float * v_near, int v_near_id, volatile float * new_config, 
        volatile int * new_config_roadmap_id,
        volatile int **roadmap_id,
        float **roadmaps,
        volatile int * roadmap_size, 
        uint32_t (***roadmap_edges), 
        float tree_edge_cost_bound, float range,
        float * sdata, int * sindex, const int num_robot, int optimize){

        static constexpr auto dim = Robot::dimension;
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int single_robot_dim = (int)(dim / num_robot);


        //if (tree_edge_cost_bound < 1e5) printf("near node cost %f\n", tree_edge_cost_bound);
        for (int robot_id=0; robot_id<num_robot; robot_id++){
            int best_nn_ind = -1;
            float best_nn_angle = 1e3;
            volatile float * working_on_roadmap = roadmaps[robot_id];

            for (int r=tid; r<roadmap_size[robot_id]; r+=blockDim.x){
                    if (!(roadmap_edges[robot_id][roadmap_id[robot_id][v_near_id]][r >> 5] >> (r & 31) & 1u) || roadmap_id[robot_id][v_near_id]==r) continue;
                    
                    /*
                    printf("here 1 bid %d tid %d num_robot %d robot id %d r_ind %d roadmap size %d config %f %f %f %f %f %f %f, %f %f %f %f %f %f %f\n", 
                            bid, tid, num_robot, robot_id, r, roadmap_size[robot_id], v_near[robot_id * single_robot_dim + 0], v_near[robot_id * single_robot_dim +1], v_near[robot_id * single_robot_dim +2], v_near[robot_id * single_robot_dim +3], v_near[robot_id * single_robot_dim +4], v_near[robot_id * single_robot_dim +5], v_near[robot_id * single_robot_dim +6],
                            roadmaps[robot_id][r * single_robot_dim + 0], roadmaps[robot_id][r * single_robot_dim + 1], roadmaps[robot_id][r * single_robot_dim + 2], roadmaps[robot_id][r * single_robot_dim + 3], roadmaps[robot_id][r * single_robot_dim + 4], roadmaps[robot_id][r * single_robot_dim + 5], roadmaps[robot_id][r * single_robot_dim + 6]);
                    */
                    
                    float edge_dist = l2_dist(v_near + robot_id * single_robot_dim, (float*)working_on_roadmap + r * single_robot_dim, single_robot_dim);
                    //if (tid==0) printf("edge dist %f tree_edge_cost_bound %f\n", edge_dist, tree_edge_cost_bound);
                    
                    if ((optimize==1 && edge_dist > tree_edge_cost_bound) || edge_dist > range) continue;
                    
                    
                    //printf("robot id %d roadmap size %d num robot %d\n", robot_id, roadmap_size[robot_id], num_robot);
                    /*
                    printf("here 2 bid %d tid %d num_robot %d robot id %d r_ind %d roadmap size %d config %f %f %f %f %f %f %f, %f %f %f %f %f %f %f\n", 
                            bid, tid, num_robot, robot_id, r, roadmap_size[robot_id], v_near[robot_id * single_robot_dim + 0], v_near[robot_id * single_robot_dim +1], v_near[robot_id * single_robot_dim +2], v_near[robot_id * single_robot_dim +3], v_near[robot_id * single_robot_dim +4], v_near[robot_id * single_robot_dim +5], v_near[robot_id * single_robot_dim +6],
                            roadmaps[robot_id][r * single_robot_dim + 0], roadmaps[robot_id][r * single_robot_dim + 1], roadmaps[robot_id][r * single_robot_dim + 2], roadmaps[robot_id][r * single_robot_dim + 3], roadmaps[robot_id][r * single_robot_dim + 4], roadmaps[robot_id][r * single_robot_dim + 5], roadmaps[robot_id][r * single_robot_dim + 6]);
                    */

                    volatile float angle_value;
                    volatile float * angle = &angle_value;
                    float * testing_node = (float*)(working_on_roadmap + r * single_robot_dim);
                    compute_angle(v_near + robot_id * single_robot_dim, config + robot_id * single_robot_dim, testing_node, angle, single_robot_dim, bid, tid);
                    //if (tid==0) printf("angle %f\n", *angle);
                    if (*angle < best_nn_angle){
                        best_nn_angle = *angle;
                        best_nn_ind = r;
                    }
                //__threadfence();
            }
            sdata[tid] = best_nn_angle;
            sindex[tid] = best_nn_ind;
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
            if (tid==0) new_config_roadmap_id[robot_id] = sindex[0];
            if (tid<single_robot_dim) new_config[robot_id * single_robot_dim + tid] = working_on_roadmap[single_robot_dim * sindex[0] + tid];
            if (sindex[0] ==-1) return false;
            tree_edge_cost_bound -= l2_dist((float *)working_on_roadmap + single_robot_dim * sindex[0], v_near + robot_id * single_robot_dim, single_robot_dim);
        }
        __syncthreads();

        //if (tid==0) printf("nn found\n");
        return true;


    }

}


#pragma once

#include "src/planning/utils.cuh"

namespace ppln::collision {

    // These specializations are defined by the robot headers in pop_planner.cu.
    // Other CUDA files use these declarations to share the same collision code.

    template <>
    __device__ void fk_approx<ppln::robots::Panda_four>(const float *q,
                                                        volatile float *sphere_pos_approx, float *T,
                                                        const int tid);

    template <>
    __device__ bool self_collision_check_approx<ppln::robots::Panda_four>(
        volatile float *sphere_pos_approx, volatile int *joint_in_collision,
        volatile int *self_sphere_to_check, const int tid);

    template <>
    __device__ bool env_collision_check_approx<ppln::robots::Panda_four>(
        volatile float *sphere_pos_approx, volatile int *joint_in_collision,
        ppln::collision::Environment<float> *env, volatile int *env_sphere_to_check, const int tid);

    template <>
    __device__ void fk<ppln::robots::Panda_four>(const float *q, volatile float *sphere_pos,
                                                 float *T, const int tid);

    template <>
    __device__ bool self_collision_check<ppln::robots::Panda_four>(volatile float *sphere_pos,
                                                                   volatile int *joint_in_collision,
                                                                   int self_batch_ind,
                                                                   int thread_interp_cnt,
                                                                   const int tid);

    template <>
    __device__ bool env_collision_check<ppln::robots::Panda_four>(
        volatile float *sphere_pos, volatile int *joint_in_collision,
        ppln::collision::Environment<float> *env, int env_batch_ind, int thread_interp_cnt,
        const int tid);

    template <>
    __device__ void fk_approx<ppln::robots::Panda_five>(const float *q,
                                                        volatile float *sphere_pos_approx, float *T,
                                                        const int tid);

    template <>
    __device__ bool self_collision_check_approx<ppln::robots::Panda_five>(
        volatile float *sphere_pos_approx, volatile int *joint_in_collision,
        volatile int *self_sphere_to_check, const int tid);

    template <>
    __device__ bool env_collision_check_approx<ppln::robots::Panda_five>(
        volatile float *sphere_pos_approx, volatile int *joint_in_collision,
        ppln::collision::Environment<float> *env, volatile int *env_sphere_to_check, const int tid);

    template <>
    __device__ void fk<ppln::robots::Panda_five>(const float *q, volatile float *sphere_pos,
                                                 float *T, const int tid);

    template <>
    __device__ bool self_collision_check<ppln::robots::Panda_five>(volatile float *sphere_pos,
                                                                   volatile int *joint_in_collision,
                                                                   int self_batch_ind,
                                                                   int thread_interp_cnt,
                                                                   const int tid);

    template <>
    __device__ bool env_collision_check<ppln::robots::Panda_five>(
        volatile float *sphere_pos, volatile int *joint_in_collision,
        ppln::collision::Environment<float> *env, int env_batch_ind, int thread_interp_cnt,
        const int tid);

} // namespace ppln::collision

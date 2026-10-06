#pragma once

#include <array>
#include <iostream>

namespace ppln::robots {
    struct Panda {
        static constexpr auto name = "panda";
        static constexpr auto dimension = 7;
        using Configuration = std::array<float, dimension>;

        // necessary to generate the scale_cfg function at compile time
        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                5.9342f, 3.6652f, 5.9342f, 3.2289f, 5.9342f, 3.9095999999999997f, 5.9342f};
            return values[i];
        }

        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {-2.9671f, -1.8326f, -2.9671f, -3.1416f,
                                        -2.9671f, -0.0873f, -2.9671f};
            return values[i];
        }

        // template metaprogramming to generate the scale_cfg function
        template <size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q) {

            q[threadIdx.x] = q[threadIdx.x] * get_s_m(threadIdx.x) + get_s_a(threadIdx.x);
        }

        __device__ __forceinline__ static void scale_cfg(float *q) {
            scale_cfg_impl(q);
        }

        template <size_t I = 0>
        __device__ __forceinline__ static void descale_cfg_impl(float *q) {
            q[threadIdx.x] = (q[threadIdx.x] - get_s_a(threadIdx.x)) / get_s_m(threadIdx.x);
        }

        __device__ __forceinline__ static void descale_cfg(float *q) {
            descale_cfg_impl(q);
        }
    };

    struct Panda_four : Panda {
        static constexpr auto name = "panda_four";
        static constexpr auto dimension = 28;
        using Configuration = std::array<float, dimension>;
        using SingleRobot = Panda;

        // necessary to generate the scale_cfg function at compile time
        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                5.9342f, 3.6652f, 5.9342f, 3.2289f, 5.9342f, 3.9095999999999997f, 5.9342f,
                5.9342f, 3.6652f, 5.9342f, 3.2289f, 5.9342f, 3.9095999999999997f, 5.9342f,
                5.9342f, 3.6652f, 5.9342f, 3.2289f, 5.9342f, 3.9095999999999997f, 5.9342f,
                5.9342f, 3.6652f, 5.9342f, 3.2289f, 5.9342f, 3.9095999999999997f, 5.9342f};
            return values[i];
        }

        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {-2.9671f, -1.8326f, -2.9671f, -3.1416f, -2.9671f, -0.0873f,
                                        -2.9671f, -2.9671f, -1.8326f, -2.9671f, -3.1416f, -2.9671f,
                                        -0.0873f, -2.9671f, -2.9671f, -1.8326f, -2.9671f, -3.1416f,
                                        -2.9671f, -0.0873f, -2.9671f, -2.9671f, -1.8326f, -2.9671f,
                                        -3.1416f, -2.9671f, -0.0873f, -2.9671f};
            return values[i];
        }

        // template metaprogramming to generate the scale_cfg function
        template <size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q) {

            q[threadIdx.x] = q[threadIdx.x] * get_s_m(threadIdx.x) + get_s_a(threadIdx.x);
        }

        __device__ __forceinline__ static void scale_cfg(float *q) {
            scale_cfg_impl(q);
        }

        template <size_t I = 0>
        __device__ __forceinline__ static void descale_cfg_impl(float *q) {
            q[threadIdx.x] = (q[threadIdx.x] - get_s_a(threadIdx.x)) / get_s_m(threadIdx.x);
        }

        __device__ __forceinline__ static void descale_cfg(float *q) {
            descale_cfg_impl(q);
        }
    };

    struct Panda_five : Panda {
        static constexpr auto name = "panda_five";
        static constexpr auto dimension = 35;
        using Configuration = std::array<float, dimension>;
        using SingleRobot = Panda;

        // necessary to generate the scale_cfg function at compile time
        __device__ static constexpr float get_s_m(int i) {
            constexpr float values[] = {
                5.9342f, 3.6652f, 5.9342f, 3.2289f, 5.9342f, 3.9095999999999997f, 5.9342f,
                5.9342f, 3.6652f, 5.9342f, 3.2289f, 5.9342f, 3.9095999999999997f, 5.9342f,
                5.9342f, 3.6652f, 5.9342f, 3.2289f, 5.9342f, 3.9095999999999997f, 5.9342f,
                5.9342f, 3.6652f, 5.9342f, 3.2289f, 5.9342f, 3.9095999999999997f, 5.9342f,
                5.9342f, 3.6652f, 5.9342f, 3.2289f, 5.9342f, 3.9095999999999997f, 5.9342f,
            };
            return values[i];
        }

        __device__ static constexpr float get_s_a(int i) {
            constexpr float values[] = {
                -2.9671f, -1.8326f, -2.9671f, -3.1416f, -2.9671f, -0.0873f, -2.9671f,
                -2.9671f, -1.8326f, -2.9671f, -3.1416f, -2.9671f, -0.0873f, -2.9671f,
                -2.9671f, -1.8326f, -2.9671f, -3.1416f, -2.9671f, -0.0873f, -2.9671f,
                -2.9671f, -1.8326f, -2.9671f, -3.1416f, -2.9671f, -0.0873f, -2.9671f,
                -2.9671f, -1.8326f, -2.9671f, -3.1416f, -2.9671f, -0.0873f, -2.9671f,
            };
            return values[i];
        }

        // template metaprogramming to generate the scale_cfg function
        template <size_t I = 0>
        __device__ __forceinline__ static void scale_cfg_impl(float *q) {

            q[threadIdx.x] = q[threadIdx.x] * get_s_m(threadIdx.x) + get_s_a(threadIdx.x);
        }

        __device__ __forceinline__ static void scale_cfg(float *q) {
            scale_cfg_impl(q);
        }

        template <size_t I = 0>
        __device__ __forceinline__ static void descale_cfg_impl(float *q) {
            q[threadIdx.x] = (q[threadIdx.x] - get_s_a(threadIdx.x)) / get_s_m(threadIdx.x);
        }

        __device__ __forceinline__ static void descale_cfg(float *q) {
            descale_cfg_impl(q);
        }
    };

} // namespace ppln::robots

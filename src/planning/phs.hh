#pragma once

#include <cmath>
#include "utils.cuh"
#include "Planners.hh"
#include "Robots.hh"


namespace phs {
    using namespace ppln;

    template<typename Robot>
    class PHS {
        static constexpr int dim = Robot::dimension;

    public:
        __device__ PHS(float* p1, float* p2, const int bid, const int tid) : focus1(p1), focus2(p2) {
            if (tid==0) min_transverse_diameter = device_utils::l2_dist(focus1, focus2, dim);
            //printf("min_tranverse_diameter: %f\n", min_transverse_diameter);
            for (int d=0; d<dim; d++){
                center[d] = (focus1[d] + focus2[d]) / 2.0f;
                //printf("center[%d]=%f\n", d, center[d]);
            }
            //__syncthreads();
            update_rotation(bid, tid);
        }

        __device__ void next(curandStateXORWOW_t* states, volatile float* config, const int bid, const int tid) {
            this->uniform_in_ball(states, config, bid, tid);
            //__syncthreads();
            this->transform(config, bid, tid);

            //printf("PHS sample before descale: %f %f %f %f %f %f %f %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6]);
            //descale config
            Robot::descale_cfg((float *)config);
            //printf("PHS sample after descale: %f %f %f %f %f %f %f %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6]);
            
            for (int d=0; d<dim; d++){
                config[d] = max(min(1.0f, config[d]), 0.0f);
                //Robot::scale_cfg((float *)config);
            }
            //__syncthreads();
        }

        __device__ void set_transverse_diameter(float transverse_diameter_in, const int bid, const int tid) {
            if (tid==0) transverse_diameter = transverse_diameter_in;
            update_transformation(bid, tid);
        }

        
    private:
        float* focus1;
        float* focus2;
        float center[dim];
        float transverse_diameter{0.};
        float min_transverse_diameter;
        float phs_measure;

        float rot_world_from_ellipse[dim][dim];
        float tf_world_from_ellipse[dim][dim];


        __device__ void update_rotation(const int bid, const int tid){
            static constexpr float circle_tolerance = 1e-6;
            if (min_transverse_diameter<circle_tolerance){
                // treat this as a circle
                if (tid==0){
                    for (int r=0; r<dim; r++){
                        for (int c=0; c<dim; c++){
                            this->rot_world_from_ellipse[r][c] = (r==c) ? 1.0f : 0.0f;
                        }
                    }
                }
                //__syncthreads();
            }
            else{
                if (tid==0){
                    float tranverse_axis[dim];
                    for (int i=0; i<dim; i++){
                        tranverse_axis[i] = (this->focus2[i]-this->focus1[i])/min_transverse_diameter;
                        //printf("transverse_axis[%d]=%f\n", i, tranverse_axis[i]);
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
                    for (int c = 0; c < dim; ++c) {
                        float vc = (c == 0) ? (1.0f - tranverse_axis[0]) : (-tranverse_axis[c]);
                        for (int r = 0; r < dim; ++r) {
                            float vr = (r == 0) ? (1.0f - tranverse_axis[0]) : (-tranverse_axis[r]);
                            float val = -beta * vr * vc;
                            if (r == c) val += 1.0f;
                            this->rot_world_from_ellipse[r][c] = val;
                        }
                    }

                    // Det fix: Householder has det = -1, flip one column => det +1
                    // Flip last column:
                    for (int r = 0; r < dim; ++r) {
                        this->rot_world_from_ellipse[r][dim-1] = -this->rot_world_from_ellipse[r][dim-1];
                    }
                }
                //__syncthreads();
            }
        }

        __device__ void update_transformation(const int bid, const int tid){
            const float conjugate_diamater = sqrt(transverse_diameter * transverse_diameter - min_transverse_diameter * min_transverse_diameter);
            float diag[dim] = {0.};
            diag[0] = transverse_diameter / 2.0f;
            for (int i=1; i<dim; i++){
                diag[i] = conjugate_diamater / 2.0f;
            }
            //printf("diag: %f %f %f %f %f %f %f %f\n", diag[0], diag[1], diag[2], diag[3], diag[4], diag[5], diag[6], diag[7]);

            for (int r=0; r<dim; r++){
                for (int c=0; c<dim; c++){
                    tf_world_from_ellipse[r][c] = rot_world_from_ellipse[r][c] * diag[c];
                }
            }
        }

        __device__ float transform(volatile float * config, const int bid, const int tid){
            for (int r=0; r<dim; r++){
                float val = 0.0f;
                for (int c=0; c<dim; c++){
                    val += tf_world_from_ellipse[r][c] * config[c];
                }
                config[r] = val + center[r];
            }
        }

        __device__ float logit(curandStateXORWOW_t* states,
                               const int bid, const int tid) {
            curandStateXORWOW_t local = states[bid*dim+tid];
            float rnd = curand_uniform(&local);
            states[bid*dim+tid] = local;
            return logf(rnd * (__frcp_rn(1.0f - rnd))) * sqrtf(M_PI / 8.0f);
        }

        __device__ void uniform_on_ball(curandStateXORWOW_t* states,
                                        volatile float* config, const int bid, const int tid) {
            unsigned mask = 0xFFFFFFFF;
            for (int d=0; d<dim; d++) {
                const auto unv = this->logit(states, bid, tid);
                config[d] = unv;
            }
            //__syncthreads();
            float norm = 0.0f;

            if (tid == 0) {
                for (int d = 0; d < dim; d++) {
                    norm += config[d] * config[d];
                }
                norm = sqrtf(norm);
            }
            //__syncthreads();
            //norm = __shfl_sync(mask, norm, 0);
            
            for (int d=0; d<dim; d++) {
                config[d] = config[d] / norm;
            }
            
            //__syncthreads();
        }

        __device__ void uniform_in_ball(curandStateXORWOW_t* states,
                                        volatile float* config, const int bid, const int tid) {
            unsigned mask = 0xFFFFFFFF;
            float radius = 0.0f;
            if (tid == 0) {
                curandStateXORWOW_t local = states[bid*dim+tid];
                float rnd = curand_uniform(&local);
                states[bid*dim+tid] = local;
                radius = powf(rnd, 1.0f / (float)dim);
            }
            //__syncthreads();
            //radius = __shfl_sync(mask, radius, 0);
            //float norm=0;
            for (int d=0; d<dim; d++) {
                this->uniform_on_ball(states, config, bid, tid);
                config[d] = config[d] * radius;
                //norm+= config[d] * config[d];
            }
            //printf("radius %f\n", radius);
            //printf("PHS sample in ball: %f %f %f %f %f %f %f %f\n", config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7]);

            //printf("config norm: %f\n", sqrt(norm));
            //__syncthreads();
        }
    };
}
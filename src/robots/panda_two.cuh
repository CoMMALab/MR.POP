namespace ppln::collision {


    #define ROBOT_COUNT 2
    #define PANDA2_APPROX_SPHERE_COUNT 11
    #define PANDA2_APPROX_JOINT_COUNT 8
    #define PANDA2_APPROX_SELF_CC_RANGE_COUNT 5
    #define FIXED -1
    #define X_PRISM 0
    #define Y_PRISM 1
    #define Z_PRISM 2
    #define X_ROT 3
    #define Y_ROT 4
    #define Z_ROT 5
    #define BATCH_SIZE 64
    
    __device__ __constant__ float robot_base_transform[] = {
        // robot 0
        -1.0,  0.0,  0.0,  0.5,
        0.0, -1.0,  0.0,  0.0,
        0.0,  0.0,  1.0,  0.1,
        0.0,  0.0,  0.0,  1.0,

        // robot 1
        1.0, 0.0, 0.0, -0.5,
        0.0, 1.0, 0.0,  0.0,
        0.0, 0.0, 1.0,  0.1,
        0.0, 0.0, 0.0,  1.0
    };
    
    __device__ __constant__ float4 panda2_approx_spheres_array[11] = {
        { 0.0f, 0.0f, 0.05f, 0.08f },
        { -0.001f, -0.039f, -0.085f, 0.154f },
        { 0.0f, -0.085f, 0.04f, 0.154f },
        { 0.039f, 0.028f, -0.052f, 0.128f },
        { -0.042f, 0.049f, 0.029f, 0.126f },
        { -0.001f, 0.037f, -0.11f, 0.176f },
        { 0.042f, 0.014f, 0.0f, 0.095f },
        { 0.015f, 0.015f, 0.075f, 0.072f },
        { 0.0f, 0.0f, 0.129f, 0.104f },
        { 0.054447f, 0.054447f, 0.1984f, 0.024f },
        { -0.054447f, -0.054447f, 0.1984f, 0.024f }
    };
    
    __device__ __constant__ float panda2_approx_fixed_transforms[] = {
        // joint 0 [base transform, following is placeholder not actual base transform]
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 1
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.333,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 2
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 3
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, -0.316,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 4
        1.0, 0.0, 0.0, 0.0825,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 5
        1.0, 0.0, 0.0, -0.0825,
        0.0, 0.0, 1.0, 0.384,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 6
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 7
        1.0, 0.0, 0.0, 0.088,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        
    };
    
    __device__ __constant__ int panda2_approx_sphere_to_joint[11] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        7,
        7,
        7
    };
    
    __device__ __constant__ int panda2_approx_joint_to_sphere_count[8] = {
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        4
    };
    
    __device__ __constant__ int panda2_approx_flattened_joint_to_spheres[11] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
    };
    
    __device__ __constant__ int panda2_approx_joint_types[] = {
        3,
        5,
        5,
        5,
        5,
        5,
        5,
        5
    };
    
    __device__ __constant__ int panda2_approx_self_cc_ranges[5][3] = {
        { 0, 5, 10 },
        { 1, 5, 10 },
        { 2, 5, 5 },
        { 2, 7, 10 },
        { 5, 7, 10 }
    };
    
    __device__ __constant__ int panda2_approx_joint_parents[8] = {
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6
    };
    
    __device__ __constant__ int panda2_approx_T_memory_idx[8] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    };
    
    __device__ __constant__ int panda2_approx_dfs_order[8] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
    };
    
    __device__ __constant__ int panda2_approx_joint_id_to_dof[8] = {
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6
    };


    __device__ __constant__ float4 panda2_spheres_array[59] = {
        { 0.0f, 0.0f, 0.05f, 0.08f },
        { 0.0f, -0.08f, 0.0f, 0.06f },
        { 0.0f, -0.03f, 0.0f, 0.06f },
        { 0.0f, 0.0f, -0.12f, 0.06f },
        { 0.0f, 0.0f, -0.17f, 0.06f },
        { 0.0f, 0.0f, 0.03f, 0.06f },
        { 0.0f, 0.0f, 0.08f, 0.06f },
        { 0.0f, -0.12f, 0.0f, 0.06f },
        { 0.0f, -0.17f, 0.0f, 0.06f },
        { 0.0f, 0.0f, -0.1f, 0.06f },
        { 0.0f, 0.0f, -0.06f, 0.05f },
        { 0.08f, 0.06f, 0.0f, 0.055f },
        { 0.08f, 0.02f, 0.0f, 0.055f },
        { -0.08f, 0.095f, 0.0f, 0.06f },
        { 0.0f, 0.0f, 0.02f, 0.055f },
        { 0.0f, 0.0f, 0.06f, 0.055f },
        { -0.08f, 0.06f, 0.0f, 0.055f },
        { 0.0f, 0.055f, 0.0f, 0.06f },
        { 0.0f, 0.075f, 0.0f, 0.06f },
        { 0.0f, 0.0f, -0.22f, 0.06f },
        { 0.0f, 0.05f, -0.18f, 0.05f },
        { 0.01f, 0.08f, -0.14f, 0.025f },
        { 0.01f, 0.085f, -0.11f, 0.025f },
        { 0.01f, 0.09f, -0.08f, 0.025f },
        { 0.01f, 0.095f, -0.05f, 0.025f },
        { -0.01f, 0.08f, -0.14f, 0.025f },
        { -0.01f, 0.085f, -0.11f, 0.025f },
        { -0.01f, 0.09f, -0.08f, 0.025f },
        { -0.01f, 0.095f, -0.05f, 0.025f },
        { 0.0f, 0.0f, 0.0f, 0.05f },
        { 0.08f, -0.01f, 0.0f, 0.05f },
        { 0.08f, 0.035f, 0.0f, 0.052f },
        { 0.0f, 0.0f, 0.07f, 0.05f },
        { 0.02f, 0.04f, 0.08f, 0.025f },
        { 0.04f, 0.02f, 0.08f, 0.025f },
        { 0.04f, 0.06f, 0.085f, 0.02f },
        { 0.06f, 0.04f, 0.085f, 0.02f },
        { -0.053033f, -0.053033f, 0.117f, 0.028f },
        { -0.03182f, -0.03182f, 0.117f, 0.028f },
        { -0.010607f, -0.010607f, 0.117f, 0.028f },
        { 0.010607f, 0.010607f, 0.117f, 0.028f },
        { 0.03182f, 0.03182f, 0.117f, 0.028f },
        { 0.053033f, 0.053033f, 0.117f, 0.028f },
        { -0.053033f, -0.053033f, 0.137f, 0.026f },
        { -0.03182f, -0.03182f, 0.137f, 0.026f },
        { -0.010607f, -0.010607f, 0.137f, 0.026f },
        { 0.010607f, 0.010607f, 0.137f, 0.026f },
        { 0.03182f, 0.03182f, 0.137f, 0.026f },
        { 0.053033f, 0.053033f, 0.137f, 0.026f },
        { -0.053033f, -0.053033f, 0.157f, 0.024f },
        { -0.03182f, -0.03182f, 0.157f, 0.024f },
        { -0.010607f, -0.010607f, 0.157f, 0.024f },
        { 0.010607f, 0.010607f, 0.157f, 0.024f },
        { 0.03182f, 0.03182f, 0.157f, 0.024f },
        { 0.053033f, 0.053033f, 0.157f, 0.024f },
        { 0.056569f, 0.056569f, 0.1874f, 0.012f },
        { 0.051619f, 0.051619f, 0.2094f, 0.012f },
        { -0.056569f, -0.056569f, 0.1874f, 0.012f },
        { -0.051619f, -0.051619f, 0.2094f, 0.012f }
    };
    
    __device__ __constant__ float panda2_fixed_transforms[] = {
        // joint 0 [base transform, following is placeholder not actual base transform]
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 1
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.333,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 2
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 3
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, -0.316,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 4
        1.0, 0.0, 0.0, 0.0825,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 5
        1.0, 0.0, 0.0, -0.0825,
        0.0, 0.0, 1.0, 0.384,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 6
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        // joint 7
        1.0, 0.0, 0.0, 0.088,
        0.0, 0.0, -1.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        
        
    };
    
    __device__ __constant__ int panda2_sphere_to_joint[59] = {
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7
    };
    
    __device__ __constant__ int panda2_joint_to_sphere_count[8] = {
        1,
        4,
        4,
        4,
        4,
        12,
        3,
        27
    };
    
    __device__ __constant__ int panda2_flattened_joint_to_spheres[59] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58
    };
    
    __device__ __constant__ int panda2_joint_types[] = {
        3,
        5,
        5,
        5,
        5,
        5,
        5,
        5
    };
    
    __device__ __constant__ int panda2_self_cc_ranges[25][3] = {
        { 0, 17, 58 },
        { 1, 17, 58 },
        { 2, 17, 58 },
        { 3, 17, 58 },
        { 4, 17, 58 },
        { 5, 17, 28 },
        { 5, 32, 58 },
        { 6, 17, 28 },
        { 6, 32, 58 },
        { 7, 17, 28 },
        { 7, 32, 58 },
        { 8, 17, 28 },
        { 8, 32, 58 },
        { 17, 32, 58 },
        { 18, 32, 58 },
        { 19, 32, 58 },
        { 20, 32, 58 },
        { 21, 32, 58 },
        { 22, 32, 58 },
        { 23, 32, 58 },
        { 24, 32, 58 },
        { 25, 32, 58 },
        { 26, 32, 58 },
        { 27, 32, 58 },
        { 28, 32, 58 }
    };
    
    __device__ __constant__ int panda2_joint_parents[8] = {
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6
    };
    
    __device__ __constant__ int panda2_T_memory_idx[8] = {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    };
    
    __device__ __constant__ int panda2_dfs_order[8] = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7
    };
    
    __device__ __constant__ int panda2_joint_id_to_dof[8] = {
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6
    };
    
    template <>
    __device__ void fk_approx<ppln::robots::Panda_dual>(
        const float* q,
        volatile float* sphere_pos_approx, // 11 spheres x 16 robots x 3 coordinates (each column is a robot)
        float *T, // 16 robots x 1 x 4x4 transform matrix , column major
        const int tid
    )
    {
        int robot_ind=0;
        // every 4 threads are responsible for one column of the transform matrix T
        // make_transform will calculate the necessary column of T_step needed for the thread
        const int col_ind = tid % 4;
        const int batch_ind = tid / 4;
    
        for (int robot_ind=0; robot_ind<ROBOT_COUNT; robot_ind++){
            int T_offset = batch_ind * 1 * 16;
            float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
            float *T_base = T + T_offset; // 4x4 transform matrix for the batch
            
            
            #pragma unroll
            for (int i = 0; i < 1; ++i) {
                float *T_col_i = T_base + i * 16 + col_ind * 4;
                for (int r=0; r<4; r++) {
                    T_col_i[r] = 0.0f;
                }
                T_col_i[col_ind] = 1.0f;
            }
            
            __syncthreads();
        
            int transformed_sphere_ind = 0;
        
            for (int j = 0; j < PANDA2_APPROX_JOINT_COUNT; ++j) {
                int i = panda2_approx_dfs_order[j];
                float T_col_tmp[4];
                int parent_idx = panda2_approx_joint_parents[i];
                int T_memory_idx_parent = panda2_approx_T_memory_idx[parent_idx];
                int T_memory_idx = panda2_approx_T_memory_idx[i];
                int q_idx = panda2_approx_joint_id_to_dof[i] + robot_ind * 7;
                if (j==0){
                    for (int c=0; c<4; c++){
                        T_step_col[c] = robot_base_transform[robot_ind*16 + 4 * c + col_ind];
                    }
                }
                if (j > 0) {
                    int ft_addr_start = i * 16;
                    int joint_type = panda2_approx_joint_types[i];
        
                    if (joint_type <= Z_PRISM) {
                        prism_fn(&panda2_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
                    }
                    else if (joint_type == X_ROT) {
                        xrot_fn(&panda2_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                    }
                    else if (joint_type == Y_ROT) {
                        yrot_fn(&panda2_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                    }
                    else if (joint_type == Z_ROT) {
                        zrot_fn(&panda2_approx_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                    }
                }
                    
                for (int r=0; r<4; r++){
                    T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
                }
                for (int r=0; r<4; r++){
                    T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
                }
                
                __syncwarp();
                int sphere_count = panda2_approx_joint_to_sphere_count[i];
                for (int s = transformed_sphere_ind + col_ind; s < transformed_sphere_ind + sphere_count; s += 4) {
                    for (int c = 0; c < 3; c++) {
                        sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + s * BATCH_SIZE * 3 + batch_ind * 3 + c] = 
                            T_base[T_memory_idx*16 + c] * panda2_approx_spheres_array[s].x +
                            T_base[T_memory_idx*16 + c + M] * panda2_approx_spheres_array[s].y +
                            T_base[T_memory_idx*16 + c + M*2] * panda2_approx_spheres_array[s].z +
                            T_base[T_memory_idx*16 + c + M*3];
                    }
                }
                transformed_sphere_ind += sphere_count;
                __syncthreads();
            }
        }
    }
    
    // 4 threads per discretized motion for self-collision check
    template <>
    __device__ bool self_collision_check_approx<ppln::robots::Panda_dual>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, volatile int * self_sphere_to_check, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool out = true;


        for (int robot_ind=0; robot_ind<ROBOT_COUNT; robot_ind++){
            for (int i=thread_ind; i<PANDA2_APPROX_SPHERE_COUNT; i+=4){
                int sphere1_ind = i;
                float sphere_1[3] = {
                    sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + sphere1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + sphere1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + sphere1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                };
                for (int other_robot_ind = robot_ind + 1; other_robot_ind < ROBOT_COUNT; other_robot_ind++) {
                    for (int j = 0; j < PANDA2_APPROX_SPHERE_COUNT; j++) {
                        float sphere_2[3] = {
                            sphere_pos_approx[other_robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                            sphere_pos_approx[other_robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                            sphere_pos_approx[other_robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                        };
                        if (sphere_sphere_self_collision(
                            sphere_1[0], sphere_1[1], sphere_1[2], panda2_approx_spheres_array[sphere1_ind].w,
                            sphere_2[0], sphere_2[1], sphere_2[2], panda2_approx_spheres_array[j].w
                        )){
                            atomicOr((int*)&joint_in_collision[robot_ind * BATCH_SIZE * PANDA2_APPROX_SPHERE_COUNT + PANDA2_APPROX_SPHERE_COUNT * batch_ind + panda2_approx_sphere_to_joint[sphere1_ind]], 2);
                            out = false;
                        }
                    }
                }
            }
            __syncwarp();
            if (thread_ind==1){
                if (robot_ind==0) self_sphere_to_check[batch_ind] = 0;
                for (int j=0; j<PANDA2_APPROX_JOINT_COUNT; j++){
                    if (joint_in_collision[robot_ind * PANDA2_APPROX_SPHERE_COUNT * BATCH_SIZE + PANDA2_APPROX_SPHERE_COUNT * batch_ind + j] & 2){
                        self_sphere_to_check[batch_ind] += panda2_joint_to_sphere_count[j];
                        //printf("bid %d in func - batch ind %d self sphere %d\n", blockIdx.x, batch_ind, self_sphere_to_check[batch_ind]);
                    }
                }
            }
        }
        return out;
    }
    
    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check_approx<ppln::robots::Panda_dual>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, volatile int * env_sphere_to_check, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool out = true;

    
        for (int robot_ind=0; robot_ind<ROBOT_COUNT; robot_ind++){
            for (int i = thread_ind; i < PANDA2_APPROX_SPHERE_COUNT; i += 4){
                // sphere i, robot batch_ind (32 robots)
                if (i > 0 &&
                    sphere_environment_in_collision(
                        env,
                        sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * BATCH_SIZE * 3 + i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                        sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * BATCH_SIZE * 3 + i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                        sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * BATCH_SIZE * 3 + i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                        panda2_approx_spheres_array[i].w
                    )
                ) {
                    atomicOr((int*)&joint_in_collision[robot_ind * BATCH_SIZE * PANDA2_APPROX_SPHERE_COUNT + PANDA2_APPROX_SPHERE_COUNT * batch_ind + panda2_approx_sphere_to_joint[i]], 1);
                    //atomicAdd((int*)env_sphere_to_check + batch_ind, panda2_joint_to_sphere_count[panda2_approx_sphere_to_joint[i]]);
                    out = false;
                } 
            }
            __syncwarp();
            if (thread_ind==1){
                if (robot_ind==0) env_sphere_to_check[batch_ind] = 0;
                for (int j=0; j<PANDA2_APPROX_JOINT_COUNT; j++){
                    if (joint_in_collision[robot_ind * PANDA2_APPROX_SPHERE_COUNT * BATCH_SIZE + PANDA2_APPROX_SPHERE_COUNT * batch_ind + j] & 1){
                        env_sphere_to_check[batch_ind] += panda2_joint_to_sphere_count[j];
                    }
                }
            }
        }
        return out;
    }
    
    
    
    
    #define PANDA2_SPHERE_COUNT 59
    #define PANDA2_JOINT_COUNT 8
    #define PANDA2_SELF_CC_RANGE_COUNT 25
    #define FIXED -1
    #define X_PRISM 0
    #define Y_PRISM 1
    #define Z_PRISM 2
    #define X_ROT 3
    #define Y_ROT 4
    #define Z_ROT 5
    
    
    
    template <>
    __device__ void fk<ppln::robots::Panda_dual>(
        const float* q,
        volatile float* sphere_pos, // 59 spheres x 16 robots x 3 coordinates (each column is a robot)
        float *T, // 16 robots x 1 x 4x4 transform matrix , column major
        const int tid
    )
    {
        // every 4 threads are responsible for one column of the transform matrix T
        // make_transform will calculate the necessary column of T_step needed for the thread
        const int col_ind = tid % 4;
        const int batch_ind = tid / 4;
    
        for (int robot_ind=0; robot_ind<ROBOT_COUNT; robot_ind++){
            int T_offset = batch_ind * 1 * 16;
            float T_step_col[4]; // 4x1 column of the joint transform matrix for this thread
            float *T_base = T + T_offset; // 4x4 transform matrix for the batch
            
            #pragma unroll
            for (int i = 0; i < 1; ++i) {
                float *T_col_i = T_base + i * 16 + col_ind * 4;
                for (int r=0; r<4; r++) {
                    T_col_i[r] = 0.0f;
                }
                T_col_i[col_ind] = 1.0f;
            }
            __syncthreads();
        
            int transformed_sphere_ind = 0;
        
            for (int j = 0; j < PANDA2_JOINT_COUNT; ++j) {
                int i = panda2_dfs_order[j];
                float T_col_tmp[4];
                int parent_idx = panda2_joint_parents[i];
                int T_memory_idx_parent = panda2_T_memory_idx[parent_idx];
                int T_memory_idx = panda2_T_memory_idx[i];
                int q_idx = panda2_joint_id_to_dof[i] + robot_ind * 7;
                if (j==0){
                    for (int c=0; c<4; c++){
                        T_step_col[c] = robot_base_transform[robot_ind*16 + 4 * c + col_ind];
                    }
                }
                if (j > 0) {
                    int ft_addr_start = i * 16;
                    int joint_type = panda2_joint_types[i];
        
                    if (joint_type <= Z_PRISM) {
                        prism_fn(&panda2_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col, joint_type);
                    }
                    else if (joint_type == X_ROT) {
                        xrot_fn(&panda2_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                    }
                    else if (joint_type == Y_ROT) {
                        yrot_fn(&panda2_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                    }
                    else if (joint_type == Z_ROT) {
                        zrot_fn(&panda2_fixed_transforms[ft_addr_start], q[q_idx], col_ind, T_step_col);
                    }
                }
                
                for (int r=0; r<4; r++){
                    T_col_tmp[r] = dot4_col(&T_base[T_memory_idx_parent*16 + r], T_step_col);
                }
                for (int r=0; r<4; r++){
                    T_base[T_memory_idx*16 + col_ind*4 + r] = T_col_tmp[r];
                }
                __syncwarp();

                int sphere_count = panda2_joint_to_sphere_count[i];
                for (int s = transformed_sphere_ind + col_ind; s < transformed_sphere_ind + sphere_count; s += 4) {
                    for (int c = 0; c < 3; c++) {
                        sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + s * BATCH_SIZE * 3 + batch_ind * 3 + c] = 
                            T_base[T_memory_idx*16 + c] * panda2_spheres_array[s].x +
                            T_base[T_memory_idx*16 + c + M] * panda2_spheres_array[s].y +
                            T_base[T_memory_idx*16 + c + M*2] * panda2_spheres_array[s].z +
                            T_base[T_memory_idx*16 + c + M*3];
                    }
                }
                transformed_sphere_ind += sphere_count;
                __syncthreads();
            }
        }
    }
    
    // 4 threads per discretized motion for self-collision check
    template <>
    __device__ bool self_collision_check<ppln::robots::Panda_dual>(volatile float* sphere_pos, volatile int* joint_in_collision, 
                                                                   int self_batch_ind, int thread_interp_cnt, const int tid){
        const int thread_ind = tid % thread_interp_cnt;
        const int batch_ind = self_batch_ind;
        bool has_collision = false;
    
        for (int robot_ind=0; robot_ind<ROBOT_COUNT; robot_ind++){
            
            for (int i=thread_ind; i<PANDA2_SPHERE_COUNT; i+=thread_interp_cnt){
                if (!(joint_in_collision[robot_ind * PANDA2_APPROX_SPHERE_COUNT * BATCH_SIZE + PANDA2_APPROX_SPHERE_COUNT * batch_ind + panda2_sphere_to_joint[i]] & 2)) continue;
                float sphere_1[3] = {
                    sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + i * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                };
                for (int other_robot_ind = robot_ind + 1; other_robot_ind < ROBOT_COUNT; other_robot_ind++) {
                    for (int j = 0; j < PANDA2_SPHERE_COUNT; j++) {
                        float sphere_2[3] = {
                            sphere_pos[other_robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                            sphere_pos[other_robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                            sphere_pos[other_robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                        };
                        if (sphere_sphere_self_collision(
                            sphere_1[0], sphere_1[1], sphere_1[2], panda2_spheres_array[i].w,
                            sphere_2[0], sphere_2[1], sphere_2[2], panda2_spheres_array[j].w
                        )){
                            //printf("Batch %d, Robot %d Sphere %d in collision with Robot %d Sphere %d\n", batch_ind, robot_ind, i, other_robot_ind, j);
                            has_collision=true;
                        }
                    }
                }
                
            }
            
            
        }
        return !has_collision;
    
    }
    
    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check<ppln::robots::Panda_dual>(volatile float* sphere_pos, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, 
                                                                  int env_batch_ind, int thread_interp_cnt, const int tid){
        const int thread_ind = tid % thread_interp_cnt;
        const int batch_ind = env_batch_ind;
        bool has_collision=false;
    
        for (int robot_ind=0; robot_ind<ROBOT_COUNT; robot_ind++){
            for (int i = thread_ind; i < PANDA2_SPHERE_COUNT; i += thread_interp_cnt){
                // sphere i, robot batch_ind (16 robots)
                if (i > 0 && (joint_in_collision[robot_ind * PANDA2_APPROX_SPHERE_COUNT * BATCH_SIZE + PANDA2_APPROX_SPHERE_COUNT * batch_ind + panda2_sphere_to_joint[i]] & 1) && 
                    sphere_environment_in_collision(
                        env,
                        sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * BATCH_SIZE * 3 + i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                        sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * BATCH_SIZE * 3 + i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                        sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * BATCH_SIZE * 3 + i * BATCH_SIZE * 3 + batch_ind * 3 + 2],
                        panda2_spheres_array[i].w
                    )
                ) {
                    has_collision=true;
                } 
                if (warp_any_active_mask(has_collision)) return false;
            }
        }
        return true;
    }
    }
    
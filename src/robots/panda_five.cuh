namespace ppln::collision {


    #define ROBOT_COUNT 5
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
    
    __device__ __constant__ float robot_base_transform_5panda[] = {
        // robot 0 (panda0)  xyz=(0.6, -0.7, 0.1), yaw=1.57
        0.00079633f, -0.99999968f,  0.00000000f,  0.60000000f,
        0.99999968f,  0.00079633f,  0.00000000f, -0.70000000f,
        0.00000000f,  0.00000000f,  1.00000000f,  0.10000000f,
        0.00000000f,  0.00000000f,  0.00000000f,  1.00000000f,

        // robot 1 (panda2)  xyz=(0.0, -0.7, 0.1), yaw=1.57
        0.00079633f, -0.99999968f,  0.00000000f,  0.00000000f,
        0.99999968f,  0.00079633f,  0.00000000f, -0.70000000f,
        0.00000000f,  0.00000000f,  1.00000000f,  0.10000000f,
        0.00000000f,  0.00000000f,  0.00000000f,  1.00000000f,

        // robot 2 (panda3)  xyz=(0.6, 0.7, 0.1), yaw=-1.57
        0.00079633f,  0.99999968f,  0.00000000f,  0.60000000f,
        -0.99999968f,  0.00079633f,  0.00000000f,  0.70000000f,
        0.00000000f,  0.00000000f,  1.00000000f,  0.10000000f,
        0.00000000f,  0.00000000f,  0.00000000f,  1.00000000f,

        // robot 3 (panda5)  xyz=(0.0, 0.7, 0.1), yaw=-1.57
        0.00079633f,  0.99999968f,  0.00000000f,  0.00000000f,
        -0.99999968f,  0.00079633f,  0.00000000f,  0.70000000f,
        0.00000000f,  0.00000000f,  1.00000000f,  0.10000000f,
        0.00000000f,  0.00000000f,  0.00000000f,  1.00000000f,

        // robot 4 (panda6)  xyz=(-0.6, -0.2, 0.1), yaw=0.0
        1.00000000f,  0.00000000f,  0.00000000f, -0.60000000f,
        0.00000000f,  1.00000000f,  0.00000000f, -0.20000000f,
        0.00000000f,  0.00000000f,  1.00000000f,  0.10000000f,
        0.00000000f,  0.00000000f,  0.00000000f,  1.00000000f,
    };
    
    
    template <>
    __device__ void fk_approx<ppln::robots::Panda_five>(
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
                        T_step_col[c] = robot_base_transform_5panda[robot_ind*16 + 4 * c + col_ind];
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
    __device__ bool self_collision_check_approx<ppln::robots::Panda_five>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, volatile int * self_sphere_to_check, const int tid){
        const int thread_ind = tid % 4;
        const int batch_ind = tid / 4;
        bool out = true;

        for (int robot_ind=0; robot_ind<ROBOT_COUNT; robot_ind++){

            for (int i = thread_ind; i < PANDA2_APPROX_SELF_CC_RANGE_COUNT; i+=4) {
                int sphere1_ind = panda2_approx_self_cc_ranges[i][0];
                float sphere_1[3] = {
                    sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + sphere1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + sphere1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + sphere1_ind * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                };

                for (int j = panda2_approx_self_cc_ranges[i][1]; j <= panda2_approx_self_cc_ranges[i][2]; j++) {
                    float sphere_2[3] = {
                        sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                        sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                        sphere_pos_approx[robot_ind * PANDA2_APPROX_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
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
                    }
                }
            }
        }
        return out;
    }
    
    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check_approx<ppln::robots::Panda_five>(volatile float* sphere_pos_approx, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, volatile int * env_sphere_to_check, const int tid){
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
    __device__ void fk<ppln::robots::Panda_five>(
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
                        T_step_col[c] = robot_base_transform_5panda[robot_ind*16 + 4 * c + col_ind];
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
    __device__ bool self_collision_check<ppln::robots::Panda_five>(volatile float* sphere_pos, volatile int* joint_in_collision, int self_batch_ind, int thread_interp_cnt, const int tid){
        const int thread_ind = tid % thread_interp_cnt;
        const int batch_ind = self_batch_ind;
        bool has_collision = false;
    
        for (int robot_ind=0; robot_ind<ROBOT_COUNT; robot_ind++){

            for (int i = thread_ind; i < PANDA2_SELF_CC_RANGE_COUNT; i += thread_interp_cnt) {

                int sphere_1_ind = panda2_self_cc_ranges[i][0];
                if (!(joint_in_collision[robot_ind * PANDA2_APPROX_SPHERE_COUNT * BATCH_SIZE + PANDA2_APPROX_SPHERE_COUNT * batch_ind + panda2_sphere_to_joint[i]] & 2)) continue;
                float sphere_1[3] = {
                    sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + i * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                    sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + i * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                    sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + i * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                };
                for (int j = panda2_self_cc_ranges[i][1]; j <= panda2_self_cc_ranges[i][2]; j++) {
                    float sphere_2[3] = {
                        sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 0],
                        sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 1],
                        sphere_pos[robot_ind * PANDA2_SPHERE_COUNT * 3 * BATCH_SIZE + j * BATCH_SIZE * 3 + batch_ind * 3 + 2]
                    };
                    if (sphere_sphere_self_collision(
                        sphere_1[0], sphere_1[1], sphere_1[2], panda2_spheres_array[sphere_1_ind].w,
                        sphere_2[0], sphere_2[1], sphere_2[2], panda2_spheres_array[j].w
                    )){
                        //return false;
                        has_collision=true;
                    }
                }
            }
            
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
                        if (warp_any_active_mask(has_collision)) return false;
                    }
                }
                
            }
            
            
        }
        return !has_collision;
    
    }
    
    // 4 threads per discretized motion for env collision check
    template <>
    __device__ bool env_collision_check<ppln::robots::Panda_five>(volatile float* sphere_pos, volatile int* joint_in_collision, ppln::collision::Environment<float> *env, 
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
    
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h>

typedef unsigned char uint8_t;

// Warp size for shuffle operations
#define WARP_SIZE 32

// Optimized block size for Ampere+ GPUs
constexpr int BLOCK_SIZE = 256;

// Vectorized types for memory efficiency
typedef __half2 half2;
typedef __half4 half4;

// Infinity for half-precision
    __half INF_HALF;
    *reinterpret_cast<unsigned short*>(&INF_HALF) = 0x7C00;  // FP16 infinity
// __device__ __half INF_HALF = __float2half(INFINITY);

// --- Utility Functions ---

// Fills an array with a given value (optimized for small arrays)
template <typename T>
__device__ __forceinline__ void array_fill(T* start, T* stop, T value) {
    while (start < stop) {
        *start++ = value;
    }
}

// --- Core Algorithm with Optimizations ---

__device__ __forceinline__ int augmenting_path_parallel(
    int nr, int nc, int i,
    half2* cost_vec, __half* u, __half* v,
    int* path, int* row4col,
    __half* shortestPathCosts,
    uint8_t* SR, uint8_t* SC,
    int* remaining,
    __half* p_minVal,
    __half infinity) {

    __half minVal = __float2half(0.0f);
    int num_remaining = nc;

    // Initialize data structures
    #pragma unroll 4
    for (int it = 0; it < num_remaining; ++it) {
        SC[it] = 0;
        remaining[it] = num_remaining - it - 1;
        shortestPathCosts[it] = infinity;
    }
    array_fill(SR, SR + nr, (uint8_t)0);

    int sink = -1;
    while (sink == -1) {
        int index = -1;
        __half lowest = infinity;
        SR[i] = 1;

        // Process two columns at once using half2
        half2* cost_row_vec = cost_vec + i * (nc / 2);
        __half base_r = __hsub(minVal, u[i]);

        for (int it = 0; it < num_remaining; it += 2) {
            int j1 = remaining[it];
            int j2 = remaining[it + 1];
            
            // Load two cost values at once
            half2 cost_pair = cost_row_vec[it / 2];
            __half cost_j1 = __low2half(cost_pair);
            __half cost_j2 = __high2half(cost_pair);

            // Compute reduced costs for both columns
            __half r1 = __hadd(base_r, __hsub(cost_j1, v[j1]));
            __half r2 = __hadd(base_r, __hsub(cost_j2, v[j2]));

            // Update shortest paths
            if (__hlt(r1, shortestPathCosts[j1])) {
                path[j1] = i;
                shortestPathCosts[j1] = r1;
            }
            if (__hlt(r2, shortestPathCosts[j2])) {
                path[j2] = i;
                shortestPathCosts[j2] = r2;
            }

            // Find minimum in warp
            __half lane_min = __hmin(r1, r2);
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                lane_min = __hmin(lane_min, __shfl_down_sync(0xFFFFFFFF, lane_min, offset));
            }

            if (lane_min == lane_min) {  // Only one thread per warp needs to update
                if (__hlt(lane_min, lowest)) {
                    lowest = lane_min;
                    index = it;
                }
            }
        }

        minVal = lowest;
        if (__heq(minVal, infinity)) {
            return -1;
        }

        int j = remaining[index];
        if (row4col[j] == -1) {
            sink = j;
        } else {
            i = row4col[j];
            if (SR[i]) {
                return -1;  // Cycle detected
            }
        }
        SC[j] = 1;
        remaining[index] = remaining[--num_remaining];
    }
    *p_minVal = minVal;
    return sink;
}

// --- Main Solver Kernel ---

__global__ void solve_kernel_half_batch_optimized(
    int bs, int nr, int nc,
    half2* cost_vec,
    __half* u, __half* v,
    __half* shortestPathCosts,
    int* path, int* col4row, int* row4col,
    uint8_t* SR, uint8_t* SC,
    int* remaining,
    __half infinity) {

    // Each block handles multiple problems
    __shared__ __half smem_u[BLOCK_SIZE];
    __shared__ __half smem_v[BLOCK_SIZE];

    // Each thread processes 4 problems
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int problem_stride = gridDim.x * blockDim.x;

    for (int batch_idx = tid; batch_idx < bs; batch_idx += problem_stride) {
        // Load problem data into shared memory
        if (threadIdx.x < nr) {
            smem_u[threadIdx.x] = u[batch_idx * nr + threadIdx.x];
        }
        if (threadIdx.x < nc) {
            smem_v[threadIdx.x] = v[batch_idx * nc + threadIdx.x];
        }
        __syncthreads();

        // Process the assignment problem
        __half minVal;
        for (int curRow = 0; curRow < nr; ++curRow) {
            int sink = augmenting_path_parallel(
                nr, nc, curRow,
                cost_vec + batch_idx * nr * (nc / 2),
                smem_u, smem_v,
                path + batch_idx * nc,
                row4col + batch_idx * nc,
                shortestPathCosts + batch_idx * nc,
                SR + batch_idx * nr,
                SC + batch_idx * nc,
                remaining + batch_idx * nc,
                &minVal,
                infinity);

            if (sink == -1) continue;

            // Update dual variables
            smem_u[curRow] = __hadd(smem_u[curRow], minVal);
            
            // Update other rows in warp-parallel manner
            for (int i = threadIdx.x; i < nr; i += blockDim.x) {
                if (SR[batch_idx * nr + i] && i != curRow) {
                    __half update = __hsub(minVal, shortestPathCosts[batch_idx * nc + col4row[batch_idx * nr + i]]);
                    smem_u[i] = __hadd(smem_u[i], update);
                }
            }

            // Update columns
            for (int j = threadIdx.x; j < nc; j += blockDim.x) {
                if (SC[batch_idx * nc + j]) {
                    __half update = __hsub(minVal, shortestPathCosts[batch_idx * nc + j]);
                    smem_v[j] = __hsub(smem_v[j], update);
                }
            }
            __syncthreads();

            // Update assignments
            int j = sink;
            int iterations = 0;
            while (iterations++ < nc) {
                int i = path[batch_idx * nc + j];
                if (i == -1) break;
                
                row4col[batch_idx * nc + j] = i;
                int swap = j;
                j = col4row[batch_idx * nr + i];
                col4row[batch_idx * nr + i] = swap;
            }
        }

        // Store results back to global memory
        if (threadIdx.x < nr) {
            u[batch_idx * nr + threadIdx.x] = smem_u[threadIdx.x];
        }
        if (threadIdx.x < nc) {
            v[batch_idx * nc + threadIdx.x] = smem_v[threadIdx.x];
        }
        __syncthreads();
    }
}

// --- Python Interface ---

torch::Tensor solve_half_batch_optimized(torch::Tensor cost, torch::Tensor col4row, torch::Tensor row4col) {
    auto sizes = cost.sizes();
    int bs = sizes[0], nr = sizes[1], nc = sizes[2];
    int device_index = cost.device().index();
    
    cudaSetDevice(device_index);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    // Convert cost matrix to half2 for vectorized access
    auto cost_vec = torch::empty({bs, nr, nc / 2}, 
        torch::dtype(torch::kFloat16).device(torch::kCUDA, device_index));
    
    // Initialize tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, device_index);
    torch::Tensor u = torch::zeros({bs, nr}, options);
    torch::Tensor v = torch::zeros({bs, nc}, options);
    torch::Tensor shortestPathCosts = torch::full({bs, nc}, INF_HALF, options);

    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_index);
    torch::Tensor path = torch::full({bs, nc}, -1, int_options);
    torch::Tensor remaining = torch::empty({bs, nc}, int_options);

    auto byte_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_index);
    torch::Tensor SR = torch::empty({bs, nr}, byte_options);
    torch::Tensor SC = torch::empty({bs, nc}, byte_options);

    // Launch optimized kernel
    int grid_size = (bs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    solve_kernel_half_batch_optimized<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        bs, nr, nc,
        reinterpret_cast<half2*>(cost_vec.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(u.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(shortestPathCosts.data_ptr<at::Half>()),
        path.data_ptr<int>(),
        col4row.data_ptr<int>(),
        row4col.data_ptr<int>(),
        SR.data_ptr<uint8_t>(),
        SC.data_ptr<uint8_t>(),
        remaining.data_ptr<int>(),
        INF_HALF
    );

    // Error checking
    AT_CUDA_CHECK(cudaGetLastError());
    return col4row;
}
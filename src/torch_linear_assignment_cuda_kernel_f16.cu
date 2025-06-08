/*
  Implementation is based on the algorithm presented in pages 1685-1686 of:

  DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems
    52(4):1679-1696, August 2016
    doi: 10.1109/TAES.2016.140952
*/

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h> 

typedef unsigned char uint8_t;

// Optimized for modern GPUs (Ampere+)
// constexpr int BLOCK_SIZE = 256;  // Better for half-precision operations

int SMPCoresf16(int device_index)
  {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device_index);
    switch (devProp.major){
    case 2: // Fermi
      if (devProp.minor == 1)
        return 48;
      else return 32;
    case 3: // Kepler
      return 192;
    case 5: // Maxwell
      return 128;
    case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) return 128;
      else if (devProp.minor == 0) return 64;
    case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) return 64;
    case 8: // Ampere
      if (devProp.minor == 0) return 64;
      else if (devProp.minor == 6) return 128;
      else if (devProp.minor == 9) return 128; // ada lovelace
    case 9: // Hopper
      if (devProp.minor == 0) return 128;
    // Unknown device;
    }
    return 128;
  }


template <typename uint8_t>
__device__ __forceinline__
void array_fill(uint8_t* start, uint8_t* stop, uint8_t value) {
    for (; start < stop; ++start) {
        *start = value;
    }
}

// __device__ __forceinline__
// int prune_costs_half(int nr, int nc, __half* cost) {
//     __half padVal = cost[nc - 1];
//     for (int c = 0; c < nc; c++) {
//         if (__hne(cost[c], padVal)) continue;
        
//         bool allPad = true;
//         for (int r = 0; r < nr; r++) {
//             if (__hne(cost[r * nr + c], padVal)) {
//                 allPad = false;
//                 break;
//             }
//         }
//         if (allPad) return c;
//     }
//     return nc;
// }

__device__ __forceinline__
int augmenting_path_half(int nr, int nc, int i,
                        __half* cost, __half* u, __half* v,
                        int* path, int* row4col,
                        __half* shortestPathCosts,
                        uint8_t* SR, uint8_t* SC,
                        int* remaining,
                        __half* p_minVal,
                        __half infinity
                        ) {


    __half minVal = __float2half(0.0f);
    int num_remaining = nc;

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

        __half* cost_row = cost + i * nc;
        __half base_r = __hsub(minVal, u[i]);
        
        for (int it = 0; it < num_remaining; it++) {
            int j = remaining[it];
            __half r = __hadd(base_r, __hsub(cost_row[j], v[j]));
            
            if (__hlt(r, shortestPathCosts[j])) {
                path[j] = i;
                shortestPathCosts[j] = r;
            }
            
            if (__hlt(shortestPathCosts[j], lowest) || 
                (__heq(shortestPathCosts[j], lowest) && row4col[j] == -1)) {
                lowest = shortestPathCosts[j];
                index = it;
            }
        }

        minVal = lowest;
        if (__heq(minVal, infinity)) { // maybe __heq
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
    // printf("[Thread %d,%d] Input: i=%d, limit=%d, nc=%d, nr=%d, sink=%d",
        // threadIdx.x, blockIdx.x, i, limit, nc, nr, sink);
    *p_minVal = minVal;
    return sink;
}

__device__ __forceinline__
void solve_kernel_half(int bs, int nr, int nc, // analogue of solve_cuda_kernel
                      __half* cost,
                      __half* u, __half* v,
                      __half* shortestPathCosts,
                      int* path, int* col4row, int* row4col,
                      uint8_t* SR, uint8_t* SC,
                      int* remaining,
                      __half infinity) {
    __half minVal;    
    for (int curRow = 0; curRow < nr; ++curRow) {
        int sink = augmenting_path_half(nr, nc, curRow, 
                                      cost,
                                      u,
                                      v,
                                      path,
                                      row4col,
                                      shortestPathCosts,
                                      SR,
                                      SC,
                                      remaining,
                                      &minVal,
                                      infinity);


    u[curRow] = __hadd(u[curRow], minVal);
        
    for (int i = 0; i < nr; i++) {
            if (SR[i] && i != curRow) {
                __half update = __hsub(minVal, shortestPathCosts[col4row[i]]);
                u[i] = __hadd(u[i], update);
            }
        }

        for (int j = 0; j < nc; j++) {
            if (SC[j]) {
                __half update = __hsub(minVal, shortestPathCosts[j]);
                v[j] = __hsub(v[j], update);
            }
        }
        int i = -1;
        int j = sink;
        int swap;
        int max_iterations = nc;
        int iterations = 0;


    while (i != curRow && iterations++ < max_iterations) {
            i = path[j];
            if (i == -1) break;  // Invalid path
            
            row4col[j] = i;      // Assign column j to row i
            swap = j;
            j = col4row[i];      // Get previous column assigned to row i
            col4row[i] = swap;   // Update row i's assignment to column j
        }
    }
}

// template <typename scalar_t>
__global__
void solve_kernel_half_batch(int bs, int nr, int nc,
                             __half *cost,
                             __half *u, __half *v,
                             __half *shortestPathCosts,
                             int *path, int *col4row, int *row4col,
                             uint8_t *SR, uint8_t *SC,
                             int *remaining,
                             __half infinity
                           ) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= bs) {
    return;
  }

  solve_kernel_half(bs, nr, nc,
    cost + i * nr * nc,
    u + i * nr,
    v + i * nc,
    shortestPathCosts + i * nc,
    path + i * nc,
    col4row + i * nr,
    row4col + i * nc,
    SR + i * nr,
    SC + i * nc,
    remaining + i * nc,
    infinity
   );

}

void solve_half_batch(torch::Tensor cost, 
    torch::Tensor col4row, 
    torch::Tensor row4col) {
    // printf("FLOAT16");
    auto sizes = cost.sizes();
    int bs = sizes[0], nr = sizes[1], nc = sizes[2];
    int device_index = cost.device().index();
    
    cudaSetDevice(device_index);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    // Create tensors with half-precision
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(torch::kCUDA, device_index);
    
    torch::Tensor u = torch::zeros({bs * nr}, options);
    torch::Tensor v = torch::zeros({bs * nc}, options);
    torch::Tensor shortestPathCosts = torch::empty({bs * nc}, options);
    
    // Infinity representation for half-precision
    __half infinity;
    *reinterpret_cast<unsigned short*>(&infinity) = 0x7C00;  // FP16 infinity
    
    // Integer tensors
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA, device_index);
        
    torch::Tensor path = torch::full({bs * nc}, -1, int_options);
    torch::Tensor remaining = torch::empty({bs * nc}, int_options);

    // Byte tensors
    auto byte_options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCUDA, device_index);
        
    torch::Tensor SR = torch::empty({bs * nr}, byte_options);
    torch::Tensor SC = torch::empty({bs * nc}, byte_options);

    // Launch kernel
    static const int blockSize = SMPCoresf16(device_index);
    int grid_size = (bs + blockSize - 1) / blockSize;
    solve_kernel_half_batch<<<grid_size, blockSize, 0, stream.stream()>>>(
        bs, nr, nc,
        reinterpret_cast<__half*>(cost.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(u.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(shortestPathCosts.data_ptr<at::Half>()),
        path.data_ptr<int>(),
        col4row.data_ptr<int>(),
        row4col.data_ptr<int>(),
        SR.data_ptr<uint8_t>(),
        SC.data_ptr<uint8_t>(),
        remaining.data_ptr<int>(),
        infinity
    );

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
 
std::vector<torch::Tensor> bla_half(torch::Tensor cost) {
    //  printf("FLOAT16");

    auto sizes = cost.sizes();  
    auto device = cost.device();
    auto options = torch::TensorOptions()
      .dtype(torch::kInt)
      .device(device.type(), device.index());
    torch::Tensor col4row = torch::full({sizes[0], sizes[1]}, -1, options);
    torch::Tensor row4col = torch::full({sizes[0], sizes[2]}, -1, options);
  
    // If sizes[2] is zero, then sizes[1] is also zero.
    if (sizes[0] * sizes[1] == 0) {
      return {col4row, row4col};
    }
    solve_half_batch(
        cost,
        col4row,
        row4col);
  
    // AT_DISPATCH_FLOATING_TYPES(__half, "solve_half_batch", [&] {
        
    // });
    return {col4row, row4col};
  }




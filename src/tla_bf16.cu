/*
  Implementation is based on the algorithm presented in pages 1685-1686 of:

  DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems
    52(4):1679-1696, August 2016
    doi: 10.1109/TAES.2016.140952
*/

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>

typedef unsigned char uint8_t;

template <typename uint8_t>
__device__ __forceinline__
void array_fill(uint8_t* start, uint8_t* stop, uint8_t value) {
    for (; start < stop; ++start) {
        *start = value;
    }
}

__device__ __forceinline__
int augmenting_path_bfloat16(int nr, int nc, int i,
                            nv_bfloat16* cost, nv_bfloat16* u, nv_bfloat16* v,
                            int* path, int* row4col,
                            nv_bfloat16* shortestPathCosts,
                            uint8_t* SR, uint8_t* SC,
                            int* remaining,
                            nv_bfloat16* p_minVal,
                            nv_bfloat16 infinity) {
    nv_bfloat16 minVal = __float2bfloat16(0.0f);
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
        nv_bfloat16 lowest = infinity;
        SR[i] = 1;

        nv_bfloat16* cost_row = cost + i * nc;
        nv_bfloat16 base_r = minVal - u[i];
        
        for (int it = 0; it < num_remaining; it++) {
            int j = remaining[it];
            nv_bfloat16 r = base_r + (cost_row[j] - v[j]);
            
            if (r < shortestPathCosts[j]) {
                path[j] = i;
                shortestPathCosts[j] = r;
            }
            
            if (shortestPathCosts[j] < lowest || 
                (shortestPathCosts[j] == lowest && row4col[j] == -1)) {
                lowest = shortestPathCosts[j];
                index = it;
            }
        }

        minVal = lowest;
        if (minVal == infinity) {
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

__device__ __forceinline__
void solve_kernel_bfloat16(int bs, int nr, int nc,
                          nv_bfloat16* cost,
                          nv_bfloat16* u, nv_bfloat16* v,
                          nv_bfloat16* shortestPathCosts,
                          int* path, int* col4row, int* row4col,
                          uint8_t* SR, uint8_t* SC,
                          int* remaining,
                          nv_bfloat16 infinity) {
    nv_bfloat16 minVal;    
    for (int curRow = 0; curRow < nr; ++curRow) {
        int sink = augmenting_path_bfloat16(nr, nc, curRow, 
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

        u[curRow] = u[curRow] + minVal;
        
        for (int i = 0; i < nr; i++) {
            if (SR[i] && i != curRow) {
                nv_bfloat16 update = minVal - shortestPathCosts[col4row[i]];
                u[i] = u[i] + update;
            }
        }

        for (int j = 0; j < nc; j++) {
            if (SC[j]) {
                nv_bfloat16 update = minVal - shortestPathCosts[j];
                v[j] = v[j] - update;
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

__global__
void solve_kernel_bfloat16_batch(int bs, int nr, int nc,
                                nv_bfloat16 *cost,
                                nv_bfloat16 *u, nv_bfloat16 *v,
                                nv_bfloat16 *shortestPathCosts,
                                int *path, int *col4row, int *row4col,
                                uint8_t *SR, uint8_t *SC,
                                int *remaining,
                                nv_bfloat16 infinity) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= bs) {
        return;
    }

    solve_kernel_bfloat16(bs, nr, nc,
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
        infinity);
}

void solve_bfloat16_batch(torch::Tensor cost, 
                         torch::Tensor col4row, 
                         torch::Tensor row4col) {
    auto sizes = cost.sizes();
    int bs = sizes[0], nr = sizes[1], nc = sizes[2];
    int device_index = cost.device().index();
    
    cudaSetDevice(device_index);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    auto options = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(torch::kCUDA, device_index);
    
    torch::Tensor u = torch::zeros({bs * nr}, options);
    torch::Tensor v = torch::zeros({bs * nc}, options);
    torch::Tensor shortestPathCosts = torch::empty({bs * nc}, options);
    
    // Infinity representation for bfloat16
    nv_bfloat16 infinity;
    *reinterpret_cast<unsigned short*>(&infinity) = 0x7F80;  // BF16 infinity
    
    auto int_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCUDA, device_index);
        
    torch::Tensor path = torch::full({bs * nc}, -1, int_options);
    torch::Tensor remaining = torch::empty({bs * nc}, int_options);

    auto byte_options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCUDA, device_index);
        
    torch::Tensor SR = torch::empty({bs * nr}, byte_options);
    torch::Tensor SC = torch::empty({bs * nc}, byte_options);

    static const int blockSize = 1;
    int grid_size = (bs + blockSize - 1) / blockSize;
    solve_kernel_bfloat16_batch<<<grid_size, blockSize, 0, stream.stream()>>>(
        bs, nr, nc,
        reinterpret_cast<nv_bfloat16*>(cost.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(u.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(shortestPathCosts.data_ptr<at::BFloat16>()),
        path.data_ptr<int>(),
        col4row.data_ptr<int>(),
        row4col.data_ptr<int>(),
        SR.data_ptr<uint8_t>(),
        SC.data_ptr<uint8_t>(),
        remaining.data_ptr<int>(),
        infinity
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

std::vector<torch::Tensor> bla_bfloat16(torch::Tensor cost) {
    auto sizes = cost.sizes();  
    auto device = cost.device();
    auto options = torch::TensorOptions()
        .dtype(torch::kInt)
        .device(device.type(), device.index());
    torch::Tensor col4row = torch::full({sizes[0], sizes[1]}, -1, options);
    torch::Tensor row4col = torch::full({sizes[0], sizes[2]}, -1, options);
  
    if (sizes[0] * sizes[1] == 0) {
        return {col4row, row4col};
    }
    solve_bfloat16_batch(
        cost,
        col4row,
        row4col);
  
    return {col4row, row4col};
}
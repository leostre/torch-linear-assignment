#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>

typedef unsigned char uint8_t;

// Vector types for optimized memory access
template <typename T>
struct Vec4 {
    T x, y, z, w;
};

int SMPCores(int device_index) {
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device_index);
    switch (devProp.major) {
        case 2: return (devProp.minor == 1) ? 48 : 32;  // Fermi
        case 3: return 192;  // Kepler
        case 5: return 128;  // Maxwell
        case 6: return ((devProp.minor == 1) || (devProp.minor == 2)) ? 128 : 64;  // Pascal
        case 7: return 64;   // Volta/Turing
        case 8: return (devProp.minor == 0) ? 64 : 128;  // Ampere/Ada
        case 9: return 128;  // Hopper
        default: return 128;
    }
}

template <typename T>
__device__ __forceinline__
void vectorized_fill(T* start, int size, T value) {
    constexpr int vec_size = sizeof(Vec4<T>) / sizeof(T);
    Vec4<T> vec_val = {value, value, value, value};
    
    Vec4<T>* vec_ptr = reinterpret_cast<Vec4<T>*>(start);
    int vec_iters = size / vec_size;
    
    for (int i = 0; i < vec_iters; ++i) {
        vec_ptr[i] = vec_val;
    }
    
    // Handle remaining elements
    for (int i = vec_iters * vec_size; i < size; ++i) {
        start[i] = value;
    }
}


template <typename scalar_t>
__device__ __forceinline__
int augmenting_path_cuda(int nr, int nc, int i,
                        const scalar_t* __restrict__ cost,
                        scalar_t* __restrict__ u,
                        scalar_t* __restrict__ v,
                        int* __restrict__ path,
                        const int* __restrict__ row4col,
                        scalar_t* __restrict__ shortestPathCosts,
                        uint8_t* __restrict__ SR,
                        uint8_t* __restrict__ SC,
                        int* __restrict__ remaining,
                        scalar_t* __restrict__ p_minVal,
                        scalar_t infinity) {
    scalar_t minVal = 0;
    int num_remaining = nc;
    
    // Vectorized initialization
    vectorized_fill(SC, nc, (uint8_t)0);
    vectorized_fill(shortestPathCosts, nc, infinity);
    
    // Initialize remaining with stride pattern
    for (int it = 0; it < nc; it += 32) {
      remaining[it] = nc - it - 1;
    }
    
    vectorized_fill(SR, nr, (uint8_t)0);

    int sink = -1;
    while (sink == -1) {
        int index = -1;
        scalar_t lowest = infinity;
        SR[i] = 1;

        const scalar_t* cost_row = cost + i * nc;
        scalar_t base_r = minVal - u[i];
        
        // Process remaining elements in vectorized chunks
        for (int it = 0; it < num_remaining; ++it) {
            const int j = remaining[it];
            const scalar_t cost_val = cost_row[j];
            const scalar_t v_val = v[j];
            const scalar_t r = base_r + cost_val - v_val;
            
            if (r < shortestPathCosts[j]) {
                path[j] = i;
                shortestPathCosts[j] = r;
            }
            
            const scalar_t current_cost = shortestPathCosts[j];
            if (current_cost < lowest || 
                (current_cost == lowest && row4col[j] == -1)) {
                lowest = current_cost;
                index = it;
            }
        }

        minVal = lowest;
        if (minVal == infinity) {
            return -1;
        }

        const int j = remaining[index];
        if (row4col[j] == -1) {
            sink = j;
        } else {
            i = row4col[j];
        }

        SC[j] = 1;
        remaining[index] = remaining[--num_remaining];
    }
    *p_minVal = minVal;
    return sink;
}

template <typename scalar_t>
__device__
void solve_cuda_kernel(int nr, int nc,
                      const scalar_t* __restrict__ cost,
                      scalar_t* __restrict__ u,
                      scalar_t* __restrict__ v,
                      scalar_t* __restrict__ shortestPathCosts,
                      int* __restrict__ path,
                      int* __restrict__ col4row,
                      int* __restrict__ row4col,
                      uint8_t* __restrict__ SR,
                      uint8_t* __restrict__ SC,
                      int* __restrict__ remaining,
                      scalar_t infinity) {
    scalar_t minVal;
    for (int curRow = 0; curRow < nr; ++curRow) {
        auto sink = augmenting_path_cuda(nr, nc, curRow, cost,
                                       u, v,
                                       path, row4col,
                                       shortestPathCosts,
                                       SR, SC,
                                       remaining,
                                       &minVal, infinity);

        CUDA_KERNEL_ASSERT(sink >= 0 && "Infeasible matrix");

        // Update u and v with vectorized operations
        u[curRow] += minVal;
        
        // Process SR in vectorized chunks
        for (int i = 0; i < nr; i += 4) {
            int end = min(i + 4, nr);
            for (int k = i; k < end; ++k) {
                if (SR[k] && k != curRow) {
                    u[k] += minVal - shortestPathCosts[col4row[k]];
                }
            }
        }
        
        // Process SC in vectorized chunks
        for (int j = 0; j < nc; j += 4) {
            int end = min(j + 4, nc);
            for (int k = j; k < end; ++k) {
                if (SC[k]) {
                    v[k] -= minVal - shortestPathCosts[k];
                }
            }
        }

        // Update assignments
        int j = sink;
        int swap;
        while (true) {
            int i = path[j];
            row4col[j] = i;
            swap = j;
            j = col4row[i];
            col4row[i] = swap;
            
            if (i == curRow) break;
        }
    }
}

template <typename scalar_t>
__global__
void solve_cuda_kernel_batch(int bs, int nr, int nc,
                            const scalar_t* __restrict__ cost,
                            scalar_t* __restrict__ u,
                            scalar_t* __restrict__ v,
                            scalar_t* __restrict__ shortestPathCosts,
                            int* __restrict__ path,
                            int* __restrict__ col4row,
                            int* __restrict__ row4col,
                            uint8_t* __restrict__ SR,
                            uint8_t* __restrict__ SC,
                            int* __restrict__ remaining,
                            scalar_t infinity) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= bs) return;

    solve_cuda_kernel(nr, nc,
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


template <typename scalar_t>
void solve_cuda_batch(c10::ScalarType scalar_type,
                      int device_index,
                      int bs, int nr, int nc,
                      scalar_t *cost, int *col4row, int *row4col) {
  cudaSetDevice(device_index);

  TORCH_CHECK(std::numeric_limits<scalar_t>::has_infinity, "Data type doesn't have infinity.");
  auto infinity = std::numeric_limits<scalar_t>::infinity();

  auto int_opt = torch::TensorOptions()
    .dtype(torch::kInt)
    .device(torch::kCUDA, device_index);
  auto scalar_t_opt = torch::TensorOptions()
    .dtype(scalar_type)
    .device(torch::kCUDA, device_index);
  auto uint8_opt = torch::TensorOptions()
    .dtype(torch::kUInt8)
    .device(torch::kCUDA, device_index);

  torch::Tensor u = torch::zeros({bs * nr}, scalar_t_opt);
  torch::Tensor v = torch::zeros({bs * nc}, scalar_t_opt);
  torch::Tensor shortestPathCosts = torch::empty({bs * nc}, scalar_t_opt);
  torch::Tensor path = torch::full({bs * nc}, -1, int_opt);
  torch::Tensor SR = torch::empty({bs * nr}, uint8_opt);
  torch::Tensor SC = torch::empty({bs * nc}, uint8_opt);
  torch::Tensor remaining = torch::empty({bs * nc}, int_opt);

  static const int blockSize = SMPCores(device_index);
  int gridSize = (bs + blockSize - 1) / blockSize;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device_index);
  solve_cuda_kernel_batch<<<gridSize, blockSize, 0, stream.stream()>>>(
    bs, nr, nc,
    cost,
    u.data<scalar_t>(),
    v.data<scalar_t>(),
    shortestPathCosts.data<scalar_t>(),
    path.data<int>(),
    col4row, row4col,
    SR.data<uint8_t>(),
    SC.data<uint8_t>(),
    remaining.data<int>(),
    infinity);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, cudaGetErrorString(err));
  }
}


std::vector<torch::Tensor> batch_linear_assignment_cuda(torch::Tensor cost) {
  auto sizes = cost.sizes();

  TORCH_CHECK(sizes[2] >= sizes[1], "The number of tasks must be greater or equal to the number of workers.");

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

  AT_DISPATCH_FLOATING_TYPES(cost.scalar_type(), "solve_cuda_batch", [&] {
    solve_cuda_batch<scalar_t>(
        cost.scalar_type(),
        device.index(),
        sizes[0], sizes[1], sizes[2],
        cost.data<scalar_t>(),
        col4row.data<int>(),
        row4col.data<int>());
  });
  return {col4row, row4col};
}

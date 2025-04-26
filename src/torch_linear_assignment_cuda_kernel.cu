/*
  Implementation is based on the algorihtm presented in pages 1685-1686 of:

  DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems
    52(4):1679-1696, August 2016
    doi: 10.1109/TAES.2016.140952
 */

 #include <cuda.h>
 #include <cuda_runtime.h>
 
 #include <torch/extension.h>
 #include <ATen/cuda/CUDAContext.h>
 #include <cuda_fp16.h>
 
 #include <limits>
 
 
 typedef unsigned char uint8_t;
 
 
 int SMPCores(int device_index)
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
 
//  template <typename scalar_t>
//  __device__ __forceinline__
//  void c
 
 template <typename scalar_t>
 __device__ __forceinline__
 void array_fill(scalar_t *start, scalar_t *stop, scalar_t value) {
   for (; start < stop; ++start) {
     *start = value;
   }
 }
 
 
 template <typename scalar_t>
 __device__ __forceinline__
 int augmenting_path_cuda(int nr, int nc, int i,
                          __half *cost, __half *u, __half *v,
                          int *path, int *row4col,
                          __half *shortestPathCosts,
                          uint8_t *SR, uint8_t *SC,
                          int *remaining,
                          __half *p_minVal,
                          auto infinity)
 {
     __half minVal = __float2half(0.);
     int num_remaining = nc;
     for (int it = 0; it < nc; ++it) {
         SC[it] = 0;
         remaining[it] = nc - it - 1;
         shortestPathCosts[it] = infinity;
     }
 
     array_fill(SR, SR + nr, (uint8_t) 0);
 
     int sink = -1;
     while (sink == -1) {
         int index = -1;
         __half lowest = infinity;
         SR[i] = 1;
 
         __half *cost_row = cost + i * nc;
         __half base_r = __hsub(minVal, u[i]);
         for (int it = 0; it < num_remaining; it++) {
             int j = remaining[it];
             __half r = __hsub(__hadd(base_r, cost_row[j]), v[j]);
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
         if (__heq(minVal, infinity)) {
             return -1;
         }
 
         int j = remaining[index];
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
 __device__ __forceinline__
 void solve_cuda_kernel(int nr, int nc,
                        __half *cost,
                        __half *u, __half *v,
                        __half *shortestPathCosts,
                        int *path, int *col4row, int *row4col,
                        uint8_t *SR, uint8_t *SC,
                        int *remaining,
                        scalar_t infinity)
 {
   __half minVal;
   for (int curRow = 0; curRow < nr; ++curRow) {
     auto sink = augmenting_path_cuda(nr, nc, curRow, cost,
                                      u, v,
                                      path, row4col,
                                      shortestPathCosts,
                                      SR, SC,
                                      remaining,
                                      &minVal, infinity);
 
     CUDA_KERNEL_ASSERT(sink >= 0 && "Infeasible matrix");
 
     u[curRow] = __hadd(u[curRow], minVal);
     for (int i = 0; i < nr; i++) {
       if (SR[i] && i != curRow) {
         u[i] = __hsub(__hadd(u[i], minVal), shortestPathCosts[col4row[i]]);
       }
     }
 
     for (int j = 0; j < nc; j++) {
       if (SC[j]) {
         v[j] = __hsub(v[j], __hsub(minVal, shortestPathCosts[j]));
       }
     }
 
     int i = -1;
     int j = sink;
     int swap;
     while (i != curRow) {
       i = path[j];
       row4col[j] = i;
       swap = j;
       j = col4row[i];
       col4row[i] = swap;
     }
   }
 }
 
 
 template <typename scalar_t>
 __global__
 void solve_cuda_kernel_batch(int bs, int nr, int nc,
                              scalar_t *cost,
                              __half *u, __half *v,
                              __half *shortestPathCosts,
                              int *path, int *col4row, int *row4col,
                              uint8_t *SR, uint8_t *SC,
                              int *remaining,
                              __half infinity) {
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i >= bs) {
     return;
   }
   __half* cost16 = new __half[nr * nc];
  //  std::cout << "type of cost: " << type(cost) << ", cost16: " << type(cost16);
   for (int k = 0; k < nr * nc; k++){
    cost16[k] = (cost[i * nr * nc + k]);
   }
 
   solve_cuda_kernel(nr, nc,
                    //  cost + i * nr * nc,
                    cost16,
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
  delete[] cost16;
 }

 
 template <typename T>
 __device__ __host__ T get_infinity() {
     if constexpr (std::is_same_v<T, at::Half>) {
         return __float2half(INFINITY);
     } else {
         return std::numeric_limits<T>::infinity();
     }
 }
 
// Modify your template function to handle half precision
template <typename scalar_t>
void solve_cuda_batch(c10::ScalarType scalar_type,
                     int device_index,
                     int bs, int nr, int nc,
                     scalar_t *cost, int *col4row, int *row4col) {
    cudaSetDevice(device_index);

    // Special handling for half precision infinity
    auto infinity = get_infinity<scalar_t>();

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

// Update your dispatch macro
std::vector<torch::Tensor> batch_linear_assignment_cuda(torch::Tensor cost) {
    auto sizes = cost.sizes();

    TORCH_CHECK(sizes[2] >= sizes[1], "The number of tasks must be greater or equal to the number of workers.");

    auto device = cost.device();
    auto options = torch::TensorOptions()
        .dtype(torch::kInt)
        .device(device.type(), device.index());
    torch::Tensor col4row = torch::full({sizes[0], sizes[1]}, -1, options);
    torch::Tensor row4col = torch::full({sizes[0], sizes[2]}, -1, options);

    if (sizes[0] * sizes[1] == 0) {
        return {col4row, row4col};
    }

    // Add half precision to the dispatch
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(cost.scalar_type(), "solve_cuda_batch", [&] {
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
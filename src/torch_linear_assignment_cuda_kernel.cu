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
 
 
 template <typename scalar_t>
 __device__ __forceinline__
 void array_fill(scalar_t *start, scalar_t *stop, scalar_t value) {
   for (; start < stop; ++start) {
     *start = value;
   }
 }

 template <typename scalar_t>
__device__
 int prune_costs(int nr, int nc,
                scalar_t *cost){
    
    auto padVal = cost[nc - 1];
    for (int c = 0; c < nc; c++){
      if (cost[c] != padVal){
        continue; 
      }
      bool allPad = true;
      for (int r = 0; r <nr; r++){
        if (cost[r * nr + c] != padVal){
          allPad = false;
          break;
        }
      }
      if (allPad){
        return c;
      }
    }
    return nc;
 }
 
 
 template <typename scalar_t>
 __device__ __forceinline__
 int augmenting_path_cuda(int nr, int nc, int i,
                          scalar_t *cost, scalar_t *u, scalar_t *v,
                          int *path, int *row4col,
                          scalar_t *shortestPathCosts,
                          uint8_t *SR, uint8_t *SC,
                          int *remaining,
                          scalar_t *p_minVal,
                          scalar_t infinity,
                          int limit
                        )
 {
     scalar_t minVal = 0;
     int num_remaining = min(nc, 
      limit
      // prune_costs(nr, nc, cost)
    );

     for (int it = 0; it < limit; ++it) {
         SC[it] = 0;
         remaining[it] = limit - it - 1;
         shortestPathCosts[it] = infinity;
     }
 
     array_fill(SR, SR + nr, (uint8_t) 0);
 
     int sink = -1;
     while (sink == -1) {

         int index = -1;
         scalar_t lowest = infinity;
         SR[i] = 1;
 
         scalar_t *cost_row = cost + i * nc;
         scalar_t base_r = minVal - u[i];
         for (int it = 0; it < num_remaining; it++) {
             int j = remaining[it];
             scalar_t r = base_r + cost_row[j] - v[j];
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
              return -1;  // Cycle detected (row already visited)
            }
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
                        scalar_t *cost,
                        scalar_t *u, scalar_t *v,
                        scalar_t *shortestPathCosts,
                        int *path, int *col4row, int *row4col,
                        uint8_t *SR, uint8_t *SC,
                        int *remaining,
                        scalar_t infinity,
                        int limit
                        )
 {
   scalar_t minVal;
   for (int curRow = 0; curRow < nr; ++curRow) {
     auto sink = augmenting_path_cuda(nr, nc, curRow, cost,
                                      u, v,
                                      path, row4col,
                                      shortestPathCosts,
                                      SR, SC,
                                      remaining,
                                      &minVal, infinity,
                                      limit
                                    );
  
     u[curRow] += minVal;
     for (int i = 0; i < nr; i++) {
       if (SR[i] && i != curRow) {
         u[i] += minVal - shortestPathCosts[col4row[i]];
       }
     }
 
     for (int j = 0; j < limit; j++) {
       if (SC[j]) {
         v[j] -= minVal - shortestPathCosts[j];
       }
     }
 
     int i = -1;
     int j = sink;
     int swap;

    int max_iterations = limit;  // Prevent infinite loops
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
 
 
 template <typename scalar_t>
 __global__
 void solve_cuda_kernel_batch(int bs, int nr, int nc,
                              scalar_t *cost,
                              scalar_t *u, scalar_t *v,
                              scalar_t *shortestPathCosts,
                              int *path, int *col4row, int *row4col,
                              uint8_t *SR, uint8_t *SC,
                              int *remaining,
                              scalar_t infinity,
                              int *limits
                            ) {
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if (i >= bs) {
     return;
   }
   int limit = limits[i];
 
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
                     infinity,
                     limit
                    );
 }
 
template <typename scalar_t>
__global__
 void getLimits(int bs, int nr, int nc, scalar_t *costs, int *out){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= bs) {return;}
  scalar_t *cost = costs + i * nr * nc;
  auto padVal = cost[nc * nr - 1];
  for (int c = 0; c < nc; c++){
    if (cost[c] != padVal){
      continue; 
    }
    bool allPad = true;
    for (int r = 0; r <nr; r++){
        if (cost[r * nr + c] != padVal){
            allPad = false;
            break;
        }
    }
    if (allPad){
        out[i] = c;
    }
  }
  out[i] = nc;
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
   torch::Tensor limits = torch::ones({bs}, int_opt) * nc;
   getLimits<<<gridSize, blockSize, 0, stream.stream()>>>(bs, nr, nc,
    cost, limits.data<int>()
   );
   solve_cuda_kernel_batch<<<gridSize, blockSize, 0, stream.stream()>>>(
     bs, nr, nc,
     cost,
     u.data_ptr<scalar_t>(),
     v.data_ptr<scalar_t>(),
     shortestPathCosts.data_ptr<scalar_t>(),
     path.data_ptr<int>(),
     col4row, row4col,
     SR.data_ptr<uint8_t>(),
     SC.data<uint8_t>(),
     remaining.data_ptr<int>(),
     infinity,
     limits.data_ptr<int>()
    );
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
     TORCH_CHECK(false, cudaGetErrorString(err));
   }
 }
 
 
 std::vector<torch::Tensor> batch_linear_assignment_cuda(torch::Tensor cost) {
   auto sizes = cost.sizes();
 
  //  TORCH_CHECK(sizes[2] >= sizes[1], "The number of tasks must be greater or equal to the number of workers.");
 
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
         cost.data_ptr<scalar_t>(),
         col4row.data_ptr<int>(),
         row4col.data_ptr<int>());
   });
   return {col4row, row4col};
 }

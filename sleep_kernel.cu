#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void spin_kernel(
        torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> sleep_time) {
  // see concurrentKernels CUDA sampl
  int64_t start_clock = clock64();
  int64_t clock_offset = 0;
  while (clock_offset < sleep_time[0])
  {
    clock_offset = clock64() - start_clock;
  }
}

void cuda_sleep(torch::Tensor sleep_time) {
    AT_DISPATCH_INTEGRAL_TYPES(sleep_time.type(), "sleep", ([&] {
        spin_kernel<scalar_t><<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
            sleep_time.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>());
    }));
}

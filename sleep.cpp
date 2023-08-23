#include <torch/extension.h>

void cuda_sleep(torch::Tensor sleep_time);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sleep", &cuda_sleep, "sleep (CUDA)");
}

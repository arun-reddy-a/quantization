import torch
from torch.utils.cpp_extension import load_inline
import time

device = "cuda"
N = 10_000_000

x = torch.randn(N, device=device)
w = torch.randn(N, device=device)
b = torch.randn(N, device=device)

torch.cuda.synchronize()
start = time.time()

y = torch.relu(x * w + b)

torch.cuda.synchronize()
print("PyTorch time:", time.time() - start)




cuda_src = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(
    const float* x,
    const float* w,
    const float* b,
    float* out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = x[i] * w[i] + b[i];
        out[i] = val > 0 ? val : 0;
    }
}

torch::Tensor fused_op(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b
) {
    auto out = torch::zeros_like(x);
    int n = x.numel();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused CUDA op");
}
'''

module = load_inline(
    name="fused_ext",
    cpp_sources="",
    cuda_sources=cuda_src,
    functions=None,
    verbose=False,
)

device = "cuda"
N = 10_000_000

x = torch.randn(N, device=device)
w = torch.randn(N, device=device)
b = torch.randn(N, device=device)

torch.cuda.synchronize()
start = time.time()

y = module.fused_op(x, w, b)

torch.cuda.synchronize()
print("Custom CUDA time:", time.time() - start)


"""
Triton is a language designed to allow GPU programmers to optimize e.g.
the tiling structure of an algorithm with a simple Python API rather than
having to deal with a complicated CUDA optimization problem.

The main application of Triton is in generating custom kernels that are
heavily optimized for GPU memory access and processing.
This sometimes takes the form of "fused" activation kernels which are optimized for
number of memory loads and stores.

Triton must be run on a GPU, so the code here should be moved to Colab for testing.

A matrix multiplication Triton kernel is available in the Triton tutorials and
deals with similar tiling issues as we have explored in Python and C++.
It is available here:
https://triton-lang.org/master/getting-started/tutorials/03-matrix-multiplication.html
Here we build on this tutorial to understand more features of Triton.

On an NVIDIA Tesla T4 GPU, there are 40 Streaming Multiprocessors (SM's) with
a 1.59 GHz boost clock rate (compared to the A100 1.41 GHz rate quoted here:
https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html).
With a fused-multiply-add operation rate per clock cycle of 512 for FP16 on tensor cores,
this yields 40 SM * 1.41 GHz * 512 FP16/cycle * 2 ops (fma = mult+add) = 65.13 TFLOPS.
This matches the result quoted on the reference page: 
https://www.techpowerup.com/gpu-specs/tesla-t4.c3316.
The 320 GB/s main memory bandwidth means that the T4 achieves a minimum OP/BYTE ratio
of min(OP/BYTE) = 65.13 TFLO / 320 GB ~ 200.
If data is stored in the shared L2 cache instead, this ratio would be higher.

A matrix multiplication of (M,K) x (K,N) -> (M,N) dimension requires 2*M*N*K FLO,
since we need K fused-mult-&-add for each output element.
To perform this calculation, all three matrices need to be loaded from memory,
which requires M*N + M*K + N*K FP16 loads, and since each FP16 is 2 bytes, this is
2 * (M*N + M*K + N*K) bytes.
Therefore, the arithmetic intensity of matmul is AI = M*N*K / (M*N + M*K + N*K).
For square matrices, this is N/3.

From this, we see that matmul is math-bound when N/3 > 200 but bandwidth-bound
when N/3 < 200.
It is perhaps relevant that the only N on which the Triton matmul op achieves
performance comparable to cuBLAS are the small N near the bandwidth-bound regime.
Once the computation becomes surely math-bound around N = 640, cuBLAS is clearly better.
This suggests that Triton may not be maximizing arithmetic throughput on the T4.
"""

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Matrix pointers
    a_ptr, b_ptr, c_ptr,
    # A.size() = (M,K), B.size() = (K,N), C.size() = (M,N)
    M, N, K,
    # strides to pass from one row or column to the next
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Second level blocking for L2 hits
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Pointer blocks to A and B sub-blocks
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Computation of block (m,n) in C
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # Note that for simplicity, we don't apply a mask here.
        # This means that if K is not a multiple of BLOCK_SIZE_K,
        # this will access out-of-bounds memory and produce an
        # error or (worse!) incorrect results.
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1)
    )
    return c
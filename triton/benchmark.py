"""
Benchmarking Triton reference matmul op against torch matmul (cuBLAS).

Move to Colab for testing.

Results:
On the Google Colab Tesla T4 GPU, cuBLAS outperforms Triton significantly
in TFLOPS for FP16 matmul (triton_vs_cublas_T4.pdf).
It is interesting that essentially the same code appearing in the Triton
tutorial, which achieves performance on par with cuBLAS on (what we assume is)
an A100 GPU, is outperformed on the weaker T4 GPU.
Reasoning about why this is the case may be instructive.
"""

# Uncomment for Colab.
# !pip install -q torch
# !pip install -q triton

import torch
import triton
import triton.ops as to

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 33)
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        # possible values for `line_arg``
        line_vals=['cublas', 'triton'],
        # label name for the lines
        line_names=["cuBLAS", "Triton"],
        # line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="matmul-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={},
    )
)

def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: to.matmul(a, b))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True)
import torch

import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),#, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),# num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=3),# num_stages=4, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def round_to_fp4_kernel(
        x_ptr,
        candidates_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    candidates = tl.load(candidates_ptr + tl.arange(0, 8)).to(tl.float32)
    
    errors = (x.abs()[:, None] - candidates[None, :]).abs().to(tl.float32)
    bit_args = errors.argmin(axis=1, tie_break_left=True)
    
    S = tl.where(x >= 0, 0, 1).to(tl.uint8)
    E = bit_args % 4    
    M = bit_args // 4

    output = S << 3 | E << 1 | M

    tl.store(output_ptr + offsets, output, mask=mask)


def round_to_fp4(x: torch.Tensor):
    assert x.is_contiguous()
    output = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
    n_elements = x.numel()
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)

    candidates = torch.tensor([0.0000, 1.0000, 2.0000, 4.0000, 0.5000, 1.5000, 3.0000, 6.0000]).to(x.device)
    round_to_fp4_kernel[grid](x, candidates, output, n_elements)
    return output





if __name__ == "__main__":
    ### BENCHMARK ###
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["N"],  # Argument names to use as an x-axis for the plot
            x_vals=[512, 1024, 2048], #, 4096, 8192],
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=[
                "torch",
                "kernel",
                "matmul"
                ],
            line_names=[
                "torch",
                "kernel",
                "matmul" 
                ],
            ylabel="quantization time, ms",  # Label name for the y-axis
            xlabel="Matrix size",
            plot_name="Time, ms",  # Name for the plot, used also as a file name for saving the plot.
            args={},
        ))
    
    @triton.testing.perf_report(configs)
    def benchmark(N, provider):
        x = 2 * torch.randn(N, N, dtype=torch.float32, device="cuda")

        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: MXFP4Tensor(size=x.shape, data=x), quantiles=quantiles)
        if provider == "matmul":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x @ x, quantiles=quantiles)
        if provider == "kernel":
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: round_to_fp4(x), quantiles=quantiles)
            print(round_to_fp4_kernel.best_config)

        return ms, max_ms, min_ms

    benchmark.run(show_plots=False, print_data=True)
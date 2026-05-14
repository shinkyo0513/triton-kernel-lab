import torch
import triton
from kernels.rowwise_sum import rowwise_sum

def get_bandwidth_gbps(bytes_moved: int, ms: float) -> float:
    gbps = (bytes_moved / (ms * 1e-3)) * 1e-9
    return gbps

def bench_rowwise_sum():
    assert torch.cuda.is_available(), "CUDA is not available."

    nrows = 1000 * 1000
    ncols = 1000

    x = torch.randn((nrows, ncols), device="cuda", dtype=torch.float32)

    def run_triton():
        rowwise_sum(x, block_size=1024)

    triton_ms = triton.testing.do_bench(run_triton)
    torch_ms = triton.testing.do_bench(lambda: torch.sum(x, dim=1))

    bytes_moved = nrows * ncols * 4 + nrows * 4

    print(f"#rows: {nrows}, #cols: {ncols}")
    print(f"Triton: {triton_ms:.4f} ms, {get_bandwidth_gbps(bytes_moved, triton_ms):.2f} GB/s")
    print(f"Torch: {torch_ms:.4f} ms, {get_bandwidth_gbps(bytes_moved, torch_ms):.2f} GB/s")

if __name__ == "__main__":
    bench_rowwise_sum()

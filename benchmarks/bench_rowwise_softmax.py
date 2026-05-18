import torch
import triton 
from kernels.softmax import rowwise_softmax

def bench_rowwise_softmax():
    assert torch.cuda.is_available(), "CUDA is not available."

    test_nrows = 8
    test_ncols = 1000
    x_test = torch.randn((test_nrows, test_ncols), device="cuda", dtype=torch.float32)

    out_triton = rowwise_softmax(x_test, 1024)
    out_torch = torch.nn.functional.softmax(x_test, dim=1)

    max_error = torch.max(torch.abs(out_triton - out_torch)).item()
    print(f"Correctness max error: {max_error:.6f}")

    assert torch.allclose(out_triton, out_torch, atol=1e-6), \
        "rowwise_softmax does not match torch.nn.functional.softmax"

    nrows = 1000 * 1000
    ncols = 1000
    x = torch.randn((nrows, ncols), device="cuda", dtype=torch.float32)

    triton_ms = triton.testing.do_bench(lambda: rowwise_softmax(x, 1024))
    torch_ms = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=1))

    print(f"#rows: {nrows}, #cols: {ncols}")
    print(f"Triton: {triton_ms:.4f} ms")
    print(f"Torch: {torch_ms:.4f} ms")

if __name__ == "__main__":
    bench_rowwise_softmax()

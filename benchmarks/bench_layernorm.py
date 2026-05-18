import torch
import triton 
from kernels.layernorm import layernorm

def bench_layernorm():
    assert torch.cuda.is_available(), "CUDA is not available."

    test_nrows = 1024
    test_ncols = 768
    eps=1e-5

    x_test = torch.randn((test_nrows, test_ncols), device="cuda", dtype=torch.float32)
    gamma_test = torch.randn((test_ncols,), device="cuda", dtype=torch.float32)
    beta_test = torch.randn((test_ncols,), device="cuda", dtype=torch.float32)

    out_triton = layernorm(x_test, gamma_test, beta_test, eps)
    out_torch = torch.nn.functional.layer_norm(
        x_test, 
        normalized_shape=(test_ncols,),
        weight=gamma_test,
        bias=beta_test,
        eps=eps
    )

    max_error = torch.max(torch.abs(out_triton - out_torch)).item()
    print(f"Correctness max error: {max_error:.6f}")

    torch.testing.assert_close(out_triton, out_torch, atol=1e-6)
    print(f"Correctness: PASS")

    nrows = 1 << 12
    ncols = 768
    x = torch.randn((nrows, ncols), device="cuda", dtype=torch.float32)
    gamma = torch.randn((ncols,), device="cuda", dtype=torch.float32)
    beta = torch.randn((ncols,), device="cuda", dtype=torch.float32)

    def run_triton():
        layernorm(x, gamma, beta, eps)

    def run_torch():
        torch.nn.functional.layer_norm(
            x, 
            normalized_shape=(ncols,),
            weight=gamma,
            bias=beta,
            eps=eps
        ) 
        
    triton_ms = triton.testing.do_bench(run_triton)
    torch_ms = triton.testing.do_bench(run_torch)

    print(f"#rows: {nrows}, #cols: {ncols}")
    print(f"Triton: {triton_ms:.4f} ms")
    print(f"Torch: {torch_ms:.4f} ms")

if __name__ == "__main__":
    bench_layernorm()

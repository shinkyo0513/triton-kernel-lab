import torch
from kernels.vector_add import vector_add

def get_bandwidth_gbps(bytes_moved: int, ms: float) -> float:
    gbps = (bytes_moved / (ms * 1e-3)) * 1e-9
    return gbps

def do_testing(
    call: callable,
    warmup: int = 10,
    repeat: int = 100
) -> float:
    assert torch.cuda.is_available(), "CUDA is not available."

    # Warmup
    for _ in range(warmup):
        call()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        call()
    end.record()

    torch.cuda.synchronize()
    
    total_ms = start.elapsed_time(end) 
    return total_ms / repeat

def bench_vector_add():
    n = 16 * 1024 * 1024

    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y = torch.randn(n, device='cuda', dtype=torch.float32)

    triton_ms = do_testing(lambda: vector_add(x, y))
    torch_ms = do_testing(lambda: x + y)

    bytes_moved = n * 4 * 3

    print(f"N: {n}")
    print(f"Triton: {triton_ms:.4f} ms, {get_bandwidth_gbps(bytes_moved, triton_ms):.2f} GB/s")
    print(f"Torch: {torch_ms:.4f} ms, {get_bandwidth_gbps(bytes_moved, torch_ms):.2f} GB/s")

if __name__ == "__main__":
    bench_vector_add()
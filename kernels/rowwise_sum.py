import torch
import triton 
import triton.language as tl

@triton.jit
def rowwise_sum_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * N

    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    global_offsets = row_start + col_offsets

    mask = col_offsets < N
    
    row_values = tl.load(x_ptr + global_offsets, mask=mask, other=0.0)

    row_sum = tl.sum(row_values, axis=0)
    
    tl.store(out_ptr + row_idx, row_sum)

def rowwise_sum(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_contiguous(), "Input must be contiguous"
    
    nrows, ncols = x.shape
    assert ncols > 0, \
        f"For this simple version, block_size must be > 0"
    assert ncols <= block_size, \
        f"For this simple version, block_size ({block_size}) must be >= N ({ncols})"
    assert block_size & (block_size - 1) == 0, \
        f"block_size ({block_size}) must be the power of 2"

    out = torch.empty((nrows,), device=x.device, dtype=x.dtype)

    grid = (nrows,)
    
    rowwise_sum_kernel[grid](
        x, 
        out, 
        nrows,
        ncols, 
        BLOCK_SIZE=block_size
    )

    return out

def main():
    nrows = 5
    ncols = 1000

    x = torch.ones((nrows, ncols), device="cuda", dtype=torch.float32)
    out_triton = rowwise_sum(x)
    out_torch = torch.sum(x, dim=1)

    max_error = torch.max(torch.abs(out_triton - out_torch)).item()
    print(f"Max error: {max_error:.6f}")

    assert torch.allclose(out_triton, out_torch, atol=1e-6)
    print(f"Correctness: PASS")

if __name__ == "__main__":
    main()

import torch
import triton 
import triton.language as tl

@triton.jit
def rowwise_softmax_kernel(
    x_ptr,
    out_ptr,
    nrows,
    ncols,
    BLOCK_SIZE: tl.constexpr,      
):
    row_idx = tl.program_id(axis=0)

    row_start = row_idx * ncols
    col_indices = tl.arange(0, BLOCK_SIZE)
    col_offsets = row_start + col_indices

    mask = col_indices < ncols

    row_data = tl.load(x_ptr + col_offsets, mask=mask, other=float("-inf"))

    row_max = tl.max(row_data, axis=0)

    row_shifted = row_data - row_max
    row_exp = tl.exp(row_shifted)
    row_exp = tl.where(mask, row_exp, 0.0)

    row_exp_sum = tl.sum(row_exp, axis=0)
    row_softmax = row_exp / row_exp_sum

    tl.store(out_ptr + col_offsets, row_softmax, mask=mask)

def rowwise_softmax(x: torch.tensor, block_size: int = 1024) -> torch.tensor:
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.ndim == 2, "Input must be 2D"
    assert x.is_contiguous(), "Input must be contiguous"

    nrows, ncols = x.shape
    assert ncols > 0, \
        f"For this simple version, block_size must be > 0"
    assert ncols <= block_size, \
        f"For this simple version, block_size ({block_size}) must be >= N ({ncols})"
    assert block_size & (block_size - 1) == 0, \
        f"block_size ({block_size}) must be the power of 2"
    
    out = torch.empty_like(x, device="cuda", dtype=torch.float32)

    grid = (nrows, )
    rowwise_softmax_kernel[grid](
        x, 
        out, 
        nrows, 
        ncols, 
        BLOCK_SIZE = block_size
    )

    return out


def main():
    nrows = 5
    ncols = 1000

    x = torch.randn((nrows, ncols), device="cuda", dtype=torch.float32)
    out_triton = rowwise_softmax(x)
    out_torch = torch.nn.functional.softmax(x, dim=1)

    max_error = torch.max(torch.abs(out_triton - out_torch)).item()
    print(f"Max error: {max_error:.6f}")

    assert torch.allclose(out_triton, out_torch, atol=1e-6)
    print(f"Correctness: PASS")

if __name__ == "__main__":
    main()
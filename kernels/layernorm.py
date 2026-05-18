import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    nrows,
    ncols,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)

    row_start = row_idx * ncols
    col_indices = tl.arange(0, BLOCK_SIZE)
    col_offsets = row_start + col_indices

    mask = col_indices < ncols

    # Load x[row, :]
    row_data = tl.load(x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Load gamma and beta
    gamma = tl.load(gamma_ptr + col_indices, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + col_indices, mask=mask, other=0.0).to(tl.float32)

    # Compute mean
    mean = tl.sum(row_data, axis=0) / ncols

    # Compute variance
    row_centered = row_data - mean
    row_centered = tl.where(mask, row_centered, 0.0)
    var = tl.sum(row_centered * row_centered, axis=0) / ncols

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    y = row_centered * rstd * gamma + beta

    # Store
    tl.store(y_ptr + col_offsets, y, mask=mask)

def next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def layernorm(
    x: torch.tensor,
    gamma: torch.tensor,
    beta: torch.tensor,
    eps: float = 1e-5,
) -> torch.tensor:
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert gamma.is_cuda, "Gamma must be a CUDA tensor"
    assert beta.is_cuda, "Beta must be a CUDA tensor"

    assert x.ndim == 2, "Input must be a 2D tensor"
    assert x.is_contiguous(), "Input must be contiguous"

    nrows, ncols = x.shape
    assert gamma.shape == (ncols, )
    assert beta.shape == (ncols, )

    block_size = next_power_of_2(ncols)
    grid = (nrows, )

    y = torch.empty_like(x, device="cuda")
    layernorm_kernel[grid](
        x,
        gamma,
        beta,
        y,
        nrows,
        ncols,
        eps,
        BLOCK_SIZE=block_size
    )

    return y

def main():
    nrows = 1024
    ncols = 768
    eps = 1e-5

    x = torch.randn((nrows, ncols), device="cuda", dtype=torch.float32)
    gamma = torch.randn((ncols, ), device="cuda", dtype=torch.float32)
    beta = torch.randn((ncols, ), device="cuda", dtype=torch.float32)

    out_triton = layernorm(x, gamma, beta, eps)
    
    out_torch = torch.nn.functional.layer_norm(
        x, 
        normalized_shape=(ncols,), 
        weight=gamma, 
        bias=beta, 
        eps=eps
    )

    max_error = torch.max(torch.abs(out_triton - out_torch)).item()
    print(f"Max error: {max_error:.6f}")

    torch.testing.assert_close(out_triton, out_torch, rtol=1e-4, atol=1e-4)
    print(f"Correctness: PASS")

if __name__ == "__main__":
    main()
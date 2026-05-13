import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    out = x + y

    tl.store(out_ptr + offsets, out, mask=mask)

def vector_add(x: torch.tensor, y: torch.tensor, block_size: int = 1024):
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    assert x.is_contiguous and y.is_contiguous

    out = torch.empty_like(x)
    n_elements = x.numel()

    grid = (triton.cdiv(n_elements, block_size),)

    vector_add_kernel[grid](
        x, 
        y, 
        out,
        n_elements,
        BLOCK_SIZE = block_size    
    )

    return out

def main():
    n_elements = 10000

    x = torch.ones(n_elements, device='cuda', dtype=torch.float32)
    y = torch.ones(n_elements, device='cuda', dtype=torch.float32) * 2.0

    out_triton = vector_add(x, y)
    out_torch = x + y

    max_error = torch.max(torch.abs(out_triton - out_torch)).item()
    print(f"Max error: {max_error:.6f}")

    assert torch.allclose(out_triton, out_torch, atol=1e-6)
    print(f"Correctness: PASS")

if __name__ == "__main__":
    main()
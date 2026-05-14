# Benchmark Results

This file records benchmark results for the kernels in this project.

## Vector Add

Benchmark command:

```bash
python benchmarks/bench_vector_add.py
```

Input:

| Parameter | Value |
| --- | ---: |
| N | 16,777,216 |
| dtype | float32 |
| bytes moved | 201,326,592 |

Results:

| Implementation | Time (ms) | Bandwidth (GB/s) |
| --- | ---: | ---: |
| Triton | 0.8200 | 245.53 |
| Torch | 0.8203 | 245.42 |


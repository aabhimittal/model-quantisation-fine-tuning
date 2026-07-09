"""Compare quantization schemes on a realistic weight matrix.

Run: ``python examples/02_quantize_linear.py``

Real weight matrices are mostly small values with a few large outliers. This
example shows why that matters: a single global scale collapses under the
outliers at 4 bits, block-wise scaling rescues most of the accuracy, and NF4
squeezes out a bit more by matching the normal distribution of the weights --
all at essentially the same bits-per-weight.
"""

import numpy as np

from quantune.quantization import compare_schemes

rng = np.random.default_rng(0)
w = rng.normal(0, 1, size=(512, 512))
w.reshape(-1)[:: w.size // 30] *= 10.0  # inject sparse heavy outliers

results = compare_schemes(w, block_size=64)

print(f"{'scheme':<16}{'rel_error':>12}{'max_abs_err':>14}{'bits/weight':>13}")
print("-" * 55)
for name, m in results.items():
    print(f"{name:<16}{m['rel_frobenius']:>12.4f}{m['max_abs_error']:>14.4f}"
          f"{m['bits_per_weight']:>13.2f}")

print("\nTakeaway: at ~4 bits, blockwise and NF4 give a >4x smaller error than a "
      "\nsingle global int4 scale -- this is the trick that makes QLoRA viable.")

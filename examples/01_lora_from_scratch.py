"""Train a LoRA adapter from scratch and watch it converge.

Run: ``python examples/01_lora_from_scratch.py``

We freeze a random "pretrained" weight ``W0`` and give the model a target that
differs from ``W0`` by a genuine low-rank delta. LoRA only trains the small
``A``/``B`` factors, yet recovers the delta -- demonstrating the core claim that
fine-tuning updates have low intrinsic rank.
"""

import numpy as np

from quantune.lora import LoRAConfig, LoRALinear

rng = np.random.default_rng(0)
d_out, d_in, r = 64, 48, 4

W0 = rng.normal(0, 1, size=(d_out, d_in))          # frozen "pretrained" weights
A_true = rng.normal(0, 1, size=(r, d_in))
B_true = rng.normal(0, 1, size=(d_out, r))
W_target = W0 + B_true @ A_true                    # target = base + rank-r delta

X = rng.normal(0, 1, size=(512, d_in))
Y = X @ W_target.T

layer = LoRALinear(W0, LoRAConfig(r=r, alpha=r, seed=1))  # scaling = alpha/r = 1
print(f"Base params (frozen):     {layer.num_base_params():,}")
print(f"Trainable LoRA params:    {layer.num_trainable_params():,}")
print(f"Compression ratio:        {layer.compression_ratio():.1f}x fewer params to train\n")

history = layer.fit(X, Y, lr=0.005, steps=3000)
print(f"Loss:  start={history[0]:.4f}  ->  end={history[-1]:.6f}")

merged = layer.merged_weight()
print(f"Recovered weight error:   {np.linalg.norm(merged - W_target):.6f}")
print("The frozen W0 never moved:", np.array_equal(layer.W0, W0))

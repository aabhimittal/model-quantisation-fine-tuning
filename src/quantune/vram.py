"""A first-principles VRAM estimator for fine-tuning.

The single most common question -- "will this fit on my GPU?" -- has a
surprisingly mechanical answer. Training memory is the sum of four buckets:

* **weights**       -- the model parameters themselves.
* **gradients**     -- one per *trainable* parameter.
* **optimizer state** -- Adam keeps two moments per trainable parameter, plus
  (in mixed precision) an fp32 master copy of them.
* **activations**   -- intermediate tensors saved for the backward pass, which
  scale with batch size and sequence length, not parameter count.

The punchline of LoRA/QLoRA falls straight out of the arithmetic: freezing the
base weights zeros out the gradient and optimizer buckets for 99%+ of
parameters, and quantizing the frozen weights to 4 bits shrinks the largest
remaining bucket. That is *why* a 7B model that needs ~60 GB for full
fine-tuning drops under the 24 GB of a single consumer GPU with QLoRA.

All numbers are estimates in GB (1 GB = 1e9 bytes) and deliberately
approximate; they are meant to guide hardware decisions, not to be exact.
"""

from __future__ import annotations

from dataclasses import dataclass

BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "nf4": 0.5, "int4": 0.5}


@dataclass
class ModelSpec:
    """Rough architectural description of a decoder-only transformer."""

    params_billion: float
    hidden_size: int = 4096
    num_layers: int = 32
    seq_len: int = 2048
    batch_size: int = 1

    @property
    def num_params(self) -> float:
        return self.params_billion * 1e9


def _activation_gb(spec: ModelSpec, dtype: str = "fp16", checkpointing: bool = True) -> float:
    """Very rough activation-memory estimate.

    With gradient checkpointing you store roughly one hidden state per layer;
    without it you store the handful of large intermediates inside each layer.
    We use a small multiplier to stand in for that per-layer footprint.
    """
    per_token = spec.hidden_size * spec.num_layers
    multiplier = 2 if checkpointing else 16
    elements = spec.batch_size * spec.seq_len * per_token * multiplier
    return elements * BYTES[dtype] / 1e9


def estimate(
    spec: ModelSpec,
    method: str = "full",
    *,
    weight_dtype: str | None = None,
    compute_dtype: str = "fp16",
    optimizer: str = "adam",
    lora_trainable_fraction: float = 0.01,
    gradient_checkpointing: bool = True,
) -> dict:
    """Estimate the training memory breakdown for a fine-tuning ``method``.

    ``method`` is one of ``"full"``, ``"lora"`` or ``"qlora"``.

    * ``full``  -- all weights in ``compute_dtype``, everything trainable.
    * ``lora``  -- weights in fp16, only ``lora_trainable_fraction`` trainable.
    * ``qlora`` -- frozen weights in NF4, only the adapter trainable.

    Returns a dict of the four buckets plus ``total_gb``.
    """
    method = method.lower()
    if method == "full":
        weight_dtype = weight_dtype or compute_dtype
        trainable = spec.num_params
    elif method == "lora":
        weight_dtype = weight_dtype or "fp16"
        trainable = spec.num_params * lora_trainable_fraction
    elif method == "qlora":
        weight_dtype = weight_dtype or "nf4"
        trainable = spec.num_params * lora_trainable_fraction
    else:
        raise ValueError(f"unknown method: {method!r} (use full/lora/qlora)")

    weights_gb = spec.num_params * BYTES[weight_dtype] / 1e9
    # Gradients are stored in the compute dtype, one per trainable parameter.
    grads_gb = trainable * BYTES[compute_dtype] / 1e9
    # Adam: two moments in fp32 (8 bytes) + fp32 master weights (4 bytes) per
    # trainable parameter. SGD-with-momentum keeps a single fp32 buffer.
    per_param_opt = 12.0 if optimizer == "adam" else 4.0
    optim_gb = trainable * per_param_opt / 1e9
    activations_gb = _activation_gb(spec, compute_dtype, gradient_checkpointing)

    total = weights_gb + grads_gb + optim_gb + activations_gb
    return {
        "method": method,
        "weight_dtype": weight_dtype,
        "weights_gb": round(weights_gb, 3),
        "gradients_gb": round(grads_gb, 3),
        "optimizer_gb": round(optim_gb, 3),
        "activations_gb": round(activations_gb, 3),
        "total_gb": round(total, 3),
    }


def compare_methods(spec: ModelSpec, **kwargs) -> dict:
    """Estimate all three methods for the same model -- the headline table."""
    return {m: estimate(spec, m, **kwargs) for m in ("full", "lora", "qlora")}


def fits_on(total_gb: float, gpu_vram_gb: float, safety: float = 0.9) -> bool:
    """Whether an estimated footprint fits, leaving a safety margin for fragmentation."""
    return total_gb <= gpu_vram_gb * safety

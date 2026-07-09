"""Low-Rank Adaptation (LoRA), implemented from scratch in NumPy.

LoRA (Hu et al., 2021) is the idea that the *update* a model needs during
fine-tuning has low intrinsic rank. Instead of learning a full ``d_out x d_in``
weight delta, you learn two small matrices ``A`` (r x d_in) and ``B``
(d_out x r) and use ``B @ A`` as the delta, scaled by ``alpha / r``. The base
weight ``W0`` is frozen; only ``A`` and ``B`` train, cutting trainable
parameters by orders of magnitude.

This module implements a :class:`LoRALinear` layer *and its gradients* by hand,
so you can watch a real (if tiny) LoRA fit converge without a deep-learning
framework. QLoRA = this exact adapter, but with ``W0`` stored in NF4 (see
:mod:`quantune.quantization`) instead of fp16.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoRAConfig:
    r: int = 4
    alpha: float = 8.0
    seed: int = 0


class LoRALinear:
    """A frozen linear layer ``y = x W0^T`` plus a trainable low-rank delta.

    Forward pass: ``y = x @ W0.T + (alpha / r) * (x @ A.T) @ B.T``.

    Only ``A`` and ``B`` are trainable. Following the paper, ``A`` is
    initialised with small Gaussian noise and ``B`` with zeros, so the adapter
    starts as a no-op (the delta is exactly zero) and fine-tuning begins from
    the pretrained behaviour.
    """

    def __init__(self, base_weight: np.ndarray, config: LoRAConfig | None = None):
        self.W0 = np.asarray(base_weight, dtype=np.float64)  # (d_out, d_in), frozen
        self.d_out, self.d_in = self.W0.shape
        self.cfg = config or LoRAConfig()
        rng = np.random.default_rng(self.cfg.seed)
        self.A = rng.normal(0.0, 0.01, size=(self.cfg.r, self.d_in))  # (r, d_in)
        self.B = np.zeros((self.d_out, self.cfg.r))                   # (d_out, r)
        self.scaling = self.cfg.alpha / self.cfg.r

    # -- inference ---------------------------------------------------------- #

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        base = x @ self.W0.T
        delta = (x @ self.A.T) @ self.B.T * self.scaling
        return base + delta

    __call__ = forward

    def merged_weight(self) -> np.ndarray:
        """Fold the adapter into a single dense weight for zero-overhead serving.

        ``W = W0 + (alpha / r) * B @ A``. After merging, a LoRA model costs
        exactly the same at inference time as the original -- one of LoRA's key
        practical advantages over adapters that add extra layers.
        """
        return self.W0 + self.scaling * (self.B @ self.A)

    # -- parameter accounting ---------------------------------------------- #

    def num_base_params(self) -> int:
        return self.d_out * self.d_in

    def num_trainable_params(self) -> int:
        return self.A.size + self.B.size

    def compression_ratio(self) -> float:
        """How many fewer parameters LoRA trains vs full fine-tuning."""
        return self.num_base_params() / self.num_trainable_params()

    # -- training (hand-written gradients) ---------------------------------- #

    def fit(self, x: np.ndarray, y: np.ndarray, *, lr: float = 0.05,
            steps: int = 500) -> list[float]:
        """Fit the adapter to targets ``y`` with plain SGD on ``A`` and ``B``.

        Minimises mean-squared error. ``W0`` never moves -- only the low-rank
        factors do -- which is the whole point of LoRA. Returns the loss history
        so callers/tests can confirm it actually converges.

        Gradient derivation (MSE ``L = mean((pred - y)^2)``, ``s`` = scaling)::

            H          = x @ A.T                      # (n, r)
            pred       = x @ W0.T + s * H @ B.T
            d          = pred - y                     # (n, d_out)
            dL/dB      = (2s / n) * d.T @ H
            dL/dH      = (2s / n) * d @ B
            dL/dA      = (dL/dH).T @ x
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = x.shape[0]
        base = x @ self.W0.T  # constant across steps; W0 is frozen
        history: list[float] = []
        for _ in range(steps):
            H = x @ self.A.T                       # (n, r)
            pred = base + self.scaling * (H @ self.B.T)
            d = pred - y                           # (n, d_out)
            history.append(float(np.mean(d ** 2)))
            grad_B = (2.0 * self.scaling / n) * (d.T @ H)      # (d_out, r)
            grad_H = (2.0 * self.scaling / n) * (d @ self.B)   # (n, r)
            grad_A = grad_H.T @ x                              # (r, d_in)
            self.B -= lr * grad_B
            self.A -= lr * grad_A
        return history

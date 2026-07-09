"""Glass-box quantization primitives, implemented from scratch in NumPy.

The goal of this module is *pedagogical transparency*. Production libraries
(bitsandbytes, GPTQ, AWQ) hide the arithmetic behind CUDA kernels. Here every
step is plain NumPy you can read in one sitting, so the mechanics of shrinking a
model from 16-bit floats down to 4-bit integers are fully visible.

Four schemes are provided, in increasing order of sophistication:

1. ``quantize_int8_symmetric`` -- the textbook starting point.
2. ``quantize_int8_affine``    -- asymmetric (zero-point) quantization.
3. ``quantize_blockwise``      -- per-block scales, the trick that makes low-bit
   quantization actually usable on real weight matrices with outliers.
4. ``quantize_nf4``            -- NormalFloat4, the 4-bit datatype introduced by
   QLoRA that assumes weights are roughly normally distributed.

Every function returns a :class:`QuantizedTensor` that can be dequantized back
to floating point, plus helpers to measure the error you paid for the savings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# The 16 NF4 code points from the QLoRA paper (Dettmers et al., 2023).
# They are the quantiles of a standard normal distribution, normalised so the
# codebook spans exactly [-1, 1]. Because weights in trained networks are
# approximately normally distributed, spacing the levels this way puts more
# resolution where the mass is -- near zero.
NF4_LEVELS = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float64,
)


@dataclass
class QuantizedTensor:
    """A tensor stored in low precision plus the metadata to reconstruct it.

    ``codes`` holds the integer indices/values actually stored on the "device".
    ``scale`` (and optional ``zero_point``) map those codes back to floats.
    ``scheme`` records how it was produced so :func:`dequantize` knows the rule.
    """

    codes: np.ndarray
    scale: np.ndarray
    scheme: str
    shape: tuple
    zero_point: Optional[np.ndarray] = None
    block_size: Optional[int] = None
    bits: int = 8
    codebook: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def stored_bits(self) -> int:
        """Total number of bits used to store the quantized payload.

        Counts the code array plus the per-block scale metadata. This is what
        makes the memory savings *honest*: a 4-bit scheme with tiny blocks is
        not really 4 bits per weight once you account for the scales.
        """
        n = int(np.prod(self.shape))
        code_bits = n * self.bits
        # Scales are kept in fp16 in real kernels; count them the same way.
        scale_bits = int(np.size(self.scale)) * 16
        zp_bits = 0 if self.zero_point is None else int(np.size(self.zero_point)) * 16
        return code_bits + scale_bits + zp_bits

    @property
    def bits_per_weight(self) -> float:
        return self.stored_bits / max(1, int(np.prod(self.shape)))


def dequantize(qt: QuantizedTensor) -> np.ndarray:
    """Reconstruct a float array from a :class:`QuantizedTensor`."""
    if qt.scheme == "int8_symmetric" or qt.scheme == "int4_symmetric":
        return (qt.codes.astype(np.float64) * qt.scale).reshape(qt.shape)
    if qt.scheme == "int8_affine":
        return ((qt.codes.astype(np.float64) - qt.zero_point) * qt.scale).reshape(qt.shape)
    if qt.scheme == "blockwise":
        blocks = qt.codes.astype(np.float64) * qt.scale[:, None]
        return blocks.reshape(-1)[: int(np.prod(qt.shape))].reshape(qt.shape)
    if qt.scheme == "nf4":
        values = qt.codebook[qt.codes] * qt.scale[:, None]
        return values.reshape(-1)[: int(np.prod(qt.shape))].reshape(qt.shape)
    raise ValueError(f"unknown scheme: {qt.scheme}")


def quantize_int8_symmetric(w: np.ndarray) -> QuantizedTensor:
    """Symmetric int8: a single scale, zero maps to zero.

    ``scale = max(|w|) / 127`` and ``q = round(w / scale)`` clipped to
    ``[-127, 127]``. Simple and fast, but a single global scale is wrecked by
    outliers -- one large weight forces a coarse scale for everyone else.
    """
    w = np.asarray(w, dtype=np.float64)
    amax = np.max(np.abs(w))
    scale = amax / 127.0 if amax > 0 else 1.0
    codes = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return QuantizedTensor(codes=codes, scale=np.float64(scale),
                           scheme="int8_symmetric", shape=w.shape, bits=8)


def quantize_int8_affine(w: np.ndarray) -> QuantizedTensor:
    """Asymmetric int8: a scale *and* a zero-point.

    Uses the full ``[0, 255]`` range by fitting ``[min(w), max(w)]`` exactly.
    Better for activations (e.g. post-ReLU) that are not centred on zero.
    """
    w = np.asarray(w, dtype=np.float64)
    wmin, wmax = float(np.min(w)), float(np.max(w))
    scale = (wmax - wmin) / 255.0 if wmax > wmin else 1.0
    zero_point = np.round(-wmin / scale)
    codes = np.clip(np.round(w / scale) + zero_point, 0, 255).astype(np.uint8)
    return QuantizedTensor(codes=codes, scale=np.float64(scale), zero_point=np.float64(zero_point),
                           scheme="int8_affine", shape=w.shape, bits=8)


def quantize_int4_symmetric(w: np.ndarray) -> QuantizedTensor:
    """Symmetric int4: 16 levels in ``[-7, 7]``, one global scale.

    Included mostly as a baseline to show how badly a *single* scale performs
    at 4 bits -- compare its error against block-wise and NF4 below.
    """
    w = np.asarray(w, dtype=np.float64)
    amax = np.max(np.abs(w))
    scale = amax / 7.0 if amax > 0 else 1.0
    codes = np.clip(np.round(w / scale), -7, 7).astype(np.int8)
    return QuantizedTensor(codes=codes, scale=np.float64(scale),
                           scheme="int4_symmetric", shape=w.shape, bits=4)


def _reshape_into_blocks(w: np.ndarray, block_size: int):
    flat = np.asarray(w, dtype=np.float64).reshape(-1)
    pad = (-len(flat)) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float64)])
    return flat.reshape(-1, block_size)


def quantize_blockwise(w: np.ndarray, block_size: int = 64, bits: int = 4) -> QuantizedTensor:
    """Per-block symmetric quantization -- the workhorse of low-bit inference.

    The weight tensor is flattened and chopped into contiguous blocks of
    ``block_size`` elements. Each block gets its *own* absmax scale, so a large
    outlier only degrades the 64 weights it shares a block with, not the whole
    matrix. This is the single most important idea for making 4-bit weights work.
    """
    blocks = _reshape_into_blocks(w, block_size)
    qmax = (1 << (bits - 1)) - 1  # e.g. 7 for 4-bit, 127 for 8-bit
    absmax = np.max(np.abs(blocks), axis=1)
    scale = np.where(absmax > 0, absmax / qmax, 1.0)
    codes = np.clip(np.round(blocks / scale[:, None]), -qmax, qmax).astype(np.int8)
    return QuantizedTensor(codes=codes, scale=scale, scheme="blockwise",
                           shape=np.asarray(w).shape, bits=bits, block_size=block_size)


def quantize_nf4(w: np.ndarray, block_size: int = 64) -> QuantizedTensor:
    """NormalFloat4 (NF4) quantization -- the datatype behind QLoRA.

    Each block is normalised by its absmax into ``[-1, 1]`` and every value is
    snapped to the nearest of the 16 :data:`NF4_LEVELS`. Because the levels are
    normal-distribution quantiles rather than evenly spaced integers, NF4 is
    "information-theoretically optimal" for normally distributed weights and
    beats plain 4-bit integers at the same bit budget.
    """
    blocks = _reshape_into_blocks(w, block_size)
    absmax = np.max(np.abs(blocks), axis=1)
    scale = np.where(absmax > 0, absmax, 1.0)
    normed = blocks / scale[:, None]
    # Nearest-neighbour assignment onto the codebook (vectorised argmin).
    dists = np.abs(normed[:, :, None] - NF4_LEVELS[None, None, :])
    codes = np.argmin(dists, axis=2).astype(np.uint8)
    return QuantizedTensor(codes=codes, scale=scale, scheme="nf4",
                           shape=np.asarray(w).shape, bits=4, block_size=block_size,
                           codebook=NF4_LEVELS)


# --------------------------------------------------------------------------- #
# Error metrics -- how much accuracy did the memory savings cost?
# --------------------------------------------------------------------------- #

def quantization_error(original: np.ndarray, qt: QuantizedTensor) -> dict:
    """Return MSE, max abs error, and relative Frobenius error of a scheme."""
    original = np.asarray(original, dtype=np.float64)
    recon = dequantize(qt)
    diff = original - recon
    denom = np.linalg.norm(original) or 1.0
    return {
        "mse": float(np.mean(diff ** 2)),
        "max_abs_error": float(np.max(np.abs(diff))),
        "rel_frobenius": float(np.linalg.norm(diff) / denom),
        "bits_per_weight": qt.bits_per_weight,
    }


def compare_schemes(w: np.ndarray, block_size: int = 64) -> dict:
    """Quantize ``w`` every way and return a comparison table.

    Handy for the "why bother with NF4?" demonstration: run it on a realistic
    weight matrix and watch the relative error fall as the scheme gets smarter,
    even though bits-per-weight stays roughly constant.
    """
    schemes = {
        "int8_symmetric": quantize_int8_symmetric(w),
        "int8_affine": quantize_int8_affine(w),
        "int4_symmetric": quantize_int4_symmetric(w),
        "blockwise_int4": quantize_blockwise(w, block_size=block_size, bits=4),
        "nf4": quantize_nf4(w, block_size=block_size),
    }
    return {name: quantization_error(w, qt) for name, qt in schemes.items()}

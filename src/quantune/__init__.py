"""quantune -- a glass-box, decision-first toolkit for fine-tuning & quantization.

Public API:

* :mod:`quantune.advisor`      -- decide *whether/how* to fine-tune.
* :mod:`quantune.vram`         -- estimate GPU memory for each method.
* :mod:`quantune.lora`         -- from-scratch LoRA adapter with training.
* :mod:`quantune.quantization` -- from-scratch int8/int4/block-wise/NF4 quantizers.
"""

from .advisor import Recommendation, Scenario, advise
from .lora import LoRAConfig, LoRALinear
from .quantization import (
    NF4_LEVELS,
    QuantizedTensor,
    compare_schemes,
    dequantize,
    quantization_error,
    quantize_blockwise,
    quantize_int4_symmetric,
    quantize_int8_affine,
    quantize_int8_symmetric,
    quantize_nf4,
)
from .vram import ModelSpec, compare_methods, estimate, fits_on

__version__ = "0.1.0"

__all__ = [
    "advise",
    "Scenario",
    "Recommendation",
    "LoRALinear",
    "LoRAConfig",
    "ModelSpec",
    "estimate",
    "compare_methods",
    "fits_on",
    "QuantizedTensor",
    "NF4_LEVELS",
    "dequantize",
    "quantization_error",
    "compare_schemes",
    "quantize_int8_symmetric",
    "quantize_int8_affine",
    "quantize_int4_symmetric",
    "quantize_blockwise",
    "quantize_nf4",
    "__version__",
]

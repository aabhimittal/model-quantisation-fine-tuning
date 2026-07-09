"""quantune -- a glass-box, decision-first toolkit for fine-tuning & quantization.

Public API:

* :mod:`quantune.advisor`      -- decide *whether/how* to fine-tune.
* :mod:`quantune.vram`         -- estimate GPU memory for each method.
* :mod:`quantune.lora`         -- from-scratch LoRA adapter with training.
* :mod:`quantune.quantization` -- from-scratch int8/int4/block-wise/NF4 quantizers.
* :mod:`quantune.deploy`       -- decide *where* to serve (NIM/vLLM/TGI/Bedrock).
* :mod:`quantune.serving`      -- actually generate text on GPU cloud (OpenAI wire).
"""

from .advisor import Recommendation, Scenario, advise
from .deploy import (
    DeploymentPlan,
    DeploymentScenario,
    advise_deployment,
    render_config,
)
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
from .serving import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    GROUNDING_SYSTEM_PROMPT,
    GenerationResult,
    OpenAICompatClient,
    ServingError,
    groundedness,
)
from .vram import ModelSpec, compare_methods, estimate, fits_on, serving_vram

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
    "serving_vram",
    "OpenAICompatClient",
    "GenerationResult",
    "ServingError",
    "groundedness",
    "GROUNDING_SYSTEM_PROMPT",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DeploymentScenario",
    "DeploymentPlan",
    "advise_deployment",
    "render_config",
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

"""A glass-box advisor for the *last* decision: where do I actually serve this?

:mod:`quantune.advisor` stops at "serve it in fp16/int8/nf4". This module picks up
there and answers the question the README's GPU-cloud brief is really asking --
**which serving stack, and how do I launch it?** -- with the same transparent,
reasoned rules the training advisor uses.

The four backends it chooses between are the ones practitioners actually reach for:

* **NVIDIA NIM** (build.nvidia.com) -- hosted, GPU-cloud, free key, *no GPU of your
  own*. The default when you just want fast tokens now.
* **vLLM** -- self-hosted, highest throughput via paged-attention/continuous
  batching. The pick when you own GPUs and care about tokens/sec and cost.
* **Hugging Face TGI** -- self-hosted, tight Hub integration, production-friendly.
* **AWS Bedrock** -- fully managed, no servers, pay-per-token; natural inside AWS.

Crucially, NIM/vLLM/TGI all speak the OpenAI wire format, so whatever this advisor
recommends, :class:`quantune.serving.OpenAICompatClient` can actually call it --
you only swap a ``base_url``. :func:`render_config` prints the exact launch artifact
for each backend so the recommendation is copy-paste runnable (generated, never
executed here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .serving import DEFAULT_BASE_URL, DEFAULT_MODEL
from .vram import ModelSpec, serving_vram

BACKENDS = ("nvidia_nim", "vllm", "tgi", "bedrock")


@dataclass
class DeploymentScenario:
    """What a practitioner knows when deciding where to serve."""

    model_params_b: float = 8.0
    has_own_gpu: bool = False       # do you have GPUs to run a server on?
    gpu_vram_gb: float = 24.0       # per-GPU budget, if you have one
    task: str = "style"             # "style" | "format" | "knowledge" | "reasoning"
    latency_sensitive: bool = False
    budget_sensitive: bool = False
    wants_managed: bool = False     # prefer "no servers to babysit"
    in_aws: bool = False            # already on AWS


@dataclass
class DeploymentPlan:
    backend: str                    # one of BACKENDS
    serving_dtype: str              # "fp16" | "int8" | "nf4"
    est_serving_vram_gb: float
    gpus_needed: int                # 0 for fully hosted/managed backends
    grounded: bool = False          # answer only from retrieved sources (anti-hallucination)
    self_consistency: int = 1       # sample N times and vote; 1 = single greedy answer
    reasons: List[str] = field(default_factory=list)
    launch_config: str = ""

    def summary(self) -> str:
        pretty = {
            "nvidia_nim": "NVIDIA NIM (hosted GPU cloud)",
            "vllm": "vLLM (self-hosted)",
            "tgi": "Hugging Face TGI (self-hosted)",
            "bedrock": "AWS Bedrock (managed)",
        }[self.backend]
        gpu_line = (
            f"GPUs:      {self.gpus_needed} x self-hosted"
            if self.gpus_needed
            else "GPUs:      none of your own (provider-hosted)"
        )
        grounding_line = (
            "grounded (answer only from retrieved sources, cite, abstain)"
            if self.grounded
            else "off (model answers from its own weights)"
        )
        sc_line = (
            f"{self.self_consistency}x sample-and-vote (self-consistency)"
            if self.self_consistency > 1
            else "single answer"
        )
        lines = [
            f"Backend:   {pretty}",
            f"Serving:   {self.serving_dtype}",
            f"Serve VRAM:{self.est_serving_vram_gb} GB (weights + KV cache)",
            gpu_line,
            f"Grounding: {grounding_line}",
            f"Decoding:  {sc_line}",
            "Why:",
        ]
        lines += [f"  - {r}" for r in self.reasons]
        if self.launch_config:
            lines += ["", "Launch it:", self.launch_config]
        return "\n".join(lines)


def _pick_serving_dtype(scenario: DeploymentScenario, reasons: List[str]) -> str:
    """Same spirit as the training advisor's serving choice, restated for clarity."""
    if scenario.latency_sensitive:
        reasons.append(
            "Latency-sensitive: int8 weights roughly halve memory bandwidth with "
            "negligible quality loss for most models."
        )
        return "int8"
    if scenario.model_params_b >= 13:
        reasons.append(
            "Large model: 4-bit (NF4) weights cut the footprint ~4x so it fits on "
            "cheaper/fewer GPUs."
        )
        return "nf4"
    reasons.append("No tight memory/latency pressure: keep fp16 for maximum quality.")
    return "fp16"


def advise_deployment(scenario: DeploymentScenario) -> DeploymentPlan:
    """Return a fully-reasoned :class:`DeploymentPlan` for a serving scenario."""
    reasons: List[str] = []
    dtype = _pick_serving_dtype(scenario, reasons)
    spec = ModelSpec(params_billion=scenario.model_params_b)
    est_vram = serving_vram(spec, dtype)

    # -- Decision: which backend? ----------------------------------------- #
    if not scenario.has_own_gpu:
        if scenario.in_aws or scenario.wants_managed:
            backend = "bedrock"
            reasons.append(
                "No GPUs of your own and you want a managed, serverless path -- AWS "
                "Bedrock serves foundation models pay-per-token with nothing to run."
            )
            gpus = 0
        else:
            backend = "nvidia_nim"
            reasons.append(
                "No GPUs of your own -- NVIDIA NIM hosts the model on its cloud GPUs "
                "behind a free, OpenAI-compatible endpoint, so you can serve today "
                "with zero infrastructure."
            )
            gpus = 0
    else:
        gpus = max(1, math.ceil(est_vram / (scenario.gpu_vram_gb * 0.9)))
        if scenario.wants_managed:
            backend = "bedrock"
            reasons.append(
                "You have GPUs but prefer no ops -- Bedrock removes the server "
                "management entirely (at the cost of pay-per-token pricing)."
            )
            gpus = 0
        elif scenario.budget_sensitive or scenario.latency_sensitive:
            backend = "vllm"
            reasons.append(
                "You own GPUs and care about throughput/cost -- vLLM's paged-attention "
                "and continuous batching give the best tokens/sec per GPU-dollar."
            )
        else:
            backend = "tgi"
            reasons.append(
                "You own GPUs and want a production server that's tightly integrated "
                "with the Hugging Face Hub -- TGI is the straightforward choice."
            )
        if gpus > 1:
            reasons.append(
                f"~{est_vram} GB in {dtype} exceeds one {scenario.gpu_vram_gb} GB GPU, so "
                f"plan for ~{gpus} GPUs (tensor-parallel) or a smaller/quantized model."
            )

    # -- Anti-hallucination: how should we generate for this task? --------- #
    grounded, self_consistency = _pick_generation_strategy(scenario.task, reasons)

    plan = DeploymentPlan(
        backend=backend,
        serving_dtype=dtype,
        est_serving_vram_gb=est_vram,
        gpus_needed=gpus,
        grounded=grounded,
        self_consistency=self_consistency,
        reasons=reasons,
    )
    plan.launch_config = render_config(backend, model=_default_model_for(backend), dtype=dtype, gpus=max(1, gpus))
    return plan


def _pick_generation_strategy(task: str, reasons: List[str]) -> tuple[bool, int]:
    """Decide grounding and self-consistency from the task -- the anti-hallucination knobs.

    Mirrors the training advisor's "RAG for facts" rule at serving time: fact-heavy
    tasks get grounded generation so answers come from sources, not weights;
    multi-step reasoning gets self-consistency (sample several times and vote) since
    a single greedy chain is the easiest thing for a model to get confidently wrong.
    """
    if task == "knowledge":
        reasons.append(
            "Knowledge/fact task: serve with grounded generation -- answer only from "
            "retrieved sources (cite, and abstain when they don't cover it) so facts "
            "come from context, not hallucinated from the weights."
        )
        return True, 1
    if task == "reasoning":
        reasons.append(
            "Reasoning task: use self-consistency -- sample the answer ~5x and take the "
            "majority. Voting over independent chains cancels one-off reasoning slips a "
            "single greedy pass would commit to."
        )
        return False, 5
    reasons.append(
        "Behaviour/format/style task: a single grounded-optional answer is fine; no "
        "extra fact-checking needed at serving time."
    )
    return False, 1


def _default_model_for(backend: str) -> str:
    if backend == "bedrock":
        return "meta.llama3-1-8b-instruct-v1:0"
    return DEFAULT_MODEL


def render_config(backend: str, *, model: str = DEFAULT_MODEL, dtype: str = "fp16", gpus: int = 1) -> str:
    """Return a copy-paste launch artifact for ``backend``. Generated, not run."""
    backend = backend.lower()
    if backend == "nvidia_nim":
        return (
            "# Hosted GPU cloud -- no server to run. Get a free key at build.nvidia.com,\n"
            "# then either use quantune's client or the OpenAI SDK:\n"
            "export NVIDIA_API_KEY=nvapi-...\n"
            f'quantune serve --prompt "Hello!" --model {model} --stream\n'
            "# or, equivalently, with the OpenAI SDK:\n"
            "#   from openai import OpenAI\n"
            f'#   client = OpenAI(base_url="{DEFAULT_BASE_URL}", api_key=os.environ["NVIDIA_API_KEY"])\n'
            f'#   client.chat.completions.create(model="{model}", messages=[...])'
        )
    if backend == "vllm":
        tp = f" --tensor-parallel-size {gpus}" if gpus > 1 else ""
        quant = " --quantization bitsandbytes" if dtype in ("int8", "nf4") else ""
        return (
            "# Self-hosted, OpenAI-compatible server on your own GPU(s):\n"
            "pip install vllm\n"
            f"python -m vllm.entrypoints.openai.api_server \\\n"
            f"    --model {model}{tp}{quant} \\\n"
            "    --host 0.0.0.0 --port 8000\n"
            "# then point quantune at it:\n"
            '#   quantune serve --base-url http://localhost:8000/v1 --prompt "Hello!"'
        )
    if backend == "tgi":
        shard = f" --num-shard {gpus}" if gpus > 1 else ""
        quant = " --quantize bitsandbytes" if dtype in ("int8", "nf4") else ""
        return (
            "# Hugging Face Text Generation Inference (Docker), OpenAI-compatible route:\n"
            "docker run --gpus all --shm-size 1g -p 8080:80 \\\n"
            "    ghcr.io/huggingface/text-generation-inference:latest \\\n"
            f"    --model-id {model}{shard}{quant}\n"
            "# then point quantune at it:\n"
            '#   quantune serve --base-url http://localhost:8080/v1 --prompt "Hello!"'
        )
    if backend == "bedrock":
        return (
            "# Fully managed -- no servers. Uses the AWS SDK (boto3), not OpenAI wire:\n"
            "pip install boto3\n"
            "import boto3, json\n"
            'brt = boto3.client("bedrock-runtime", region_name="us-east-1")\n'
            f'resp = brt.invoke_model(modelId="{model}",\n'
            '    body=json.dumps({"prompt": "Hello!", "max_gen_len": 256}))\n'
            'print(json.loads(resp["body"].read()))'
        )
    raise ValueError(f"unknown backend: {backend!r} (use one of {BACKENDS})")

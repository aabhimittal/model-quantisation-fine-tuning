"""A transparent decision engine for the question *before* the code.

Most fine-tuning tutorials start by assuming you should fine-tune. The more
valuable and more often-skipped question is *whether* to -- and if so, with
which method on which hardware. This module encodes that judgement as an
explicit, inspectable rule set rather than a black box: every recommendation
comes with the reasons that produced it, so you can argue with it.

The two decisions it makes:

1. **Adapt at all?**  Fine-tuning teaches *behaviour, format, and style*. It is
   the wrong tool for injecting *fresh facts* -- that is what retrieval (RAG) is
   for. With very few examples, prompting usually wins.
2. **Which method?**  Given the model size and available VRAM, pick full
   fine-tuning, LoRA, or QLoRA, and a serving quantization -- backed by the
   :mod:`quantune.vram` estimator so the answer respects your actual GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .vram import ModelSpec, estimate, fits_on


@dataclass
class Scenario:
    """The inputs a practitioner actually has on hand."""

    task: str = "style"          # "style" | "format" | "knowledge" | "reasoning"
    num_examples: int = 1000     # size of the labelled dataset
    model_params_b: float = 7.0  # base model size, in billions of params
    gpu_vram_gb: float = 24.0    # per-GPU memory budget
    latency_sensitive: bool = False
    data_changes_often: bool = False


@dataclass
class Recommendation:
    should_fine_tune: bool
    approach: str                       # "prompting" | "rag" | "fine-tuning" | "rag+fine-tuning"
    method: str                         # "n/a" | "full" | "lora" | "qlora"
    serving_quantization: str           # "fp16" | "int8" | "nf4"
    estimated_vram_gb: float
    fits_budget: bool
    reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Approach:  {self.approach}",
            f"Method:    {self.method}",
            f"Serving:   {self.serving_quantization}",
            f"Est. VRAM: {self.estimated_vram_gb} GB "
            f"({'fits' if self.fits_budget else 'DOES NOT fit'} the budget)",
            "Why:",
        ]
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def _pick_method(spec: ModelSpec, gpu_vram_gb: float, reasons: List[str]) -> tuple[str, float]:
    """Choose the cheapest method whose estimated footprint fits the GPU."""
    for method in ("full", "lora", "qlora"):
        est = estimate(spec, method)
        if fits_on(est["total_gb"], gpu_vram_gb):
            if method == "full":
                reasons.append(
                    f"Full fine-tuning fits (~{est['total_gb']} GB <= "
                    f"{gpu_vram_gb} GB), so no need to trade accuracy for memory."
                )
            else:
                reasons.append(
                    f"{method.upper()} chosen: it fits in ~{est['total_gb']} GB by "
                    f"freezing base weights"
                    + (" and storing them in 4-bit NF4." if method == "qlora"
                       else " and training a small low-rank adapter.")
                )
            return method, est["total_gb"]
    # Nothing fits: recommend QLoRA anyway and flag the shortfall.
    est = estimate(spec, "qlora")
    reasons.append(
        f"Even QLoRA needs ~{est['total_gb']} GB, above the {gpu_vram_gb} GB budget -- "
        "use a smaller base model, shorter sequences, or multi-GPU/offloading."
    )
    return "qlora", est["total_gb"]


def advise(scenario: Scenario) -> Recommendation:
    """Return a fully-reasoned recommendation for a :class:`Scenario`."""
    reasons: List[str] = []
    spec = ModelSpec(params_billion=scenario.model_params_b)

    # -- Decision 1: should we fine-tune at all? --------------------------- #
    if scenario.task == "knowledge" and scenario.data_changes_often:
        reasons.append(
            "The task is fact injection over frequently-changing data. Fine-tuning "
            "bakes facts into weights and goes stale; retrieval (RAG) keeps them "
            "fresh and citable."
        )
        return Recommendation(
            should_fine_tune=False, approach="rag", method="n/a",
            serving_quantization="int8" if scenario.latency_sensitive else "fp16",
            estimated_vram_gb=round(estimate(spec, "full")["weights_gb"], 3),
            fits_budget=True, reasons=reasons,
        )

    if scenario.num_examples < 50:
        reasons.append(
            f"Only {scenario.num_examples} examples -- too few to fine-tune reliably. "
            "Start with few-shot prompting; revisit fine-tuning once you have "
            "hundreds to thousands of clean examples."
        )
        return Recommendation(
            should_fine_tune=False, approach="prompting", method="n/a",
            serving_quantization="fp16", estimated_vram_gb=0.0,
            fits_budget=True, reasons=reasons,
        )

    if scenario.task == "knowledge":
        reasons.append(
            "Task is knowledge-heavy but data is stable, so pair retrieval (RAG) "
            "for the facts with light fine-tuning for the answer style/format."
        )
        approach = "rag+fine-tuning"
    else:
        reasons.append(
            f"Task is '{scenario.task}' (behaviour/format/style) with "
            f"{scenario.num_examples} examples -- a good fit for fine-tuning, which "
            "excels at teaching how to respond rather than new facts."
        )
        approach = "fine-tuning"

    # -- Decision 2: which method fits the hardware? ----------------------- #
    method, vram = _pick_method(spec, scenario.gpu_vram_gb, reasons)

    # -- Serving quantization ---------------------------------------------- #
    if scenario.latency_sensitive:
        serving = "int8"
        reasons.append(
            "Latency-sensitive serving: int8 weights roughly halve memory "
            "bandwidth with negligible quality loss for most models."
        )
    elif scenario.model_params_b >= 13:
        serving = "nf4"
        reasons.append(
            "Large model for serving: 4-bit (NF4) weights cut the memory "
            "footprint ~4x so it fits on cheaper hardware."
        )
    else:
        serving = "fp16"
        reasons.append("No tight memory/latency pressure at serving time: keep fp16 for max quality.")

    return Recommendation(
        should_fine_tune=True, approach=approach, method=method,
        serving_quantization=serving, estimated_vram_gb=round(vram, 3),
        fits_budget=fits_on(vram, scenario.gpu_vram_gb), reasons=reasons,
    )

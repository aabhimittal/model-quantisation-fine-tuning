"""Cut hallucination by grounding the model in sources -- and measure the drop.

Run: ``python examples/05_grounded_generation.py``

The demo in ``04`` hallucinated ("Quantum LoRA", "nuclear physics") because a small
model was answering a factual question from memory alone. quantune's thesis is
"RAG for facts": don't trust the weights, hand the model the source text and make it
answer only from that. This example shows the same question three ways and prints the
groundedness score so the improvement is visible, not asserted.

Needs a free NVIDIA key (https://build.nvidia.com); auto-skips the live calls without
one::

    export NVIDIA_API_KEY=nvapi-...
    python examples/05_grounded_generation.py
"""

import os

from quantune.serving import OpenAICompatClient, ServingError

# A couple of source snippets (straight from docs/concepts.md) to ground on.
CONTEXT = [
    "QLoRA fine-tunes a large model cheaply by keeping the frozen base weights in "
    "4-bit NF4 and training only a small low-rank LoRA adapter. This collapses the "
    "gradient and optimizer memory buckets and quarters the weight bucket, so a 7B "
    "model that needs ~113 GB for full fine-tuning drops under a single 24 GB GPU.",
    "LoRA freezes the pretrained weights and learns a low-rank delta (A and B "
    "matrices), training 100-1000x fewer parameters than full fine-tuning.",
]

QUESTION = "In one sentence, what problem does QLoRA solve and how?"
UNANSWERABLE = "What learning rate schedule does QLoRA recommend?"

if not (os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")):
    print(
        "No NVIDIA_API_KEY set -- skipping the live calls.\n"
        "Get a free key at https://build.nvidia.com, then:\n"
        "    export NVIDIA_API_KEY=nvapi-...\n"
        "    python examples/05_grounded_generation.py"
    )
    raise SystemExit(0)

client = OpenAICompatClient()


def show(title: str, **kwargs) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    try:
        # temperature=0 -> deterministic, the right default for factual answers.
        result = client.generate(QUESTION, temperature=0.0, max_tokens=100, **kwargs)
    except ServingError as exc:
        print(f"live call failed: {exc}\n")
        return
    print(result.text.strip())
    print(f"\n[{result.summary()}]\n")


# 1) Ungrounded: the model answers from memory -- watch the groundedness be low
#    (and the content often wrong).
show("UNGROUNDED (memory only)")

# 2) Grounded: same question, but the model may only use CONTEXT.
show("GROUNDED (answer only from sources)", context=CONTEXT)

# 3) Grounded but unanswerable from the sources: a well-behaved model abstains
#    instead of inventing an answer.
print("=" * 72)
print("GROUNDED + UNANSWERABLE (should say \"I don't know.\")")
print("=" * 72)
try:
    res = client.generate(UNANSWERABLE, context=CONTEXT, temperature=0.0, max_tokens=60)
    print(res.text.strip())
    print(f"\n[{res.summary()}]")
except ServingError as exc:
    print(f"live call failed: {exc}")

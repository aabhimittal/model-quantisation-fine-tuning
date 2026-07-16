"""Cut hallucination a second way: sample several answers and vote.

Run: ``python examples/06_self_consistency.py``

Grounding (example 05) fixes *fact* questions by handing the model sources. But for
*reasoning* -- where the risk is a wrong chain of steps, not a missing fact -- the
better lever is **self-consistency**: sample the answer several times with the
temperature up, then keep the majority. Independent samples rarely make the *same*
mistake, so voting cancels one-off slips that a single greedy pass would commit to.
The agreement score is also a live confidence signal: when the samples scatter, the
model is unsure and you should not trust any single answer.

Needs a free NVIDIA key (https://build.nvidia.com); auto-skips without one::

    export NVIDIA_API_KEY=nvapi-...
    python examples/06_self_consistency.py
"""

import os

from quantune.serving import OpenAICompatClient, ServingError

# A short word problem -- the kind of multi-step question a single greedy decode can
# get confidently wrong, and where majority voting helps.
QUESTION = (
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
    "How much does the ball cost? Answer with just the amount."
)

if not (os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")):
    print(
        "No NVIDIA_API_KEY set -- skipping the live call.\n"
        "Get a free key at https://build.nvidia.com, then:\n"
        "    export NVIDIA_API_KEY=nvapi-...\n"
        "    python examples/06_self_consistency.py"
    )
    raise SystemExit(0)

client = OpenAICompatClient()
try:
    result = client.self_consistency(QUESTION, n=5, temperature=0.7, max_tokens=60)
except ServingError as exc:
    print(f"live call failed: {exc}")
    raise SystemExit(1)

print("Question:")
print(f"  {QUESTION}\n")
print(f"Sampled {result.n_samples} answers:")
for i, s in enumerate(result.samples, start=1):
    print(f"  [{i}] {s.strip()}")
print()
print(f"Voted answer ({result.votes}/{result.n_samples} agree): {result.text.strip()}")
print(f"\n[{result.summary()}]")
print(
    "\nLow agreement here would be the signal to distrust any single sample -- "
    "exactly when a lone greedy answer is most likely to be hallucinated."
)

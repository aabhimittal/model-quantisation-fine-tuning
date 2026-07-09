"""Serve a model on GPU cloud -- pick a backend, then actually generate text.

Run: ``python examples/04_serve_on_nvidia_nim.py``

Part 1 always runs (it's pure NumPy/stdlib): the deployment advisor reasons about
where to serve a few scenarios and prints copy-paste launch configs.

Part 2 makes a *real* call to NVIDIA NIM's hosted GPUs -- but only if you've set a
free API key. Get one at https://build.nvidia.com (Get API Key -> nvapi-...), then::

    export NVIDIA_API_KEY=nvapi-...
    python examples/04_serve_on_nvidia_nim.py

The GPU lives in NVIDIA's cloud; nothing here needs a local GPU or any dependency
beyond NumPy + the Python standard library.
"""

import os

from quantune.deploy import DeploymentScenario, advise_deployment
from quantune.serving import OpenAICompatClient, ServingError

# --- Part 1: where should I serve? (always runs, no network) --------------- #
scenarios = {
    "8B, no GPU of my own, want it now": DeploymentScenario(model_params_b=8, has_own_gpu=False),
    "8B, already on AWS, no servers please": DeploymentScenario(
        model_params_b=8, has_own_gpu=False, in_aws=True),
    "70B, I own 24GB GPUs, care about cost": DeploymentScenario(
        model_params_b=70, has_own_gpu=True, gpu_vram_gb=24, budget_sensitive=True),
}
for title, scenario in scenarios.items():
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(advise_deployment(scenario).summary(), "\n")

# --- Part 2: actually generate on GPU cloud (needs a free key) ------------- #
print("=" * 72)
print("Live generation on NVIDIA NIM")
print("=" * 72)
if not (os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")):
    print(
        "No NVIDIA_API_KEY set -- skipping the live call.\n"
        "Get a free key at https://build.nvidia.com, then:\n"
        "    export NVIDIA_API_KEY=nvapi-...\n"
        "    python examples/04_serve_on_nvidia_nim.py"
    )
else:
    client = OpenAICompatClient()
    try:
        result = client.generate(
            "In one sentence, what problem does QLoRA solve?",
            stream=True,
            max_tokens=80,
            on_token=lambda t: print(t, end="", flush=True),
        )
        print("\n\n" + result.summary())
    except ServingError as exc:
        print(f"live call failed: {exc}")

"""Ask the advisor what to do, then check the VRAM math behind it.

Run: ``python examples/03_advisor_and_vram.py``

Walks through a few realistic scenarios and prints the recommendation plus the
memory breakdown that justifies the chosen method.
"""

from quantune.advisor import Scenario, advise
from quantune.vram import ModelSpec, compare_methods

scenarios = {
    "10 examples, style task": Scenario(task="style", num_examples=10),
    "changing product docs (facts)": Scenario(
        task="knowledge", num_examples=8000, data_changes_often=True),
    "brand voice, 7B on a 24GB GPU": Scenario(
        task="style", num_examples=5000, model_params_b=7, gpu_vram_gb=24),
    "70B reasoning on a 24GB GPU": Scenario(
        task="reasoning", num_examples=20000, model_params_b=70, gpu_vram_gb=24),
}

for title, scenario in scenarios.items():
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(advise(scenario).summary(), "\n")

print("=" * 70)
print("VRAM breakdown for a 7B model (GB)")
print("=" * 70)
table = compare_methods(ModelSpec(params_billion=7))
for method, e in table.items():
    print(f"{method:<6} weights={e['weights_gb']:>6}  grads={e['gradients_gb']:>5}  "
          f"optim={e['optimizer_gb']:>5}  acts={e['activations_gb']:>5}  "
          f"TOTAL={e['total_gb']:>7}")

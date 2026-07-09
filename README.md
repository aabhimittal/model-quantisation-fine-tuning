# quantune — a glass-box, decision-first toolkit for fine-tuning & quantization

> Fine-tuning (LoRA / QLoRA) and quantization (int8 / int4 / NF4) for open-source
> LLMs — with the mechanics **exposed, not hidden**, and the **"should I even
> fine-tune?"** question answered *before* you write any training code.

Most fine-tuning repos are a thin wrapper over a black-box library call: you run
a script, weights come out, and you learn nothing about *why*. `quantune` takes
the opposite bet. It is built around two ideas that the usual tutorials skip:

1. **Decision first.** A transparent rule engine tells you *whether* to
   fine-tune (vs. prompting or RAG), *which* method fits your GPU (full / LoRA /
   QLoRA), and *how* to serve it — every answer comes with its reasoning.
2. **Glass box.** LoRA and every quantization scheme (including **NF4**, the
   datatype behind QLoRA) are implemented **from scratch in ~600 lines of pure
   NumPy** with hand-derived gradients. No CUDA, no `bitsandbytes`, no GPU
   required — you can read and step through the actual arithmetic.

The result is a repo you *learn from*, that also gives you a practical VRAM
calculator and advisor you can use on real projects.

---

## Why this is a different approach

| Typical tutorial repo | `quantune` |
| --- | --- |
| Assumes you should fine-tune | **Advisor** decides fine-tune vs. prompt vs. RAG first |
| `peft`/`bitsandbytes` black box | LoRA + NF4 quantization **from scratch**, unit-tested |
| Needs a GPU to run anything | Runs on a **laptop CPU** in seconds |
| "It uses less memory" (hand-wave) | **VRAM estimator** with the four-bucket breakdown |
| No way to see the accuracy cost | **Error metrics** comparing every quant scheme |

---

## Install

```bash
git clone https://github.com/aabhimittal/model-quantisation-fine-tuning.git
cd model-quantisation-fine-tuning
pip install -e .          # installs the `quantune` CLI + library (numpy only)
```

## 60-second tour (CLI)

**1. Should I fine-tune, and how?**

```bash
$ quantune advise --task style --examples 2000 --model-b 7 --vram 24
Approach:  fine-tuning
Method:    lora
Serving:   fp16
Est. VRAM: 16.054 GB (fits the budget)
Why:
  - Task is 'style' (behaviour/format/style) with 2000 examples -- a good fit
    for fine-tuning, which excels at teaching how to respond rather than facts.
  - LORA chosen: it fits in ~16.054 GB by freezing base weights and training a
    small low-rank adapter.
  - No tight memory/latency pressure at serving time: keep fp16 for max quality.
```

**2. Will it fit my GPU?**

```bash
$ quantune vram --model-b 7
method     weights    grads    optim     acts     TOTAL
-------------------------------------------------------
full          14.0     14.0     84.0    1.074   113.074
lora          14.0     0.14     0.84    1.074    16.054
qlora          3.5     0.14     0.84    1.074     5.554
```

That single table is the whole QLoRA pitch: a 7B model goes from **113 GB**
(needs a cluster) to **5.5 GB** (fits a laptop GPU).

**3. What does quantization cost me in accuracy?**

```bash
$ quantune quantize --rows 512 --cols 512
scheme            rel_error    max_abs   bits/wt
------------------------------------------------
int8_symmetric       0.0272     0.0474      8.00
int8_affine          0.0246     0.0428      8.00
int4_symmetric       0.4955     0.8605      4.00   ← one global scale collapses
blockwise_int4       0.1125     0.8600      4.25   ← per-block scales rescue it
nf4                  0.0937     0.5605      4.25   ← normal-aware levels win
```

**4. Where do I serve it, and how do I launch that?**

```bash
$ quantune deploy --model-b 8            # no GPU of your own? -> hosted GPU cloud
Backend:   NVIDIA NIM (hosted GPU cloud)
Serving:   fp16
Serve VRAM:17.074 GB (weights + KV cache)
GPUs:      none of your own (provider-hosted)
Why:
  - No GPUs of your own -- NVIDIA NIM hosts the model on its cloud GPUs behind a
    free, OpenAI-compatible endpoint, so you can serve today with zero infrastructure.
...
$ quantune deploy --model-b 70 --has-gpu --budget-sensitive   # own GPUs -> vLLM, + launch cmd
$ quantune deploy --emit-config tgi --gpus 2                   # just print a TGI docker run
```

The advisor's four backends — **NVIDIA NIM**, **vLLM**, **Hugging Face TGI**, **AWS
Bedrock** — are the same ones the brief names, and `deploy` prints a copy-paste
launch config for each.

## Serve it for real on GPU cloud (NVIDIA NIM) — no GPU required

quantune runs on your laptop, but it can *drive* a real GPU by talking the
OpenAI-compatible wire format to a hosted endpoint. NIM, vLLM, and TGI all speak it,
so one client + a swapped `base_url` reaches all three. Get a **free** key at
[build.nvidia.com](https://build.nvidia.com) (Get API Key → `nvapi-...`; no credit
card, no GPU), then:

```bash
export NVIDIA_API_KEY=nvapi-...
quantune serve --prompt "Explain NF4 in one sentence." --stream
#   ...streams tokens from NVIDIA's cloud GPUs...
#   [model=meta/llama-3.1-8b-instruct  tokens=39  ttft=520 ms  speed=51.5 tok/s  total=0.76s]
```

Same client, self-hosted server — just change one flag:

```bash
quantune serve --base-url http://localhost:8000/v1 --prompt "Hello!"   # vLLM / local NIM
```

Because "fast, low-latency" is a *measurable* claim, every call reports
**time-to-first-token** and **tokens/sec**. This uses only the Python standard
library — the `numpy`-only install is unchanged.

### Reduce hallucination — ground it

A small model answering from memory invents facts (in the demo above it called NF4
"nuclear physics"). quantune's whole thesis is **"RAG for facts, don't bake them into
weights"** — so at serving time, hand the model the source text and make it answer
**only** from that, or abstain. Pass `--context`/`--context-file` (or `context=[...]`
in code) and each answer comes back with a transparent **groundedness score** — the
share of the answer's content words actually found in your sources:

```bash
# ungrounded: answers from memory, often wrong
quantune serve --prompt "What problem does QLoRA solve?" --temperature 0

# grounded: answer must come from the sources, and gets scored
quantune serve --prompt "What problem does QLoRA solve?" --temperature 0 \
    --context-file docs/concepts.md
#   ...QLoRA keeps frozen base weights in 4-bit NF4 and trains a small LoRA adapter... [1]
#   [model=meta/llama-3.1-8b-instruct  ttft=310 ms  speed=48 tok/s  grounded=92%]
```

If the answer isn't in the sources, a grounded model replies `"I don't know."`
(`abstained=True`) instead of guessing. Use `--temperature 0` for deterministic,
factual answers. The score is an honest lexical proxy — it flags unsupported text, it
doesn't certify truth. See `examples/05_grounded_generation.py` for a before/after.

## 60-second tour (library)

```python
import numpy as np
from quantune import LoRALinear, LoRAConfig, quantize_nf4, dequantize, advise, Scenario

# --- LoRA: train a rank-4 adapter on a frozen base weight ---
W0 = np.random.randn(64, 48)
layer = LoRALinear(W0, LoRAConfig(r=4, alpha=8))
layer.fit(X, Y, lr=0.005, steps=2000)     # only A, B move; W0 stays frozen
merged = layer.merged_weight()            # fold back for zero-overhead serving

# --- NF4: quantize a weight matrix to 4 bits and measure the error ---
qt = quantize_nf4(W0, block_size=64)
reconstructed = dequantize(qt)            # ~4 bits/weight, normal-aware

# --- Advisor: get a reasoned recommendation ---
rec = advise(Scenario(task="knowledge", num_examples=8000, data_changes_often=True))
print(rec.approach)                       # -> "rag"  (don't bake changing facts into weights)

# --- Deploy: pick a serving backend, then generate on GPU cloud ---
from quantune import DeploymentScenario, advise_deployment, OpenAICompatClient
plan = advise_deployment(DeploymentScenario(model_params_b=8, has_own_gpu=False))
print(plan.backend)                       # -> "nvidia_nim"

client = OpenAICompatClient()             # reads NVIDIA_API_KEY; no local GPU needed
out = client.generate("Say hi in French.", stream=True)
print(out.text, out.tokens_per_s)         # real tokens from NVIDIA's cloud GPUs
```

## What's inside

```
src/quantune/
├── advisor.py       # decision engine: fine-tune vs prompt vs RAG, + method choice
├── vram.py          # four-bucket training-memory estimator (weights/grads/optim/acts)
├── lora.py          # LoRALinear from scratch: forward, merge, hand-derived SGD
├── quantization.py  # int8 (sym/affine), int4, block-wise, and NF4 — all from scratch
├── deploy.py        # serving advisor: NIM/vLLM/TGI/Bedrock + copy-paste launch configs
├── serving.py       # OpenAI-compatible client: real GPU-cloud generation, stdlib only
└── cli.py           # `quantune advise | vram | quantize | deploy | serve`
examples/            # 5 runnable demos (04/05 make real NIM calls if a key is set)
tests/               # pytest suite covering every claim above (serving mocked, no network)
docs/concepts.md     # the theory, mapped to the code
```

## Run the examples & tests

```bash
python examples/01_lora_from_scratch.py     # watch a LoRA fit converge to ~0 loss
python examples/02_quantize_linear.py       # int4 vs blockwise vs NF4 on real-ish weights
python examples/03_advisor_and_vram.py      # end-to-end scenarios + VRAM tables

pip install -e ".[dev]" && pytest -q        # 27 passed
```

## The mental model in one paragraph

Fine-tune to change **behaviour/format/style**; use **RAG** for **facts** that
move. If you fine-tune, **LoRA** freezes the pretrained weights and learns a
tiny low-rank delta (100–1000× fewer trainable params), and **QLoRA** goes
further by storing those frozen weights in **4-bit NF4** — collapsing the two
memory buckets (gradients + Adam optimizer state) that dominate full
fine-tuning, and cutting the weight bucket 4×. That is how a 7B model becomes
trainable on a single consumer GPU. Read `docs/concepts.md` for the full story,
then read the code — it's short on purpose.

## References

- Hu et al., **LoRA** (2021) — arXiv:2106.09685
- Dettmers et al., **QLoRA** (2023) — arXiv:2305.14314
- Dettmers et al., **LLM.int8()** (2022) — arXiv:2208.07339
- Frantar et al., **GPTQ** (2022) — arXiv:2210.17323

## License

MIT — see [LICENSE](LICENSE).

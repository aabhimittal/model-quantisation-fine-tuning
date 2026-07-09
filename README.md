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
```

## What's inside

```
src/quantune/
├── advisor.py       # decision engine: fine-tune vs prompt vs RAG, + method choice
├── vram.py          # four-bucket training-memory estimator (weights/grads/optim/acts)
├── lora.py          # LoRALinear from scratch: forward, merge, hand-derived SGD
├── quantization.py  # int8 (sym/affine), int4, block-wise, and NF4 — all from scratch
└── cli.py           # `quantune advise | vram | quantize`
examples/            # 3 runnable, CPU-only demos
tests/               # 27 pytest tests covering every claim above
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

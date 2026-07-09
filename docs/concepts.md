# Concepts: Fine-Tuning & Quantization

A compact reference for the ideas this repo implements. Everything here is
backed by runnable code in `src/quantune/`.

## 1. When to fine-tune (and when *not* to)

Fine-tuning changes a model's weights on your data. It is powerful but often
reached for too early. A useful mental model:

| You want to change… | Best tool | Why |
| --- | --- | --- |
| **Behaviour, format, style, tone** | Fine-tuning | These are patterns of *how* to respond — exactly what gradient descent on examples teaches. |
| **Fresh or frequently-changing facts** | RAG (retrieval) | Facts baked into weights go stale and can't be cited. Retrieval keeps them live. |
| **A handful of examples (<~50)** | Prompting / few-shot | Not enough signal to fine-tune reliably; a good prompt is faster and cheaper. |
| **Stable domain knowledge + house style** | RAG **and** light fine-tuning | Retrieve the facts, fine-tune the voice. |

`quantune.advisor` encodes exactly this table as an inspectable rule set.

## 2. LoRA — Low-Rank Adaptation

Full fine-tuning learns a dense weight update `ΔW` with the same shape as `W`
(millions to billions of numbers). LoRA's insight: that update has **low
intrinsic rank**, so approximate it as a product of two skinny matrices.

```
W_effective = W0 + (alpha / r) · B · A
              └┬─┘   └────┬──────┘
            frozen   trainable low-rank delta
```

- `W0` — the pretrained weight, **frozen**.
- `A` ∈ ℝ^(r×d_in), `B` ∈ ℝ^(d_out×r) — the only trainable parameters.
- `r` — the rank (typically 4–64). `B` starts at zero so training begins as a
  no-op from the pretrained behaviour.
- `alpha / r` — a fixed scaling that decouples the learning rate from `r`.

Because `r ≪ d`, you train **100–1000× fewer parameters**. At serving time you
can fold the delta back in (`merged_weight()`), so there is zero inference
overhead. See `src/quantune/lora.py` for a from-scratch implementation with
hand-derived gradients.

## 3. Quantization — shrinking the weights

Quantization maps high-precision floats (fp16, 2 bytes) to low-precision
integers (int8 = 1 byte, int4 = 0.5 bytes), cutting memory and bandwidth.

- **Symmetric int8**: `q = round(w / scale)`, `scale = max(|w|) / 127`. One
  global scale; simple but wrecked by outliers.
- **Affine (asymmetric)**: adds a zero-point to use the full range — good for
  non-centred distributions like post-ReLU activations.
- **Block-wise**: give every block of ~64 weights its *own* scale. An outlier
  now only hurts its own block. **This is the key trick** that makes 4-bit
  weights usable.
- **NF4 (NormalFloat4)**: a 4-bit datatype whose 16 levels are the quantiles of
  a normal distribution. Since trained weights are ≈ normal, NF4 places
  resolution where the mass is and beats plain int4 at the same bit budget.

`src/quantune/quantization.py` implements all four, plus error metrics so you
can *measure* the accuracy/size trade-off.

## 4. QLoRA — the combination

**QLoRA = LoRA adapters on top of an NF4-quantized frozen base.** The frozen
weights (99%+ of the model) drop to 4 bits, and only the tiny adapter trains in
higher precision. The result: fine-tune a 7B model in **~5.5 GB** instead of
**~113 GB** — a single consumer GPU instead of a cluster.

Why the savings are so dramatic (see `quantune.vram`):

| Bucket | Full FT | LoRA | QLoRA |
| --- | --- | --- | --- |
| Weights | fp16 (2 B/param) | fp16 | **NF4 (0.5 B/param)** |
| Gradients | all params | adapter only | adapter only |
| Optimizer (Adam) | 12 B × all params | 12 B × adapter | 12 B × adapter |

The optimizer state — two moments plus an fp32 master copy per trainable
parameter — is the real memory killer in full fine-tuning. Freezing the base
weights makes it vanish for 99% of the model.

## References

1. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021), arXiv:2106.09685.
2. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023), arXiv:2305.14314.
3. Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (2022), arXiv:2208.07339.
4. Frantar et al., *GPTQ: Accurate Post-Training Quantization* (2022), arXiv:2210.17323.

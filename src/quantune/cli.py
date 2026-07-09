"""Command-line interface: ``python -m quantune ...``.

Subcommands mirror the library:

* ``advise``   -- get a fine-tune / method / serving recommendation.
* ``vram``     -- print the memory breakdown for full/LoRA/QLoRA.
* ``quantize`` -- benchmark every quantization scheme on random weights.
"""

from __future__ import annotations

import argparse

import numpy as np

from .advisor import Scenario, advise
from .quantization import compare_schemes
from .vram import ModelSpec, compare_methods


def _cmd_advise(args: argparse.Namespace) -> None:
    scenario = Scenario(
        task=args.task,
        num_examples=args.examples,
        model_params_b=args.model_b,
        gpu_vram_gb=args.vram,
        latency_sensitive=args.latency_sensitive,
        data_changes_often=args.data_changes_often,
    )
    print(advise(scenario).summary())


def _cmd_vram(args: argparse.Namespace) -> None:
    spec = ModelSpec(params_billion=args.model_b, seq_len=args.seq_len, batch_size=args.batch)
    table = compare_methods(spec)
    header = f"{'method':<8} {'weights':>9} {'grads':>8} {'optim':>8} {'acts':>8} {'TOTAL':>9}"
    print(f"Model: {args.model_b}B params, seq_len={args.seq_len}, batch={args.batch}\n")
    print(header)
    print("-" * len(header))
    for name, e in table.items():
        print(f"{name:<8} {e['weights_gb']:>9} {e['gradients_gb']:>8} "
              f"{e['optimizer_gb']:>8} {e['activations_gb']:>8} {e['total_gb']:>9}")


def _cmd_quantize(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(0)
    # Mostly-normal weights with a few heavy outliers, like a real layer.
    w = rng.normal(0, 1, size=(args.rows, args.cols))
    w.reshape(-1)[:: max(1, w.size // 20)] *= 8.0
    results = compare_schemes(w, block_size=args.block_size)
    header = f"{'scheme':<16} {'rel_error':>10} {'max_abs':>10} {'bits/wt':>9}"
    print(f"Random {args.rows}x{args.cols} weights with outliers, block_size={args.block_size}\n")
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<16} {m['rel_frobenius']:>10.4f} {m['max_abs_error']:>10.4f} "
              f"{m['bits_per_weight']:>9.2f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantune", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("advise", help="recommend whether/how to fine-tune")
    a.add_argument("--task", default="style", choices=["style", "format", "knowledge", "reasoning"])
    a.add_argument("--examples", type=int, default=1000)
    a.add_argument("--model-b", type=float, default=7.0, help="base model size in billions")
    a.add_argument("--vram", type=float, default=24.0, help="GPU VRAM budget in GB")
    a.add_argument("--latency-sensitive", action="store_true")
    a.add_argument("--data-changes-often", action="store_true")
    a.set_defaults(func=_cmd_advise)

    v = sub.add_parser("vram", help="estimate training memory per method")
    v.add_argument("--model-b", type=float, default=7.0)
    v.add_argument("--seq-len", type=int, default=2048)
    v.add_argument("--batch", type=int, default=1)
    v.set_defaults(func=_cmd_vram)

    q = sub.add_parser("quantize", help="benchmark quantization schemes")
    q.add_argument("--rows", type=int, default=512)
    q.add_argument("--cols", type=int, default=512)
    q.add_argument("--block-size", type=int, default=64)
    q.set_defaults(func=_cmd_quantize)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

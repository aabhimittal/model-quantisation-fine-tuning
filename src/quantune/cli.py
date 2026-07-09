"""Command-line interface: ``python -m quantune ...``.

Subcommands mirror the library:

* ``advise``   -- get a fine-tune / method / serving recommendation.
* ``vram``     -- print the memory breakdown for full/LoRA/QLoRA.
* ``quantize`` -- benchmark every quantization scheme on random weights.
* ``deploy``   -- recommend a serving backend (NIM/vLLM/TGI/Bedrock) + launch config.
* ``serve``    -- actually generate text on GPU cloud via an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .advisor import Scenario, advise
from .deploy import DeploymentScenario, _default_model_for, advise_deployment, render_config
from .quantization import compare_schemes
from .serving import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatClient, ServingError
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


def _cmd_deploy(args: argparse.Namespace) -> None:
    if args.emit_config:
        # Pick the backend's natural default model id unless the user overrode --model.
        model = args.model if args.model != DEFAULT_MODEL else _default_model_for(args.emit_config)
        print(render_config(args.emit_config, model=model, dtype=args.dtype, gpus=args.gpus))
        return
    scenario = DeploymentScenario(
        model_params_b=args.model_b,
        has_own_gpu=args.has_gpu,
        gpu_vram_gb=args.vram,
        latency_sensitive=args.latency_sensitive,
        budget_sensitive=args.budget_sensitive,
        wants_managed=args.managed,
        in_aws=args.in_aws,
    )
    print(advise_deployment(scenario).summary())


def _cmd_serve(args: argparse.Namespace) -> None:
    client = OpenAICompatClient(base_url=args.base_url, timeout=args.timeout)
    try:
        if args.list_models:
            for mid in client.list_models():
                print(mid)
            return
        if not args.prompt:
            raise ServingError("nothing to do: pass --prompt \"...\" (or --list-models).")
        result = client.generate(
            args.prompt,
            model=args.model,
            system=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=args.stream,
            on_token=(lambda t: (sys.stdout.write(t), sys.stdout.flush())) if args.stream else None,
        )
    except ServingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if args.stream:
        print()  # newline after the streamed tokens
    else:
        print(result.text)
    print(f"\n[{result.summary()}]", file=sys.stderr)


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

    d = sub.add_parser("deploy", help="recommend a serving backend + launch config")
    d.add_argument("--model-b", type=float, default=8.0, help="model size in billions")
    d.add_argument("--has-gpu", action="store_true", help="you have GPUs to self-host on")
    d.add_argument("--vram", type=float, default=24.0, help="per-GPU VRAM budget in GB")
    d.add_argument("--latency-sensitive", action="store_true")
    d.add_argument("--budget-sensitive", action="store_true")
    d.add_argument("--managed", action="store_true", help="prefer a no-servers managed backend")
    d.add_argument("--in-aws", action="store_true", help="already running on AWS")
    d.add_argument("--emit-config", choices=list(("nvidia_nim", "vllm", "tgi", "bedrock")),
                   help="just print the launch config for this backend and exit")
    d.add_argument("--model", default=DEFAULT_MODEL, help="model id for --emit-config")
    d.add_argument("--dtype", default="fp16", choices=["fp16", "int8", "nf4"], help="serving dtype for --emit-config")
    d.add_argument("--gpus", type=int, default=1, help="GPU count for --emit-config")
    d.set_defaults(func=_cmd_deploy)

    s = sub.add_parser("serve", help="generate text on GPU cloud (OpenAI-compatible endpoint)")
    s.add_argument("--prompt", help="the user prompt to generate from")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="OpenAI-compatible /v1 root (NVIDIA NIM by default; use localhost for vLLM/TGI)")
    s.add_argument("--system", default=None, help="optional system prompt")
    s.add_argument("--max-tokens", type=int, default=256)
    s.add_argument("--temperature", type=float, default=0.2)
    s.add_argument("--stream", action="store_true", help="stream tokens and measure time-to-first-token")
    s.add_argument("--timeout", type=float, default=60.0)
    s.add_argument("--list-models", action="store_true", help="list available model ids and exit")
    s.set_defaults(func=_cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

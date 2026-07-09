"""Tests for the deployment advisor and launch-config generator (no network)."""

import pytest

from quantune.deploy import (
    BACKENDS,
    DeploymentScenario,
    advise_deployment,
    render_config,
)
from quantune.vram import ModelSpec, serving_vram


def test_no_gpu_recommends_hosted_nim():
    plan = advise_deployment(DeploymentScenario(model_params_b=8, has_own_gpu=False))
    assert plan.backend == "nvidia_nim"
    assert plan.gpus_needed == 0


def test_no_gpu_in_aws_recommends_bedrock():
    plan = advise_deployment(DeploymentScenario(model_params_b=8, has_own_gpu=False, in_aws=True))
    assert plan.backend == "bedrock"


def test_no_gpu_wants_managed_recommends_bedrock():
    plan = advise_deployment(DeploymentScenario(has_own_gpu=False, wants_managed=True))
    assert plan.backend == "bedrock"


def test_own_gpu_budget_recommends_vllm():
    plan = advise_deployment(
        DeploymentScenario(model_params_b=8, has_own_gpu=True, budget_sensitive=True)
    )
    assert plan.backend == "vllm"
    assert plan.gpus_needed >= 1


def test_own_gpu_default_recommends_tgi():
    plan = advise_deployment(DeploymentScenario(model_params_b=8, has_own_gpu=True))
    assert plan.backend == "tgi"


def test_own_gpu_but_managed_still_bedrock():
    plan = advise_deployment(DeploymentScenario(has_own_gpu=True, wants_managed=True))
    assert plan.backend == "bedrock"
    assert plan.gpus_needed == 0


def test_latency_sensitive_uses_int8():
    plan = advise_deployment(DeploymentScenario(model_params_b=7, latency_sensitive=True))
    assert plan.serving_dtype == "int8"


def test_large_model_uses_nf4():
    plan = advise_deployment(DeploymentScenario(model_params_b=70, has_own_gpu=False))
    assert plan.serving_dtype == "nf4"


def test_big_model_needs_multiple_gpus():
    plan = advise_deployment(
        DeploymentScenario(model_params_b=70, has_own_gpu=True, gpu_vram_gb=24, budget_sensitive=True)
    )
    assert plan.gpus_needed > 1


def test_every_plan_has_reasons_and_launch_config():
    plan = advise_deployment(DeploymentScenario())
    assert plan.reasons
    assert plan.launch_config
    assert isinstance(plan.summary(), str)


@pytest.mark.parametrize(
    "backend,needle",
    [
        ("nvidia_nim", "build.nvidia.com"),
        ("vllm", "vllm.entrypoints.openai.api_server"),
        ("tgi", "text-generation-inference"),
        ("bedrock", "bedrock-runtime"),
    ],
)
def test_render_config_contains_expected_command(backend, needle):
    assert needle in render_config(backend, model="meta/llama-3.1-8b-instruct")


def test_render_config_rejects_unknown_backend():
    with pytest.raises(ValueError):
        render_config("sagemaker")


def test_render_config_covers_all_backends():
    for backend in BACKENDS:
        assert render_config(backend)  # non-empty


def test_serving_vram_is_smaller_than_training():
    spec = ModelSpec(params_billion=7)
    # fp16 serving = ~14 GB weights + a little KV cache; far below full-FT training.
    assert serving_vram(spec, "fp16") < 30
    # nf4 weights are 4x smaller than fp16, so total serving VRAM must drop.
    assert serving_vram(spec, "nf4") < serving_vram(spec, "fp16")

from quantune.vram import ModelSpec, compare_methods, estimate, fits_on


def test_method_ordering_qlora_cheapest():
    spec = ModelSpec(params_billion=7)
    table = compare_methods(spec)
    assert table["qlora"]["total_gb"] < table["lora"]["total_gb"] < table["full"]["total_gb"]


def test_lora_zeroes_optimizer_bucket_relative_to_full():
    spec = ModelSpec(params_billion=7)
    full = estimate(spec, "full")
    lora = estimate(spec, "lora")
    # Optimizer state is the killer for full FT; LoRA nearly eliminates it.
    assert lora["optimizer_gb"] < full["optimizer_gb"] / 50


def test_qlora_weight_bucket_is_quarter_of_fp16():
    spec = ModelSpec(params_billion=7)
    full = estimate(spec, "full", compute_dtype="fp16")
    qlora = estimate(spec, "qlora")
    # NF4 (0.5 B/param) vs fp16 (2 B/param) => ~4x smaller weight bucket.
    assert qlora["weights_gb"] == full["weights_gb"] / 4


def test_qlora_7b_fits_consumer_gpu():
    spec = ModelSpec(params_billion=7)
    est = estimate(spec, "qlora")
    assert fits_on(est["total_gb"], gpu_vram_gb=24.0)


def test_full_7b_does_not_fit_consumer_gpu():
    spec = ModelSpec(params_billion=7)
    est = estimate(spec, "full")
    assert not fits_on(est["total_gb"], gpu_vram_gb=24.0)


def test_activation_memory_scales_with_batch():
    small = estimate(ModelSpec(params_billion=7, batch_size=1), "lora")
    big = estimate(ModelSpec(params_billion=7, batch_size=8), "lora")
    assert big["activations_gb"] > small["activations_gb"]

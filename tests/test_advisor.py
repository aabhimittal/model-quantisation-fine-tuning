from quantune.advisor import Scenario, advise


def test_tiny_dataset_recommends_prompting():
    rec = advise(Scenario(task="style", num_examples=10))
    assert not rec.should_fine_tune
    assert rec.approach == "prompting"


def test_changing_facts_recommend_rag():
    rec = advise(Scenario(task="knowledge", num_examples=5000, data_changes_often=True))
    assert not rec.should_fine_tune
    assert rec.approach == "rag"


def test_stable_knowledge_recommends_rag_plus_fine_tuning():
    rec = advise(Scenario(task="knowledge", num_examples=5000, data_changes_often=False))
    assert rec.should_fine_tune
    assert rec.approach == "rag+fine-tuning"


def test_style_task_recommends_fine_tuning():
    rec = advise(Scenario(task="style", num_examples=2000, model_params_b=7, gpu_vram_gb=24))
    assert rec.should_fine_tune
    assert rec.approach == "fine-tuning"
    assert rec.method in {"lora", "qlora"}  # 7B full FT won't fit 24 GB


def test_small_model_big_gpu_allows_full_fine_tuning():
    rec = advise(Scenario(task="format", num_examples=5000, model_params_b=0.5, gpu_vram_gb=80))
    assert rec.method == "full"


def test_big_model_tight_gpu_picks_qlora():
    rec = advise(Scenario(task="reasoning", num_examples=5000, model_params_b=70, gpu_vram_gb=24))
    assert rec.method == "qlora"


def test_latency_sensitive_serving_uses_int8():
    rec = advise(Scenario(task="style", num_examples=2000, latency_sensitive=True))
    assert rec.serving_quantization == "int8"


def test_every_recommendation_carries_reasons():
    rec = advise(Scenario())
    assert rec.reasons
    assert isinstance(rec.summary(), str)

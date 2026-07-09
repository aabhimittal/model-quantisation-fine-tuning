import numpy as np

from quantune.lora import LoRAConfig, LoRALinear


def test_zero_init_is_a_noop():
    rng = np.random.default_rng(0)
    W0 = rng.normal(0, 1, size=(8, 6))
    x = rng.normal(0, 1, size=(4, 6))
    layer = LoRALinear(W0, LoRAConfig(r=2, alpha=4))
    # B is zero-initialised, so the adapter starts as the frozen base map.
    np.testing.assert_allclose(layer.forward(x), x @ W0.T)


def test_trainable_params_are_much_smaller_than_base():
    W0 = np.zeros((4096, 4096))
    layer = LoRALinear(W0, LoRAConfig(r=8))
    assert layer.num_trainable_params() == (8 * 4096) * 2
    assert layer.compression_ratio() > 200  # >200x fewer params to train


def test_fit_recovers_low_rank_delta():
    rng = np.random.default_rng(1)
    d_out, d_in, r = 10, 8, 2
    W0 = rng.normal(0, 1, size=(d_out, d_in))
    # Ground-truth target uses a rank-r delta LoRA should be able to represent.
    A_true = rng.normal(0, 1, size=(r, d_in))
    B_true = rng.normal(0, 1, size=(d_out, r))
    x = rng.normal(0, 1, size=(200, d_in))
    y = x @ (W0 + B_true @ A_true).T

    layer = LoRALinear(W0, LoRAConfig(r=r, alpha=r, seed=0))  # scaling = 1
    history = layer.fit(x, y, lr=0.05, steps=2000)
    assert history[-1] < history[0]           # loss decreased
    assert history[-1] < 1e-3                 # and converged near zero


def test_frozen_base_weight_never_changes():
    rng = np.random.default_rng(2)
    W0 = rng.normal(0, 1, size=(6, 5))
    original = W0.copy()
    x = rng.normal(0, 1, size=(20, 5))
    y = rng.normal(0, 1, size=(20, 6))
    layer = LoRALinear(W0, LoRAConfig(r=2))
    layer.fit(x, y, lr=0.01, steps=50)
    np.testing.assert_array_equal(layer.W0, original)


def test_merged_weight_matches_forward():
    rng = np.random.default_rng(3)
    W0 = rng.normal(0, 1, size=(7, 9))
    x = rng.normal(0, 1, size=(5, 9))
    layer = LoRALinear(W0, LoRAConfig(r=3, alpha=6))
    layer.A = rng.normal(0, 0.5, size=layer.A.shape)
    layer.B = rng.normal(0, 0.5, size=layer.B.shape)
    # Serving via a single merged weight must equal the two-path forward pass.
    np.testing.assert_allclose(x @ layer.merged_weight().T, layer.forward(x))

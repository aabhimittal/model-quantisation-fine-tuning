import numpy as np
import pytest

from quantune import quantization as q


def test_int8_symmetric_roundtrip_bounded():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1, size=(64, 64))
    qt = q.quantize_int8_symmetric(w)
    err = q.quantization_error(w, qt)
    # int8 over a normal matrix should be very close.
    assert err["rel_frobenius"] < 0.02
    assert qt.codes.dtype == np.int8


def test_int8_affine_exact_endpoints():
    w = np.linspace(-3.0, 5.0, num=256).reshape(16, 16)
    qt = q.quantize_int8_affine(w)
    recon = q.dequantize(qt)
    # Asymmetric quantization fits min/max exactly (up to one step).
    step = qt.scale
    assert abs(recon.min() - w.min()) <= step
    assert abs(recon.max() - w.max()) <= step


def test_blockwise_beats_global_with_outliers():
    rng = np.random.default_rng(1)
    w = rng.normal(0, 1, size=(32, 32))
    w[0, 0] = 50.0  # a single outlier that wrecks a global scale
    global_err = q.quantization_error(w, q.quantize_int4_symmetric(w))
    block_err = q.quantization_error(w, q.quantize_blockwise(w, block_size=16, bits=4))
    assert block_err["rel_frobenius"] < global_err["rel_frobenius"]


def test_nf4_beats_int4_on_normal_weights():
    rng = np.random.default_rng(2)
    w = rng.normal(0, 1, size=(128, 128))
    int4 = q.quantization_error(w, q.quantize_blockwise(w, block_size=64, bits=4))
    nf4 = q.quantization_error(w, q.quantize_nf4(w, block_size=64))
    # NF4's normal-distribution-aware levels should win on Gaussian weights.
    assert nf4["rel_frobenius"] < int4["rel_frobenius"]


def test_nf4_levels_are_sorted_and_span_unit_interval():
    assert np.all(np.diff(q.NF4_LEVELS) > 0)
    assert q.NF4_LEVELS[0] == pytest.approx(-1.0)
    assert q.NF4_LEVELS[-1] == pytest.approx(1.0)
    assert 0.0 in q.NF4_LEVELS  # zero is exactly representable
    assert len(q.NF4_LEVELS) == 16


def test_shape_preserved_across_schemes():
    rng = np.random.default_rng(3)
    w = rng.normal(0, 1, size=(30, 17))  # not a multiple of block_size
    for qt in (
        q.quantize_int8_symmetric(w),
        q.quantize_int8_affine(w),
        q.quantize_blockwise(w, block_size=64),
        q.quantize_nf4(w, block_size=64),
    ):
        assert q.dequantize(qt).shape == w.shape


def test_bits_per_weight_accounts_for_scale_overhead():
    w = np.ones((256, 256))
    small_blocks = q.quantize_nf4(w, block_size=16)
    big_blocks = q.quantize_nf4(w, block_size=256)
    # Smaller blocks mean more scales => higher effective bits/weight.
    assert small_blocks.bits_per_weight > big_blocks.bits_per_weight
    assert big_blocks.bits_per_weight >= 4.0


def test_zero_tensor_does_not_divide_by_zero():
    w = np.zeros((8, 8))
    for qt in (q.quantize_int8_symmetric(w), q.quantize_nf4(w), q.quantize_blockwise(w)):
        assert np.all(np.isfinite(q.dequantize(qt)))

"""Tests for device-resident input: any object exposing
``__cuda_array_interface__`` may stand in for X, y, or the weights, and X is
then binned on the GPU where it already lives.

The protocol side is exercised with a synthetic producer over raw ``cudaMalloc``
memory (through ctypes), because the CI image carries no cupy or torch. The
producer is exactly what the protocol asks for, so a real cupy array reaches the
same code path.

    PYTHONPATH=build-cuda/python pytest python/tests/test_device_input.py
"""

from __future__ import annotations

import ctypes
import hashlib
import tempfile
import types

import bonsai
import numpy as np
import pytest

CUDA_MEMCPY_HOST_TO_DEVICE = 1
PAIRS = [("booster.n_iters", "12"), ("tree.max_depth", "5")]

requires_cuda = pytest.mark.skipif(
    not bonsai.cuda_available(), reason="no CUDA build or no visible device"
)


@pytest.fixture
def to_device():
    """Factory copying a numpy array to fresh device memory, freed at teardown."""
    runtime = ctypes.CDLL("libcudart.so")
    owned: list[ctypes.c_void_p] = []

    def _copy(a: np.ndarray) -> int:
        ptr = ctypes.c_void_p()
        assert runtime.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(a.nbytes)) == 0
        assert (
            runtime.cudaMemcpy(
                ptr,
                a.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(a.nbytes),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
            == 0
        )
        owned.append(ptr)
        return ptr.value

    yield _copy
    for ptr in owned:
        runtime.cudaFree(ptr)


def _reg_data(n=20000, f=8, seed=0):
    """(X, y, w) float32 arrays with signal in the first two columns."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, f), dtype=np.float32)
    y = (X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, n)).astype(np.float32)
    w = (0.5 + rng.random(n)).astype(np.float32)
    return X, y, w


def _cai(ptr, shape, typestr="<f4", stream=None, strides=None, mask=None):
    """A minimal __cuda_array_interface__ producer: the protocol, nothing else."""
    iface = {
        "shape": tuple(shape),
        "typestr": typestr,
        "data": (ptr, True),
        "version": 3,
        "strides": strides,
        "stream": stream,
        "mask": mask,
    }
    return types.SimpleNamespace(__cuda_array_interface__=iface)


def _model_sha(model) -> str:
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as f:
        model.save(f.name)
        return hashlib.sha256(open(f.name, "rb").read()).hexdigest()[:16]


# protocol validation ==============================================================================


def test_device_input_rejects_non_float32():
    with pytest.raises(Exception, match="float32"):
        bonsai.train(PAIRS, _cai(4096, (10, 3), typestr="<f8"), np.zeros(10, np.float32))


def test_device_input_rejects_wrong_rank():
    with pytest.raises(Exception, match="2-dimensional"):
        bonsai.train(PAIRS, _cai(4096, (10,)), np.zeros(10, np.float32))


def test_device_input_rejects_strided_matrix():
    strided = _cai(4096, (10, 3), strides=(4, 40))
    with pytest.raises(Exception, match="C-contiguous"):
        bonsai.train(PAIRS, strided, np.zeros(10, np.float32))


def test_device_input_rejects_mask():
    with pytest.raises(Exception, match="mask"):
        bonsai.train(PAIRS, _cai(4096, (10, 3), mask=8192), np.zeros(10, np.float32))


def test_device_input_rejects_null_pointer():
    with pytest.raises(Exception, match="empty"):
        bonsai.train(PAIRS, _cai(0, (10, 3)), np.zeros(10, np.float32))


@requires_cuda
def test_device_input_rejects_a_host_pointer():
    """A protocol-valid interface over memory that is not a device allocation.

    The protocol carries an address and nothing that says where it lives, so
    the placement check is the only thing standing between a host pointer and
    a kernel reading it.
    """
    with pytest.raises(Exception, match="device memory"):
        bonsai.train(PAIRS, _cai(4096, (10, 3)), np.zeros(10, np.float32))


@pytest.mark.skipif(bonsai.cuda_available(), reason="needs a CUDA-less build or host")
def test_device_input_raises_without_a_device():
    with pytest.raises(Exception, match="cuda_available"):
        bonsai.train(PAIRS, _cai(4096, (10, 3)), np.zeros(10, np.float32))


# device-resident fits =============================================================================


@requires_cuda
@pytest.mark.parametrize("grower", ["depthwise", "leafwise"])
def test_train_on_device_input_is_byte_identical_to_host(to_device, grower):
    """Same model bits from a device pointer as from the numpy array it holds.

    A CPU grower reads the bins the device wrote, materialized on the host, so
    equal model bytes are equal bin ids: the parity claim for device binning.
    """
    X, y, _ = _reg_data()
    pairs = [*PAIRS, ("dispatch.grower_name", grower)]
    host = bonsai.train(pairs, X, y)
    dev = bonsai.train(pairs, _cai(to_device(X), X.shape), y)

    assert _model_sha(dev) == _model_sha(host)
    np.testing.assert_array_equal(np.asarray(dev.predict(X)), np.asarray(host.predict(X)))


@requires_cuda
def test_cuda_grower_on_device_input_matches_host(to_device):
    """A GPU fit is equal to tolerance, not bit for bit.

    Device histogram atomics accumulate in arbitrary order, so two GPU fits of
    the same host array already differ in their last bits; 1e-4 is the bound
    the CUDA suite holds every GPU-versus-host comparison to.
    """
    X, y, _ = _reg_data()
    pairs = [*PAIRS, ("dispatch.grower_name", "cuda_depthwise")]
    host = np.asarray(bonsai.train(pairs, X, y).predict(X))
    dev = np.asarray(bonsai.train(pairs, _cai(to_device(X), X.shape), y).predict(X))

    np.testing.assert_allclose(dev, host, atol=1e-4)


@requires_cuda
def test_train_accepts_device_labels_and_weights(to_device):
    X, y, w = _reg_data()
    host = bonsai.train(PAIRS, X, y, sample_weight=w)
    dev = bonsai.train(
        PAIRS,
        _cai(to_device(X), X.shape),
        _cai(to_device(y), y.shape),
        sample_weight=_cai(to_device(w), w.shape),
    )
    assert _model_sha(dev) == _model_sha(host)


@requires_cuda
def test_train_honors_the_producer_stream(to_device):
    """A legacy-default-stream handle is waited on, not ignored."""
    X, y, _ = _reg_data()
    host = bonsai.train(PAIRS, X, y)
    dev = bonsai.train(PAIRS, _cai(to_device(X), X.shape, stream=1), y)
    assert _model_sha(dev) == _model_sha(host)


@requires_cuda
def test_dataset_from_device_input_bins_on_the_device(to_device):
    X, y, _ = _reg_data()
    ds = bonsai.Dataset(_cai(to_device(X), X.shape), y, max_bin=255)

    assert ds.device == "cuda"
    assert (ds.n_rows, ds.n_features) == X.shape
    host = bonsai.train(PAIRS, X, y)
    assert _model_sha(bonsai.train(PAIRS, ds)) == _model_sha(host)


@requires_cuda
def test_dataset_from_device_input_gathers_a_sample(to_device):
    """The sampled arm: cuts fitted on rows gathered out of the device matrix.

    Every other test here sits below the default n_samples, where the sampler
    names no rows and the gather is a straight download, so this is the only
    one that runs the gather kernel. Byte-identical models pin both which rows
    it picked and the order it wrote them in.
    """
    X, y, _ = _reg_data()
    dev = bonsai.Dataset(_cai(to_device(X), X.shape), y, n_samples=1024)
    host = bonsai.Dataset(X, y, n_samples=1024)

    assert _model_sha(bonsai.train(PAIRS, dev)) == _model_sha(bonsai.train(PAIRS, host))


@requires_cuda
def test_dataset_rejects_a_host_hint_for_device_input(to_device):
    X, y, _ = _reg_data()
    with pytest.raises(Exception, match="device-resident"):
        bonsai.Dataset(_cai(to_device(X), X.shape), y, device="cpu")


@requires_cuda
def test_device_input_refuses_a_disagreeing_device_id(to_device):
    """A fit placed elsewhere raises rather than migrating the matrix.

    The message differs by fleet: a second device makes it the residency
    mismatch, a single-device host makes it the device_id range check.
    """
    X, y, _ = _reg_data()
    with pytest.raises(Exception, match="device"):
        bonsai.train(
            [*PAIRS, ("parallel.device_id", "1")], _cai(to_device(X), X.shape), y
        )

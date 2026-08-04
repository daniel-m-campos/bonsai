"""Tests for device-resident input: any CUDA array supporting DLPack may stand
in for X, y, or the weights, and X is then binned on the GPU where it already
lives.

The protocol side is exercised with a synthetic producer over raw ``cudaMalloc``
memory (through ctypes), because the CI image carries no cupy or torch. The
producer exports exactly the capsule DLPack specifies, so a real cupy array
reaches the same code path.

    PYTHONPATH=build-cuda/python pytest python/tests/test_device_input.py
"""

from __future__ import annotations

import ctypes
import hashlib
import tempfile

import bonsai
import numpy as np
import pytest

CUDA_MEMCPY_HOST_TO_DEVICE = 1
DL_CUDA = 2
DL_FLOAT = 2
PAIRS = [("booster.n_iters", "12"), ("tree.max_depth", "5")]

requires_cuda = pytest.mark.skipif(
    not bonsai.cuda_available(), reason="no CUDA build or no visible device"
)


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.restype = ctypes.py_object
_capsule_new.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)


class _DevicePointer:
    """A minimal DLPack producer over a device pointer: the protocol, nothing else.

    A class rather than a namespace because ``__dlpack__`` is looked up on the
    type. Every export builds a fresh capsule and keeps its structures alive on
    the producer, since the consumer reads them after the call returns.
    """

    def __init__(self, ptr, shape, bits=32, strides=None, device=(DL_CUDA, 0)):
        self.ptr = ptr
        self.shape = tuple(shape)
        self.bits = bits
        self.strides = strides
        self.device = device
        self.exports = 0
        self._alive = []

    def __dlpack_device__(self):
        return self.device

    def __dlpack__(self, **_):
        self.exports += 1
        shape = (ctypes.c_int64 * len(self.shape))(*self.shape)
        strides = (
            (ctypes.c_int64 * len(self.shape))(*self.strides) if self.strides else None
        )
        deleter = _DELETER(lambda _: None)
        managed = _DLManagedTensor()
        managed.dl_tensor.data = ctypes.c_void_p(self.ptr)
        managed.dl_tensor.device = _DLDevice(*self.device)
        managed.dl_tensor.ndim = len(self.shape)
        managed.dl_tensor.dtype = _DLDataType(DL_FLOAT, self.bits, 1)
        managed.dl_tensor.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
        managed.dl_tensor.strides = (
            ctypes.cast(strides, ctypes.POINTER(ctypes.c_int64)) if strides else None
        )
        managed.dl_tensor.byte_offset = 0
        managed.manager_ctx = None
        managed.deleter = deleter
        self._alive.append((managed, shape, strides, deleter))
        return _capsule_new(ctypes.byref(managed), b"dltensor", None)


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


def _model_sha(model) -> str:
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as f:
        model.save(f.name)
        return hashlib.sha256(open(f.name, "rb").read()).hexdigest()[:16]


# protocol validation ==============================================================================


def test_device_input_rejects_non_float32():
    device = _DevicePointer(4096, (10, 3), bits=64)
    with pytest.raises(Exception, match="float32"):
        bonsai.train(PAIRS, device, np.zeros(10, np.float32))


def test_device_input_rejects_wrong_rank():
    with pytest.raises(Exception, match="DLPack"):
        bonsai.train(PAIRS, _DevicePointer(4096, (10,)), np.zeros(10, np.float32))


def test_device_input_rejects_strided_matrix():
    strided = _DevicePointer(4096, (10, 3), strides=(1, 10))
    with pytest.raises(Exception, match="DLPack"):
        bonsai.train(PAIRS, strided, np.zeros(10, np.float32))


def test_device_input_rejects_null_pointer():
    with pytest.raises(Exception, match="empty"):
        bonsai.train(PAIRS, _DevicePointer(0, (10, 3)), np.zeros(10, np.float32))


def test_device_input_rejects_zero_columns():
    with pytest.raises(Exception, match="empty"):
        bonsai.train(PAIRS, _DevicePointer(4096, (10, 0)), np.zeros(10, np.float32))


def test_host_array_stays_on_the_host_path():
    """A numpy array must not be imported by the device arm.

    Both arms accept DLPack, and only the device tag separates them, so this
    pins that a host array still bins on the host and trains the same model.
    """
    X, y, _ = _reg_data(n=2000)

    assert bonsai.Dataset(X, y).device == "cpu"
    assert _model_sha(bonsai.train(PAIRS, X, y)) == _model_sha(
        bonsai.train(PAIRS, bonsai.Dataset(X, y))
    )


@pytest.mark.skipif(bonsai.cuda_available(), reason="needs a CUDA-less build or host")
def test_device_input_raises_without_a_device():
    with pytest.raises(Exception, match="cuda_available"):
        bonsai.train(PAIRS, _DevicePointer(4096, (10, 3)), np.zeros(10, np.float32))


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
    dev = bonsai.train(pairs, _DevicePointer(to_device(X), X.shape), y)

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
    dev = np.asarray(
        bonsai.train(pairs, _DevicePointer(to_device(X), X.shape), y).predict(X)
    )

    np.testing.assert_allclose(dev, host, atol=1e-4)


@requires_cuda
def test_train_accepts_device_labels_and_weights(to_device):
    X, y, w = _reg_data()
    host = bonsai.train(PAIRS, X, y, sample_weight=w)
    dev = bonsai.train(
        PAIRS,
        _DevicePointer(to_device(X), X.shape),
        _DevicePointer(to_device(y), y.shape),
        sample_weight=_DevicePointer(to_device(w), w.shape),
    )
    assert _model_sha(dev) == _model_sha(host)


@requires_cuda
def test_train_imports_through_the_producer(to_device):
    """Ordering is the producer's: DLPack synchronizes at export, bonsai reads
    on the default stream, so the export call is the whole contract."""
    X, y, _ = _reg_data()
    device = _DevicePointer(to_device(X), X.shape)
    host = bonsai.train(PAIRS, X, y)
    dev = bonsai.train(PAIRS, device, y)

    assert device.exports == 1
    assert _model_sha(dev) == _model_sha(host)


@requires_cuda
def test_dataset_from_device_input_bins_on_the_device(to_device):
    X, y, _ = _reg_data()
    ds = bonsai.Dataset(_DevicePointer(to_device(X), X.shape), y, max_bin=255)

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
    dev = bonsai.Dataset(_DevicePointer(to_device(X), X.shape), y, n_samples=1024)
    host = bonsai.Dataset(X, y, n_samples=1024)

    assert _model_sha(bonsai.train(PAIRS, dev)) == _model_sha(bonsai.train(PAIRS, host))


@requires_cuda
def test_dataset_rejects_a_host_hint_for_device_input(to_device):
    X, y, _ = _reg_data()
    with pytest.raises(Exception, match="device-resident"):
        bonsai.Dataset(_DevicePointer(to_device(X), X.shape), y, device="cpu")


@requires_cuda
def test_device_input_refuses_a_disagreeing_device_id(to_device):
    """A fit placed elsewhere raises rather than migrating the matrix.

    The message differs by fleet: a second device makes it the residency
    mismatch, a single-device host makes it the device_id range check.
    """
    X, y, _ = _reg_data()
    with pytest.raises(Exception, match="device"):
        bonsai.train(
            [*PAIRS, ("parallel.device_id", "1")],
            _DevicePointer(to_device(X), X.shape),
            y,
        )

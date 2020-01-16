__version__ = "0.3.0"
__all__ = ["Model", "LossType"]

from ._libomikuji import lib, ffi
from enum import Enum
from typing import Optional


class LossType(Enum):
    HINGE = lib.Hinge
    LOG = lib.Log


class Model(object):
    """A Omikuji model object."""

    def __init__(self, model_ptr, predict_thread_pool_ptr=ffi.NULL):
        """For internal use only. To get model objects, call load or train_on_data instead."""
        assert model_ptr != ffi.NULL
        self._model_ptr = ffi.gc(model_ptr, lib.free_omikuji_model)
        self._predict_thread_pool_ptr = predict_thread_pool_ptr

    @classmethod
    def _init_thread_pool(cls, n_threads: Optional[int]):
        if n_threads is None:
            return ffi.NULL

        n_threads = max(n_threads, 0)
        thread_pool_ptr = lib.init_omikuji_thread_pool(n_threads)
        return ffi.gc(thread_pool_ptr, lib.free_omikuji_thread_pool)

    def init_prediction_thread_pool(self, n_threads: int):
        """Initialize the thread pool for processing model prediction.

        Omikuji uses Rayon for parallelization. If this method is not called, the global Rayon thread pool is used.

        """
        self._predict_thread_pool_ptr = self._init_thread_pool(n_threads)

    @classmethod
    def load(cls, path: str):
        """Load Omikuji model from the given directory."""
        model_ptr = lib.load_omikuji_model(ffi.new("char[]", path.encode()))
        if model_ptr == ffi.NULL:
            raise RuntimeError("Failed to load model from %s" % (path,))

        return cls(model_ptr)

    @property
    def n_features(self):
        """Get the expected dimension of feature vectors."""
        return lib.omikuji_n_features(self._model_ptr)

    def save(self, path):
        """Save Omikuji model to the given directory."""
        assert self._model_ptr != ffi.NULL
        if (
            lib.save_omikuji_model(self._model_ptr, ffi.new("char[]", path.encode()))
            < 0
        ):
            raise RuntimeError("Failed to save model to %s" % (path,))

    def densify_weights(
        self, max_sparse_density: float = 0.1, n_threads: Optional[int] = None
    ):
        """Densify model weights to speed up prediction at the cost of more memory usage.

        Note that this method is NOT thread-safe. The caller is responsible for making sure that no other method call is
        happening at the same time.

        """
        assert self._model_ptr != ffi.NULL
        thread_pool_ptr = (
            self._predict_thread_pool_ptr
            if n_threads is None
            else self._init_thread_pool(n_threads)
        )
        lib.densify_omikuji_model(self._model_ptr, max_sparse_density, thread_pool_ptr)

    def predict(self, feature_value_pairs, beam_size=10, top_k=10):
        """Make predictions with Omikuji model."""
        assert self._model_ptr != ffi.NULL

        feature_value_pairs = sorted(feature_value_pairs, key=lambda kv: kv[0])
        n_features = self.n_features
        if feature_value_pairs:
            for (f1, _), (f2, _) in zip(feature_value_pairs, feature_value_pairs[1:]):
                assert 0 <= f1 < f2 < n_features, "Incorrect feature index"

        input_len = len(feature_value_pairs)
        feature_indices = ffi.new("uint32_t[]", input_len)
        feature_values = ffi.new("float[]", input_len)
        for i, (f, v) in enumerate(feature_value_pairs):
            feature_indices[i] = f
            feature_values[i] = v

        output_labels = ffi.new("uint32_t[]", top_k)
        output_scores = ffi.new("float[]", top_k)
        top_k = lib.omikuji_predict(
            self._model_ptr,
            beam_size,
            input_len,
            feature_indices,
            feature_values,
            top_k,
            output_labels,
            output_scores,
            self._predict_thread_pool_ptr,
        )
        output = []
        for i in range(top_k):
            output.append((output_labels[i], output_scores[i]))

        return output

    @classmethod
    def default_hyper_param(cls):
        """Get default training hyper-parameters."""
        return lib.omikuji_default_hyper_param()

    @classmethod
    def train_on_data(
        cls, data_path: str, hyper_param=None, n_threads: Optional[int] = None
    ):
        """Train a model with the given data and hyper-parameters."""
        thread_pool_ptr = cls._init_thread_pool(n_threads)
        dataset_ptr = lib.load_omikuji_data_set(
            ffi.new("char[]", data_path.encode()), thread_pool_ptr
        )
        if dataset_ptr == ffi.NULL:
            raise RuntimeError("Failed to load data from %s" % (data_path,))

        dataset_ptr = ffi.gc(dataset_ptr, lib.free_omikuji_data_set)

        if hyper_param is None:
            hyper_param = cls.default_hyper_param()

        model_ptr = lib.train_omikuji_model(dataset_ptr, hyper_param, thread_pool_ptr)
        if model_ptr == ffi.NULL:
            raise RuntimeError("Failed to train model")

        return Model(model_ptr)


def init_logger():
    """Initialize a simple logger that writes to stdout."""
    if lib.omikuji_init_logger() < 0:
        raise RuntimeWarning("Failed to initialize logger")


ffi.init_once(init_logger, "omikuji_init_logger")

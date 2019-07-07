__version__ = "0.0.1"
__all__ = ["Model", "LossType", "init_rayon_threads"]

from ._libparabel import lib, ffi
from enum import Enum


class LossType(Enum):
    HINGE = lib.Hinge
    LOG = lib.Log


class Model(object):
    """A parabel model object."""

    def __init__(self, model_ptr):
        """For internal use only. To get model objects, call load or Trainer.train_on_data instead."""
        assert model_ptr != ffi.NULL
        self._model_ptr = ffi.gc(model_ptr, lib.free_parabel_model)

    @classmethod
    def load(cls, path: str):
        """Load parabel model from file of the given path."""
        model_ptr = lib.load_parabel_model(ffi.new("char[]", path.encode()))
        if model_ptr == ffi.NULL:
            raise RuntimeError(f"Failed to load model from {path}")

        return cls(model_ptr)

    @property
    def n_features(self):
        """Get the expected dimension of feature vectors."""
        return lib.parabel_n_features(self._model_ptr)

    def save(self, path):
        """Save parabel model to file of the given path."""
        assert self._model_ptr != ffi.NULL
        if (
            lib.save_parabel_model(self._model_ptr, ffi.new("char[]", path.encode()))
            < 0
        ):
            raise RuntimeError(f"Failed to save model to {path}")

    def predict(self, feature_value_pairs, beam_size=10, top_k=10):
        """Make predictions with parabel model."""
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
        top_k = lib.parabel_predict(
            self._model_ptr,
            beam_size,
            input_len,
            feature_indices,
            feature_values,
            top_k,
            output_labels,
            output_scores,
        )
        output = []
        for i in range(top_k):
            output.append((output_labels[i], output_scores[i]))

        return output

    @classmethod
    def default_hyper_param(cls):
        """Get default training hyper-parameters."""
        return lib.parabel_default_hyper_param()

    @classmethod
    def train_on_data(cls, data_path: str, hyper_param=None):
        """Train a model with the given data and hyper-parameters."""
        dataset_ptr = lib.load_parabel_data_set(ffi.new("char[]", data_path.encode()))
        if dataset_ptr == ffi.NULL:
            raise RuntimeError(f"Failed to load data from {data_path}")

        dataset_ptr = ffi.gc(dataset_ptr, lib.free_parabel_data_set)

        if hyper_param is None:
            hyper_param = cls.default_hyper_param()

        model_ptr = lib.train_parabel_model(dataset_ptr, hyper_param)
        if model_ptr == ffi.NULL:
            raise RuntimeError(f"Failed to train model")

        return Model(model_ptr)


def init_logger():
    """Initialize a simple logger that writes to stdout."""
    if lib.parabel_init_logger() < 0:
        raise RuntimeWarning("Failed to initialize logger")


ffi.init_once(init_logger, "parabel_init_logger")


def init_rayon_threads(n_threads):
    """Optionally initialize Rayon global thread pool with certain number of threads."""
    if lib.rayon_init_threads(n_threads) < 0:
        raise RuntimeWarning("Failed to initialize Rayon thread-pool")

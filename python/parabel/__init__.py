__version__ = "0.0.1"
__all__ = ["Model", "LossType", "Trainer", "init_rayon_threads"]

from ._libparabel import lib, ffi

try:
    from dataclasses import dataclass
except ImportError:
    # Give up on the niceties of dataclass for Python <3.7
    def dataclass(cls):
        return cls


from enum import Enum


class Model(object):
    """A parabel model object."""

    def __init__(self, model_ptr):
        """For internal use only. To get model objects, call load or Trainer.train_on_data instead."""
        assert model_ptr != ffi.NULL
        self._model_ptr = ffi.gc(model_ptr, lib.free_parabel_model)

    @classmethod
    def load(cls, path):
        """Load parabel model from file of the given path."""
        model_ptr = lib.load_parabel_model(ffi.new("char[]", path.encode()))
        if model_ptr == ffi.NULL:
            raise RuntimeError(f"Failed to load model from {path}")

        return cls(model_ptr)

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


class LossType(Enum):
    HINGE = 0
    LOG = 1


@dataclass
class Trainer:
    """Trainer for parabel model."""

    n_trees: int = 3
    max_leaf_size: int = 100
    cluster_eps: float = 0.0001
    centroid_threshold: float = 0.0
    linear_loss_type: LossType = LossType.HINGE
    linear_eps: float = 0.1
    linear_c: float = 1.0
    linear_weight_threshold: float = 0.1
    linear_max_iter: int = 20

    def train_on_data(self, data_path):
        """Train parabel model on the given dataset file."""
        dataset_ptr = lib.load_parabel_data_set(ffi.new("char[]", data_path.encode()))
        if dataset_ptr == ffi.NULL:
            raise RuntimeError(f"Failed to load data from {data_path}")

        dataset_ptr = ffi.gc(dataset_ptr, lib.free_parabel_data_set)
        trainer_ptr = ffi.gc(
            lib.create_parabel_trainer(
                self.n_trees,
                self.max_leaf_size,
                self.cluster_eps,
                self.centroid_threshold,
                lib.Hinge if self.linear_loss_type == LossType.HINGE else lib.Log,
                self.linear_eps,
                self.linear_c,
                self.linear_weight_threshold,
                self.linear_max_iter,
            ),
            lib.free_parabel_trainer,
        )

        model_ptr = lib.train_parabel_model(trainer_ptr, dataset_ptr)
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

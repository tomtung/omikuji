#ifndef PARABEL_H
#define PARABEL_H

#include <cstdarg>
#include <cstdint>
#include <cstdlib>

namespace parabel {

enum class LossType {
  Hinge = 0,
  Log = 1,
};

struct ParabelDataSet;

struct ParabelModel;

struct ParabelTrainer;

extern "C" {

/// Create model trainer from the given hyper-parameters.
ParabelTrainer *create_parabel_trainer(size_t n_trees,
                                       size_t max_leaf_size,
                                       float cluster_eps,
                                       float centroid_threshold,
                                       LossType linear_loss_type,
                                       float linear_eps,
                                       float linear_c,
                                       float linear_weight_threshold,
                                       uint32_t linear_max_iter);

/// Free data set object.
void free_parabel_data_set(ParabelDataSet *dataset_ptr);

/// Free parabel model from memory.
void free_parabel_model(ParabelModel *model_ptr);

/// Free parabel trainer.
void free_parabel_trainer(ParabelTrainer *trainer_ptr);

/// Load a data file from the Extreme Classification Repository.
ParabelDataSet *load_parabel_data_set(const char *path);

/// Load parabel model from file of the given path.
ParabelModel *load_parabel_model(const char *path);

/// Initialize a simple logger that writes to stdout.
int8_t parabel_init_logger();

/// Make predictions with parabel model.
size_t parabel_predict(ParabelModel *model_ptr,
                       size_t beam_size,
                       size_t input_len,
                       const uint32_t *feature_indices,
                       const float *feature_values,
                       size_t output_len,
                       uint32_t *output_labels,
                       float *output_scores);

/// Optionally initialize Rayon global thread pool with certain number of threads.
int8_t rayon_init_threads(size_t n_threads);

/// Save parabel model to file of the given path.
int8_t save_parabel_model(ParabelModel *model_ptr, const char *path);

/// Train parabel model on the given data set.
ParabelModel *train_parabel_model(const ParabelTrainer *trainer_ptr,
                                  const ParabelDataSet *dataset_ptr);

} // extern "C"

} // namespace parabel

#endif // PARABEL_H

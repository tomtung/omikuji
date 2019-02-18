use crate::util::create_progress_bar;
use crate::{DataSet, Index, IndexValueVec, Model};
use hashbrown::HashSet;
use itertools::izip;
use log::info;
use rayon::prelude::*;
use std::sync::Mutex;

fn precision_at_k(
    max_k: usize,
    true_labels: &[HashSet<Index>],
    predicted_labels: &[IndexValueVec],
) -> Vec<f32> {
    assert_eq!(true_labels.len(), predicted_labels.len());
    let mut ps = vec![0.; max_k];
    for (truth, predictions) in izip!(true_labels, predicted_labels) {
        let mut n_correct = 0;
        for k in 0..max_k.min(predictions.len()) {
            if truth.contains(&predictions[k].0) {
                n_correct += 1;
            }
            ps[k] += n_correct as f32 / (k + 1) as f32;
        }
    }
    for p in &mut ps {
        *p /= predicted_labels.len() as f32;
    }
    ps
}

pub fn test_all(
    model: &Model,
    test_dataset: &DataSet,
    beam_size: usize,
) -> (Vec<IndexValueVec>, Vec<f32>) {
    let n_examples = test_dataset.feature_lists.len();
    let pb = Mutex::new(create_progress_bar(n_examples as u64));
    let start_t = time::precise_time_s();
    let predicted_labels = test_dataset
        .feature_lists
        .par_iter()
        .map(|feature_vec| {
            let predictions = model.predict(feature_vec, beam_size);
            pb.lock().expect("Failed to lock progress bar").add(1);
            predictions
        })
        .collect::<Vec<_>>();
    info!(
        "Done testing on {} examples; it took {:.2}s",
        n_examples,
        time::precise_time_s() - start_t
    );

    let precisions = precision_at_k(5, &test_dataset.label_sets, &predicted_labels);
    info!(
        "Precision@[1, 3, 5] = [{:.2}, {:.2}, {:.2}]",
        precisions[0] * 100.,
        precisions[2] * 100.,
        precisions[4] * 100.,
    );

    (predicted_labels, precisions)
}

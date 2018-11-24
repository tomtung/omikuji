use crate::data::{Example, Feature, Label, SparseVector};
use std::collections::HashMap;

/// Compute average feature vectors for labels in a given dataset, l2-normalized and pruned with
/// a given threshold.
fn compute_feature_vectors_per_label(
    examples: &[Example],
    threshold: f32,
) -> (Vec<Label>, Vec<SparseVector<Feature>>) {
    let mut label_to_feature_to_sum = HashMap::<Label, HashMap<Feature, f32>>::new();
    for example in examples {
        for label in &example.labels {
            let mut feature_to_sum = label_to_feature_to_sum.entry(label.to_owned()).or_default();
            for (feature, value) in &example.features.entries {
                *feature_to_sum.entry(feature.to_owned()).or_default() += value;
            }
        }
    }
    label_to_feature_to_sum
        .into_iter()
        .map(|(label, feature_to_sum)| {
            let mut v = SparseVector::from(feature_to_sum);
            v.l2_normalize();
            v.prune_with_threshold(threshold);
            (label, v)
        }).unzip()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::*;
    use std::iter::FromIterator;

    #[test]
    fn test_compute_label_vectors() {
        let examples = vec![
            Example {
                features: SparseVector::from(vec![(0, 1.), (2, 2.)]),
                labels: vec![0, 1],
            },
            Example {
                features: SparseVector::from(vec![(1, 1.), (3, 2.)]),
                labels: vec![0, 2],
            },
            Example {
                features: SparseVector::from(vec![(0, 1.), (3, 2.)]),
                labels: vec![1, 2],
            },
        ];

        let (labels, vecs) = compute_feature_vectors_per_label(&examples, 1. / 18f32.sqrt() + 1e-4);
        assert_eq!(
            HashMap::<Label, SparseVector<Feature>>::from_iter(
                vec![
                    (
                        0,
                        SparseVector::from(vec![
                            (0, 1. / 10f32.sqrt()),
                            (1, 1. / 10f32.sqrt()),
                            (2, 2. / 10f32.sqrt()),
                            (3, 2. / 10f32.sqrt()),
                        ])
                    ),
                    (
                        1,
                        SparseVector::from(vec![
                            (0, 2. / 12f32.sqrt()),
                            (2, 2. / 12f32.sqrt()),
                            (3, 2. / 12f32.sqrt()),
                        ])
                    ),
                    (
                        2,
                        SparseVector::from(vec![
                            // The first two entries are pruned by the given threshold
                            // (0, 1. / 18f32.sqrt()),
                            // (1, 1. / 18f32.sqrt()),
                            (3, 4. / 18f32.sqrt()),
                        ])
                    ),
                ].into_iter()
            ),
            HashMap::<Label, SparseVector<Feature>>::from_iter(
                labels.into_iter().zip(vecs.into_iter())
            )
        );
    }
}

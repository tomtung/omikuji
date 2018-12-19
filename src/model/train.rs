use super::{cluster, liblinear, Tree, TreeNode};
use crate::data::{Example, Feature, Label, SparseVector};
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use std::iter::FromIterator;

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
        })
        .unzip()
}

#[derive(Copy, Clone, Debug)]
struct HyperParam {
    classifier_hyper_param: liblinear::TrainHyperParam,
    cluster_eps: f32,
    max_leaf_size: usize,
}

fn train_leaf_node(
    examples: &[&Example],
    labels: &[Label],
    hyper_param: &liblinear::TrainHyperParam,
) -> TreeNode {
    assert!(!examples.is_empty());
    assert!(!labels.is_empty());
    let feature_vecs = examples.iter().map(|e| &e.features).collect::<Vec<_>>();
    let label_classifier_pairs = labels
        .iter()
        .map(|label| {
            let labels = examples
                .iter()
                .map(|e| e.labels.contains(label))
                .collect::<Vec<_>>();
            let model = liblinear::Model::train(&feature_vecs, &labels, hyper_param);
            (label.to_owned(), model)
        })
        .collect::<Vec<_>>();
    TreeNode::LeafNode {
        label_classifier_pairs,
    }
}

/// Determines the height of the tree.
///
/// We use depth instead of leaf size for termination condition. This could cause over-splitting of
/// nodes, resulting leaves with less than half of the upper bound, but also this makes the binary
/// tree complete, which simplifies beam search.
#[inline]
fn compute_tree_height(n_labels: usize, max_leaf_size: usize) -> usize {
    assert!(max_leaf_size > 1);
    ((n_labels as f32) / (max_leaf_size as f32)).log2().ceil() as usize
}

pub struct TreeTrainer<'a> {
    examples: Vec<&'a Example>,
    labels: Vec<Label>,
    label_centroids: Vec<SparseVector<Feature>>,
    tree_height: usize,
    cluster_eps: f32,
    max_leaf_size: usize,
    classifier_hyper_param: liblinear::TrainHyperParam,
}

impl<'a> TreeTrainer<'a> {
    pub fn initialize(
        examples: &'a [Example],
        centroid_threshold: f32,
        cluster_eps: f32,
        max_leaf_size: usize,
        classifier_loss_type: super::liblinear::LossType,
        classifier_eps: f32,
        classifier_c: f32,
        classifier_weight_threshold: f32,
    ) -> Self {
        let (labels, label_centroids) =
            compute_feature_vectors_per_label(&examples, centroid_threshold);
        let tree_height = compute_tree_height(labels.len(), max_leaf_size);
        Self {
            examples: examples.iter().collect_vec(),
            labels,
            label_centroids,
            tree_height,
            cluster_eps,
            max_leaf_size,
            classifier_hyper_param: liblinear::TrainHyperParam {
                loss_type: classifier_loss_type,
                eps: classifier_eps,
                C: classifier_c,
                weight_threshold: classifier_weight_threshold,
            },
        }
    }

    pub fn train(&self) -> Tree {
        Tree {
            root: self.train_subtree(
                self.tree_height,
                &self.examples,
                &self.labels,
                &self.label_centroids.iter().collect_vec(),
            ),
        }
    }

    fn train_subtree(
        &self,
        height: usize,
        examples: &[&Example],
        labels: &[Label],
        label_centroids: &[&SparseVector<Feature>],
    ) -> TreeNode {
        assert!(!examples.is_empty());
        assert!(!labels.is_empty());

        // If reached maximum depth, build and return a leaf node
        if height == 0 {
            assert!(labels.len() <= self.max_leaf_size);
            return train_leaf_node(examples, labels, &self.classifier_hyper_param);
        }

        // Split labels into 2 sets by clustering
        assert!(labels.len() > 1);
        let mut label_splits = [
            (
                Vec::<Label>::with_capacity(labels.len() / 2 + 1),
                Vec::<&SparseVector<Feature>>::with_capacity(labels.len() / 2 + 1),
            ),
            (
                Vec::<Label>::with_capacity(labels.len() / 2 + 1),
                Vec::<&SparseVector<Feature>>::with_capacity(labels.len() / 2 + 1),
            ),
        ];
        for (&label, centroid, assignment) in izip!(
            labels,
            label_centroids,
            cluster::balanced_2means(label_centroids, self.cluster_eps)
        ) {
            let (split_labels, split_centroids) = &mut label_splits[assignment as usize];
            split_labels.push(label);
            split_centroids.push(centroid);
        }

        // For each split, train an example classifier and recursively train a subtree
        let example_feature_vecs = examples.iter().map(|e| &e.features).collect::<Vec<_>>();
        let mut node_model_pairs = label_splits
            .iter()
            .map(|(split_labels, split_centroids)| {
                // An example belongs to the current split if it has any of the split labels
                let mut split_examples = Vec::<&Example>::with_capacity(examples.len());
                let mut example_in_split = vec![false; examples.len()];
                let split_label_set = HashSet::<Label>::from_iter(split_labels.iter().cloned());
                for (i, &example) in examples.iter().enumerate() {
                    if !example.labels.is_disjoint(&split_label_set) {
                        example_in_split[i] = true;
                        split_examples.push(example);
                    }
                }

                // Train a classifier that predicts whether an example belongs to the current split
                let model = liblinear::Model::train(
                    &example_feature_vecs,
                    &example_in_split,
                    &self.classifier_hyper_param,
                );

                // Train a subtree for the current split
                let subtree =
                    self.train_subtree(height - 1, &split_examples, split_labels, split_centroids);

                (Box::new(subtree), model)
            })
            .collect::<Vec<_>>();

        // Convert to fixed-length array
        assert_eq!(2, node_model_pairs.len());
        let child_classifier_pairs = [
            node_model_pairs.pop().unwrap(),
            node_model_pairs.pop().unwrap(),
        ];

        TreeNode::BranchNode {
            child_classifier_pairs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn test_compute_label_vectors() {
        let examples = vec![
            Example {
                features: SparseVector::from(vec![(0, 1.), (2, 2.)]),
                labels: HashSet::from_iter(vec![0, 1]),
            },
            Example {
                features: SparseVector::from(vec![(1, 1.), (3, 2.)]),
                labels: HashSet::from_iter(vec![0, 2]),
            },
            Example {
                features: SparseVector::from(vec![(0, 1.), (3, 2.)]),
                labels: HashSet::from_iter(vec![1, 2]),
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
                ]
                .into_iter()
            ),
            HashMap::<Label, SparseVector<Feature>>::from_iter(
                labels.into_iter().zip(vecs.into_iter())
            )
        );
    }

    #[test]
    fn test_compute_tree_height() {
        assert_eq!(compute_tree_height(2, 2), 0);
        assert_eq!(compute_tree_height(3, 2), 1);
        assert_eq!(compute_tree_height(4, 2), 1);
        assert_eq!(compute_tree_height(5, 2), 2);
        assert_eq!(compute_tree_height(6, 2), 2);
    }
}

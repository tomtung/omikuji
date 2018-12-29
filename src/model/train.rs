use super::{cluster, liblinear, Tree, TreeNode};
use crate::data::DataSet;
use crate::mat_util::*;
use crate::{Index, IndexSet, IndexValueVec, SparseMat, SparseMatView};
use hashbrown::HashMap;
use itertools::{izip, Itertools};
use std::cmp::{max, min};
use std::iter::FromIterator;

/// Compute centroid feature vectors for labels in a given dataset, pruned with the given threshold.
///
/// Assumes that dataset is well-formed.
fn compute_label_centroids(dataset: &DataSet, threshold: f32) -> (Vec<Index>, Vec<IndexValueVec>) {
    let mut label_to_feature_to_sum =
        HashMap::<Index, HashMap<Index, f32>>::with_capacity(dataset.n_labels);
    for (features, labels) in izip!(&dataset.feature_lists, &dataset.label_sets) {
        for &label in labels {
            let feature_to_sum = label_to_feature_to_sum.entry(label).or_default();
            for &(feature, value) in features {
                *feature_to_sum.entry(feature).or_default() += value;
            }
        }
    }

    label_to_feature_to_sum
        .into_iter()
        .map(|(label, feature_to_sum)| {
            let mut v = feature_to_sum.into_iter().collect_vec();
            v.l2_normalize();
            v.prune_with_threshold(threshold);
            v.sort_by_index();
            (label, v)
        })
        .unzip()
}

#[derive(Copy, Clone, Debug)]
struct HyperParam {
    pub max_leaf_size: usize,
    pub cluster_eps: f32,
    pub centroid_threshold: f32,
    pub classifier: liblinear::TrainHyperParam,
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
    example_feature_matrix: SparseMat,
    example_labels: Vec<&'a IndexSet>,
    all_labels: Vec<Index>,
    label_centroid_matrix: SparseMat,
    tree_height: usize,
    hyper_param: HyperParam,
}

impl<'a> TreeTrainer<'a> {
    /// Initialize a reusable tree trainer with the dataset and hyper-parameters.
    ///
    /// Dataset is assumed to be well-formed.
    pub(self) fn initialize(dataset: &'a DataSet, hyper_param: HyperParam) -> Self {
        assert_eq!(dataset.feature_lists.len(), dataset.label_sets.len());
        let example_feature_matrix = dataset
            .feature_lists
            .copy_normalized_with_bias_to_csrmat(dataset.n_features);
        let example_labels = dataset.label_sets.iter().collect_vec();
        let (all_labels, label_centroids) =
            compute_label_centroids(&dataset, hyper_param.centroid_threshold);
        let label_centroid_matrix = label_centroids.copy_to_csrmat(dataset.n_features);
        let tree_height = compute_tree_height(all_labels.len(), hyper_param.max_leaf_size);
        Self {
            example_feature_matrix,
            example_labels,
            all_labels,
            label_centroid_matrix,
            tree_height,
            hyper_param,
        }
    }

    pub(self) fn train(&self) -> Tree {
        let identity_map = (0..self.example_feature_matrix.cols() as Index).collect_vec();
        Tree {
            root: self.train_subtree(
                self.tree_height,
                (
                    self.example_feature_matrix.view(),
                    &self.example_labels,
                    &identity_map,
                ),
                (&self.all_labels, self.label_centroid_matrix.view()),
            ),
        }
    }

    fn train_leaf_node(
        &self,
        (example_feature_matrix, example_labels, col_index_to_feature): (
            SparseMatView,
            &[&IndexSet],
            &[Index],
        ),
        leaf_labels: &[Index],
    ) -> TreeNode {
        let n_examples = example_feature_matrix.rows();
        assert_eq!(n_examples, example_labels.len());
        assert!(n_examples > 0);
        TreeNode::LeafNode {
            label_classifier_pairs: leaf_labels
                .iter()
                .map(|&leaf_label| {
                    let labels = example_labels
                        .iter()
                        .map(|example_labels| example_labels.contains(&leaf_label))
                        .collect_vec();

                    let classifier = liblinear::Model::train(
                        &example_feature_matrix,
                        &labels,
                        &self
                            .hyper_param
                            .classifier
                            .adapt_to_sample_size(n_examples, self.example_feature_matrix.rows()),
                    )
                    .remap_features_indices(
                        col_index_to_feature,
                        self.example_feature_matrix.cols(),
                    );
                    (leaf_label, classifier)
                })
                .collect(),
        }
    }

    fn split_branch(
        &self,
        labels: &[Index],
        label_centroid_matrix: &SparseMatView,
    ) -> Vec<(Vec<Index>, SparseMat)> {
        let n_labels = labels.len();
        assert!(n_labels > 1);
        assert_eq!(n_labels, label_centroid_matrix.rows());

        let label_assignments =
            cluster::balanced_2means(&label_centroid_matrix, self.hyper_param.cluster_eps);
        assert_eq!(n_labels, label_assignments.len());

        let (true_indices, false_indices) =
            (0..n_labels).partition::<Vec<_>, _>(|&i| label_assignments[i]);
        assert!(
            max(true_indices.len(), false_indices.len())
                - min(true_indices.len(), false_indices.len())
                <= 1
        );
        [true_indices, false_indices]
            .iter()
            .map(|indices| {
                let split_labels = indices.iter().map(|&i| labels[i]).collect_vec();
                let (split_centroid_matrix, _) =
                    shrink_column_indices(label_centroid_matrix.copy_outer_dims(indices));
                (split_labels, split_centroid_matrix)
            })
            .collect()
    }

    fn find_examples_with_labels(
        example_labels: &[&IndexSet],
        split_labels: &[Index],
    ) -> Vec<usize> {
        // An example belongs to the current split if it has any of the split labels
        let split_label_set = IndexSet::from_iter(split_labels.iter().cloned());
        example_labels
            .iter()
            .enumerate()
            .filter_map(|(i, labels)| {
                if labels.is_disjoint(&split_label_set) {
                    None
                } else {
                    Some(i)
                }
            })
            .collect_vec()
    }

    fn train_branch_split_classifier(
        &self,
        example_feature_matrix: &SparseMatView,
        split_example_indices: &[usize],
    ) -> liblinear::Model {
        let n_examples = example_feature_matrix.rows();
        let mut example_in_split = vec![false; example_feature_matrix.rows()];
        for &i in split_example_indices {
            example_in_split[i] = true;
        }

        liblinear::Model::train(
            &example_feature_matrix,
            &example_in_split,
            &self
                .hyper_param
                .classifier
                .adapt_to_sample_size(n_examples, self.example_feature_matrix.rows()),
        )
    }

    fn train_subtree(
        &self,
        height: usize,
        (example_feature_matrix, example_labels, col_index_to_feature): (
            SparseMatView,
            &[&IndexSet],
            &[Index],
        ),
        (subtree_labels, subtree_label_centroid_matrix): (&[Index], SparseMatView),
    ) -> TreeNode {
        assert_eq!(example_feature_matrix.rows(), example_labels.len());
        assert!(example_feature_matrix.rows() > 0);

        // If reached maximum depth, build and return a leaf node
        if height == 0 {
            assert!(subtree_labels.len() <= self.hyper_param.max_leaf_size);
            return self.train_leaf_node(
                (example_feature_matrix, example_labels, col_index_to_feature),
                subtree_labels,
            );
        }

        let mut child_classifier_pairs = self
            .split_branch(subtree_labels, &subtree_label_centroid_matrix)
            .into_iter()
            .map(|(split_labels, split_label_centroid_matrix)| {
                let split_example_indices =
                    Self::find_examples_with_labels(example_labels, &split_labels);

                let classifier = self
                    .train_branch_split_classifier(&example_feature_matrix, &split_example_indices)
                    .remap_features_indices(
                        col_index_to_feature,
                        self.example_feature_matrix.cols(),
                    );

                // Train a subtree for the current split
                let subtree = {
                    let (split_example_feature_matrix, mut new_index_to_old) =
                        shrink_column_indices(
                            example_feature_matrix.copy_outer_dims(&split_example_indices),
                        );
                    for index in &mut new_index_to_old {
                        *index = col_index_to_feature[*index as usize];
                    }

                    let split_examples_labels = split_example_indices
                        .iter()
                        .map(|&i| example_labels[i])
                        .collect_vec();

                    self.train_subtree(
                        height - 1,
                        (
                            split_example_feature_matrix.view(),
                            &split_examples_labels,
                            &new_index_to_old,
                        ),
                        (&split_labels, split_label_centroid_matrix.view()),
                    )
                };

                (Box::new(subtree), classifier)
            })
            .collect_vec();

        assert_eq!(2, child_classifier_pairs.len());
        TreeNode::BranchNode {
            child_classifier_pairs: [
                child_classifier_pairs.pop().unwrap(),
                child_classifier_pairs.pop().unwrap(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn test_compute_label_centroids() {
        let dataset = DataSet {
            n_features: 4,
            n_labels: 3,
            feature_lists: vec![
                vec![(0, 1.), (2, 2.)],
                vec![(1, 1.), (3, 2.)],
                vec![(0, 1.), (3, 2.)],
            ],
            label_sets: vec![
                IndexSet::from_iter(vec![0, 1]),
                IndexSet::from_iter(vec![0, 2]),
                IndexSet::from_iter(vec![1, 2]),
            ],
        };

        let (labels, vecs) = compute_label_centroids(&dataset, 1. / 18f32.sqrt() + 1e-4);
        assert_eq!(
            HashMap::<Index, IndexValueVec>::from_iter(
                vec![
                    (
                        0,
                        vec![
                            (0, 1. / 10f32.sqrt()),
                            (1, 1. / 10f32.sqrt()),
                            (2, 2. / 10f32.sqrt()),
                            (3, 2. / 10f32.sqrt()),
                        ]
                    ),
                    (
                        1,
                        vec![
                            (0, 2. / 12f32.sqrt()),
                            (2, 2. / 12f32.sqrt()),
                            (3, 2. / 12f32.sqrt()),
                        ]
                    ),
                    (
                        2,
                        vec![
                            // The first two entries are pruned by the given threshold
                            // (0, 1. / 18f32.sqrt()),
                            // (1, 1. / 18f32.sqrt()),
                            (3, 4. / 18f32.sqrt()),
                        ]
                    ),
                ]
                .into_iter()
            ),
            HashMap::<Index, IndexValueVec>::from_iter(labels.into_iter().zip(vecs.into_iter()))
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

use super::{cluster, liblinear, Model, Tree, TreeNode};
use crate::data::DataSet;
use crate::mat_util::*;
use crate::{Index, IndexSet, IndexValueVec, SparseMat};
use derive_builder::Builder;
use hashbrown::HashMap;
use itertools::{izip, Itertools};
use pbr::ProgressBar;
use rayon::prelude::*;
use std::io::{stderr, Stderr};
use std::iter::FromIterator;
use std::sync::Mutex;

/// Model training hyper-parameters.
#[derive(Builder, Copy, Clone, Debug)]
pub struct HyperParam {
    #[builder(default = "3")]
    pub n_trees: usize,

    #[builder(default = "100")]
    pub max_leaf_size: usize,

    #[builder(default = "0.0001")]
    pub cluster_eps: f32,

    #[builder(default = "0.")]
    pub centroid_threshold: f32,

    #[builder(default)]
    pub linear: liblinear::HyperParam,
}

impl HyperParam {
    /// Create a builder object.
    pub fn builder() -> HyperParamBuilder {
        HyperParamBuilder::default()
    }

    /// Train a parabel model on the given dataset.
    pub fn train(&self, dataset: &DataSet) -> Model {
        let trainer = TreeTrainer::initialize(dataset, *self);
        let trees: Vec<_> = (0..self.n_trees)
            .into_par_iter()
            .map(|_| trainer.train())
            .collect();
        Model {
            trees,
            n_features: dataset.n_features,
        }
    }
}

struct TreeTrainer<'a> {
    all_examples: TrainingExamples<'a>,
    all_labels: LabelCluster,
    tree_height: usize,
    hyper_param: HyperParam,
    progress_bar: Mutex<ProgressBar<Stderr>>,
}

impl<'a> TreeTrainer<'a> {
    /// Initialize a reusable tree trainer with the dataset and hyper-parameters.
    ///
    /// Dataset is assumed to be well-formed.
    fn initialize(dataset: &'a DataSet, hyper_param: HyperParam) -> Self {
        assert_eq!(dataset.feature_lists.len(), dataset.label_sets.len());
        let (all_examples, all_labels) = rayon::join(
            || TrainingExamples::new_from_dataset(dataset),
            || LabelCluster::new_from_dataset(dataset, hyper_param.centroid_threshold),
        );

        let tree_height = Self::compute_tree_height(all_labels.len(), hyper_param.max_leaf_size);
        let progress_bar =
            Self::create_progress_bar(all_labels.len(), tree_height, hyper_param.n_trees);
        Self {
            all_examples,
            all_labels,
            tree_height,
            hyper_param,
            progress_bar,
        }
    }

    fn create_progress_bar(
        n_labels: usize,
        tree_height: usize,
        n_trees: usize,
    ) -> Mutex<ProgressBar<Stderr>> {
        let n_classifiers_per_tree = 2usize.pow(tree_height as u32 + 1) - 2 + n_labels;
        Mutex::new(ProgressBar::on(
            stderr(),
            (n_trees * n_classifiers_per_tree) as u64,
        ))
    }

    /// Determines the height of the tree.
    ///
    /// We use depth instead of leaf size for termination condition. This could cause over-splitting
    /// of nodes, resulting leaves with less than half of the upper bound, but this also makes the
    /// binary tree complete, which simplifies beam search.
    #[inline]
    fn compute_tree_height(n_labels: usize, max_leaf_size: usize) -> usize {
        assert!(max_leaf_size > 1);
        ((n_labels as f32) / (max_leaf_size as f32)).log2().ceil() as usize
    }

    #[inline]
    fn classifier_hyper_param(&self, n_examples: usize) -> liblinear::HyperParam {
        self.hyper_param
            .linear
            .adapt_to_sample_size(n_examples, self.all_examples.len())
    }

    fn train(&self) -> Tree {
        Tree {
            root: self.train_subtree(self.tree_height, &self.all_examples, &self.all_labels),
        }
    }

    fn train_subtree(
        &self,
        height: usize,
        examples: &TrainingExamples,
        label_cluster: &LabelCluster,
    ) -> TreeNode {
        if height == 0 {
            // If reached maximum depth, build and return a leaf node
            assert!(label_cluster.labels.len() <= self.hyper_param.max_leaf_size);
            self.train_leaf_node(&examples, &label_cluster.labels)
        } else {
            // Otherwise, branch and train subtrees recursively
            let [left_cluster, right_cluster] = label_cluster.split(self.hyper_param.cluster_eps);
            let (left_pair, right_pair) = rayon::join(
                || self.train_branch_split(height, examples, &left_cluster),
                || self.train_branch_split(height, examples, &right_cluster),
            );
            TreeNode::BranchNode {
                child_classifier_pairs: [left_pair, right_pair],
            }
        }
    }

    fn train_leaf_node(&self, examples: &TrainingExamples, leaf_labels: &[Index]) -> TreeNode {
        let label_classifier_pairs = leaf_labels
            .par_iter()
            .map(|&leaf_label| (leaf_label, self.train_leaf_classifier(examples, leaf_label)))
            .collect();

        self.progress_bar
            .lock()
            .unwrap()
            .add(leaf_labels.len() as u64);

        TreeNode::LeafNode {
            label_classifier_pairs,
        }
    }

    fn train_leaf_classifier(
        &self,
        examples: &TrainingExamples,
        leaf_label: Index,
    ) -> liblinear::Model {
        // Train classifier on whether each example has the given label
        let classifier_labels = examples
            .label_sets
            .iter()
            .map(|example_labels| example_labels.contains(&leaf_label))
            .collect_vec();

        self.train_classifier(examples, &classifier_labels)
    }

    fn train_branch_split(
        &self,
        height: usize,
        examples: &TrainingExamples,
        split_label_cluster: &LabelCluster,
    ) -> (Box<TreeNode>, liblinear::Model) {
        let split_example_indices = examples.find_examples_with_labels(&split_label_cluster.labels);
        let (subtree, classifier) = rayon::join(
            || {
                let split_examples = examples.take_examples_by_indices(&split_example_indices);
                self.train_subtree(height - 1, &split_examples, split_label_cluster)
            },
            || self.train_branch_split_classifier(examples, &split_example_indices),
        );

        (Box::new(subtree), classifier)
    }

    fn train_branch_split_classifier(
        &self,
        examples: &TrainingExamples,
        split_example_indices: &[usize],
    ) -> liblinear::Model {
        // Train classifier on whether each example belongs to the current split
        let mut classifier_labels = vec![false; examples.len()];
        for &i in split_example_indices {
            classifier_labels[i] = true;
        }

        let model = self.train_classifier(examples, &classifier_labels);

        self.progress_bar.lock().unwrap().add(1);

        model
    }

    fn train_classifier(
        &self,
        examples: &TrainingExamples,
        classifier_labels: &[bool],
    ) -> liblinear::Model {
        liblinear::Model::train(
            &examples.feature_matrix.view(),
            &classifier_labels,
            &self.classifier_hyper_param(examples.len()),
        )
        .remap_features_indices(&examples.index_to_feature, self.all_examples.n_features())
    }
}

/// Internal representation of training examples for training a subtree.
struct TrainingExamples<'a> {
    feature_matrix: SparseMat,
    index_to_feature: Vec<Index>,
    label_sets: Vec<&'a IndexSet>,
}

impl<'a> TrainingExamples<'a> {
    #[inline]
    fn new(
        feature_matrix: SparseMat,
        index_to_feature: Vec<Index>,
        label_sets: Vec<&'a IndexSet>,
    ) -> Self {
        assert_eq!(feature_matrix.rows(), label_sets.len());
        assert!(!label_sets.is_empty());
        assert_eq!(feature_matrix.cols(), index_to_feature.len());
        Self {
            feature_matrix,
            index_to_feature,
            label_sets,
        }
    }

    fn new_from_dataset(dataset: &'a DataSet) -> Self {
        let feature_matrix = dataset
            .feature_lists
            .copy_normalized_with_bias_to_csrmat(dataset.n_features);
        let index_to_feature = (0..feature_matrix.cols() as Index).collect_vec();
        let label_sets = dataset.label_sets.iter().collect_vec();

        Self::new(feature_matrix, index_to_feature, label_sets)
    }

    #[inline]
    fn len(&self) -> usize {
        self.feature_matrix.rows()
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.feature_matrix.cols()
    }

    fn find_examples_with_labels(&self, labels: &[Index]) -> Vec<usize> {
        let labels = IndexSet::from_iter(labels.iter().cloned());
        self.label_sets
            .par_iter()
            .enumerate()
            .filter_map(|(i, example_labels)| {
                if example_labels.is_disjoint(&labels) {
                    None
                } else {
                    Some(i)
                }
            })
            .collect()
    }

    fn take_examples_by_indices(&self, indices: &[usize]) -> Self {
        let (new_feature_matrix, mut new_index_to_feature) = self
            .feature_matrix
            .copy_outer_dims(&indices)
            .shrink_column_indices();
        for index in &mut new_index_to_feature {
            *index = self.index_to_feature[*index as usize];
        }

        let new_label_sets = indices.iter().map(|&i| self.label_sets[i]).collect_vec();
        Self::new(new_feature_matrix, new_index_to_feature, new_label_sets)
    }
}

/// Internal representation of label cluster for building the structure of a subtree.
struct LabelCluster {
    labels: Vec<Index>,
    feature_matrix: SparseMat,
}

impl LabelCluster {
    fn new(labels: Vec<Index>, feature_matrix: SparseMat) -> Self {
        assert_eq!(labels.len(), feature_matrix.rows());
        assert!(!labels.is_empty());
        Self {
            labels,
            feature_matrix,
        }
    }

    fn new_from_dataset(dataset: &DataSet, centroid_threshold: f32) -> Self {
        let (labels, label_centroids) = Self::compute_label_centroids(&dataset, centroid_threshold);
        Self::new(labels, label_centroids.copy_to_csrmat(dataset.n_features))
    }

    /// Compute centroid feature vectors for labels in a given dataset, pruned with the given threshold.
    ///
    /// Assumes that dataset is well-formed.
    fn compute_label_centroids(
        dataset: &DataSet,
        threshold: f32,
    ) -> (Vec<Index>, Vec<IndexValueVec>) {
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

    fn take_labels_by_indices(&self, indices: &[usize]) -> Self {
        let new_labels = indices.iter().map(|&i| self.labels[i]).collect_vec();
        let (new_feature_matrix, _) = self
            .feature_matrix
            .copy_outer_dims(indices)
            .shrink_column_indices();

        Self::new(new_labels, new_feature_matrix)
    }

    #[inline]
    fn len(&self) -> usize {
        self.feature_matrix.rows()
    }

    fn split(&self, cluster_eps: f32) -> [Self; 2] {
        let label_assignments = cluster::balanced_2means(&self.feature_matrix.view(), cluster_eps);
        assert_eq!(self.labels.len(), label_assignments.len());

        let (true_indices, false_indices) =
            (0..self.labels.len()).partition::<Vec<_>, _>(|&i| label_assignments[i]);
        assert!(((true_indices.len() as i64) - (false_indices.len() as i64)).abs() <= 1);

        [
            self.take_labels_by_indices(&true_indices),
            self.take_labels_by_indices(&false_indices),
        ]
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

        let (labels, vecs) =
            LabelCluster::compute_label_centroids(&dataset, 1. / 18f32.sqrt() + 1e-4);
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
        assert_eq!(TreeTrainer::compute_tree_height(2, 2), 0);
        assert_eq!(TreeTrainer::compute_tree_height(3, 2), 1);
        assert_eq!(TreeTrainer::compute_tree_height(4, 2), 1);
        assert_eq!(TreeTrainer::compute_tree_height(5, 2), 2);
        assert_eq!(TreeTrainer::compute_tree_height(6, 2), 2);
    }
}

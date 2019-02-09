use super::{cluster, liblinear, Model, Tree, TreeNode};
use crate::data::DataSet;
use crate::mat_util::*;
use crate::util::{create_progress_bar, ProgressBar};
use crate::{Index, IndexSet, IndexValueVec, Mat, SparseMat};
use derive_builder::Builder;
use hashbrown::HashMap;
use itertools::{izip, Itertools};
use log::info;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::iter::FromIterator;
use std::sync::Mutex;

/// Model training hyper-parameters.
#[derive(Builder, Copy, Clone, Debug, Serialize, Deserialize)]
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
    ///
    /// Here we take ownership of the dataset object to perform necessary prepossessing. One can
    /// choose to clone a dataset before passing it in to avoid losing the original data.
    pub fn train(&self, mut dataset: DataSet) -> Model {
        info!("Training Parabel model with hyper-parameters {:?}", self);
        let start_t = time::precise_time_s();

        info!("Initializing tree trainer");
        let trainer = TreeTrainer::initialize(&mut dataset, *self);

        info!("Start training forest");
        let trees: Vec<_> = (0..self.n_trees)
            .into_par_iter()
            .map(|_| trainer.train())
            .collect();

        info!(
            "Parabel model training complete; it took {:.2}s",
            time::precise_time_s() - start_t
        );
        Model {
            trees,
            n_features: dataset.n_features,
            hyper_parm: *self,
        }
    }
}

struct TreeTrainer<'a> {
    all_examples: TrainingExamples<'a>,
    all_labels: LabelCluster,
    tree_height: usize,
    hyper_param: HyperParam,
    progress_bar: Mutex<ProgressBar>,
}

impl<'a> TreeTrainer<'a> {
    /// Initialize a reusable tree trainer with the dataset and hyper-parameters.
    ///
    /// Dataset is assumed to be well-formed.
    fn initialize(dataset: &'a mut DataSet, hyper_param: HyperParam) -> Self {
        assert_eq!(dataset.feature_lists.len(), dataset.label_sets.len());
        // l2-normalize all examples in the dataset
        dataset
            .feature_lists
            .par_iter_mut()
            .for_each(|v| v.l2_normalize());
        // Initialize label clusters
        let all_labels = LabelCluster::new_from_dataset(dataset, hyper_param.centroid_threshold);

        // Append bias term to each vector to make training linear classifiers easier
        let bias_index = dataset.n_features as Index;
        dataset
            .feature_lists
            .iter_mut()
            .for_each(|v| v.push((bias_index, 1.)));
        // Initialize examples set
        let all_examples = TrainingExamples::new_from_dataset(dataset);

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
    ) -> Mutex<ProgressBar> {
        let n_classifiers_per_tree = 2usize.pow(tree_height as u32 + 1) - 2 + n_labels;
        Mutex::new(create_progress_bar(
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
            let label_clusters = label_cluster.split(self.hyper_param.cluster_eps);
            debug_assert_eq!(2, label_clusters.len());

            let example_index_lists = label_clusters
                .iter()
                .map(|cluster| examples.find_examples_with_labels(&cluster.labels))
                .collect::<Vec<_>>();

            let (weight_matrix, children) = rayon::join(
                || self.train_classifier_group(examples, &example_index_lists),
                || self.train_child_nodes(height, examples, &label_clusters, &example_index_lists),
            );

            TreeNode::BranchNode {
                weight_matrix,
                children,
            }
        }
    }

    fn train_child_nodes(
        &self,
        height: usize,
        examples: &TrainingExamples,
        label_clusters: &[LabelCluster],
        example_index_lists: &[Vec<usize>],
    ) -> Vec<TreeNode> {
        label_clusters
            .par_iter()
            .zip_eq(example_index_lists.par_iter())
            .map(|(label_cluster, example_indices)| {
                self.train_subtree(
                    height - 1,
                    &examples.take_examples_by_indices(example_indices),
                    label_cluster,
                )
            })
            .collect()
    }

    fn train_leaf_node(&self, examples: &TrainingExamples, leaf_labels: &[Index]) -> TreeNode {
        let example_index_lists = leaf_labels
            .par_iter()
            .map(|&label| examples.find_examples_with_label(label))
            .collect::<Vec<_>>();
        let weight_matrix = self.train_classifier_group(examples, &example_index_lists);
        TreeNode::LeafNode {
            weight_matrix,
            labels: leaf_labels.to_vec(),
        }
    }

    fn train_classifier_group(
        &self,
        examples: &TrainingExamples,
        index_lists: &[Vec<usize>],
    ) -> Mat {
        let weight_matrix = liblinear::train_classifier_group(
            &examples.feature_matrix.view(),
            index_lists,
            &self.classifier_hyper_param(examples.len()),
        )
        .remap_column_indices(&examples.index_to_feature, self.all_examples.n_features());

        assert_eq!(weight_matrix.rows(), index_lists.len());
        self.progress_bar
            .lock()
            .unwrap()
            .add(index_lists.len() as u64);

        // Store as dense matrix if not sparse enough, which greatly speeds up prediction
        let density =
            weight_matrix.nnz() as f32 / (weight_matrix.rows() * weight_matrix.cols()) as f32;
        if density < 0.25 {
            Mat::Sparse(weight_matrix)
        } else {
            Mat::Dense(weight_matrix.to_dense())
        }
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
        let feature_matrix = dataset.feature_lists.copy_to_csrmat(dataset.n_features + 1); // + 1 because we added bias term
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

    fn find_examples_with_label(&self, label: Index) -> Vec<usize> {
        self.label_sets
            .par_iter()
            .enumerate()
            .filter_map(|(i, example_labels)| {
                if example_labels.contains(&label) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
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

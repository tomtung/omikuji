mod cluster;
pub mod eval;
pub mod liblinear;
pub mod train;

use crate::mat_util::*;
use crate::{Index, IndexValueVec};
use hashbrown::HashMap;
use itertools::Itertools;
use log::info;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io;
use std::mem::swap;

/// Model training hyper-parameters.
pub type TrainHyperParam = train::HyperParam;

/// A Parabel model, which contains a forest of trees.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    trees: Vec<Tree>,
    n_features: usize,
}

impl Model {
    /// Returns a ranked list of predictions for the given input example.
    ///
    /// # Arguments
    ///
    /// * `feature_vec` - An input vector for prediction, assumed to be ordered by indices and have
    /// no duplicate or out-of-range indices
    /// * `beam_size` - Beam size for beam search.
    pub fn predict(&self, feature_vec: &[(Index, f32)], beam_size: usize) -> IndexValueVec {
        let feature_vec = self.prepare_feature_vec(feature_vec);
        let mut label_to_total_score = HashMap::<Index, f32>::new();
        let tree_predictions: Vec<_> = self
            .trees
            .par_iter()
            .map(|tree| tree.predict(&feature_vec, beam_size))
            .collect();
        for label_score_pairs in tree_predictions {
            for (label, score) in label_score_pairs {
                let total_score = label_to_total_score.entry(label).or_insert(0.);
                *total_score += score;
            }
        }

        let mut label_score_pairs = label_to_total_score
            .iter()
            .map(|(&label, &total_score)| (label, total_score / self.trees.len() as f32))
            .collect_vec();
        label_score_pairs.sort_unstable_by(|(_, score1), (_, score2)| {
            score2.partial_cmp(score1).unwrap_or_else(|| {
                panic!("Numeric error: unable to compare {} and {}", score1, score2)
            })
        });
        label_score_pairs
    }

    /// The expected dimension of feature vectors.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Prepare the feature vector in both dense and sparse forms to make prediction more efficient.
    fn prepare_feature_vec(&self, sparse_vec: &[(Index, f32)]) -> SparseVec {
        let norm = sparse_vec
            .iter()
            .map(|(_, v)| v.powi(2))
            .sum::<f32>()
            .sqrt();

        let (mut indices, mut data): (Vec<_>, Vec<_>) = sparse_vec
            .iter()
            .cloned()
            .map(|(i, v)| (i, v / norm))
            .unzip();

        indices.push(self.n_features as Index);
        data.push(1.);

        SparseVec::new(self.n_features + 1, indices, data)
    }

    /// Serialize model.
    pub fn save<W: io::Write>(&self, writer: W) -> io::Result<()> {
        info!("Saving model...");
        let start_t = time::precise_time_s();

        bincode::serialize_into(writer, self)
            .or_else(|e| Err(io::Error::new(io::ErrorKind::Other, e)))?;

        info!(
            "Model saved; it took {:.2}s",
            time::precise_time_s() - start_t
        );
        Ok(())
    }

    /// Deserialize model.
    pub fn load<R: io::Read>(reader: R) -> io::Result<Self> {
        info!("Loading model...");
        let start_t = time::precise_time_s();

        let model: Self = bincode::deserialize_from(reader)
            .or_else(|e| Err(io::Error::new(io::ErrorKind::Other, e)))?;
        info!(
            "Model loaded; it took {:.2}s",
            time::precise_time_s() - start_t
        );
        Ok(model)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Tree {
    root: TreeNode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum TreeNode {
    BranchNode {
        classifier: liblinear::MultiLabelClassifier,
        children: Vec<TreeNode>,
    },
    LeafNode {
        classifier: liblinear::MultiLabelClassifier,
        labels: Vec<Index>,
    },
}

impl TreeNode {
    fn is_leaf(&self) -> bool {
        if let TreeNode::LeafNode { .. } = self {
            true
        } else {
            false
        }
    }
}

impl Tree {
    fn predict(&self, feature_vec: &SparseVec, beam_size: usize) -> IndexValueVec {
        assert!(beam_size > 0);
        let mut curr_level = Vec::<(&TreeNode, f32)>::with_capacity(beam_size * 2);
        let mut next_level = Vec::<(&TreeNode, f32)>::with_capacity(beam_size * 2);

        curr_level.push((&self.root, 0.));

        // Iterate until only leaves are left
        while curr_level.iter().any(|(node, _)| !node.is_leaf()) {
            assert!(!curr_level.is_empty());
            next_level.clear();
            for &(node, node_score) in &curr_level {
                match node {
                    TreeNode::BranchNode {
                        classifier,
                        children,
                    } => {
                        let mut child_scores = classifier.predict(feature_vec);
                        child_scores += node_score;
                        next_level
                            .extend(children.iter().zip_eq(child_scores.into_iter().cloned()));
                    }
                    TreeNode::LeafNode { .. } => {
                        next_level.push((node, node_score));
                    }
                }
            }

            swap(&mut curr_level, &mut next_level);
            if curr_level.len() > beam_size {
                curr_level.sort_unstable_by(|(_, score1), (_, score2)| {
                    score2.partial_cmp(score1).unwrap_or_else(|| {
                        panic!("Numeric error: unable to compare {} and {}", score1, score2)
                    })
                });
                curr_level.truncate(beam_size);
            }
        }

        curr_level
            .iter()
            .flat_map(|&(leaf, leaf_score)| match leaf {
                TreeNode::LeafNode { classifier, labels } => {
                    let mut label_scores = classifier.predict(feature_vec);
                    label_scores.mapv_inplace(|v| (v + leaf_score).exp());
                    labels
                        .iter()
                        .cloned()
                        .zip_eq(label_scores.into_iter().cloned())
                        .collect_vec()
                }
                _ => unreachable!(),
            })
            .collect_vec()
    }
}

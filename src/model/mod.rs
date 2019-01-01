mod cluster;
pub mod eval;
pub mod liblinear;
pub mod train;

use crate::{mat_util::*, Index, IndexValueVec, SparseVecView};
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
    hyper_parm: TrainHyperParam,
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
        let feature_vec = feature_vec.copy_normalized_with_bias_to_csvec(self.n_features);
        let mut label_to_total_score = HashMap::<Index, f32>::new();
        let tree_predictions: Vec<_> = self
            .trees
            .par_iter()
            .map(|tree| tree.predict(feature_vec.view(), beam_size))
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
        label_score_pairs
            .sort_unstable_by(|(_, score1), (_, score2)| score2.partial_cmp(score1).unwrap());
        label_score_pairs
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
        child_classifier_pairs: [(Box<TreeNode>, liblinear::Model); 2],
    },
    LeafNode {
        label_classifier_pairs: Vec<(Index, liblinear::Model)>,
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
    fn predict(&self, feature_vec: SparseVecView, beam_size: usize) -> IndexValueVec {
        assert!(beam_size > 0);

        let mut curr_level = Vec::<(&TreeNode, f32)>::with_capacity(beam_size * 2);
        let mut next_level = Vec::<(&TreeNode, f32)>::with_capacity(beam_size * 2);

        curr_level.push((&self.root, 0.));
        loop {
            assert!(!curr_level.is_empty());

            if curr_level.len() > beam_size {
                curr_level.sort_unstable_by(|(_, score1), (_, score2)| {
                    score2.partial_cmp(score1).unwrap()
                });
                curr_level.truncate(beam_size);
            }

            if curr_level.first().unwrap().0.is_leaf() {
                break;
            }

            next_level.clear();
            for &(node, score) in &curr_level {
                match node {
                    TreeNode::BranchNode {
                        child_classifier_pairs,
                    } => {
                        for (child, classifier) in child_classifier_pairs {
                            next_level.push((child, score + classifier.predict_score(feature_vec)));
                        }
                    }
                    _ => unreachable!("The tree is not a complete binary tree."),
                }
            }

            swap(&mut curr_level, &mut next_level);
        }

        curr_level
            .iter()
            .flat_map(|&(leaf, score)| match leaf {
                TreeNode::LeafNode {
                    label_classifier_pairs,
                } => label_classifier_pairs
                    .iter()
                    .map(|(label, classifier)| {
                        let label_score = (score + classifier.predict_score(feature_vec)).exp();
                        (*label, label_score)
                    })
                    .collect_vec(),
                _ => unreachable!("The tree is not a complete binary tree."),
            })
            .collect_vec()
    }
}

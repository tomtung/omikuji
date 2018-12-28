mod cluster;
mod liblinear;
mod train;

use crate::{Index, IndexValueVec, SparseVecView};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::mem::swap;

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
    pub fn is_leaf(&self) -> bool {
        if let TreeNode::LeafNode { .. } = self {
            true
        } else {
            false
        }
    }
}

impl Tree {
    pub fn predict(&self, feature_vec: SparseVecView, beam_size: usize) -> IndexValueVec {
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
                if let TreeNode::BranchNode {
                    child_classifier_pairs,
                } = node
                {
                    for (child, classifier) in child_classifier_pairs {
                        next_level.push((child, score + classifier.predict_score(feature_vec)));
                    }
                } else {
                    unreachable!("The tree is not a complete binary tree.");
                }
            }

            swap(&mut curr_level, &mut next_level);
        }

        let mut label_score_pairs = curr_level
            .iter()
            .flat_map(|&(leaf, score)| {
                if let TreeNode::LeafNode {
                    label_classifier_pairs,
                } = leaf
                {
                    label_classifier_pairs
                        .iter()
                        .map(|(label, classifier)| {
                            (*label, score + classifier.predict_score(feature_vec))
                        })
                        .collect_vec()
                } else {
                    unreachable!("The tree is not a complete binary tree.");
                }
            })
            .collect_vec();
        label_score_pairs
            .sort_unstable_by(|(_, score1), (_, score2)| score2.partial_cmp(score1).unwrap());
        label_score_pairs
    }
}

mod cluster;
pub mod eval;
pub mod liblinear;
pub mod train;

use crate::mat_util::*;
use crate::{Index, IndexValueVec};
use hashbrown::HashMap;
use itertools::Itertools;
use log::info;
use ordered_float::NotNan;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::io;
use std::mem::swap;

/// Model training hyper-parameters.
pub type TrainHyperParam = train::HyperParam;

#[derive(Eq, PartialEq, Clone, Copy, Debug, Serialize, Deserialize)]
struct Settings {
    n_features: usize,
    classifier_loss_type: liblinear::LossType,
}

/// A Omikuji model, which contains a forest of trees.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    trees: Vec<TreeNode>,
    settings: Settings,
}

static MODEL_SETTINGS_FILE_NAME: &str = "settings.json";
static TREE_FILE_NAME_PREFIX: &str = "tree";

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
            .map(|tree| tree.predict(self.settings.classifier_loss_type, &feature_vec, beam_size))
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
        label_score_pairs.sort_unstable_by_key(|&(_, score)| Reverse(NotNan::new(score).unwrap()));
        label_score_pairs
    }

    /// The expected dimension of feature vectors.
    pub fn n_features(&self) -> usize {
        self.settings.n_features
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

        indices.push(self.settings.n_features as Index);
        data.push(1.);

        SparseVec::new(self.settings.n_features + 1, indices, data)
    }

    /// Serialize model into the directory with the given path.
    pub fn save<P: AsRef<std::path::Path>>(&self, dir_path: P) -> io::Result<()> {
        info!("Saving model...");
        let start_t = time::precise_time_s();

        let dir_path = dir_path.as_ref();
        if !dir_path.exists() {
            std::fs::create_dir_all(dir_path)?;
        } else if !dir_path.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "file with the given name already exists",
            ));
        }

        let settings_path = dir_path.join(MODEL_SETTINGS_FILE_NAME);
        if settings_path.exists() {
            let reader = std::io::BufReader::new(std::fs::File::open(settings_path)?);
            let existing_settings = serde_json::from_reader(reader)?;
            if self.settings != existing_settings {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "a model with different settings is already saved in the given directory",
                ));
            } else {
                info!(
                    "A model is already saved at {}; trees will be added to the existing model",
                    dir_path.display(),
                );
            }
        } else {
            let writer = std::io::BufWriter::new(std::fs::File::create(settings_path)?);
            serde_json::to_writer_pretty(writer, &self.settings).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Unable to serialize settings: {}", e),
                )
            })?;
        }

        let index_to_tree_path =
            |index: usize| dir_path.join(format!("{}{}.cbor", TREE_FILE_NAME_PREFIX, index));
        let mut curr_index = 0usize;
        for tree in &self.trees {
            let mut tree_path = index_to_tree_path(curr_index);
            while tree_path.exists() {
                info!(
                    "File {} already exists, skipping to keep it",
                    tree_path.display()
                );
                curr_index += 1;
                tree_path = index_to_tree_path(curr_index);
            }

            info!("Saving tree to {}", tree_path.display());
            let writer = std::io::BufWriter::new(std::fs::File::create(tree_path)?);
            serde_cbor::to_writer(writer, tree).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Unable to serialize tree: {}", e),
                )
            })?;
            curr_index += 1;
        }

        info!(
            "Model saved; it took {:.2}s",
            time::precise_time_s() - start_t
        );
        Ok(())
    }

    /// Deserialize model from the given directory.
    pub fn load<P: AsRef<std::path::Path>>(dir_path: P) -> io::Result<Self> {
        info!("Loading model...");
        let start_t = time::precise_time_s();

        let dir_path = dir_path.as_ref();
        let settings = {
            let settings_path = dir_path.join(MODEL_SETTINGS_FILE_NAME);
            info!("Loading model settings from {}...", settings_path.display());
            let reader = std::io::BufReader::new(std::fs::File::open(settings_path)?);
            serde_json::from_reader(reader)?
        };
        info!("Loaded model settings {:?}...", settings);

        let mut trees = Vec::<TreeNode>::new();
        for entry in dir_path.read_dir()? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let file_name = entry.file_name();
                let file_name_str = file_name.to_string_lossy();
                if file_name_str.starts_with(TREE_FILE_NAME_PREFIX)
                    && file_name_str.ends_with(".cbor")
                {
                    info!("Loading tree from {}...", entry.path().display());
                    let reader = std::io::BufReader::new(std::fs::File::open(entry.path())?);
                    let tree: TreeNode = serde_cbor::from_reader(reader).map_err(|e| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Unable to deserialize tree: {}", e),
                        )
                    })?;
                    if !tree.is_valid(settings) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Tree loaded from {} is invalid", file_name_str),
                        ));
                    }
                    trees.push(tree);
                }
            }
        }

        info!(
            "Model loaded; it took {:.2}s",
            time::precise_time_s() - start_t
        );
        Ok(Self { settings, trees })
    }

    /// Densify model weights to speed up prediction at the cost of more memory usage.
    pub fn densify_weights(&mut self, max_sparse_density: f32) {
        info!("Densifying model weights...");
        let start_t = time::precise_time_s();

        self.trees
            .par_iter_mut()
            .for_each(|tree| tree.densify_weights(max_sparse_density));

        info!(
            "Model weights densified; it took {:.2}s",
            time::precise_time_s() - start_t
        );
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum TreeNode {
    Branch {
        weights: Vec<Option<Vector>>,
        children: Vec<TreeNode>,
    },
    Leaf {
        weights: Vec<Option<Vector>>,
        labels: Vec<Index>,
    },
}

impl TreeNode {
    fn is_valid(&self, settings: Settings) -> bool {
        let is_weight_vec_valid = |w: &Option<Vector>| {
            if let Some(ref v) = w {
                v.dim() == settings.n_features + 1 // +1 because it includes bias
            } else {
                true
            }
        };
        match self {
            TreeNode::Branch {
                ref weights,
                ref children,
            } => {
                weights.len() == children.len()
                    && weights.iter().all(is_weight_vec_valid)
                    && children.iter().all(|c| c.is_valid(settings))
            }
            TreeNode::Leaf {
                ref weights,
                ref labels,
            } => weights.len() == labels.len() && weights.iter().all(is_weight_vec_valid),
        }
    }

    fn is_leaf(&self) -> bool {
        if let TreeNode::Leaf { .. } = self {
            true
        } else {
            false
        }
    }

    fn densify_weights(&mut self, max_sparse_density: f32) {
        fn densify(weights: &mut [Option<Vector>], max_sparse_density: f32) {
            for w in weights.iter_mut() {
                if let Some(w) = w {
                    if !w.is_dense() && w.density() > max_sparse_density {
                        w.densify();
                    }
                }
            }
        }

        match self {
            TreeNode::Branch {
                ref mut weights,
                ref mut children,
            } => {
                densify(weights, max_sparse_density);
                children
                    .par_iter_mut()
                    .for_each(|child| child.densify_weights(max_sparse_density));
            }
            TreeNode::Leaf {
                ref mut weights, ..
            } => {
                densify(weights, max_sparse_density);
            }
        }
    }

    fn predict(
        &self,
        classifier_loss_type: liblinear::LossType,
        feature_vec: &SparseVec,
        beam_size: usize,
    ) -> IndexValueVec {
        assert!(beam_size > 0);
        let mut curr_level = Vec::<(&TreeNode, f32)>::with_capacity(beam_size * 2);
        let mut next_level = Vec::<(&TreeNode, f32)>::with_capacity(beam_size * 2);

        curr_level.push((&self, 0.));

        // Iterate until only leaves are left
        while curr_level.iter().any(|(node, _)| !node.is_leaf()) {
            assert!(!curr_level.is_empty());
            next_level.clear();
            for &(node, node_score) in &curr_level {
                match node {
                    TreeNode::Branch { weights, children } => {
                        let mut child_scores =
                            liblinear::predict(weights, classifier_loss_type, feature_vec);
                        child_scores += node_score;
                        next_level
                            .extend(children.iter().zip_eq(child_scores.into_iter().cloned()));
                    }
                    TreeNode::Leaf { .. } => {
                        next_level.push((node, node_score));
                    }
                }
            }

            swap(&mut curr_level, &mut next_level);
            if curr_level.len() > beam_size {
                curr_level.sort_unstable_by_key(|&(_, score)| Reverse(NotNan::new(score).unwrap()));
                curr_level.truncate(beam_size);
            }
        }

        curr_level
            .iter()
            .flat_map(|&(leaf, leaf_score)| match leaf {
                TreeNode::Leaf { weights, labels } => {
                    let mut label_scores =
                        liblinear::predict(weights, classifier_loss_type, feature_vec);
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

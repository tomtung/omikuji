use crate::mat_util::*;
use crate::model::train::LabelCluster;
use crate::{Index, IndexSet, IndexValueVec};
use itertools::Itertools;
use log::info;
use rayon::prelude::*;
use std::fs;
use std::io::{Error, ErrorKind, Result};
use time;

/// A training dataset loaded in memory.
#[derive(Clone)]
pub struct DataSet {
    pub(crate) n_features: usize,
    pub(crate) n_labels: usize,
    pub(crate) feature_lists: Vec<IndexValueVec>,
    pub(crate) label_sets: Vec<IndexSet>,
}

/// Parse a line in a data file from the Extreme Classification Repository
///
/// The line should be in the following format:
/// label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
fn parse_xc_repo_data_line(line: &str, n_features: usize) -> Result<(IndexValueVec, IndexSet)> {
    let mut token_iter = line.split(' ');

    let mut labels = IndexSet::new();
    {
        let labels_str = token_iter.next().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!("Failed to find labels in line: \"{}\"", line),
            )
        })?;
        for label_str in labels_str.split(',') {
            if !label_str.is_empty() {
                labels.insert(label_str.parse::<Index>().map_err(|_| {
                    Error::new(
                        ErrorKind::InvalidData,
                        format!("Failed to parse label {} in line \"{}\"", label_str, line),
                    )
                })?);
            }
        }
        labels.shrink_to_fit();
    }

    let mut features = Vec::new();
    {
        for feature_value_pair_str in token_iter {
            let mut feature_value_pair_iter = feature_value_pair_str.split(':');
            let feature = feature_value_pair_iter
                .next()
                .and_then(|s| s.parse::<Index>().ok())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidData,
                        format!("Failed to parse feature {}", feature_value_pair_str),
                    )
                })?;
            let value = feature_value_pair_iter
                .next()
                .and_then(|s| s.parse::<f32>().ok())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidData,
                        format!("Failed to parse feature value {}", feature_value_pair_str),
                    )
                })?;
            if feature_value_pair_iter.next().is_some() {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Failed to parse feature {}", feature_value_pair_str),
                ));
            }
            features.push((feature, value));
        }
        features.sort_by_index();
        if !features.is_valid_sparse_vec(n_features) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Feature vector is invalid in line {}", line),
            ));
        }
    }

    Ok((features, labels))
}

impl DataSet {
    /// Load a data file from the Extreme Classification Repository
    pub fn load_xc_repo_data_file(path: &str) -> Result<Self> {
        info!("Loading data from {}", path);
        let start_t = time::precise_time_s();

        let file_content = fs::read_to_string(path)?;
        info!("Parsing data");
        let lines: Vec<&str> = file_content.par_lines().collect();
        let (n_examples, n_features, n_labels) = {
            let tokens = lines[0].split_whitespace().collect_vec();
            if tokens.len() != 3 {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Expect header line with 3 space-separated tokens, found {} instead",
                        tokens.len()
                    ),
                ));
            }

            let n_examples = tokens[0].parse::<usize>().or_else(|_| {
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "Failed to parse number of examples",
                ))
            })?;
            let n_features = tokens[1].parse::<usize>().or_else(|_| {
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "Failed to parse number of features",
                ))
            })?;
            let n_labels = tokens[1].parse::<usize>().or_else(|_| {
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "Failed to parse number of labels",
                ))
            })?;

            (n_examples, n_features, n_labels)
        };

        let lines: Vec<_> = lines
            .into_par_iter()
            .skip(1)
            .map(|line| parse_xc_repo_data_line(line, n_features))
            .collect::<Result<_>>()?;
        let (feature_lists, label_sets): (Vec<_>, Vec<_>) = lines.into_iter().unzip();

        if n_examples != feature_lists.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Expected {} examples, but read {}",
                    n_examples,
                    feature_lists.len()
                ),
            ));
        }

        info!(
            "Loaded {} examples; it took {:.2}s",
            n_examples,
            time::precise_time_s() - start_t
        );
        Ok(Self {
            n_features,
            n_labels,
            feature_lists,
            label_sets,
        })
    }
}

pub fn load_label_features_file(path: &str) -> Result<LabelCluster> {
    info!("Loading label freatures from {}", path);
    let start_t = time::precise_time_s();

    let file_content = fs::read_to_string(path)?;
    info!("Parsing label features");
    let lines: Vec<&str> = file_content.par_lines().collect();
    let (n_labels, n_features) = {
        let tokens = lines[0].split_whitespace().collect_vec();
        if tokens.len() != 2 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Expect header line with 2 space-separated tokens, found {} instead",
                    tokens.len()
                ),
            ));
        }

        let n_labels = tokens[0].parse::<usize>().or_else(|_| {
            Err(Error::new(
                ErrorKind::InvalidData,
                "Failed to parse number of labels",
            ))
        })?;
        if n_labels + 1 != lines.len() {
            Err(Error::new(
                ErrorKind::InvalidData,
                format!("Expected {} lines, but read {}", n_labels + 1, lines.len()),
            ))?;
        }

        let n_features = tokens[1].parse::<usize>().or_else(|_| {
            Err(Error::new(
                ErrorKind::InvalidData,
                "Failed to parse number of features",
            ))
        })?;

        (n_labels, n_features)
    };

    let (labels, feature_lists) = {
        let lines: Vec<(IndexValueVec, IndexSet)> = lines
            .into_par_iter()
            .skip(1)
            .map(|line| parse_xc_repo_data_line(line, n_features))
            .collect::<Result<_>>()?;

        let (mut feature_lists, label_sets): (Vec<_>, Vec<_>) = lines.into_iter().unzip();

        feature_lists.iter_mut().for_each(|v| {
            v.l2_normalize();
            v.sort_by_index();
        });

        let labels = {
            let mut labels = Vec::<Index>::with_capacity(n_labels);
            let mut labels_seen = IndexSet::with_capacity(n_labels);
            for (i, label_set) in label_sets.into_iter().enumerate() {
                if label_set.len() != 1 {
                    Err(Error::new(
                        ErrorKind::InvalidData,
                        format!(
                            "Each data row should contain exactly 1 label, but the {}-th row contains {}",
                            i + 1, label_set.len()),
                    ))?;
                }

                let label = label_set.into_iter().next().unwrap();
                if labels_seen.contains(&label) {
                    Err(Error::new(
                        ErrorKind::InvalidData,
                        format!(
                            "Label {} should only be in one data row, but appears a second time in row {}-th row",
                            label, i + 1),
                    ))?;
                }
                labels_seen.insert(label);
                labels.push(label);
            }
            labels
        };

        assert_eq!(labels.len(), n_labels);
        assert_eq!(labels.len(), feature_lists.len());
        (labels, feature_lists)
    };

    info!(
        "Loaded features for {} labels; it took {:.2}s",
        n_labels,
        time::precise_time_s() - start_t
    );

    let feature_matrix = csrmat_from_index_value_pair_lists(feature_lists, n_features);
    Ok(LabelCluster::new(labels, feature_matrix))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn test_parse_xc_repo_data_line() {
        assert_eq!(
            (
                vec![(21, 1.), (23, 2.), (24, 3.)],
                IndexSet::from_iter(vec![11, 12]),
            ),
            parse_xc_repo_data_line("11,12 21:1 23:2 24:3", 25).unwrap()
        );
    }
}

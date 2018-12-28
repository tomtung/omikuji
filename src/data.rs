use crate::mat_util::*;
use crate::{Index, IndexSet, IndexValueVec};
use log::info;
use pbr::ProgressBar;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, Error, ErrorKind, Result};
use time;

pub struct DataSet {
    pub n_features: usize,
    pub n_labels: usize,
    pub feature_lists: Vec<IndexValueVec>,
    pub label_sets: Vec<IndexSet>,
}

impl DataSet {
    /// Parse a line in a data file from the Extreme Classification Repository
    ///
    /// The line should be in the following format:
    /// label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
    fn parse_xc_repo_data_line(line: &str, n_features: usize) -> Result<(IndexValueVec, IndexSet)> {
        let mut token_iter = line.split(' ');

        let mut labels = IndexSet::new();
        {
            let labels_str = token_iter.next().ok_or(ErrorKind::InvalidData)?;
            for label_str in labels_str.split(',') {
                if !label_str.is_empty() {
                    labels.insert(
                        label_str
                            .parse::<Index>()
                            .ok()
                            .ok_or(ErrorKind::InvalidData)?,
                    );
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
                    .ok_or(ErrorKind::InvalidData)?;
                let value = feature_value_pair_iter
                    .next()
                    .and_then(|s| s.parse::<f32>().ok())
                    .ok_or(ErrorKind::InvalidData)?;
                if feature_value_pair_iter.next().is_some() {
                    Err(ErrorKind::InvalidData)?;
                }
                features.push((feature, value));
            }
            features.sort_by_index();
            if !features.is_valid_sparse_vec(n_features) {
                Err(ErrorKind::InvalidData)?;
            }
            features.shrink_to_fit();
        }

        Ok((features, labels))
    }

    /// Load a data file from the Extreme Classification Repository
    pub fn load_xc_repo_data_file(path: &str) -> Result<Self> {
        info!("Loading data from {}", path);
        let start_t = time::precise_time_s();

        let mut lines = BufReader::new(File::open(path)?).lines();

        let (n_examples, n_features, n_labels) = {
            let header_line = lines.next().ok_or(ErrorKind::InvalidData)??;
            let mut token_iter = header_line.split_whitespace();
            let n_examples = token_iter
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or(ErrorKind::InvalidData)?;
            let n_features = token_iter
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or(ErrorKind::InvalidData)?;
            let n_labels = token_iter
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .ok_or(ErrorKind::InvalidData)?;
            if token_iter.next().is_some() {
                Err(ErrorKind::InvalidData)?;
            }

            (n_examples, n_features, n_labels)
        };

        let mut pb = ProgressBar::on(::std::io::stderr(), n_examples as u64);
        let mut feature_lists = Vec::with_capacity(n_examples);
        let mut label_sets = Vec::with_capacity(n_examples);
        for line in lines {
            let (features, labels) = Self::parse_xc_repo_data_line(&line?, n_features)?;
            feature_lists.push(features);
            label_sets.push(labels);
            pb.inc();
        }

        assert_eq!(feature_lists.len(), label_sets.len());

        if n_examples != feature_lists.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Expected {} examples, only read {} lines",
                    n_examples,
                    feature_lists.len()
                ),
            ));
        }

        pb.finish();
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
            DataSet::parse_xc_repo_data_line("11,12 21:1 23:2 24:3", 25).unwrap()
        );
    }
}

use hashbrown::HashSet;
use pbr::ProgressBar;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, Error, ErrorKind, Result};
use time;

mod sparse_vector;
pub use self::sparse_vector::SparseVector;

pub type Feature = u32;

pub type Label = u32;

#[derive(Clone, Debug, PartialEq)]
pub struct Example {
    pub features: SparseVector<Feature>,
    pub labels: HashSet<Label>,
}

pub struct DataSet {
    pub n_features: u32,
    pub n_labels: u32,
    pub examples: Vec<Example>,
}

impl DataSet {
    /// Parse a line in a data file from the Extreme Classification Repository
    ///
    /// The line should be in the following format:
    /// label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
    fn parse_xc_repo_data_line(line: &str) -> Result<Example> {
        let mut token_iter = line.split(' ');

        let mut labels = HashSet::new();
        let labels_str = token_iter.next().ok_or(ErrorKind::InvalidData)?;
        for label_str in labels_str.split(',') {
            if !label_str.is_empty() {
                labels.insert(
                    label_str
                        .parse::<Label>()
                        .ok()
                        .ok_or(ErrorKind::InvalidData)?,
                );
            }
        }
        labels.shrink_to_fit();

        let features = {
            let mut features = Vec::new();
            for feature_value_pair_str in token_iter {
                let mut feature_value_pair_iter = feature_value_pair_str.split(':');
                let feature = feature_value_pair_iter
                    .next()
                    .and_then(|s| s.parse::<Feature>().ok())
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
            features.shrink_to_fit();
            SparseVector::from(features)
        };

        Ok(Example { features, labels })
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
                .and_then(|s| s.parse::<u32>().ok())
                .ok_or(ErrorKind::InvalidData)?;
            let n_features = token_iter
                .next()
                .and_then(|s| s.parse::<u32>().ok())
                .ok_or(ErrorKind::InvalidData)?;
            let n_labels = token_iter
                .next()
                .and_then(|s| s.parse::<u32>().ok())
                .ok_or(ErrorKind::InvalidData)?;
            if token_iter.next().is_some() {
                Err(ErrorKind::InvalidData)?;
            }

            (n_examples, n_features, n_labels)
        };

        let mut pb = ProgressBar::on(::std::io::stderr(), n_examples.into());
        let mut examples = Vec::with_capacity(n_examples as usize);
        for line in lines {
            examples.push(Self::parse_xc_repo_data_line(&line?)?);
            pb.inc();
        }
        examples.shrink_to_fit();

        if n_examples as usize != examples.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Expected {} examples, only read {} lines",
                    n_examples,
                    examples.len()
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
            examples,
        })
    }
}

pub struct DataSplits {
    index_lists: Vec<Vec<usize>>,
}

impl DataSplits {
    /// Parse a line from a data split file, which are a list of space separated indices.
    fn parse_xc_repo_data_split_line(line: &str) -> Result<Vec<usize>> {
        line.split_whitespace()
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| Error::from(ErrorKind::InvalidData))
            })
            .collect()
    }

    pub fn parse_xc_repo_data_split_file(path: &str) -> Result<Self> {
        info!("Loading data splits from {}", path);
        let start_t = time::precise_time_s();

        let mut index_lists = Vec::<Vec<usize>>::new();
        for line in BufReader::new(File::open(path)?).lines() {
            let indices = Self::parse_xc_repo_data_split_line(&line?)?;
            assert!(!indices.is_empty());
            if index_lists.is_empty() {
                index_lists.resize(indices.len(), Vec::new());
            } else if indices.len() != index_lists.len() {
                Err(ErrorKind::InvalidData)?;
            }

            for (i, index) in indices.into_iter().enumerate() {
                if index == 0 {
                    Err(ErrorKind::InvalidData)?;
                }
                index_lists[i].push(index - 1);
            }
        }

        info!(
            "Loaded data splits from {}; it took {:.2}s",
            path,
            time::precise_time_s() - start_t
        );
        Ok(Self { index_lists })
    }

    pub fn num_splits(&self) -> usize {
        self.index_lists.len()
    }

    fn create_dataset_split(dataset: &DataSet, indices: &[usize]) -> DataSet {
        let examples = indices
            .iter()
            .map(|&i| dataset.examples[i].clone())
            .collect();
        DataSet {
            n_features: dataset.n_features,
            n_labels: dataset.n_labels,
            examples,
        }
    }

    pub fn split_dataset(&self, dataset: &DataSet, split_index: usize) -> (DataSet, DataSet) {
        let indices = &self.index_lists[split_index];
        let other_indices: Vec<_> = {
            let index_set: HashSet<_> = indices.iter().cloned().collect();
            (0..dataset.examples.len())
                .filter(|i| !index_set.contains(i))
                .collect()
        };
        (
            Self::create_dataset_split(dataset, indices),
            Self::create_dataset_split(dataset, &other_indices),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn test_parse_xc_repo_data_line() {
        assert_eq!(
            super::Example {
                features: super::SparseVector::from(vec![(21, 1.), (23, 2.), (24, 3.)]),
                labels: HashSet::from_iter(vec![11, 12]),
            },
            super::DataSet::parse_xc_repo_data_line("11,12 21:1 23:2 24:3").unwrap()
        );
    }

    #[test]
    fn test_parse_xc_repo_data_split_line() {
        assert_eq!(
            vec![1, 2, 3, 2, 1],
            super::DataSplits::parse_xc_repo_data_split_line("1 2 3 2 1").unwrap()
        )
    }
}

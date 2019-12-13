use crate::mat_util::*;
use itertools::{izip, Itertools};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, ScalarOperand, ShapeBuilder};
use num_traits::Float;
use order_stat::kth;
use ordered_float::NotNan;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use sprs::prod::csr_mulacc_dense_colmaj;
use sprs::{CsMatViewI, SpIndex};
use std::cmp::Reverse;
use std::fmt::Display;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign};

/// Hyper-parameter settings for clustering.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct HyperParam {
    pub k: usize,
    pub balanced: bool,
    pub eps: f32,
    pub min_size: usize,
}

impl Default for HyperParam {
    fn default() -> Self {
        Self {
            k: 2,
            balanced: true,
            eps: 0.0001,
            min_size: 2,
        }
    }
}

impl HyperParam {
    /// Check if the hyper-parameter settings are valid.
    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 {
            Err(format!("k must be positive, but is {}", self.k))
        } else if self.eps <= 0. {
            Err(format!("eps must be positive, but is {}", self.eps))
        } else if self.min_size == 0 {
            Err(format!(
                "min_size must be positive, but is {}",
                self.min_size
            ))
        } else {
            Ok(())
        }
    }

    /// Find clusters from the given data.
    pub fn train<N, I, Iptr>(&self, feature_matrix: &CsMatViewI<N, I, Iptr>) -> Vec<Vec<usize>>
    where
        I: SpIndex,
        Iptr: SpIndex,
        N: Float + AddAssign + DivAssign + ScalarOperand + Display + Sum,
    {
        assert!(feature_matrix.is_csr());

        let n_examples = feature_matrix.rows();
        assert!(n_examples > 0);

        // Randomly pick examples as initial centroids
        let mut centroids = initialize_centroids(feature_matrix, self.k);

        let mut partitions = vec![self.k; n_examples]; // Initialize to out-of-bound value
        let mut similarities = Array2::zeros((n_examples, self.k));

        let mut prev_avg_similarity =
            N::from(-2.).expect("Failed to convert -2. to generic type N");
        loop {
            // Compute cosine similarities between each label vector and both centroids
            // as well as their difference
            calculate_similarities_to_centroids(
                feature_matrix,
                centroids.view(),
                similarities.view_mut(),
            );

            self.update_partitions(similarities.view(), &mut partitions);

            // Calculate average similarities
            let avg_similarity = partitions
                .iter()
                .enumerate()
                .map(|(i, &p)| similarities[[i, p]])
                .sum::<N>()
                / N::from(feature_matrix.rows()).unwrap();

            // Stop iteration if converged
            if avg_similarity - prev_avg_similarity < N::from(self.eps).unwrap() {
                break;
            } else {
                prev_avg_similarity = avg_similarity;
                update_centroids(feature_matrix, &partitions, centroids.view_mut());
            }
        }

        let mut clusters = vec![Vec::new(); self.k];
        for (i, p) in partitions.into_iter().enumerate() {
            clusters[p].push(i);
        }

        // Disband clusters smaller than the given threshold
        loop {
            // Find the smallest, non-empty cluster
            let p = (0..self.k)
                .filter(|&p| !clusters[p].is_empty())
                .min_by_key(|&p| clusters[p].len())
                .unwrap();

            // Break if the smallest cluster is large enough, or if it already contains all examples
            if clusters[p].len() >= self.min_size || clusters[p].len() == n_examples {
                break;
            }

            similarities.column_mut(p).fill(N::neg_infinity());
            while let Some(i) = clusters[p].pop() {
                let (s, new_p) = find_max(similarities.row(i)).unwrap();
                assert!(N::is_finite(s));
                assert_ne!(p, new_p);
                clusters[new_p].push(i);
            }
        }

        // Only keep non-empty clusters
        clusters.retain(|c| !c.is_empty());
        assert_eq!(n_examples, clusters.iter().map(|c| c.len()).sum::<usize>());

        clusters
    }

    fn update_partitions<N>(&self, similarities: ArrayView2<N>, partitions: &mut [usize])
    where
        N: Float + Display,
    {
        let update_fn = if !self.balanced {
            kmeans_update_partitions
        } else if self.k == 2 {
            balanced_2means_update_partitions
        } else {
            balanced_kmeans_update_partitions
        };

        update_fn(similarities, partitions);
    }
}

fn initialize_centroids<N, I, Iptr>(feature_matrix: &CsMatViewI<N, I, Iptr>, k: usize) -> Array2<N>
where
    I: SpIndex,
    Iptr: SpIndex,
    N: Float + AddAssign,
{
    let mut centroids = Array2::zeros((feature_matrix.cols(), k).f());
    for (i, c) in izip!(
        rand::seq::index::sample(&mut thread_rng(), feature_matrix.rows(), k).into_iter(),
        centroids.gencolumns_mut()
    ) {
        dense_add_assign_csvec(
            c,
            feature_matrix.outer_view(i).unwrap_or_else(|| {
                panic!(
                    "Failed to take {}-th outer view for feature_matrix of shape {:?}",
                    i,
                    feature_matrix.shape()
                )
            }),
        );
    }

    centroids
}

fn calculate_similarities_to_centroids<N: Float, I: SpIndex, Iptr: SpIndex>(
    feature_matrix: &CsMatViewI<N, I, Iptr>,
    centroids: ArrayView2<N>,
    mut similarities: ArrayViewMut2<N>,
) {
    debug_assert!(feature_matrix.is_csr());
    debug_assert_eq!(similarities.nrows(), feature_matrix.rows());
    debug_assert_eq!(centroids.nrows(), feature_matrix.cols());
    debug_assert_eq!(similarities.ncols(), centroids.ncols());

    similarities.fill(N::zero());
    csr_mulacc_dense_colmaj(
        feature_matrix.view(),
        centroids.view(),
        similarities.view_mut(),
    );
}

fn kmeans_update_partitions<N>(similarities: ArrayView2<N>, partitions: &mut [usize])
where
    N: Float + Display,
{
    debug_assert_eq!(similarities.nrows(), partitions.len());
    assert!(similarities.ncols() > 0);

    for (s, p) in similarities
        .axis_iter(Axis(0))
        .zip_eq(partitions.iter_mut())
    {
        let (_, i) = find_max(s).unwrap();
        *p = i;
    }
}

fn balanced_kmeans_update_partitions<N>(similarities: ArrayView2<N>, partitions: &mut [usize])
where
    N: Float + Display,
{
    debug_assert_eq!(similarities.nrows(), partitions.len());

    // Make a copy because we'll be making modifications
    let mut similarities = similarities.to_owned();

    let k_clusters = similarities.ncols();
    assert!(k_clusters > 0);

    let max_cluster_size = ((partitions.len() as f64) / (k_clusters as f64)).ceil() as usize;
    assert!(max_cluster_size > 0);

    // For each cluster, create a min-heap of (similarity, index) pairs
    let mut clusters = vec![
        std::collections::binary_heap::BinaryHeap::<(
            Reverse<NotNan<N>>,
            usize
        )>::with_capacity(max_cluster_size + 1);
        k_clusters
    ];

    for i in 0..similarities.nrows() {
        let mut j = i;
        loop {
            let (s, p) = find_max(similarities.row(j)).unwrap();
            assert!(N::is_finite(s));

            partitions[j] = p;
            let cluster = &mut clusters[p];
            cluster.push((Reverse(NotNan::new(s).unwrap()), j));

            // If after adding the current item to the corresponding cluster doesn't make it
            // over-sized, continue to the next item (by breaking the inner loop)
            if cluster.len() <= max_cluster_size {
                break;
            }

            // Otherwise remove the least similar item from the current cluster
            let (_, next_j) = cluster.pop().unwrap();
            similarities[[next_j, p]] = N::neg_infinity();
            j = next_j;
        }
    }
}

fn balanced_2means_update_partitions<N>(similarities: ArrayView2<N>, partitions: &mut [usize])
where
    N: Float + Display,
{
    debug_assert_eq!(similarities.nrows(), partitions.len());
    debug_assert_eq!(similarities.ncols(), 2);

    let mut diff_index_pairs = similarities
        .axis_iter(Axis(0))
        .map(|row| {
            assert_eq!(2, row.len());
            Reverse(NotNan::new(row[[0]] - row[[1]]).unwrap())
        })
        .enumerate()
        .map(|(i, d)| (d, i))
        .collect_vec();

    // Reorder by differences, where the two halves will be assigned different partitions
    let mid_rank = partitions.len() / 2 - 1;
    kth(&mut diff_index_pairs, mid_rank);

    for (r, &(_, i)) in diff_index_pairs.iter().enumerate() {
        // Update partition assignment
        partitions[i] = (r > mid_rank) as usize;
    }
}

fn update_centroids<N, I, Iptr>(
    feature_matrix: &CsMatViewI<N, I, Iptr>,
    partitions: &[usize],
    mut centroids: ArrayViewMut2<N>,
) where
    I: SpIndex,
    Iptr: SpIndex,
    N: Float + AddAssign + DivAssign + ScalarOperand,
{
    debug_assert_eq!(feature_matrix.rows(), partitions.len());
    debug_assert_eq!(feature_matrix.cols(), centroids.nrows());

    // Update centroids for next iteration
    centroids.fill(N::zero());
    for (i, &p) in partitions.iter().enumerate() {
        debug_assert!(p < centroids.ncols());

        // Update centroid
        dense_add_assign_csvec(
            centroids.column_mut(p),
            feature_matrix.outer_view(i).unwrap_or_else(|| {
                panic!(
                    "Failed to take {}-th outer view for feature_matrix of shape {:?}",
                    i,
                    feature_matrix.shape()
                )
            }),
        );
    }
    centroids.iter().for_each(|s| assert!(!s.is_nan()));
    // Normalize to get the new centroids
    centroids
        .gencolumns_mut()
        .into_iter()
        .for_each(dense_vec_l2_normalize);
}

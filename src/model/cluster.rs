use crate::mat_util::*;
use itertools::{izip, Itertools};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, ScalarOperand, ShapeBuilder};
use num_traits::Float;
use order_stat::kth_by;
use rand::prelude::*;
use sprs::prod::csr_mulacc_dense_colmaj;
use sprs::{CsMatViewI, SpIndex};
use std::fmt::Display;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign};

fn initialize_centroids<N, I>(feature_matrix: &CsMatViewI<N, I>, k: usize) -> Array2<N>
where
    I: SpIndex,
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

fn calculate_similarities_to_centroids<N, I>(
    feature_matrix: &CsMatViewI<N, I>,
    centroids: ArrayView2<N>,
    mut similarities: ArrayViewMut2<N>,
) where
    I: SpIndex,
    N: Float,
{
    debug_assert!(feature_matrix.is_csr());
    debug_assert_eq!(similarities.rows(), feature_matrix.rows());
    debug_assert_eq!(centroids.rows(), feature_matrix.cols());
    debug_assert_eq!(similarities.cols(), centroids.cols());

    similarities.fill(N::zero());
    csr_mulacc_dense_colmaj(
        feature_matrix.view(),
        centroids.view(),
        similarities.view_mut(),
    );
}

fn balanced_2means_update_partitions<N>(similarities: ArrayView2<N>, partitions: &mut [usize])
where
    N: Float + Display,
{
    debug_assert_eq!(similarities.rows(), partitions.len());
    debug_assert_eq!(similarities.cols(), 2);

    let mut index_diff_pairs = similarities
        .axis_iter(Axis(0))
        .map(|row| {
            assert_eq!(2, row.len());
            row[[0]] - row[[1]]
        })
        .enumerate()
        .collect_vec();

    // Reorder by differences, where the two halves will be assigned different partitions
    let mid_rank = partitions.len() / 2 - 1;
    kth_by(&mut index_diff_pairs, mid_rank, |(_, ld), (_, rd)| {
        rd.partial_cmp(ld)
            .unwrap_or_else(|| panic!("Numeric error: unable to compare {} and {}", ld, rd))
    });

    for (r, &(i, _)) in index_diff_pairs.iter().enumerate() {
        // Update partition assignment
        partitions[i] = (r > mid_rank) as usize;
    }
}

fn update_centroids<N, I>(
    feature_matrix: &CsMatViewI<N, I>,
    partitions: &[usize],
    mut centroids: ArrayViewMut2<N>,
) where
    I: SpIndex,
    N: Float + AddAssign + DivAssign + ScalarOperand,
{
    debug_assert_eq!(feature_matrix.rows(), partitions.len());
    debug_assert_eq!(feature_matrix.cols(), centroids.rows());

    // Update centroids for next iteration
    centroids.fill(N::zero());
    for (i, &p) in partitions.iter().enumerate() {
        debug_assert!(p < centroids.cols());

        // Update centroid
        dense_add_assign_csvec(
            centroids.index_axis_mut(Axis(1), p),
            feature_matrix.outer_view(i).unwrap_or_else(|| {
                panic!(
                    "Failed to take {}-th outer view for feature_matrix of shape {:?}",
                    i,
                    feature_matrix.shape()
                )
            }),
        );
    }
    // Normalize to get the new centroids
    centroids
        .gencolumns_mut()
        .into_iter()
        .for_each(dense_vec_l2_normalize);
}

/// Cluster vectors into 2 balanced subsets.
///
/// Each row of the feature matrix is assumed to be l2-normalized.
pub fn balanced_2means<N, I>(feature_matrix: &CsMatViewI<N, I>, threshold: N) -> Vec<Vec<usize>>
where
    I: SpIndex,
    N: Float + AddAssign + DivAssign + ScalarOperand + Display + Sum,
{
    assert!(feature_matrix.is_csr());

    let k = 2;
    let n_examples = feature_matrix.rows();
    assert!(n_examples >= k);

    // Randomly pick 2 vectors as initial centroids
    let mut centroids = initialize_centroids(feature_matrix, k);

    let mut partitions = vec![0; n_examples];
    let mut similarities = Array2::zeros((n_examples, k));

    let mut prev_avg_similarity = N::from(-2.).expect("Failed to convert -2. to generic type N");
    loop {
        // Compute cosine similarities between each label vector and both centroids
        // as well as their difference
        calculate_similarities_to_centroids(
            feature_matrix,
            centroids.view(),
            similarities.view_mut(),
        );

        // Update partition assignments
        balanced_2means_update_partitions(similarities.view(), &mut partitions);

        // Calculate average similarities
        let avg_similarity = partitions
            .iter()
            .enumerate()
            .map(|(i, &p)| similarities[[i, p]])
            .sum::<N>()
            / N::from(feature_matrix.rows()).unwrap();

        assert!(
            avg_similarity + N::from(1e-3).expect("Failed to convert 1e-3 to generic type N")
                >= prev_avg_similarity
        );

        // Stop iteration if converged
        if avg_similarity - prev_avg_similarity < threshold {
            break;
        } else {
            prev_avg_similarity = avg_similarity;
            update_centroids(feature_matrix, &partitions, centroids.view_mut());
        }
    }

    let mut clusters = vec![Vec::new(); k];
    for (i, p) in partitions.into_iter().enumerate() {
        clusters[p].push(i);
    }

    return clusters;
}

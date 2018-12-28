use crate::mat_util::*;
use itertools::{izip, Itertools};
use ndarray::{Array2, ArrayViewMut2, Axis, ScalarOperand, ShapeBuilder};
use num_traits::Float;
use order_stat::kth_by;
use rand::prelude::*;
use sprs::prod::csr_mulacc_dense_colmaj;
use sprs::{CsMatViewI, SpIndex};
use std::ops::{AddAssign, DivAssign};

fn balanced_2means_iterate<N, I>(
    feature_matrix: &CsMatViewI<N, I>,
    partitions: &mut [bool],
    mut centroids: ArrayViewMut2<N>,
    mut similarities: ArrayViewMut2<N>,
    index_diff_pairs: &mut Vec<(usize, N)>,
) -> N
where
    I: SpIndex,
    N: Float + AddAssign + DivAssign + ScalarOperand,
{
    debug_assert!(feature_matrix.is_csr());
    debug_assert!(feature_matrix.rows() >= 2);
    debug_assert_eq!(feature_matrix.rows(), partitions.len());
    debug_assert_eq!(2, centroids.shape()[1]);
    debug_assert!(
        !centroids.is_standard_layout(),
        "Centroid matrix should be in column major order"
    );

    let n_examples = feature_matrix.rows();

    // Compute cosine similarities between each label vector and both centroids
    // as well as their difference
    similarities.fill(N::zero());
    csr_mulacc_dense_colmaj(
        feature_matrix.view(),
        centroids.view(),
        similarities.view_mut(),
    );

    index_diff_pairs.clear();
    index_diff_pairs.extend(
        similarities
            .axis_iter(Axis(0))
            .map(|row| {
                assert_eq!(2, row.len());
                row[[0]] - row[[1]]
            })
            .enumerate(),
    );

    // Reorder by differences, where the two halves will be assigned different partitions
    let mid_rank = n_examples / 2 - 1;
    kth_by(index_diff_pairs, mid_rank, |(_, ld), (_, rd)| {
        rd.partial_cmp(ld).unwrap()
    });

    // Re-assign partitions and compute new centroids accordingly
    let mut total_similarities = N::zero();
    centroids.fill(N::zero());
    for (r, &(i, _)) in index_diff_pairs.iter().enumerate() {
        let p = r > mid_rank;
        let c = p as usize;

        // Update partition assignment
        partitions[i] = p;

        // Update sum of cosine similarities to assigned centroid
        total_similarities += similarities[[i, c]];

        // Update centroid
        dense_add_assign_csvec(
            centroids.subview_mut(Axis(1), c),
            feature_matrix.outer_view(i).unwrap(),
        );
    }

    // Normalize to get the new centroids
    centroids
        .gencolumns_mut()
        .into_iter()
        .foreach(dense_vec_l2_normalize);

    total_similarities / N::from(n_examples).unwrap()
}

/// Cluster vectors into 2 balanced subsets.
///
/// Each row of the feature matrix is assumed to be l2-normalized.
pub fn balanced_2means<N, I>(feature_matrix: &CsMatViewI<N, I>, threshold: N) -> Vec<bool>
where
    I: SpIndex,
    N: Float + AddAssign + DivAssign + ScalarOperand,
{
    assert!(feature_matrix.is_csr());

    let n_examples = feature_matrix.rows();
    let n_features = feature_matrix.cols();
    assert!(n_examples >= 2);

    // Randomly pick 2 vectors as initial centroids
    let mut centroids = Array2::zeros((n_features, 2).f());
    for (i, c) in izip!(
        rand::seq::index::sample(&mut thread_rng(), n_examples, 2).into_iter(),
        centroids.gencolumns_mut()
    ) {
        dense_add_assign_csvec(c, feature_matrix.outer_view(i).unwrap());
    }

    let mut prev_avg_similarity = N::from(-2.).unwrap();
    let mut partitions = vec![false; n_examples];

    // Temporary workspace; declared here to only allocate once
    let mut similarities = Array2::zeros((n_examples, 2));
    let mut index_diff_pairs = Vec::<(usize, N)>::with_capacity(n_examples);
    loop {
        let avg_similarity = balanced_2means_iterate(
            feature_matrix,
            &mut partitions,
            centroids.view_mut(),
            similarities.view_mut(),
            &mut index_diff_pairs,
        );
        assert!(avg_similarity + N::from(1e-3).unwrap() >= prev_avg_similarity);
        // Stop iteration if converged
        if avg_similarity - prev_avg_similarity < threshold {
            return partitions;
        } else {
            prev_avg_similarity = avg_similarity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use ndarray::array;
    use sprs::csr_from_dense;

    #[test]
    fn test_balanced_2means_iterate() {
        let feature_matrix = csr_from_dense(
            array![
                [1., 0.],
                [0., -1.],
                [0.5, 0.75f32.sqrt()],
                [-0.75f32.sqrt(), -0.5],
            ]
            .view(),
            1e-3,
        );
        let mut partitions = vec![false; 4];
        let mut centroids = array![
            [0.5f32.sqrt(), 0.5f32.sqrt()],
            [-0.5f32.sqrt(), -0.5f32.sqrt()],
        ]
        .reversed_axes();
        assert_approx_eq!(
            0.836516303737808,
            balanced_2means_iterate(
                &feature_matrix.view(),
                &mut partitions,
                centroids.view_mut(),
                Array2::zeros((4, 2)).view_mut(),
                &mut Vec::new(),
            )
        );
        assert_eq!(vec![false, true, false, true], partitions);
        assert!(centroids.all_close(
            &array![[0.75f32.sqrt(), 0.5], [-0.5, -0.75f32.sqrt()]].reversed_axes(),
            1e-5
        ));
    }

    #[test]
    fn test_balanced_2means() {
        let feature_matrix = csr_from_dense(
            array![
                [1., 0.],
                [0., -1.],
                [0.5, 0.75f32.sqrt()],
                [-0.75f32.sqrt(), -0.5],
                [1., 0.],
                [0., -1.],
                [0.5, 0.75f32.sqrt()],
                [-0.75f32.sqrt(), -0.5],
            ]
            .view(),
            1e-3,
        );
        let partitions = balanced_2means(&feature_matrix.view(), 1e-4);
        assert_eq!(4, partitions.iter().cloned().filter(|&p| p).count());
        assert_eq!(4, partitions.iter().cloned().filter(|&p| !p).count());
    }
}

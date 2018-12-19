use crate::data::{Feature, SparseVector};
use hashbrown::HashMap;
use order_stat::kth_by;
use rand::prelude::*;

fn balanced_2means_iterate(
    vectors: &[&SparseVector<Feature>],
    partitions: &mut [bool],
    centroids: &mut [SparseVector<Feature>; 2],
) -> f32 {
    assert_eq!(vectors.len(), partitions.len());
    assert!(vectors.len() >= 2);

    // Compute cosine similarities between each label vector and both centroids
    // as well as their difference
    let mut similarities = Vec::<[f32; 2]>::with_capacity(vectors.len());
    let mut index_diff_pairs = Vec::<(usize, f32)>::with_capacity(vectors.len());
    for (i, v) in vectors.iter().enumerate() {
        let s = [centroids[0].dot(&v), centroids[1].dot(&v)];
        assert!(-1. - 1e-4 < s[0] && s[0] < 1. + 1e-4 && -1. - 1e-4 < s[1] && s[1] < 1. + 1e-4);
        similarities.push(s);
        index_diff_pairs.push((i, s[0] - s[1]));
    }

    // Reorder by differences, where the two halves will be assigned different partitions
    let mid_rank = vectors.len() / 2 - 1;
    kth_by(&mut index_diff_pairs, mid_rank, |(_, ld), (_, rd)| {
        rd.partial_cmp(ld).unwrap()
    });

    let mut total_similarities = 0.;
    let mut centroid_builder_maps = vec![HashMap::<Feature, f32>::new(); 2];
    for (r, &(i, _)) in index_diff_pairs.iter().enumerate() {
        // Update partition
        partitions[i] = r > mid_rank;

        // Update sum of cosine similarities to assigned centroid
        total_similarities += similarities[i][partitions[i] as usize];

        // Prepare to update new centroids
        for &(index, value) in &vectors[i].entries {
            *centroid_builder_maps[partitions[i] as usize]
                .entry(index)
                .or_default() += value;
        }
    }

    // Update new centroids
    for (i, map) in centroid_builder_maps.into_iter().enumerate() {
        let mut v = SparseVector::from(map);
        v.l2_normalize();
        centroids[i] = v;
    }

    total_similarities / vectors.len() as f32
}

/// Cluster vectors into 2 balanced subsets.
pub fn balanced_2means(vectors: &[&SparseVector<Feature>], epsilon: f32) -> Vec<bool> {
    // Randomly pick 2 vectors as initial centroids
    let mut centroids: [SparseVector<Feature>; 2] = {
        assert!(vectors.len() >= 2);
        let mut iter = vectors.choose_multiple(&mut thread_rng(), 2);
        [
            (*iter.next().unwrap()).clone(),
            (*iter.next().unwrap()).clone(),
        ]
    };
    let mut prev_avg_similarity = -2.;
    let mut partitions = vec![false; vectors.len()];

    loop {
        let avg_similarity = balanced_2means_iterate(vectors, &mut partitions, &mut centroids);
        assert!(avg_similarity + 1e-3 >= prev_avg_similarity);
        // Stop iteration if converged
        if avg_similarity - prev_avg_similarity < epsilon {
            return partitions;
        } else {
            prev_avg_similarity = avg_similarity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_2means_iterate() {
        let vectors = vec![
            SparseVector::from(vec![(0, 1.)]),
            SparseVector::from(vec![(1, -1.)]),
            SparseVector::from(vec![(0, 0.5), (1, 0.75f32.sqrt())]),
            SparseVector::from(vec![(0, -0.75f32.sqrt()), (1, -0.5)]),
        ];
        let mut partitions = vec![false; vectors.len()];
        let mut centroids = [
            SparseVector::from(vec![(0, 0.5f32.sqrt()), (1, 0.5f32.sqrt())]),
            SparseVector::from(vec![(0, -0.5f32.sqrt()), (1, -0.5f32.sqrt())]),
        ];

        assert_approx_eq!(
            0.836516303737808,
            balanced_2means_iterate(
                &vectors.iter().collect::<Vec<_>>(),
                &mut partitions,
                &mut centroids
            )
        );

        assert_eq!(vec![false, true, false, true], partitions);

        assert_eq!(centroids[0].entries.len(), 2);
        assert_approx_eq!(centroids[0].entries[0].1, 0.75f32.sqrt());
        assert_approx_eq!(centroids[0].entries[1].1, 0.5);

        assert_eq!(centroids[1].entries.len(), 2);
        assert_approx_eq!(centroids[1].entries[0].1, -0.5);
        assert_approx_eq!(centroids[1].entries[1].1, -0.75f32.sqrt());
    }

    #[test]
    fn test_balanced_2means() {
        let vectors = vec![
            SparseVector::from(vec![(0, 1.)]),
            SparseVector::from(vec![(1, -1.)]),
            SparseVector::from(vec![(0, 0.5), (1, 0.75f32.sqrt())]),
            SparseVector::from(vec![(0, -0.75f32.sqrt()), (1, -0.5)]),
            SparseVector::from(vec![(0, 1.)]),
            SparseVector::from(vec![(1, -1.)]),
            SparseVector::from(vec![(0, 0.5), (1, 0.75f32.sqrt())]),
            SparseVector::from(vec![(0, -0.75f32.sqrt()), (1, -0.5)]),
        ];
        let partitions = balanced_2means(&vectors.iter().collect::<Vec<_>>(), 1e-4);
        assert_eq!(4, partitions.iter().cloned().filter(|&p| p).count());
        assert_eq!(4, partitions.iter().cloned().filter(|&p| !p).count());
    }
}

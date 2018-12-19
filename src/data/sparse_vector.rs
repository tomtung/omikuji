use hashbrown::HashMap;
use std;
use std::cmp::{max, min};
use std::hash::{BuildHasher, Hash};

/// Simple sparse vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct SparseVector<T: Ord> {
    /// A list of (index, value) pairs, sorted by index.
    pub entries: Vec<(T, f32)>,
}

/// Create a sparse vector from a list of potentially unsorted index-value pairs.
///
/// Note that we assume there are no duplicate indices.
impl<T: Ord> From<Vec<(T, f32)>> for SparseVector<T> {
    fn from(mut index_value_pairs: Vec<(T, f32)>) -> Self {
        index_value_pairs.sort_unstable_by(|l, r| l.0.cmp(&r.0));
        index_value_pairs.shrink_to_fit();
        Self::from_sorted(index_value_pairs)
    }
}

/// Create a sparse vector from a HashMap.
impl<T: Ord + Hash, S: BuildHasher> From<HashMap<T, f32, S>> for SparseVector<T> {
    fn from(index_to_value: HashMap<T, f32, S>) -> Self {
        SparseVector::from(index_to_value.into_iter().collect::<Vec<_>>())
    }
}

impl<T: Ord> SparseVector<T> {
    /// Create a list of index-value pairs already sorted by index.
    pub fn from_sorted(sorted_index_value_pairs: Vec<(T, f32)>) -> Self {
        Self {
            entries: sorted_index_value_pairs,
        }
    }

    /// Rescale the length of the vector to be 1.
    ///
    ///     # use parabel::data::SparseVector;
    ///     let mut v = SparseVector::from(
    ///         vec![(1, 1.), (5, 2.), (50, 4.), (100, 6.), (1000, 8.)]);
    ///     v.l2_normalize();
    ///     assert_eq!(vec![
    ///         (1, 1. / 11.),
    ///         (5, 2. / 11.),
    ///         (50, 4. / 11.),
    ///         (100, 6. / 11.),
    ///         (1000, 8. / 11.),
    ///     ], v.entries);
    ///
    /// If the vector is has length 0, it remains unchanged.
    ///
    ///     # use parabel::data::SparseVector;
    ///     let mut v = SparseVector::from(
    ///         vec![(1, 0.), (5, 0.), (50, 0.), (100, 0.), (1000, 0.)]);
    ///     v.l2_normalize();
    ///     assert_eq!(vec![(1, 0.), (5, 0.), (50, 0.), (100, 0.), (1000, 0.)], v.entries);
    ///
    pub fn l2_normalize(&mut self) {
        let mut length = 0f32;
        for (_, v) in &self.entries {
            length += v.powi(2);
        }
        length = length.sqrt();

        if length > 0. {
            for entry in &mut self.entries {
                entry.1 /= length;
            }
        }
    }

    /// Remove entries with values smaller than the given threshold.
    ///
    ///     # use parabel::data::SparseVector;
    ///     let mut v = SparseVector::from(
    ///         vec![(1, 0.0001), (5, 0.001), (50, 0.01), (100, 0.1)]);
    ///     v.prune_with_threshold(0.01);
    ///     assert_eq!(vec![(50, 0.01), (100, 0.1)], v.entries);
    ///
    pub fn prune_with_threshold(&mut self, epsilon: f32) {
        self.entries.retain(|&(_, v)| v >= epsilon);
        self.entries.shrink_to_fit();
    }

    /// Compute the dot product with another sparse vector.
    ///
    ///     # use parabel::data::SparseVector;
    ///     let x = SparseVector::from(vec![(1, 2.), (4, 5.), (6, 3.), (7, 10.)]);
    ///     let y = SparseVector::from(vec![(1, 3.), (5, 5.), (7, 3.), (8, 10.), (10, 100.)]);
    ///     assert_eq!(2. * 3. + 10. * 3., x.dot(&y));
    ///
    pub fn dot(&self, other: &Self) -> f32 {
        let mut sum = 0f32;

        if self.entries.is_empty() || other.entries.is_empty() {
            return 0.;
        }

        if self.entries.len() + other.entries.len()
            < min(self.entries.len(), other.entries.len())
                * (max(self.entries.len(), other.entries.len()) as f32).log2() as usize
        {
            let mut self_iter = self.entries.iter().peekable();
            let mut other_iter = other.entries.iter().peekable();
            while let (Some((self_i, self_v)), Some((other_i, other_v))) =
                (self_iter.peek(), other_iter.peek())
            {
                if self_i == other_i {
                    sum += self_v * other_v;
                    self_iter.next();
                    other_iter.next();
                } else if self_i < other_i {
                    self_iter.next();
                } else {
                    other_iter.next();
                }
            }
        } else {
            let mut l = &self.entries[..];
            let mut r = &other.entries[..];

            if l.len() > r.len() {
                std::mem::swap(&mut l, &mut r);
            }
            while !l.is_empty() && !r.is_empty() {
                match r.binary_search_by(|x| x.0.cmp(&l[0].0)) {
                    Ok(i) => {
                        sum += l[0].1 * r[i].1;
                        r = &r[i + 1..];
                    }
                    Err(i) => {
                        r = &r[i..];
                    }
                };
                l = &l[1..];
            }
        }

        sum
    }
}

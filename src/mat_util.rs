use bit_set::BitSet;
use ndarray::ArrayViewMut1;
use num_traits::{Float, Num, Unsigned};
use sprs::{CsMatBase, CsMatI, CsVecI, CsVecViewI, SpIndex, SparseMat};
use std::ops::{AddAssign, Deref, DerefMut, DivAssign};

pub trait IndexValuePairs<IndexT: SpIndex + Unsigned, ValueT: Copy>:
    Deref<Target = [(IndexT, ValueT)]>
{
    fn is_valid_sparse_vec(&self, length: usize) -> bool {
        // If empty, always valid
        if self.is_empty() {
            return true;
        }
        // Check if:
        // - All indices are smaller than max index
        // - Pairs are sorted by indices
        // - There are no duplicate indices
        if self[0].0.index() >= length {
            return false;
        }
        if self.len() > 1 {
            for ((i, _), (j, _)) in self.iter().skip(1).zip(self.iter()) {
                if i.index() >= length || i <= j {
                    return false;
                }
            }
        }

        true
    }

    /// Copy data to a new sprs sparse vector object.
    ///
    /// This assumes that is_valid_sparse_vec would return true.
    fn copy_to_csvec(
        &self,
        mut length: usize,
        maybe_append_bias: Option<ValueT>,
    ) -> CsVecI<ValueT, IndexT> {
        let (mut indices, mut data): (Vec<IndexT>, Vec<ValueT>) = self.iter().cloned().unzip();
        if let Some(bias) = maybe_append_bias {
            indices.push(IndexT::from_usize(length));
            data.push(bias);
            length += 1;
        }
        CsVecI::new(length.index(), indices, data)
    }
}

impl<IndexT, ValueT, PairsT> IndexValuePairs<IndexT, ValueT> for PairsT
where
    IndexT: SpIndex + Unsigned,
    ValueT: Copy,
    PairsT: Deref<Target = [(IndexT, ValueT)]>,
{
}

pub trait IndexValuePairsMut<IndexT, ValueT>: DerefMut<Target = [(IndexT, ValueT)]> {
    fn sort_by_index(&mut self)
    where
        IndexT: Ord,
    {
        self.sort_unstable_by(|l, r| l.0.cmp(&r.0));
    }

    fn l2_normalize(&mut self)
    where
        ValueT: Float + AddAssign + DivAssign,
    {
        let mut length = ValueT::zero();
        for (_, v) in self.iter() {
            length += v.powi(2);
        }

        if !length.is_zero() {
            length = length.sqrt();
            for (_, v) in self.iter_mut() {
                *v /= length;
            }
        }
    }
}

impl<IndexT, ValueT, PairsT> IndexValuePairsMut<IndexT, ValueT> for PairsT where
    PairsT: DerefMut<Target = [(IndexT, ValueT)]>
{
}

pub trait OwnedIndexValuePairs<IndexT, ValueT> {
    fn prune_with_threshold(&mut self, threshold: ValueT)
    where
        ValueT: Float;
}

impl<IndexT, ValueT> OwnedIndexValuePairs<IndexT, ValueT> for Vec<(IndexT, ValueT)> {
    fn prune_with_threshold(&mut self, threshold: ValueT)
    where
        ValueT: Float,
    {
        self.retain(|&(_, v)| v.abs() >= threshold);
    }
}

pub trait IndexValuePairLists<IndexT, ValueT, RowT>: Deref<Target = [RowT]>
where
    RowT: Deref<Target = [(IndexT, ValueT)]>,
{
    /// Copy data to a new sprs CSR matrix object.
    ///
    /// This assumes that is_valid_sparse_vec would return true for each row.
    fn copy_to_csrmat(
        &self,
        mut n_col: usize,
        maybe_append_bias: Option<ValueT>,
    ) -> sprs::CsMatI<ValueT, IndexT>
    where
        IndexT: SpIndex,
        ValueT: Copy,
    {
        let mut indptr: Vec<IndexT> = Vec::with_capacity(self.len() + 1);
        let mut indices: Vec<IndexT> = Vec::new();
        let mut data: Vec<ValueT> = Vec::new();

        indptr.push(IndexT::zero());
        for row in self.iter() {
            for &(i, v) in row.iter() {
                assert!(i.index() < n_col);
                indices.push(i);
                data.push(v);
            }
            if let Some(bias) = maybe_append_bias {
                indices.push(IndexT::from_usize(n_col));
                data.push(bias);
            }
            indptr.push(IndexT::from_usize(indices.len()));
        }

        if maybe_append_bias.is_some() {
            n_col += 1;
        }
        sprs::CsMatI::new((self.len(), n_col), indptr, indices, data)
    }
}

impl<IndexT, ValueT, RowT, T> IndexValuePairLists<IndexT, ValueT, RowT> for T
where
    T: Deref<Target = [RowT]>,
    RowT: Deref<Target = [(IndexT, ValueT)]>,
{
}

pub trait SparseMatTools<DataT, IndexT: SpIndex>: SparseMat {
    fn copy_outer_dims(&self, indices: &[usize]) -> CsMatI<DataT, IndexT>;
}

impl<N, I, IptrStorage, IndStorage, DataStorage> SparseMatTools<N, I>
    for CsMatBase<N, I, IptrStorage, IndStorage, DataStorage>
where
    I: SpIndex,
    N: Copy,
    IptrStorage: Deref<Target = [I]>,
    IndStorage: Deref<Target = [I]>,
    DataStorage: Deref<Target = [N]>,
{
    fn copy_outer_dims(&self, indices: &[usize]) -> CsMatI<N, I> {
        let mut iptr = Vec::<I>::with_capacity(indices.len() + 1);
        let mut ind = Vec::<I>::with_capacity(indices.len() * 2);
        let mut data = Vec::<N>::with_capacity(indices.len() * 2);

        iptr.push(I::zero());
        for &i in indices {
            if let Some(v) = self.outer_view(i) {
                for &i in v.indices() {
                    ind.push(i);
                }
                for &v in v.data() {
                    data.push(v);
                }
            }

            iptr.push(I::from_usize(ind.len()));
        }

        CsMatI::new((indices.len(), self.inner_dims()), iptr, ind, data)
    }
}

/// Remap column indices according to the given mapping.
///
/// The mapping is assumed to be well-formed, i.e. sorted, within range, and without duplicates.
pub fn remap_column_indices<N, I>(
    csr_mat: CsMatI<N, I>,
    old_index_to_new: &[I],
    n_columns: usize,
) -> CsMatI<N, I>
where
    I: SpIndex,
    N: Copy,
{
    assert!(csr_mat.is_csr());
    let (n_rows, _) = csr_mat.shape();

    let (indptr, mut indices, data) = csr_mat.into_raw_storage();
    for index in &mut indices {
        *index = old_index_to_new[index.index()];
    }
    CsMatI::new((n_rows, n_columns), indptr, indices, data)
}

/// Remap indices according to the given mapping.
///
/// The mapping is assumed to be well-formed, i.e. sorted, within range, and without duplicates.
pub fn remap_csvec_indices<N, I>(
    csvec: CsVecI<N, I>,
    old_index_to_new: &[I],
    dim: usize,
) -> CsVecI<N, I>
where
    I: SpIndex,
    N: Copy,
{
    let (mut indices, data) = csvec.into_raw_storage();
    for index in &mut indices {
        *index = old_index_to_new[index.index()];
    }
    CsVecI::new(dim, indices, data)
}

/// Shrinks column indices of a CSR matrix.
///
/// The operation can be reversed by calling map_column_indices on the returned matrix and mapping.
pub fn shrink_column_indices<N, I>(csr_mat: CsMatI<N, I>) -> (CsMatI<N, I>, Vec<I>)
where
    I: SpIndex,
    N: Copy,
{
    assert!(csr_mat.is_csr());
    let shape = csr_mat.shape();

    let new_index_to_old = {
        let mut old_indices = Vec::with_capacity(shape.1);
        let mut index_set = BitSet::with_capacity(shape.1);
        for &i in csr_mat.indices() {
            if index_set.insert(i.index()) {
                old_indices.push(i);
            }
        }
        old_indices.sort_unstable();
        old_indices
    };

    let old_index_to_new = {
        let mut lookup = vec![I::zero(); shape.1];
        for (new_index, &old_index) in new_index_to_old.iter().enumerate() {
            lookup[old_index.index()] = I::from_usize(new_index);
        }
        lookup
    };

    let mat = remap_column_indices(csr_mat, &old_index_to_new, new_index_to_old.len());
    (mat, new_index_to_old)
}

pub fn dense_add_assign_csvec<N, I>(mut dense_vec: ArrayViewMut1<N>, csvec: CsVecViewI<N, I>)
where
    I: sprs::SpIndex,
    N: Num + Copy + AddAssign,
{
    assert_eq!(dense_vec.len(), csvec.dim());
    for (i, &v) in csvec.iter() {
        dense_vec[[i]] += v;
    }
}

pub fn dense_vec_l2_normalize<N>(mut vec: ArrayViewMut1<N>)
where
    N: Float + DivAssign + ndarray::ScalarOperand,
{
    let length = vec.dot(&vec).sqrt();
    vec /= length;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_is_valid_sparse_vec() {
        assert!(Vec::<(usize, f64)>::new().is_valid_sparse_vec(0));
        assert!(Vec::<(usize, f64)>::new().is_valid_sparse_vec(123));

        assert!(vec![(123u32, 123.)].is_valid_sparse_vec(124));
        assert!(!vec![(123u32, 123.)].is_valid_sparse_vec(123));

        assert!(vec![(1u32, 0.), (3, 0.), (5, 0.)].is_valid_sparse_vec(6));
        assert!(!vec![(1u32, 0.), (3, 0.), (5, 0.)].is_valid_sparse_vec(5));
        assert!(!vec![(1u32, 0.), (5, 0.), (3, 0.)].is_valid_sparse_vec(6));
    }

    #[test]
    fn test_copy_to_csvec() {
        assert_eq!(
            CsVecI::new(10, vec![1u32, 3, 5], vec![1., 2., 3.]),
            vec![(1u32, 1.), (3, 2.), (5, 3.)].copy_to_csvec(10, None),
        );

        assert_eq!(
            CsVecI::new(11, vec![1u32, 3, 5, 10], vec![1., 2., 3., 1.]),
            vec![(1u32, 1.), (3, 2.), (5, 3.)].copy_to_csvec(10, Some(1.)),
        );
    }

    #[test]
    fn test_sort_by_index() {
        let mut pairs = vec![(1, 123.), (3, 321.), (2, 213.), (4, 432.)];
        pairs.sort_by_index();
        assert_eq!(vec![(1, 123.), (2, 213.), (3, 321.), (4, 432.)], pairs);
    }

    #[test]
    fn test_l2_normalize() {
        let mut pairs = vec![(1, 1.), (5, 2.), (50, 4.), (100, 6.), (1000, 8.)];
        pairs.l2_normalize();
        assert_eq!(
            vec![
                (1, 1. / 11.),
                (5, 2. / 11.),
                (50, 4. / 11.),
                (100, 6. / 11.),
                (1000, 8. / 11.),
            ],
            pairs
        );

        let mut pairs = vec![(1, 0.), (5, 0.), (50, 0.), (100, 0.), (1000, 0.)];
        pairs.l2_normalize();
        assert_eq!(
            vec![(1, 0.), (5, 0.), (50, 0.), (100, 0.), (1000, 0.),],
            pairs
        );
    }

    #[test]
    fn test_prune_with_threshold() {
        let mut v = vec![(1, 0.0001), (5, 0.001), (50, 0.01), (100, -0.1)];
        v.prune_with_threshold(0.01);
        assert_eq!(vec![(50, 0.01), (100, -0.1)], v);
    }

    #[test]
    fn test_copy_to_csrmat() {
        let mat = vec![
            vec![(0usize, 1), (1, 2)],
            vec![(0, 3), (2, 4)],
            vec![(2, 5)],
        ];
        assert_eq!(
            sprs::CsMat::new(
                (3, 5),
                vec![0, 2, 4, 5],
                vec![0, 1, 0, 2, 2],
                vec![1, 2, 3, 4, 5],
            ),
            mat.copy_to_csrmat(5, None)
        );

        assert_eq!(
            sprs::CsMat::new(
                (3, 6),
                vec![0, 3, 6, 8],
                vec![0, 1, 5, 0, 2, 5, 2, 5],
                vec![1, 2, 1, 3, 4, 1, 5, 1],
            ),
            mat.copy_to_csrmat(5, Some(1))
        );
    }

    #[test]
    fn test_copy_outer_dims() {
        let mat = sprs::CsMat::new(
            (3, 3),
            vec![0, 2, 4, 5],
            vec![0, 1, 0, 2, 2],
            vec![1, 2, 3, 4, 5],
        );
        assert_eq!(
            sprs::CsMat::new(
                (4, 3),
                vec![0, 2, 3, 3, 5],
                vec![0, 1, 2, 0, 2],
                vec![1, 2, 5, 3, 4],
            ),
            mat.copy_outer_dims(&[0, 2, 3, 1])
        );
    }

    #[test]
    fn test_remap_column_indices() {
        let mat = sprs::CsMat::new(
            (3, 3),
            vec![0, 2, 4, 5],
            vec![0, 1, 0, 2, 2],
            vec![1, 2, 3, 4, 5],
        );

        assert_eq!(
            sprs::CsMat::new(
                (3, 2000),
                vec![0, 2, 4, 5],
                vec![10, 100, 10, 1000, 1000],
                vec![1, 2, 3, 4, 5],
            ),
            remap_column_indices(mat, &vec![10, 100, 1000], 2000,)
        );
    }

    #[test]
    fn test_remap_csvec_indices() {
        let vec = CsVecI::new(10, vec![1u32, 3, 5], vec![1., 2., 3.]);
        assert_eq!(
            CsVecI::new(2000, vec![10, 100, 1000], vec![1., 2., 3.]),
            remap_csvec_indices(vec, &[0, 10, 0, 100, 0, 1000], 2000)
        );
    }

    #[test]
    fn test_shrink_column_indices() {
        let mat = sprs::CsMat::new(
            (3, 2000),
            vec![0, 2, 4, 5],
            vec![10, 100, 10, 1000, 1000],
            vec![1, 2, 3, 4, 5],
        );
        assert_eq!(
            (
                sprs::CsMat::new(
                    (3, 3),
                    vec![0, 2, 4, 5],
                    vec![0, 1, 0, 2, 2],
                    vec![1, 2, 3, 4, 5],
                ),
                vec![10, 100, 1000]
            ),
            shrink_column_indices(mat)
        )
    }

    #[test]
    fn test_dense_add_assign_csvec() {
        let mut dense = array![1, 2, 3, 4, 5];
        let sparse = CsVecI::new(5, vec![1, 3], vec![6, 7]);
        dense_add_assign_csvec(dense.view_mut(), sparse.view());
        assert_eq!(array![1, 2 + 6, 3, 4 + 7, 5], dense);
    }

    #[test]
    fn test_dense_vec_l2_normalize() {
        let mut v = array![1., 2., 4., 6., 8.];
        dense_vec_l2_normalize(v.view_mut());
        assert_eq!(array![1. / 11., 2. / 11., 4. / 11., 6. / 11., 8. / 11.], v);
    }
}

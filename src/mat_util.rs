use crate::Index;
use hashbrown::HashSet;
use ndarray::ArrayViewMut1;
use num_traits::{Float, Num, Unsigned};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use sprs::{CsMatBase, CsMatI, CsVecViewI, SpIndex};
use std::fmt::Display;
use std::ops::{AddAssign, Deref, DerefMut, DivAssign};

pub type SparseVec = sprs::CsVecI<f32, Index>;
pub type SparseMat = sprs::CsMatI<f32, Index, usize>;
pub type SparseMatView<'a> = sprs::CsMatViewI<'a, f32, Index, usize>;
pub type DenseVec = ndarray::Array1<f32>;

/// A vector, can be either dense or sparse.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum Vector {
    Dense(Vec<f32>),
    Sparse(SparseVec),
}

impl Vector {
    pub fn dim(&self) -> usize {
        match self {
            Vector::Dense(this) => this.len(),
            Vector::Sparse(this) => this.dim(),
        }
    }

    pub fn dot(&self, that: &SparseVec) -> f32 {
        match self {
            Vector::Dense(this) => that.dot_dense(this),
            Vector::Sparse(this) => that.dot(this),
        }
    }

    pub fn is_dense(&self) -> bool {
        match self {
            Vector::Dense(_) => true,
            Vector::Sparse(_) => false,
        }
    }

    pub fn density(&self) -> f32 {
        match self {
            Vector::Dense(_) => 1.,
            Vector::Sparse(v) => v.nnz() as f32 / v.dim() as f32,
        }
    }

    pub fn densify(&mut self) {
        *self = match self {
            Vector::Dense(_) => {
                return; // Already dense, do nothing
            }
            Vector::Sparse(sparse_v) => {
                let mut dense_v = vec![0.0; sparse_v.dim()];
                sparse_v.scatter(&mut dense_v);
                Vector::Dense(dense_v)
            }
        };
    }
}

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

pub fn csrmat_from_index_value_pair_lists<IndexT, ValueT>(
    pair_lists: Vec<Vec<(IndexT, ValueT)>>,
    n_col: usize,
) -> sprs::CsMatI<ValueT, IndexT, usize>
where
    IndexT: SpIndex,
    ValueT: Copy,
{
    let n_row = pair_lists.len();
    let mut indptr: Vec<usize> = Vec::with_capacity(n_row + 1);
    let mut indices: Vec<IndexT> = Vec::new();
    let mut data: Vec<ValueT> = Vec::new();

    indptr.push(0);
    for row in pair_lists.into_iter() {
        for (i, v) in row.into_iter() {
            assert!(i.index() < n_col);
            indices.push(i);
            data.push(v);
        }
        indptr.push(indices.len());
    }

    sprs::CsMatI::new((n_row, n_col), indptr, indices, data)
}

pub trait CsMatBaseTools<DataT, IndexT: SpIndex, Iptr: SpIndex>: sprs::SparseMat {
    fn copy_outer_dims(&self, indices: &[usize]) -> CsMatI<DataT, IndexT, Iptr>;
}

impl<N, I, Iptr, IptrStorage, IndStorage, DataStorage> CsMatBaseTools<N, I, Iptr>
    for CsMatBase<N, I, IptrStorage, IndStorage, DataStorage, Iptr>
where
    I: SpIndex,
    N: Copy,
    IptrStorage: Deref<Target = [Iptr]>,
    IndStorage: Deref<Target = [I]>,
    DataStorage: Deref<Target = [N]>,
    Iptr: SpIndex,
{
    fn copy_outer_dims(&self, indices: &[usize]) -> CsMatI<N, I, Iptr> {
        let mut iptr = Vec::<Iptr>::with_capacity(indices.len() + 1);
        let mut ind = Vec::<I>::with_capacity(indices.len() * 2);
        let mut data = Vec::<N>::with_capacity(indices.len() * 2);

        iptr.push(Iptr::zero());
        for &i in indices {
            if let Some(v) = self.outer_view(i) {
                for &i in v.indices() {
                    ind.push(i);
                }
                for &v in v.data() {
                    data.push(v);
                }
            }

            iptr.push(
                Iptr::from::<usize>(ind.len()).unwrap_or_else(|| {
                    panic!("Failed to convert usize {} to index type", ind.len())
                }),
            );
        }

        CsMatI::new((indices.len(), self.inner_dims()), iptr, ind, data)
    }
}

pub trait CsMatITools<DataT: Copy, IndexT: SpIndex>: sprs::SparseMat + Sized {
    fn shrink_inner_indices(self) -> (Self, Vec<IndexT>);
    fn remap_inner_indices(self, old_index_to_new: &[IndexT], n_columns: usize) -> Self;
}

impl<N, I, Iptr> CsMatITools<N, I> for CsMatI<N, I, Iptr>
where
    I: SpIndex,
    N: Copy,
    Iptr: SpIndex,
{
    /// Shrinks inner indices of a Sparse matrix.
    ///
    /// The operation can be reversed by calling remap_inner_indices on the returned
    /// matrix and mapping.
    fn shrink_inner_indices(self) -> (Self, Vec<I>) {
        let new_index_to_old = {
            let mut old_indices = Vec::with_capacity(self.inner_dims());
            let mut index_set = HashSet::with_capacity(self.inner_dims());
            for &i in self.indices() {
                if index_set.insert(i.index()) {
                    old_indices.push(i);
                }
            }
            old_indices.sort_unstable();
            old_indices
        };

        let old_index_to_new = {
            let mut lookup = vec![I::zero(); self.inner_dims()];
            for (new_index, &old_index) in new_index_to_old.iter().enumerate() {
                lookup[old_index.index()] = I::from::<usize>(new_index).unwrap_or_else(|| {
                    panic!("Failed to convert usize {} to index type", new_index)
                });
            }
            lookup
        };

        let mat = self.remap_inner_indices(&old_index_to_new, new_index_to_old.len());
        (mat, new_index_to_old)
    }

    /// Remap inner indices according to the given mapping.
    ///
    /// The mapping is assumed to be well-formed, i.e. sorted, within range, and without duplicates.
    fn remap_inner_indices(self, old_index_to_new: &[I], new_inner_dims: usize) -> Self {
        let outer_dims = self.outer_dims();
        let is_csr = self.is_csr();

        let (indptr, mut indices, data) = self.into_raw_storage();
        for index in &mut indices {
            *index = old_index_to_new[index.index()];
        }
        let new_mat = CsMatI::new((outer_dims, new_inner_dims), indptr, indices, data);
        if is_csr {
            new_mat
        } else {
            new_mat.transpose_into()
        }
    }
}

pub fn csvec_dot_self<N, I>(vec: &CsVecViewI<N, I>) -> N
where
    I: SpIndex,
    N: Num + AddAssign + Copy,
{
    let mut prod = N::zero();
    for &val in vec.data() {
        prod += val * val;
    }
    prod
}

pub fn dense_add_assign_csvec<N, I>(mut dense_vec: ArrayViewMut1<N>, csvec: CsVecViewI<N, I>)
where
    I: sprs::SpIndex,
    N: Num + Copy + AddAssign,
{
    assert_eq!(dense_vec.len(), csvec.dim());
    for (i, &v) in csvec.iter() {
        // This is safe because we checked length above
        unsafe {
            *dense_vec.uget_mut(i) += v;
        }
    }
}

pub fn dense_add_assign_csvec_mul_scalar<N, I>(
    mut dense_vec: ArrayViewMut1<N>,
    csvec: CsVecViewI<N, I>,
    scalar: N,
) where
    I: sprs::SpIndex,
    N: Num + Copy + AddAssign,
{
    assert_eq!(dense_vec.len(), csvec.dim());
    for (i, &v) in csvec.iter() {
        // This is safe because we checked length above
        unsafe {
            *dense_vec.uget_mut(i) += v * scalar;
        }
    }
}

pub fn dense_vec_l2_normalize<N>(mut vec: ArrayViewMut1<N>)
where
    N: Float + DivAssign + ndarray::ScalarOperand,
{
    let length = vec.dot(&vec).sqrt();
    if length > N::from(1e-5).unwrap() {
        vec /= length;
    } else {
        vec.fill(N::zero());
    }
}

pub fn find_max<N>(arr: ndarray::ArrayView1<N>) -> Option<(N, usize)>
where
    N: Float + Display,
{
    if let Some((i, &v)) = arr
        .indexed_iter()
        .max_by_key(|(_, &l)| NotNan::new(l).unwrap())
    {
        Some((v, i))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use sprs::CsVecI;

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
    fn test_csrmat_from_index_value_pair_lists() {
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
            csrmat_from_index_value_pair_lists(mat, 5)
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
    fn test_remap_inner_indices() {
        let mat = sprs::CsMat::new(
            (3, 3),
            vec![0, 2, 4, 5],
            vec![0, 1, 0, 2, 2],
            vec![1, 2, 3, 4, 5],
        );
        let expected_mat = sprs::CsMat::new(
            (3, 2000),
            vec![0, 2, 4, 5],
            vec![10, 100, 10, 1000, 1000],
            vec![1, 2, 3, 4, 5],
        );

        assert_eq!(
            expected_mat.clone(),
            mat.clone().remap_inner_indices(&vec![10, 100, 1000], 2000)
        );
        assert_eq!(
            expected_mat.transpose_into(),
            mat.transpose_into()
                .remap_inner_indices(&vec![10, 100, 1000], 2000)
        );
    }

    #[test]
    fn test_shrink_inner_indices() {
        let mat = sprs::CsMat::new(
            (3, 2000),
            vec![0, 2, 4, 5],
            vec![10, 100, 10, 1000, 1000],
            vec![1, 2, 3, 4, 5],
        );
        let expected_mat = sprs::CsMat::new(
            (3, 3),
            vec![0, 2, 4, 5],
            vec![0, 1, 0, 2, 2],
            vec![1, 2, 3, 4, 5],
        );
        assert_eq!(
            (expected_mat.clone(), vec![10, 100, 1000]),
            mat.clone().shrink_inner_indices()
        );
        assert_eq!(
            (expected_mat.transpose_into(), vec![10, 100, 1000]),
            mat.transpose_into().shrink_inner_indices()
        );
    }

    #[test]
    fn test_dense_add_assign_csvec() {
        let mut dense = array![1, 2, 3, 4, 5];
        let sparse = CsVecI::new(5, vec![1, 3], vec![6, 7]);
        dense_add_assign_csvec(dense.view_mut(), sparse.view());
        assert_eq!(array![1, 2 + 6, 3, 4 + 7, 5], dense);
    }

    #[test]
    fn test_dense_add_assign_csvec_mul_scalar() {
        let mut dense = array![1, 2, 3, 4, 5];
        let sparse = CsVecI::new(5, vec![1, 3], vec![6, 7]);
        dense_add_assign_csvec_mul_scalar(dense.view_mut(), sparse.view(), 2);
        assert_eq!(array![1, 2 + 6 * 2, 3, 4 + 7 * 2, 5], dense);
    }

    #[test]
    fn test_dense_vec_l2_normalize() {
        let mut v = array![1., 2., 4., 6., 8.];
        dense_vec_l2_normalize(v.view_mut());
        assert_eq!(array![1. / 11., 2. / 11., 4. / 11., 6. / 11., 8. / 11.], v);
    }

    #[test]
    fn test_find_max() {
        assert_eq!(Some((3., 0)), find_max(array![3.].view()));
        assert_eq!(
            Some((10., 4)),
            find_max(array![3., 5., 1., 5., 10., 0.].view())
        );
        assert_eq!(None, find_max(DenseVec::zeros(0).view()));
    }
}

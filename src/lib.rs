pub type Index = u32;
pub type IndexValueVec = Vec<(Index, f32)>;
pub type IndexSet = hashbrown::HashSet<Index>;
pub type SparseVec = sprs::CsVecI<f32, Index>;
pub type SparseVecView<'a> = sprs::CsVecViewI<'a, f32, Index>;
pub type SparseMat = sprs::CsMatI<f32, Index>;
pub type SparseMatView<'a> = sprs::CsMatViewI<'a, f32, Index>;
pub type DenseVec = ndarray::Array1<f32>;
pub type DenseVecView<'a> = ndarray::ArrayView1<'a, f32>;
pub type DenseMat = ndarray::Array2<f32>;
pub type DataSet = data::DataSet;
pub type Model = model::Model;
pub use mat_util::Mat;

pub mod data;
mod mat_util;
pub mod model;
mod util;

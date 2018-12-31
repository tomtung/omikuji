#![allow(dead_code)]

pub type Index = u32;
pub type IndexValueVec = Vec<(Index, f32)>;
pub type IndexSet = hashbrown::HashSet<Index>;
pub type SparseVec = sprs::CsVecI<f32, Index>;
pub type SparseVecView<'a> = sprs::CsVecViewI<'a, f32, Index>;
pub type SparseMat = sprs::CsMatI<f32, Index>;
pub type SparseMatView<'a> = sprs::CsMatViewI<'a, f32, Index>;

pub mod data;
mod mat_util;
pub mod model;
mod util;

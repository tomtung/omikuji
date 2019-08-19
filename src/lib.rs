/*!

Omikuji provides an efficient implementation of Partitioned Label Trees and its variations
for extreme multi-label classification.

*/

pub type Index = u32;
pub type IndexValueVec = Vec<(Index, f32)>;
pub type IndexSet = hashbrown::HashSet<Index>;
pub type DataSet = data::DataSet;
pub type Model = model::Model;

pub mod data;
mod mat_util;
pub mod model;
mod util;

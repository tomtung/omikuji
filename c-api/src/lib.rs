use itertools::Itertools;
use libc::size_t;
use omikuji::{rayon, Index};
use std::convert::TryInto;
use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_void};
use std::slice;
use omikuji::{IndexSet, IndexValueVec};

#[repr(C)]
pub struct Model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct DataSet {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ThreadPool {
    _private: [u8; 0],
}

/// Initialize a thread pool for later use.
///
/// # Safety
/// The caller is responsible for freeing the returned pointer by calling
/// [free_omikuji_thread_pool()].
///
#[no_mangle]
pub unsafe extern "C" fn init_omikuji_thread_pool(n_threads: usize) -> *mut ThreadPool {
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .stack_size(32 * 1024 * 1024)
        .build()
        .unwrap();
    Box::into_raw(Box::new(thread_pool)) as *mut ThreadPool
}

/// Free the thread pool object.
///
/// # Safety
/// The input pointer must have been obtained by calling [init_omikuji_thread_pool()]. The caller
/// is also responsible for ensuring not freeing the same pointer more than once.
///
#[no_mangle]
pub unsafe extern "C" fn free_omikuji_thread_pool(ptr: *mut ThreadPool) {
    if !ptr.is_null() {
        let ptr = ptr as *mut c_void as *mut rayon::ThreadPool;
        drop(Box::from_raw(ptr));
    }
}

unsafe fn maybe_run_with_thread_pool<OP, R>(thread_pool_ptr: *const ThreadPool, op: OP) -> R
where
    OP: FnOnce() -> R + Send,
    R: Send,
{
    let thread_pool_ptr = thread_pool_ptr as *const c_void as *const rayon::ThreadPool;
    if thread_pool_ptr.is_null() {
        op()
    } else {
        (*thread_pool_ptr).install(op)
    }
}

/// Load omikuji model from the given directory.
///
/// # Safety
/// The path pointer must point to a valid C string.
/// The caller is responsible for freeing the returned pointer by calling [free_omikuji_model()].
///
#[no_mangle]
pub unsafe extern "C" fn load_omikuji_model(path: *const c_char) -> *mut Model {
    assert!(!path.is_null(), "Path should not be null");
    let maybe_model = CStr::from_ptr(path)
        .to_str()
        .map_err(|e| format!("Failed to parse path: {}", e))
        .and_then(|path| {
            omikuji::Model::load(path).map_err(|e| format!("Failed to load model: {}", e))
        });

    match maybe_model {
        Ok(model) => Box::into_raw(Box::new(model)) as *mut Model,
        Err(msg) => {
            eprintln!("{}", msg);
            std::ptr::null_mut()
        }
    }
}

/// Save omikuji model to the given directory.
///
/// # Safety
/// The input model pointer must have been obtained by calling [load_omikuji_model()] or
/// [train_omikuji_model()]. The path pointer must point to a valid C string.
///
#[no_mangle]
pub unsafe extern "C" fn save_omikuji_model(model_ptr: *mut Model, path: *const c_char) -> i8 {
    assert!(!model_ptr.is_null(), "Model should not be null");
    assert!(!path.is_null(), "Path should not be null");
    let model_ptr = model_ptr as *mut c_void as *mut omikuji::Model;
    if let Err(msg) = CStr::from_ptr(path)
        .to_str()
        .map_err(|e| format!("Failed to parse path: {}", e))
        .and_then(|path| {
            (*model_ptr)
                .save(path)
                .map_err(|e| format!("Failed to save model: {}", e))
        })
    {
        eprintln!("{}", msg);
        -1
    } else {
        0
    }
}

/// Free omikuji model from memory.
///
/// # Safety
/// The input model pointer must have been obtained by calling [load_omikuji_model()] or
/// [train_omikuji_model()]. The caller is also responsible for ensuring not freeing the same
/// pointer more than once.
///
#[no_mangle]
pub unsafe extern "C" fn free_omikuji_model(model_ptr: *mut Model) {
    if !model_ptr.is_null() {
        let model_ptr = model_ptr as *mut c_void as *mut omikuji::Model;
        drop(Box::from_raw(model_ptr));
    }
}

/// Densify model weights to speed up prediction at the cost of more memory usage.
///
/// # Safety
/// The model pointer must have been obtained by calling [load_omikuji_model()] or
/// [train_omikuji_model()]. The thread pool pointer must have been obtained by calling
/// [init_omikuji_thread_pool()].
///
#[no_mangle]
pub unsafe extern "C" fn densify_omikuji_model(
    model_ptr: *mut Model,
    max_sparse_density: f32,
    thread_pool_ptr: *const ThreadPool,
) {
    assert!(!model_ptr.is_null(), "Model should not be null");
    let model = &mut *(model_ptr as *mut c_void as *mut omikuji::Model);
    maybe_run_with_thread_pool(thread_pool_ptr, || {
        model.densify_weights(max_sparse_density)
    });
}

/// Get the expected dimension of feature vectors.
///
/// # Safety
/// The model pointer must have been obtained by calling [load_omikuji_model()] or
/// [train_omikuji_model()].
///
#[no_mangle]
pub unsafe extern "C" fn omikuji_n_features(model_ptr: *const Model) -> size_t {
    assert!(!model_ptr.is_null(), "Model should not be null");
    let model_ptr = model_ptr as *const c_void as *const omikuji::Model;
    (*model_ptr).n_features()
}

/// The number of trees in the forest model.
///
/// # Safety
/// The model pointer must have been obtained by calling [load_omikuji_model()] or
/// [train_omikuji_model()].
///
#[no_mangle]
pub unsafe extern "C" fn omikuji_n_trees(model_ptr: *const Model) -> size_t {
    assert!(!model_ptr.is_null(), "Model should not be null");
    let model_ptr = model_ptr as *const c_void as *const omikuji::Model;
    (*model_ptr).n_trees()
}

/// Make predictions with omikuji model.
///
/// # Safety
/// The model pointer must have been obtained by calling [load_omikuji_model()] or
/// [train_omikuji_model()]. The thread pool pointer must have been obtained by calling
/// [init_omikuji_thread_pool()]. [feature_indices], [feature_values], [output_labels], and
/// [output_scores] must point to valid arrays of their respective type.
///
#[no_mangle]
pub unsafe extern "C" fn omikuji_predict(
    model_ptr: *const Model,
    beam_size: size_t,
    input_len: size_t,
    feature_indices: *const u32,
    feature_values: *const c_float,
    output_len: size_t,
    output_labels: *mut u32,
    output_scores: *mut c_float,
    thread_pool_ptr: *const ThreadPool,
) -> size_t {
    assert!(!model_ptr.is_null(), "Model should not be null");
    let model = &*(model_ptr as *mut c_void as *mut omikuji::Model);
    let feature_vec = {
        let feature_indices = slice::from_raw_parts(feature_indices, input_len);
        let feature_values = slice::from_raw_parts(feature_values, input_len);
        feature_indices
            .iter()
            .cloned()
            .zip_eq(feature_values.iter().cloned())
            .collect_vec()
    };

    let predictions =
        maybe_run_with_thread_pool(thread_pool_ptr, || model.predict(&feature_vec, beam_size));

    let output_len = output_len.min(predictions.len());
    let output_labels = slice::from_raw_parts_mut(output_labels, output_len);
    let output_scores = slice::from_raw_parts_mut(output_scores, output_len);
    for (i, (label, score)) in predictions.into_iter().take(output_len).enumerate() {
        output_labels[i] = label;
        output_scores[i] = score;
    }

    output_len
}

/// Load a data file from the Extreme Classification Repository.
///
/// # Safety
/// The thread pool pointer must have been obtained by calling [init_omikuji_thread_pool()].
/// The path pointer must point to a valid C string. The caller is responsible for freeing
/// the returned pointer by calling [free_omikuji_data_set()].
///
#[no_mangle]
pub unsafe extern "C" fn load_omikuji_data_set(
    path: *const c_char,
    thread_pool_ptr: *const ThreadPool,
) -> *mut DataSet {
    assert!(!path.is_null(), "Path should not be null");
    match CStr::from_ptr(path)
        .to_str()
        .map_err(|_| "Failed to parse path")
        .and_then(|path| {
            maybe_run_with_thread_pool(thread_pool_ptr, || {
                omikuji::DataSet::load_xc_repo_data_file(path)
                    .map_err(|_| "Failed to load data file")
            })
        }) {
        Ok(dataset) => Box::into_raw(Box::new(dataset)) as *mut DataSet,
        Err(msg) => {
            eprintln!("{}", msg);
            std::ptr::null_mut()
        }
    }
}

pub fn extract_pairs<T: Copy>(indices: &Vec<u32>, indptr: &Vec<u32>, data: &Vec<T>) -> Vec<(u32, T)> {
    let mut pairs: Vec<(u32, T)> = Vec::with_capacity(data.len()); // Preallocate memory
    
    for row in 0..indptr.len() - 1 {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        
        for i in start..end {
            pairs.push((indices[i], data[i]));
        }
    }
    pairs
}


pub fn extract_label_sets(indices: &Vec<u32>, indptr: &Vec<u32>, data: &Vec<u32>) -> Vec<IndexSet> {
    let mut label_sets: Vec<IndexSet> = Vec::with_capacity(indptr.len() - 1); // Preallocate memory for label_sets

    for row in 0..indptr.len() - 1 {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;

        let mut label_set: IndexSet = IndexSet::with_capacity(end - start); // Preallocate memory for label_set
        
        // Iterate directly over non-zero elements
        for (&index, &value) in indices[start..end].iter().zip(&data[start..end]) {
            if value != 0 {
                label_set.insert(index);
            }
        }
        label_sets.push(label_set);
    }
    label_sets
}


#[no_mangle]
pub unsafe extern "C" fn load_omikuji_data_set_from_features_labels(
    num_features: size_t,
    num_labels: size_t,
    num_nnz_features: size_t,
    num_nnz_labels: size_t,
    num_rows: size_t,
    feature_indices: * const u32,
    feature_indptr: * const u32,
    feature_data: * const c_float,
    label_indices: * const u32,
    label_indptr: * const u32,
    label_data: *const u32,
    thread_pool_ptr: *const ThreadPool,
) -> *mut DataSet {

    // features
    let vec_feature_indices = {
        slice::from_raw_parts(feature_indices, num_nnz_features).iter().cloned().collect_vec()
    };
    let vec_feature_indptr = {
        slice::from_raw_parts(feature_indptr, num_rows+1).iter().cloned().collect_vec()
    };
    let vec_feature_data = {
        slice::from_raw_parts(feature_data, num_nnz_features).iter().cloned().collect_vec()
    };
    // labels
    let vec_labels_indices = {
        slice::from_raw_parts(label_indices, num_nnz_labels).iter().cloned().collect_vec()
    };
    let vec_labels_indptr = {
        slice::from_raw_parts(label_indptr, num_rows+1).iter().cloned().collect_vec()
    };
    let vec_labels_data = {
        slice::from_raw_parts(label_data, num_nnz_labels).iter().cloned().collect_vec()
    };

    let features_list: Vec<IndexValueVec> = vec_feature_indptr
    .windows(2)
    .map(|window| {
        let start = window[0] as usize;
        let end = window[1] as usize;
        extract_pairs(
            &vec_feature_indices[start..end].to_vec(),
            &vec![0, (end - start).try_into().unwrap()],
            &vec_feature_data[start..end].to_vec(),
        )
    })
    .collect();

    let label_sets: Vec<IndexSet> = extract_label_sets(&vec_labels_indices, &vec_labels_indptr, &vec_labels_data);

    // // For debugging purpose
    // println!("features_list={:?}", features_list);
    // println!("labels_set={:?}", label_sets);

    // println!("==== RUST Features ====");
    // println!("indices={:?}", vec_feature_indices);
    // println!("indptr={:?}", vec_feature_indptr);
    // println!("data={:?}", vec_feature_data);

    // println!("==== RUST Labels ====");
    // println!("indices={:?}", vec_labels_indices);
    // println!("indptr={:?}", vec_labels_indptr);
    // println!("data={:?}", vec_labels_data);
    
    // Construct the DataSet and return a raw pointer
    let dataset = maybe_run_with_thread_pool(thread_pool_ptr, || {omikuji::DataSet::from_x_y(num_features, num_labels, features_list, label_sets).map_err(|_| "Failed passing data to Rust")});
    return Box::into_raw(Box::new(dataset)) as *mut DataSet;
}

/// Free data set object.
///
/// # Safety
/// The input pointer must have been obtained by calling [load_omikuji_data_set()]. The caller
/// is also responsible for ensuring not freeing the same pointer more than once.
///
#[no_mangle]
pub unsafe extern "C" fn free_omikuji_data_set(dataset_ptr: *mut DataSet) {
    if !dataset_ptr.is_null() {
        let dataset_ptr = dataset_ptr as *mut c_void as *mut omikuji::DataSet;
        drop(Box::from_raw(dataset_ptr));
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub enum LossType {
    Hinge = 0,
    Log = 1,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct HyperParam {
    pub n_trees: size_t,
    pub min_branch_size: size_t,
    pub max_depth: size_t,
    pub centroid_threshold: f32,
    pub collapse_every_n_layers: size_t,
    pub tree_structure_only: bool,
    pub train_trees_1_by_1: bool,
    pub linear_loss_type: LossType,
    pub linear_eps: c_float,
    pub linear_c: c_float,
    pub linear_weight_threshold: c_float,
    pub linear_max_iter: u32,
    pub cluster_k: size_t,
    pub cluster_balanced: bool,
    pub cluster_eps: f32,
    pub cluster_min_size: usize,
}

impl From<omikuji::model::TrainHyperParam> for HyperParam {
    fn from(hyper_param: omikuji::model::TrainHyperParam) -> Self {
        Self {
            n_trees: hyper_param.n_trees,
            min_branch_size: hyper_param.min_branch_size,
            max_depth: hyper_param.max_depth,
            centroid_threshold: hyper_param.centroid_threshold,
            collapse_every_n_layers: hyper_param.collapse_every_n_layers,
            linear_loss_type: match hyper_param.linear.loss_type {
                omikuji::model::liblinear::LossType::Hinge => LossType::Hinge,
                omikuji::model::liblinear::LossType::Log => LossType::Log,
            },
            linear_eps: hyper_param.linear.eps,
            linear_c: hyper_param.linear.c,
            linear_weight_threshold: hyper_param.linear.weight_threshold,
            linear_max_iter: hyper_param.linear.max_iter,
            cluster_k: hyper_param.cluster.k,
            cluster_balanced: hyper_param.cluster.balanced,
            cluster_eps: hyper_param.cluster.eps,
            cluster_min_size: hyper_param.cluster.min_size,
            tree_structure_only: hyper_param.tree_structure_only,
            train_trees_1_by_1: hyper_param.train_trees_1_by_1,
        }
    }
}

impl TryInto<omikuji::model::TrainHyperParam> for HyperParam {
    type Error = String;

    fn try_into(self) -> Result<omikuji::model::TrainHyperParam, Self::Error> {
        let hyper_param = omikuji::model::train::HyperParam {
            n_trees: self.n_trees,
            min_branch_size: self.min_branch_size,
            max_depth: self.max_depth,
            centroid_threshold: self.centroid_threshold,
            collapse_every_n_layers: self.collapse_every_n_layers,
            tree_structure_only: self.tree_structure_only,
            train_trees_1_by_1: self.train_trees_1_by_1,
            linear: omikuji::model::liblinear::HyperParam {
                loss_type: match self.linear_loss_type {
                    LossType::Hinge => omikuji::model::liblinear::LossType::Hinge,
                    LossType::Log => omikuji::model::liblinear::LossType::Log,
                },
                eps: self.linear_eps,
                c: self.linear_c,
                weight_threshold: self.linear_weight_threshold,
                max_iter: self.linear_max_iter,
            },
            cluster: omikuji::model::cluster::HyperParam {
                k: self.cluster_k,
                balanced: self.cluster_balanced,
                eps: self.cluster_eps,
                min_size: self.cluster_min_size,
            },
        };

        if let Err(msg) = hyper_param.validate() {
            Err(msg)
        } else {
            Ok(hyper_param)
        }
    }
}

/// Get the default training hyper-parameters
#[no_mangle]
pub extern "C" fn omikuji_default_hyper_param() -> HyperParam {
    omikuji::model::train::HyperParam::default().into()
}

/// Train omikuji model on the given data set and hyper-parameters.
///
/// # Safety
/// The dataset pointer must have been obtained by calling [load_omikuji_data_set()].
/// The thread pool pointer must have been obtained by calling [init_omikuji_thread_pool()].
/// The caller is responsible for freeing the returned pointer by calling [free_omikuji_model()].
///
#[no_mangle]
pub unsafe extern "C" fn train_omikuji_model(
    dataset_ptr: *const DataSet,
    hyper_param: HyperParam,
    thread_pool_ptr: *const ThreadPool,
) -> *mut Model {
    assert!(!dataset_ptr.is_null(), "Dataset should not be null");
    let result: Result<omikuji::model::TrainHyperParam, String> = hyper_param.try_into();
    match result {
        Ok(hyper_param) => {
            let dataset_ptr = dataset_ptr as *const c_void as *const omikuji::DataSet;
            // Clone the dataset so that the pointer remains valid
            let dataset = (*dataset_ptr).clone();

            let model = maybe_run_with_thread_pool(thread_pool_ptr, || hyper_param.train(dataset));

            Box::into_raw(Box::new(model)) as *mut Model
        }
        Err(msg) => {
            eprintln!("Failed to set hyper-parameters: {}", msg);
            std::ptr::null_mut()
        }
    }
}

/// Initialize a simple logger that writes to stdout.
#[no_mangle]
pub extern "C" fn omikuji_init_logger() -> i8 {
    match simple_logger::init() {
        Ok(_) => 0,
        Err(_) => {
            eprintln!("Failed to initialize logger");
            -1
        }
    }
}

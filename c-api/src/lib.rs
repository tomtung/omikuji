use itertools::Itertools;
use libc::{c_void, size_t};
use std::convert::TryInto;
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::os::raw::{c_char, c_float};
use std::slice;

#[repr(C)]
pub struct Model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct DataSet {
    _private: [u8; 0],
}

/// Load parabel model from file of the given path.
#[no_mangle]
pub unsafe extern "C" fn load_parabel_model(
    path: *const c_char,
    max_sparse_density: f32,
) -> *mut Model {
    assert!(!path.is_null(), "Path should not be null");
    let maybe_model = CStr::from_ptr(path)
        .to_str()
        .map_err(|_| "Failed to parse path")
        .and_then(|path| File::open(path).map_err(|_| "Failed to open file"))
        .and_then(|file| {
            parabel::Model::load(BufReader::new(file))
                .map(|mut model| {
                    model.densify_weights(max_sparse_density);
                    model
                })
                .map_err(|_| "Failed to load model")
        });

    match maybe_model {
        Ok(model) => Box::into_raw(Box::new(model)) as *mut Model,
        Err(msg) => {
            eprintln!("{}", msg);
            std::ptr::null_mut()
        }
    }
}

/// Save parabel model to file of the given path.
#[no_mangle]
pub unsafe extern "C" fn save_parabel_model(model_ptr: *mut Model, path: *const c_char) -> i8 {
    assert!(!model_ptr.is_null(), "Model should not be null");
    assert!(!path.is_null(), "Path should not be null");
    let model_ptr = model_ptr as *mut c_void as *mut parabel::Model;
    if let Err(msg) = CStr::from_ptr(path)
        .to_str()
        .map_err(|_| "Failed to parse path")
        .and_then(|path| File::create(path).map_err(|_| "Failed to open file"))
        .and_then(|file| {
            (*model_ptr)
                .save(BufWriter::new(file))
                .map_err(|_| "Failed to load model")
        })
    {
        eprintln!("{}", msg);
        -1
    } else {
        0
    }
}

/// Free parabel model from memory.
#[no_mangle]
pub unsafe extern "C" fn free_parabel_model(model_ptr: *mut Model) {
    if !model_ptr.is_null() {
        let model_ptr = model_ptr as *mut c_void as *mut parabel::Model;
        drop(Box::from_raw(model_ptr));
    }
}

/// Densify model weights to speed up prediction at the cost of more memory usage.
#[no_mangle]
pub unsafe extern "C" fn densify_parabel_model(model_ptr: *mut Model, max_sparse_density: f32) {
    assert!(!model_ptr.is_null(), "Model should not be null");
    let model_ptr = model_ptr as *mut c_void as *mut parabel::Model;
    (*model_ptr).densify_weights(max_sparse_density);
}

/// Get the expected dimension of feature vectors.
#[no_mangle]
pub unsafe extern "C" fn parabel_n_features(model_ptr: *const Model) -> size_t {
    assert!(!model_ptr.is_null(), "Model should not be null");
    let model_ptr = model_ptr as *const c_void as *const parabel::Model;
    (*model_ptr).n_features()
}

/// Make predictions with parabel model.
#[no_mangle]
pub unsafe extern "C" fn parabel_predict(
    model_ptr: *const Model,
    beam_size: size_t,
    input_len: size_t,
    feature_indices: *const u32,
    feature_values: *const c_float,
    output_len: size_t,
    output_labels: *mut u32,
    output_scores: *mut c_float,
) -> size_t {
    assert!(!model_ptr.is_null(), "Model should not be null");
    let model_ptr = model_ptr as *const c_void as *const parabel::Model;
    let feature_vec = {
        let feature_indices = slice::from_raw_parts(feature_indices, input_len);
        let feature_values = slice::from_raw_parts(feature_values, input_len);
        feature_indices
            .iter()
            .cloned()
            .zip_eq(feature_values.iter().cloned())
            .collect_vec()
    };

    let predictions = (*model_ptr).predict(&feature_vec, beam_size as usize);

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
#[no_mangle]
pub unsafe extern "C" fn load_parabel_data_set(
    path: *const c_char,
    n_threads: usize,
) -> *mut DataSet {
    assert!(!path.is_null(), "Path should not be null");
    match CStr::from_ptr(path)
        .to_str()
        .map_err(|_| "Failed to parse path")
        .and_then(|path| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .unwrap()
                .install(|| {
                    parabel::DataSet::load_xc_repo_data_file(path)
                        .map_err(|_| "Failed to laod data file")
                })
        }) {
        Ok(dataset) => Box::into_raw(Box::new(dataset)) as *mut DataSet,
        Err(msg) => {
            eprintln!("{}", msg);
            std::ptr::null_mut()
        }
    }
}

/// Free data set object.
#[no_mangle]
pub unsafe extern "C" fn free_parabel_data_set(dataset_ptr: *mut DataSet) {
    if !dataset_ptr.is_null() {
        let dataset_ptr = dataset_ptr as *mut c_void as *mut parabel::DataSet;
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

impl From<parabel::model::TrainHyperParam> for HyperParam {
    fn from(hyperparam: parabel::model::TrainHyperParam) -> Self {
        Self {
            n_trees: hyperparam.n_trees,
            min_branch_size: hyperparam.min_branch_size,
            max_depth: hyperparam.max_depth,
            centroid_threshold: hyperparam.centroid_threshold,
            linear_loss_type: match hyperparam.linear.loss_type {
                parabel::model::liblinear::LossType::Hinge => LossType::Hinge,
                parabel::model::liblinear::LossType::Log => LossType::Log,
            },
            linear_eps: hyperparam.linear.eps,
            linear_c: hyperparam.linear.c,
            linear_weight_threshold: hyperparam.linear.weight_threshold,
            linear_max_iter: hyperparam.linear.max_iter,
            cluster_k: hyperparam.cluster.k,
            cluster_balanced: hyperparam.cluster.balanced,
            cluster_eps: hyperparam.cluster.eps,
            cluster_min_size: hyperparam.cluster.min_size,
        }
    }
}

impl TryInto<parabel::model::TrainHyperParam> for HyperParam {
    type Error = String;

    fn try_into(self) -> Result<parabel::model::TrainHyperParam, Self::Error> {
        let mut hyper_param = parabel::model::train::HyperParam::default();
        hyper_param.n_trees = self.n_trees;
        hyper_param.min_branch_size = self.min_branch_size;
        hyper_param.max_depth = self.max_depth;
        hyper_param.centroid_threshold = self.centroid_threshold;
        hyper_param.linear.loss_type = match self.linear_loss_type {
            LossType::Hinge => parabel::model::liblinear::LossType::Hinge,
            LossType::Log => parabel::model::liblinear::LossType::Log,
        };

        hyper_param.linear.eps = self.linear_eps;
        hyper_param.linear.c = self.linear_c;
        hyper_param.linear.weight_threshold = self.linear_weight_threshold;
        hyper_param.linear.max_iter = self.linear_max_iter;

        hyper_param.cluster.k = self.cluster_k;
        hyper_param.cluster.balanced = self.cluster_balanced;
        hyper_param.cluster.eps = self.cluster_eps;
        hyper_param.cluster.min_size = self.cluster_min_size;

        if let Err(msg) = hyper_param.validate() {
            Err(msg)
        } else {
            Ok(hyper_param)
        }
    }
}

/// Get the default training hyper-parameters
#[no_mangle]
pub extern "C" fn parabel_default_hyper_param() -> HyperParam {
    parabel::model::train::HyperParam::default().into()
}

/// Train parabel model on the given data set and hyper-parameters.
#[no_mangle]
pub unsafe extern "C" fn train_parabel_model(
    dataset_ptr: *const DataSet,
    hyper_param: HyperParam,
    n_threads: usize,
) -> *mut Model {
    assert!(!dataset_ptr.is_null(), "Dataset should not be null");
    let result: Result<parabel::model::TrainHyperParam, String> = hyper_param.try_into();
    match result {
        Ok(hyper_param) => {
            let dataset_ptr = dataset_ptr as *const c_void as *const parabel::DataSet;
            // Clone the dataset so that the pointer remains valid
            let dataset = (*dataset_ptr).clone();

            let model = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .unwrap()
                .install(|| Box::new(hyper_param.train(dataset)));

            Box::into_raw(model) as *mut Model
        }
        Err(msg) => {
            eprintln!("Failed to set hyper-parameters: {}", msg);
            std::ptr::null_mut()
        }
    }
}

/// Initialize a simple logger that writes to stdout.
#[no_mangle]
pub extern "C" fn parabel_init_logger() -> i8 {
    match simple_logger::init() {
        Ok(_) => 0,
        Err(_) => {
            eprintln!("Failed to initialize logger");
            -1
        }
    }
}

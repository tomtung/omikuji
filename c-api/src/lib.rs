use itertools::Itertools;
use libc::{int8_t, size_t, uint32_t};
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::os::raw::{c_char, c_float};
use std::slice;

#[no_mangle]
pub enum ParabelModel {}

#[no_mangle]
pub enum ParabelDataSet {}

/// Load parabel model from file of the given path.
#[no_mangle]
pub unsafe extern "C" fn load_parabel_model(path: *const c_char) -> *mut ParabelModel {
    assert!(!path.is_null(), "Path should not be null");
    let maybe_model = CStr::from_ptr(path)
        .to_str()
        .map_err(|_| "Failed to parse path")
        .and_then(|path| File::open(path).map_err(|_| "Failed to open file"))
        .and_then(|file| {
            parabel::Model::load(BufReader::new(file)).map_err(|_| "Failed to load model")
        });

    match maybe_model {
        Ok(model) => Box::into_raw(Box::new(model)) as *mut ParabelModel,
        Err(msg) => {
            eprintln!("{}", msg);
            std::ptr::null_mut()
        }
    }
}

/// Save parabel model to file of the given path.
#[no_mangle]
pub unsafe extern "C" fn save_parabel_model(
    model_ptr: *mut ParabelModel,
    path: *const c_char,
) -> int8_t {
    assert!(!model_ptr.is_null(), "Model should not be null");
    assert!(!path.is_null(), "Path should not be null");
    let model_ptr = model_ptr as *mut parabel::Model;
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
pub unsafe extern "C" fn free_parabel_model(model_ptr: *mut ParabelModel) {
    if !model_ptr.is_null() {
        let model_ptr = model_ptr as *mut parabel::Model;
        drop(Box::from_raw(model_ptr));
    }
}

/// Make predictions with parabel model.
#[no_mangle]
pub unsafe extern "C" fn parabel_predict(
    model_ptr: *mut ParabelModel,
    beam_size: size_t,
    input_len: size_t,
    feature_indices: *const uint32_t,
    feature_values: *const c_float,
    output_len: size_t,
    output_labels: *mut uint32_t,
    output_scores: *mut c_float,
) -> size_t {
    assert!(!model_ptr.is_null(), "Model should not be null");
    let model_ptr = model_ptr as *mut parabel::Model;
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

    return output_len;
}

/// Load a data file from the Extreme Classification Repository.
#[no_mangle]
pub unsafe extern "C" fn load_parabel_data_set(path: *const c_char) -> *mut ParabelDataSet {
    assert!(!path.is_null(), "Path should not be null");
    match CStr::from_ptr(path)
        .to_str()
        .map_err(|_| "Failed to parse path")
        .and_then(|path| {
            parabel::DataSet::load_xc_repo_data_file(path).map_err(|_| "Failed to laod data file")
        }) {
        Ok(dataset) => Box::into_raw(Box::new(dataset)) as *mut ParabelDataSet,
        Err(msg) => {
            eprintln!("{}", msg);
            std::ptr::null_mut()
        }
    }
}

/// Free data set object.
#[no_mangle]
pub unsafe extern "C" fn free_parabel_data_set(dataset_ptr: *mut ParabelDataSet) {
    if !dataset_ptr.is_null() {
        let dataset_ptr = dataset_ptr as *mut parabel::DataSet;
        drop(Box::from_raw(dataset_ptr));
    }
}

#[repr(C)]
pub enum LossType {
    Hinge = 0,
    Log = 1,
}

/// Train parabel model on the given data set and hyper-parameters.
#[no_mangle]
pub unsafe extern "C" fn train_parabel_model(
    n_trees: size_t,
    max_leaf_size: size_t,
    cluster_eps: c_float,
    centroid_threshold: c_float,
    linear_loss_type: LossType,
    linear_eps: c_float,
    linear_c: c_float,
    linear_weight_threshold: c_float,
    linear_max_iter: uint32_t,
    dataset_ptr: *const ParabelDataSet,
) -> *mut ParabelModel {
    assert!(!dataset_ptr.is_null(), "Dataset should not be null");
    match parabel::model::liblinear::HyperParam::builder()
        .loss_type(match linear_loss_type {
            LossType::Hinge => parabel::model::liblinear::LossType::Hinge,
            LossType::Log => parabel::model::liblinear::LossType::Log,
        })
        .eps(linear_eps)
        .c(linear_c)
        .weight_threshold(linear_weight_threshold)
        .max_iter(linear_max_iter)
        .build()
        .and_then(|liblinear_hyperparam| {
            parabel::model::TrainHyperParam::builder()
                .linear(liblinear_hyperparam)
                .n_trees(n_trees)
                .max_leaf_size(max_leaf_size)
                .cluster_eps(cluster_eps)
                .centroid_threshold(centroid_threshold)
                .build()
        }) {
        Ok(hyperparam) => {
            let dataset_ptr = dataset_ptr as *const parabel::DataSet;
            let model = hyperparam.train((*dataset_ptr).clone());
            Box::into_raw(Box::new(model)) as *mut ParabelModel
        }
        Err(msg) => {
            eprintln!("Failed to set hyper-parameters: {}", msg);
            std::ptr::null_mut()
        }
    }
}

/// Initialize a simple logger that writes to stdout.
#[no_mangle]
pub extern "C" fn parabel_init_logger() -> int8_t {
    match simple_logger::init() {
        Ok(_) => 0,
        Err(_) => {
            eprintln!("Failed to initialize logger");
            -1
        }
    }
}

/// Optionally initialize Rayon global thread pool with certain number of threads.
#[no_mangle]
pub extern "C" fn rayon_init_threads(n_threads: size_t) -> int8_t {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads as usize)
        .build_global()
    {
        Ok(_) => 0,
        Err(_) => {
            eprintln!("Failed to initialize Rayon global thread-pool");
            -1
        }
    }
}

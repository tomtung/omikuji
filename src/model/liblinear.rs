use crate::mat_util::*;
use crate::{Index, SparseMatView, SparseVec, SparseVecView};
use itertools::izip;
use serde::{Deserialize, Serialize};
use std::os::raw::{c_char, c_double, c_int};
use std::ptr;
use std::slice;

/// The loss function used by liblinear model.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum LossType {
    /// Log loss: min_w w^Tw/2 + C \sum log(1 + exp(-y_i w^Tx_i))
    Log,
    /// Squared hinge loss: min_w w^Tw/2 + C \sum max(0, 1- y_i w^Tx_i)^2
    Hinge,
}

/// Hyper-parameter settings for training liblinear model.
#[derive(Copy, Clone, Debug)]
#[allow(non_snake_case)]
pub struct TrainHyperParam {
    pub loss_type: LossType,
    pub eps: f32,
    pub C: f32,
    pub weight_threshold: f32,
}

impl TrainHyperParam {
    pub fn adapt_to_sample_size(&self, n_curr_examples: usize, n_total_examples: usize) -> Self {
        match self.loss_type {
            LossType::Hinge => *self,
            LossType::Log => Self {
                C: n_total_examples as f32 / n_curr_examples as f32,
                ..*self
            },
        }
    }
}

/// A binary classifier trained with liblinear.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    weights: SparseVec,
    loss_type: LossType,
}

/// Mute prints from liblinear.
pub fn mute_liblinear_print() {
    unsafe {
        ffi::set_print_string_function(ffi::print_null);
    }
}

impl Model {
    /// Remap indices of weight vector.
    ///
    /// The mapping is assumed to be well-formed, i.e. sorted, within range, and without duplicates.
    pub fn remap_features_indices(self, index_to_feature: &[Index], n_cols: usize) -> Self {
        let Self { weights, loss_type } = self;
        let weights = remap_csvec_indices(weights, index_to_feature, n_cols);
        Self { weights, loss_type }
    }

    /// Compute score for a given example.
    ///
    /// Note: the last element of feature vector is assumed to be a bias term with value 1.
    pub fn predict_score(&self, feature_vec: SparseVecView) -> f32 {
        let p = self.weights.dot(feature_vec);
        match self.loss_type {
            LossType::Log => -(-p).exp().ln_1p(),
            LossType::Hinge => -(1f32 - p).max(0.).powi(2),
        }
    }

    /// Train a binary classifier with liblinear.
    ///
    /// Note: the last column of feature matrix is assumed to be bias terms with value 1.
    pub fn train(
        feature_matrix: &SparseMatView,
        labels: &[bool],
        hyper_param: &TrainHyperParam,
    ) -> Model {
        assert!(feature_matrix.is_csr());
        assert_eq!(feature_matrix.outer_dims(), labels.len());

        // Used to as inputs into liblinear
        let mut x_vals = Vec::<Vec<ffi::feature_node>>::with_capacity(feature_matrix.outer_dims());
        let mut y_vals = Vec::<c_double>::with_capacity(labels.len());

        for (feature_vec, &label) in izip!(feature_matrix.outer_iterator(), labels) {
            // For binary classification, liblinear treats +1 as positive and -1 as negative
            y_vals.push(if label { 1. } else { -1. });

            // We need to use the smallest possible feature indices to minimize the number of
            // parameters of the liblinear model
            let mut feature_nodes = Vec::<ffi::feature_node>::with_capacity(feature_vec.nnz() + 1);
            for (feature, &value) in feature_vec.iter() {
                let index = (feature + 1) as c_int; // Index starts at 1
                let value = c_double::from(value);
                feature_nodes.push(ffi::feature_node { index, value });
            }
            feature_nodes.push(ffi::feature_node {
                index: -1,
                value: 0.,
            });
            x_vals.push(feature_nodes);
        }

        assert_eq!(x_vals.len(), feature_matrix.outer_dims());
        assert_eq!(y_vals.len(), feature_matrix.outer_dims());

        // Construct input problem & parameter for liblinear
        let x_ptrs: Vec<_> = x_vals.iter().map(|v| v.as_ptr()).collect();
        let prob = ffi::problem {
            l: feature_matrix.outer_dims() as c_int,
            n: feature_matrix.cols() as c_int, // Already includes bias term
            y: y_vals.as_ptr(),
            x: x_ptrs.as_ptr(),
            bias: 1.,
        };

        let param = ffi::parameter {
            solver_type: match hyper_param.loss_type {
                LossType::Log => ffi::L2R_LR,
                LossType::Hinge => ffi::L2R_L2LOSS_SVC,
            },
            eps: c_double::from(hyper_param.eps),
            C: c_double::from(hyper_param.C),
            nr_weight: 0,
            weight_label: ptr::null(),
            weight: ptr::null(),
            p: 0.,
            init_sol: ptr::null(),
        };

        mute_liblinear_print();
        unsafe {
            // Call liblinear to train the model
            assert!(ffi::check_parameter(&prob, &param).is_null());
            let mut p_model = ffi::train(&prob, &param);

            assert_eq!(2, (*p_model).nr_class);
            assert_eq!([1, -1], slice::from_raw_parts((*p_model).label, 2));
            assert_eq!((feature_matrix.cols() - 1) as c_int, (*p_model).nr_feature);

            // Collect resulting weights and bias
            let weights = {
                let (indices, values) = slice::from_raw_parts((*p_model).w, feature_matrix.cols())
                    .iter()
                    .take(feature_matrix.cols())
                    .enumerate()
                    .filter_map(|(index, &value)| {
                        if value.abs() <= c_double::from(hyper_param.weight_threshold) {
                            None
                        } else {
                            Some((index as u32, value as f32))
                        }
                    })
                    .unzip();

                SparseVec::new(feature_matrix.cols(), indices, values)
            };

            // Free liblinear model memory before we go
            ffi::free_and_destroy_model(&mut p_model);

            Model {
                weights,
                loss_type: hyper_param.loss_type,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_train_and_predict() {
        let feature_matrix = vec![
            vec![(200, 0.47712)],
            vec![(100, 2.1), (200, 0.30103)],
            vec![(100, 3.4), (200, 0.34242)],
            vec![(100, 4.5), (200, 0.27875)],
            vec![(100, 5.), (200, 0.27875)],
            vec![(100, 1.7), (200, 0.30103)],
            vec![(100, 2.2), (200, 0.4624)],
            vec![(100, 4.), (200, 0.50515)],
            vec![(100, 10.), (200, -1.52288)],
            vec![(100, 9.), (200, -1.30103)],
            vec![(100, 10.5), (200, -1.04576)],
            vec![(100, 8.7), (200, -1.1549)],
            vec![(100, 7.1), (200, -0.82391)],
            vec![(100, 9.), (200, -1.39794)],
            vec![(100, 8.5), (200, -1.1549)],
            vec![(100, 9.3), (200, -0.92082)],
            vec![(100, 12.), (200, -1.22185)],
        ]
        .copy_to_csrmat(201, Some(1.));
        let labels = [
            false, false, false, false, false, false, false, false, true, true, true, true, true,
            true, true, true, true,
        ];

        // Log loss
        {
            let model = Model::train(
                &feature_matrix.view(),
                &labels,
                &TrainHyperParam {
                    loss_type: LossType::Log,
                    eps: 0.01,
                    C: 1.,
                    weight_threshold: 0.,
                },
            );
            assert_eq!(3, model.weights.nnz());
            assert_eq!(100, model.weights.indices()[0]);
            assert_approx_eq!(0.20762629796386239, model.weights.data()[0]);
            assert_eq!(200, model.weights.indices()[1]);
            assert_approx_eq!(-1.4655500091007376, model.weights.data()[1]);
            assert_eq!(201, model.weights.indices()[2]);
            assert_approx_eq!(-1.1900622116954989, model.weights.data()[2]);
            assert_approx_eq!(
                0.232326,
                model
                    .predict_score(feature_matrix.outer_view(1).unwrap())
                    .exp()
            );
        }

        // Log loss + weight threshold
        {
            let model = Model::train(
                &feature_matrix.view(),
                &labels,
                &TrainHyperParam {
                    loss_type: LossType::Log,
                    eps: 0.01,
                    C: 1.,
                    weight_threshold: 0.25, // this filters out the first feature
                },
            );
            assert_eq!(2, model.weights.nnz());
            assert_eq!(200, model.weights.indices()[0]);
            assert_approx_eq!(-1.4655500091007376, model.weights.data()[0]);
            assert_eq!(201, model.weights.indices()[1]);
            assert_approx_eq!(-1.1900622116954989, model.weights.data()[1]);
        }

        // Hinge loss
        {
            let model = Model::train(
                &feature_matrix.view(),
                &labels,
                &TrainHyperParam {
                    loss_type: LossType::Hinge,
                    eps: 0.01,
                    C: 1.,
                    weight_threshold: 0.,
                },
            );
            assert_eq!(3, model.weights.nnz());
            assert_eq!(100, model.weights.indices()[0]);
            assert_approx_eq!(0.06818952064451736, model.weights.data()[0]);
            assert_eq!(200, model.weights.indices()[1]);
            assert_approx_eq!(-1.1165663235675385, model.weights.data()[1]);
            assert_eq!(201, model.weights.indices()[2]);
            assert_approx_eq!(-0.7422697157666398, model.weights.data()[2]);
            assert_approx_eq!(
                -3.744966849165483,
                model.predict_score(feature_matrix.outer_view(1).unwrap())
            );
        }
    }
}

mod ffi {
    use super::*;

    #[repr(C)]
    #[allow(non_camel_case_types)]
    #[derive(Copy, Clone, Debug)]
    pub struct feature_node {
        pub index: c_int,
        pub value: c_double,
    }

    #[repr(C)]
    #[allow(non_camel_case_types)]
    #[derive(Debug)]
    pub struct problem {
        pub l: c_int,
        pub n: c_int,
        pub y: *const c_double,
        pub x: *const *const feature_node,
        pub bias: c_double,
    }

    pub static L2R_LR: c_int = 0;
    pub static L2R_L2LOSS_SVC: c_int = 2;

    #[repr(C)]
    #[allow(non_camel_case_types)]
    #[allow(non_snake_case)]
    #[derive(Debug)]
    pub struct parameter {
        pub solver_type: c_int,
        pub eps: c_double,
        pub C: c_double,
        pub nr_weight: c_int,
        pub weight_label: *const c_int,
        pub weight: *const c_double,
        pub p: c_double,
        pub init_sol: *const c_double,
    }

    #[repr(C)]
    #[allow(non_camel_case_types)]
    #[derive(Debug)]
    pub struct model {
        pub param: parameter,
        pub nr_class: c_int,
        pub nr_feature: c_int,
        pub w: *const c_double,
        pub label: *const c_int,
        pub bias: c_double,
    }

    /// Used to suppress output from liblinear.
    pub extern "C" fn print_null(_cstr: *const c_char) {}

    extern "C" {
        #[cfg(test)]
        #[allow(non_upper_case_globals)]
        static liblinear_version: i32;

        pub fn train(prob: *const problem, param: *const parameter) -> *mut model;
        pub fn check_parameter(prob: *const problem, param: *const parameter) -> *const c_char;
        pub fn free_and_destroy_model(model_ptr_ptr: *mut *mut model);
        pub fn set_print_string_function(print_func: extern "C" fn(*const c_char));
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_version() {
            unsafe {
                assert_eq!(liblinear_version, 221);
            }
        }
    }
}

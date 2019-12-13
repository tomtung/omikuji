use crate::mat_util::*;
use itertools::Itertools;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::{INFINITY, NEG_INFINITY};
use std::iter::FromIterator;
use std::ops::Deref;

/// The loss function used by liblinear model.
#[derive(Eq, PartialEq, Copy, Clone, Debug, Serialize, Deserialize)]
pub enum LossType {
    /// Log loss: min_w w^Tw/2 + C \sum log(1 + exp(-y_i w^Tx_i))
    Log,
    /// Squared hinge loss: min_w w^Tw/2 + C \sum max(0, 1- y_i w^Tx_i)^2
    Hinge,
}

/// Hyper-parameter settings for training liblinear model.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct HyperParam {
    pub loss_type: LossType,
    pub eps: f32,
    pub c: f32,
    pub weight_threshold: f32,
    pub max_iter: u32,
}

impl Default for HyperParam {
    fn default() -> Self {
        Self {
            loss_type: LossType::Hinge,
            eps: 0.1,
            c: 1.,
            weight_threshold: 0.1,
            max_iter: 20,
        }
    }
}

impl HyperParam {
    /// Check if the hyper-parameter settings are valid.
    pub fn validate(&self) -> Result<(), String> {
        if self.eps <= 0. {
            Err(format!("eps must be positive, but is {}", self.eps))
        } else if self.c <= 0. {
            Err(format!("c must be positive, but is {}", self.c))
        } else if self.weight_threshold < 0. {
            Err(format!(
                "weight_threshold must be non-negative, but is {}",
                self.weight_threshold
            ))
        } else if self.max_iter == 0 {
            Err(format!(
                "max_iter must be positive, but is {}",
                self.max_iter
            ))
        } else {
            Ok(())
        }
    }

    /// Adapt regularization based on sample size relative to overall training data size.
    pub(crate) fn adapt_to_sample_size(
        &self,
        n_curr_examples: usize,
        n_total_examples: usize,
    ) -> Self {
        match self.loss_type {
            LossType::Hinge => *self,
            LossType::Log => Self {
                c: self.c * n_total_examples as f32 / n_curr_examples as f32,
                ..*self
            },
        }
    }

    /// Train a one-vs-all multi-label classifier with the given data.
    pub(crate) fn train<Indices: Deref<Target = [usize]> + Sync>(
        &self,
        feature_matrix: &SparseMatView,
        label_to_example_indices: &[Indices],
    ) -> Vec<Option<Vector>> {
        self.validate().unwrap();

        assert!(feature_matrix.is_csr());
        // Remove empty columns from features matrix to speed up training
        let n_features = feature_matrix.inner_dims();
        let (feature_matrix, index_to_feature) = feature_matrix.to_owned().shrink_inner_indices();

        let solver = match self.loss_type {
            LossType::Hinge => solve_l2r_l2_svc,
            LossType::Log => solve_l2r_lr_dual,
        };
        label_to_example_indices
            .par_iter()
            .map(|indices| {
                // For the current classifier, an example is positive iff its index is in the given list
                let mut labels = vec![false; feature_matrix.rows()];
                let mut n_pos = 0;
                for &i in indices.iter() {
                    labels[i] = true;
                    n_pos += 1;
                }
                assert_ne!(n_pos, 0);
                // Don't train if all examples are positives
                if n_pos == labels.len() {
                    return None;
                }

                // Train the classifier
                let mut w = {
                    let (indices, data) = solver(
                        &feature_matrix.view(),
                        &labels,
                        self.eps,
                        self.c,
                        self.c,
                        self.max_iter,
                    )
                    .indexed_iter()
                    .filter_map(|(index, &value)| {
                        if value.abs() <= self.weight_threshold {
                            None
                        } else {
                            Some((index_to_feature[index], value))
                        }
                    })
                    .unzip();

                    Vector::Sparse(SparseVec::new(n_features, indices, data))
                };

                // Only store in sparse format if density is lower than half to save space
                if w.density() > 0.5 {
                    w.densify();
                }

                Some(w)
            })
            .collect()
    }
}

pub(crate) fn predict(
    weights: &[Option<Vector>],
    loss_type: LossType,
    feature_vec: &SparseVec,
) -> DenseVec {
    DenseVec::from_iter(weights.iter().map(|w| {
        if let Some(w) = w {
            let score = w.dot(feature_vec);
            match loss_type {
                LossType::Log => -(-score).exp().ln_1p(),
                LossType::Hinge => -(1. - score).max(0.).powi(2),
            }
        } else {
            0.
        }
    }))
}

/// A coordinate descent solver for L2-loss SVM dual problems.
///
/// This is pretty much a line-by-line port from liblinear (with some simplification) to avoid
/// unnecessary ffi-related overhead.
///
///  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
///    s.t.      0 <= \alpha_i <= upper_bound_i,
///
///  where Qij = yi yj xi^T xj and
///  D is a diagonal matrix
///
/// In L1-SVM case (omitted):
/// 		upper_bound_i = Cp if y_i = 1
/// 		upper_bound_i = Cn if y_i = -1
/// 		D_ii = 0
/// In L2-SVM case:
/// 		upper_bound_i = INF
/// 		D_ii = 1/(2*Cp)	if y_i = 1
/// 		D_ii = 1/(2*Cn)	if y_i = -1
///
/// Given:
/// x, y, Cp, Cn
/// eps is the stopping tolerance
///
/// See Algorithm 3 of Hsieh et al., ICML 2008.
#[allow(clippy::many_single_char_names)]
fn solve_l2r_l2_svc(
    x: &SparseMatView,
    y: &[bool],
    eps: f32,
    cp: f32,
    cn: f32,
    max_iter: u32,
) -> DenseVec {
    assert!(x.is_csr());
    assert_eq!(x.rows(), y.len());

    let l = x.rows();
    let w_size = x.cols();
    let mut w = DenseVec::zeros(w_size);

    let mut active_size = l;

    // PG: projected gradient, for shrinking and stopping
    let mut pg: f32;
    let mut pgmax_old = INFINITY;
    let mut pgmax_new: f32;
    let mut pgmin_new: f32;

    // default solver_type: L2R_L2LOSS_SVC_DUAL
    let diag: [f32; 2] = [0.5 / cn, 0.5 / cp];

    // Note that 0 <= alpha[i] <= upper_bound[y[i]]
    let mut alpha = vec![0.; l];

    let mut index = (0..l).collect_vec();
    let qd = x
        .outer_iterator()
        .zip(y.iter())
        .map(|(xi, &yi)| diag[yi as usize] + csvec_dot_self(&xi))
        .collect_vec();

    let mut iter = 0;
    let mut rng = thread_rng();
    while iter < max_iter {
        pgmax_new = NEG_INFINITY;
        pgmin_new = INFINITY;

        index.shuffle(&mut rng);

        let mut s = 0;
        while s < active_size {
            let i = index[s];
            let yi = y[i];
            let yi_sign = if yi { 1. } else { -1. };
            let xi = x.outer_view(i).unwrap_or_else(|| {
                panic!(
                    "Failed to take {}-th outer view for matrix x of shape {:?}",
                    i,
                    x.shape()
                )
            });
            let alpha_i = &mut alpha[i];

            let g = yi_sign * xi.dot_dense(w.view()) - 1. + *alpha_i * diag[yi as usize];

            pg = 0.;
            if *alpha_i == 0. {
                if g > pgmax_old {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                } else if g < 0. {
                    pg = g;
                }
            } else {
                pg = g;
            }

            pgmax_new = pgmax_new.max(pg);
            pgmin_new = pgmin_new.min(pg);

            if pg.abs() > 1e-12 {
                let alpha_old = *alpha_i;
                *alpha_i = (*alpha_i - g / qd[i]).max(0.);
                let d = (*alpha_i - alpha_old) * yi_sign;
                dense_add_assign_csvec_mul_scalar(w.view_mut(), xi, d);
            }

            s += 1;
        }

        iter += 1;

        if pgmax_new - pgmin_new <= eps {
            if active_size == l {
                break;
            } else {
                active_size = l;
                pgmax_old = INFINITY;
                continue;
            }
        }
        pgmax_old = pgmax_new;
        if pgmax_old <= 0. {
            pgmax_old = INFINITY;
        }
    }

    w
}

/// A coordinate descent solver for the dual of L2-regularized logistic regression problems.
///
/// This is pretty much a line-by-line port from liblinear (with some simplification) to avoid
/// unnecessary ffi-related overhead.
///
///  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
///    s.t.      0 <= \alpha_i <= upper_bound_i,
///
///  where Qij = yi yj xi^T xj and
///  upper_bound_i = Cp if y_i = 1
///  upper_bound_i = Cn if y_i = -1
///
/// Given:
/// x, y, Cp, Cn
/// eps is the stopping tolerance
///
/// See Algorithm 5 of Yu et al., MLJ 2010.
#[allow(clippy::many_single_char_names)]
fn solve_l2r_lr_dual(
    x: &SparseMatView,
    y: &[bool],
    eps: f32,
    cp: f32,
    cn: f32,
    max_iter: u32,
) -> DenseVec {
    assert!(x.is_csr());
    assert_eq!(x.rows(), y.len());

    let l = x.rows();
    let w_size = x.cols();

    let max_inner_iter = 100; // for inner Newton
    let mut innereps = 1e-2;
    let innereps_min = eps.min(1e-8);
    let upper_bound = [cn, cp];

    // store alpha and C - alpha. Note that
    // 0 < alpha[i] < upper_bound[GETI(i)]
    // alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
    let mut alpha = y
        .iter()
        .flat_map(|&yi| {
            let c = upper_bound[yi as usize];
            let alpha = (0.001 * c).min(1e-8);
            vec![alpha, c - alpha]
        })
        .collect_vec();

    let xtx = x
        .outer_iterator()
        .map(|xi| csvec_dot_self(&xi))
        .collect_vec();

    let mut w = DenseVec::zeros(w_size);
    for (i, (xi, &yi)) in x.outer_iterator().zip(y.iter()).enumerate() {
        let yi_sign = if yi { 1. } else { -1. };
        dense_add_assign_csvec_mul_scalar(w.view_mut(), xi, yi_sign * alpha[2 * i]);
    }

    let mut index = (0..l).collect_vec();

    let mut iter = 0;
    let mut rng = thread_rng();
    while iter < max_iter {
        index.shuffle(&mut rng);
        let mut newton_iter = 0;
        let mut gmax = 0f32;
        for &i in &index {
            let yi = y[i];
            let yi_sign = if yi { 1. } else { -1. };
            let c = upper_bound[yi as usize];
            let xi = x.outer_view(i).unwrap_or_else(|| {
                panic!(
                    "Failed to take {}-th outer view for matrix x of shape {:?}",
                    i,
                    x.shape()
                )
            });
            let a = xtx[i];
            let b = yi_sign * xi.dot_dense(w.view());

            // Decide to minimize g_1(z) or g_2(z)
            let (ind1, ind2, sign) = if 0.5 * a * (alpha[2 * i + 1] - alpha[2 * i]) + b < 0. {
                (2 * i + 1, 2 * i, -1.)
            } else {
                (2 * i, 2 * i + 1, 1.)
            };

            //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
            let alpha_old = alpha[ind1];
            let mut z = if c - alpha_old < 0.5 * c {
                0.1 * alpha_old
            } else {
                alpha_old
            };
            let mut gp = a * (z - alpha_old) + sign * b + (z / (c - z)).ln();
            gmax = gmax.max(gp.abs());

            // Newton method on the sub-problem
            let eta = 0.1; // xi in the paper
            let mut inner_iter = 0;
            while inner_iter <= max_inner_iter {
                if gp.abs() < innereps {
                    break;
                }
                let gpp = a + c / (c - z) / z;
                let tmpz = z - gp / gpp;
                if tmpz <= 0. {
                    z *= eta;
                } else {
                    // tmpz in (0, C)
                    z = tmpz;
                }
                gp = a * (z - alpha_old) + sign * b + (z / (c - z)).ln();
                newton_iter += 1;
                inner_iter += 1;
            }

            if inner_iter > 0 {
                // update w
                alpha[ind1] = z;
                alpha[ind2] = c - z;
                dense_add_assign_csvec_mul_scalar(
                    w.view_mut(),
                    xi,
                    sign * (z - alpha_old) * yi_sign,
                );
            }
        }

        iter += 1;

        if gmax < eps {
            break;
        }

        if newton_iter <= l / 10 {
            innereps = innereps_min.max(0.1 * innereps);
        }
    }

    w
}

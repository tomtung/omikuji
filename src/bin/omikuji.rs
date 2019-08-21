extern crate clap;
extern crate omikuji;
extern crate rayon;

use clap::value_t;
use std::fs::File;
use std::io::{BufWriter, Write};

fn set_num_threads(matches: &clap::ArgMatches) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(value_t!(matches, "n_threads", usize).unwrap())
        .stack_size(32 * 1024 * 1024)
        .build_global()
        .unwrap();
}

fn parse_train_hyper_param(matches: &clap::ArgMatches) -> omikuji::model::TrainHyperParam {
    let mut hyper_param = omikuji::model::train::HyperParam::default();

    hyper_param.n_trees = value_t!(matches, "n_trees", usize).unwrap();
    hyper_param.min_branch_size = value_t!(matches, "min_branch_size", usize).unwrap();
    hyper_param.max_depth = value_t!(matches, "max_depth", usize).unwrap();
    hyper_param.centroid_threshold = value_t!(matches, "centroid_threshold", f32).unwrap();
    hyper_param.collapse_every_n_layers =
        value_t!(matches, "collapse_every_n_layers", usize).unwrap();
    hyper_param.tree_structure_only = matches.occurrences_of("tree_structure_only") > 0;

    hyper_param.linear.loss_type = match matches.value_of("linear.loss").unwrap() {
        "hinge" => omikuji::model::liblinear::LossType::Hinge,
        "log" => omikuji::model::liblinear::LossType::Log,
        _ => unreachable!(),
    };
    hyper_param.linear.eps = value_t!(matches, "linear.eps", f32).unwrap();
    hyper_param.linear.c = value_t!(matches, "linear.c", f32).unwrap();
    hyper_param.linear.weight_threshold =
        value_t!(matches, "linear.weight_threshold", f32).unwrap();
    hyper_param.linear.max_iter = value_t!(matches, "linear.max_iter", u32).unwrap();

    hyper_param.cluster.k = value_t!(matches, "cluster.k", usize).unwrap();
    hyper_param.cluster.balanced = matches.occurrences_of("cluster.unbalanced") == 0;
    hyper_param.cluster.eps = value_t!(matches, "cluster.eps", f32).unwrap();
    hyper_param.cluster.min_size = value_t!(matches, "cluster.min_size", usize).unwrap();

    hyper_param.validate().unwrap();
    hyper_param
}

fn train(matches: &clap::ArgMatches) {
    set_num_threads(matches);
    let train_hyperparam = parse_train_hyper_param(matches);

    let training_dataset = {
        let path = matches.value_of("training_data").unwrap();
        omikuji::DataSet::load_xc_repo_data_file(path).expect("Failed to load training data")
    };

    let model = train_hyperparam.train(training_dataset);
    if let Some(model_path) = matches.value_of("model_path") {
        model.save(model_path).expect("Failed to save model");
    }
}

fn test(matches: &clap::ArgMatches) {
    set_num_threads(matches);

    let model = {
        let model_path = matches.value_of("model_path").unwrap();
        let mut model = omikuji::Model::load(model_path).expect("Failed to load model");
        let max_sparse_density = value_t!(matches, "max_sparse_density", f32).unwrap();
        model.densify_weights(max_sparse_density);
        model
    };

    if let Some(test_path) = matches.value_of("test_data") {
        let test_dataset =
            omikuji::DataSet::load_xc_repo_data_file(test_path).expect("Failed to load test data");

        let (predictions, _) = {
            let beam_size = value_t!(matches, "beam_size", usize).unwrap();
            omikuji::model::eval::test_all(&model, &test_dataset, beam_size)
        };
        if let Some(out_path) = matches.value_of("out_path") {
            let k_top = value_t!(matches, "k_top", usize).unwrap();

            let mut writer =
                BufWriter::new(File::create(out_path).expect("Failed to create output file"));
            for prediction in predictions {
                for (i, &(ref label, score)) in prediction.iter().take(k_top).enumerate() {
                    if i > 0 {
                        write!(&mut writer, "\t").unwrap();
                    }
                    write!(&mut writer, "{} {:.3}", label, score).unwrap();
                }
                writeln!(&mut writer).unwrap();
            }
        }
    }
}

fn main() {
    simple_logger::init().unwrap();

    let default_hyperparam = omikuji::model::train::HyperParam::default();
    let default_n_trees = default_hyperparam.n_trees.to_string();
    let default_min_branch_size = default_hyperparam.min_branch_size.to_string();
    let default_max_depth = default_hyperparam.max_depth.to_string();
    let default_centroid_threshold = default_hyperparam.centroid_threshold.to_string();
    let default_collapse_every_n_layers = default_hyperparam.collapse_every_n_layers.to_string();
    let default_linear_eps = default_hyperparam.linear.eps.to_string();
    let default_linear_c = default_hyperparam.linear.c.to_string();
    let default_linear_weight_threshold = default_hyperparam.linear.weight_threshold.to_string();
    let default_linear_max_iter = default_hyperparam.linear.max_iter.to_string();
    let default_cluster_k = default_hyperparam.cluster.k.to_string();
    let default_cluster_eps = default_hyperparam.cluster.eps.to_string();
    let default_cluster_min_size = default_hyperparam.cluster.min_size.to_string();

    let arg_matches = clap::App::new("omikuji")
        .about("Omikuji: an efficient implementation of Partitioned Label Trees and its variations \
                for extreme multi-label classification")
        .subcommand(
            clap::SubCommand::with_name("train")
                .about("Train a new model")
                .arg(
                    clap::Arg::with_name("training_data")
                        .index(1)
                        .help("Path to training dataset file (in the format of the Extreme Classification Repository)")
                        .required(true)
                        .value_name("TRAINING_DATA_PATH")
                )
                .arg(
                    clap::Arg::with_name("model_path")
                        .long("model_path")
                        .help("Optional path of the directory where the trained model will be saved if provided; \
                               if an model with compatible settings is already saved in the given directory, \
                               the newly trained trees will be added to the existing model")
                        .takes_value(true)
                        .value_name("PATH")
                        .required(false)
                )
                .arg(
                    clap::Arg::with_name("n_threads")
                        .long("n_threads")
                        .help("Number of worker threads. If 0, the number is selected automatically")
                        .takes_value(true)
                        .value_name("T")
                        .default_value("0")
                )
                .arg(
                    clap::Arg::with_name("n_trees")
                        .long("n_trees")
                        .help("Number of trees")
                        .takes_value(true)
                        .value_name("N")
                        .default_value(&default_n_trees)
                )
                .arg(
                    clap::Arg::with_name("min_branch_size")
                        .long("min_branch_size")
                        .help("Number of labels below which no further clustering & branching is done")
                        .takes_value(true)
                        .value_name("SIZE")
                        .default_value(&default_min_branch_size)
                )
                .arg(
                    clap::Arg::with_name("max_depth")
                        .long("max_depth")
                        .help("Maximum tree depth")
                        .takes_value(true)
                        .value_name("DEPTH")
                        .default_value(&default_max_depth)
                )
                .arg(
                    clap::Arg::with_name("centroid_threshold")
                        .long("centroid_threshold")
                        .help("Threshold for pruning label centroid vectors")
                        .takes_value(true)
                        .value_name("THRESHOLD")
                        .default_value(&default_centroid_threshold)
                )
                .arg(
                    clap::Arg::with_name("collapse_every_n_layers")
                        .long("collapse_every_n_layers")
                        .help("Number of adjacent layers to collapse, \
                                  which increases tree arity and decreases tree depth")
                        .takes_value(true)
                        .value_name("N")
                        .default_value(&default_collapse_every_n_layers)
                )
                .arg(
                    clap::Arg::with_name("tree_structure_only")
                        .long("tree_structure_only")
                        .help("Build the trees without training classifiers; \
                                  useful when a downstream user needs the tree structures only")
                        .takes_value(false)
                )
                .arg(
                    clap::Arg::with_name("linear.loss")
                        .long("linear.loss")
                        .help("Loss function used by linear classifiers")
                        .takes_value(true)
                        .value_name("LOSS")
                        .default_value(match default_hyperparam.linear.loss_type {
                            omikuji::model::liblinear::LossType::Hinge => "hinge",
                            omikuji::model::liblinear::LossType::Log => "log",
                        })
                        .possible_values(&["hinge", "log"])
                )
                .arg(
                    clap::Arg::with_name("linear.eps")
                        .long("linear.eps")
                        .help("Epsilon value for determining linear classifier convergence")
                        .takes_value(true)
                        .value_name("EPS")
                        .default_value(& default_linear_eps)
                )
                .arg(
                    clap::Arg::with_name("linear.c")
                        .long("linear.c")
                        .help("Cost co-efficient for regularizing linear classifiers")
                        .takes_value(true)
                        .value_name("C")
                        .default_value(&default_linear_c)
                )
                .arg(
                    clap::Arg::with_name("linear.weight_threshold")
                        .long("linear.weight_threshold")
                        .help("Threshold for pruning weight vectors of linear classifiers")
                        .takes_value(true)
                        .value_name("THRESHOLD")
                        .default_value(&default_linear_weight_threshold)
                )
                .arg(
                    clap::Arg::with_name("linear.max_iter")
                        .long("linear.max_iter")
                        .help("Max number of iterations for training each linear classifier")
                        .takes_value(true)
                        .value_name("M")
                        .default_value(&default_linear_max_iter)
                )
                .arg(
                    clap::Arg::with_name("cluster.k")
                        .long("cluster.k")
                        .help("Number of clusters")
                        .takes_value(true)
                        .value_name("K")
                        .default_value(&default_cluster_k)
                )
                .arg(
                    clap::Arg::with_name("cluster.unbalanced")
                        .long("cluster.unbalanced")
                        .help("Perform regular k-means clustering instead of balanced k-means clustering")
                        .takes_value(false)
                )
                .arg(
                    clap::Arg::with_name("cluster.eps")
                        .long("cluster.eps")
                        .help("Epsilon value for determining clustering convergence")
                        .takes_value(true)
                        .value_name("EPS")
                        .default_value(&default_cluster_eps)
                )
                .arg(
                    clap::Arg::with_name("cluster.min_size")
                        .long("cluster.min_size")
                        .help("Labels in clusters with sizes smaller than this threshold are reassigned to other clusters instead")
                        .takes_value(true)
                        .value_name("SIZE")
                        .default_value(&default_cluster_min_size)
                )
            )
        .subcommand(
            clap::SubCommand::with_name("test")
                .about("Test an existing model")
                .arg(
                    clap::Arg::with_name("model_path")
                        .index(1)
                        .help("Path of the directory where the trained model is saved")
                        .required(true)
                        .value_name("MODEL_PATH")
                )
                .arg(
                    clap::Arg::with_name("test_data")
                        .index(2)
                        .help("Path to test dataset file (in the format of the Extreme Classification Repository)")
                        .required(true)
                        .value_name("TEST_DATA_PATH")
                )
                .arg(
                    clap::Arg::with_name("n_threads")
                        .long("n_threads")
                        .help("Number of worker threads. If 0, the number is selected automatically")
                        .takes_value(true)
                        .value_name("T")
                        .default_value("0")
                )
                .arg(
                    clap::Arg::with_name("max_sparse_density")
                        .long("max_sparse_density")
                        .help("Density threshold above which sparse weight vectors are converted to dense format. Lower values speed up prediction at the cost of more memory usage")
                        .takes_value(true)
                        .value_name("DENSITY")
                        .default_value("0.1")
                )
                .arg(
                    clap::Arg::with_name("beam_size")
                        .long("beam_size")
                        .help("Beam size for beam search")
                        .takes_value(true)
                        .default_value("10")
                )
                .arg(
                    clap::Arg::with_name("k_top")
                        .long("k_top")
                        .help("Number of top predictions to write out for each test example")
                        .takes_value(true)
                        .value_name("K")
                        .default_value("5")
                )
                .arg(
                    clap::Arg::with_name("out_path")
                        .long("out_path")
                        .help("Path to the which predictions will be written, if provided")
                        .takes_value(true)
                        .value_name("PATH")
                        .required(false)
                )
            )
        .get_matches();

    if let Some(arg_matches) = arg_matches.subcommand_matches("train") {
        train(&arg_matches);
    } else if let Some(arg_matches) = arg_matches.subcommand_matches("test") {
        test(&arg_matches);
    } else {
        println!("{}", arg_matches.usage());
    }
}

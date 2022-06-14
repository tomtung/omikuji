use clap::{ArgEnum, Args, Parser, Subcommand};
use const_default::ConstDefault;
use omikuji::model::liblinear::LossType;
use omikuji::model::TrainHyperParam;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[clap(version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new omikuji model
    Train(TrainArgs),

    /// Test an existing omikuji model
    Test(TestArgs),
}

#[derive(Args)]
#[clap(rename_all = "snake_case")]
struct TrainArgs {
    /// Path to training dataset file
    ///
    /// The dataset file is expected to be in the format of the Extreme Classification
    /// Repository.
    #[clap(value_parser, required = true)]
    training_data_path: PathBuf,

    /// Optional path of the directory where the trained model will be saved if provided
    ///
    /// If an model with compatible settings is already saved in the given directory,
    /// the newly trained trees will be added to the existing model")
    #[clap(value_parser, long)]
    model_path: Option<PathBuf>,

    /// Number of worker threads
    ///
    /// If 0, the number is selected automatically.
    #[clap(value_parser, long, default_value_t = 0)]
    n_threads: usize,

    /// Number of trees.
    #[clap(value_parser, long, default_value_t = TrainHyperParam::DEFAULT.n_trees)]
    n_trees: usize,

    /// Number of labels below which no further clustering & branching is done
    #[clap(value_parser, long, name = "SIZE", default_value_t = TrainHyperParam::DEFAULT.min_branch_size)]
    min_branch_size: usize,

    /// Maximum tree depth
    #[clap(value_parser, long, name = "DEPTH", default_value_t = TrainHyperParam::DEFAULT.max_depth)]
    max_depth: usize,

    /// Threshold for pruning label centroid vectors
    #[clap(value_parser, long, name = "THRESHOLD", default_value_t = TrainHyperParam::DEFAULT.centroid_threshold)]
    centroid_threshold: f32,

    /// Number of adjacent layers to collapse
    ///
    /// This increases tree arity and decreases tree depth.
    #[clap(value_parser, long, name = "N_LAYERS", default_value_t = TrainHyperParam::DEFAULT.collapse_every_n_layers)]
    collapse_every_n_layers: usize,

    /// Build the trees without training classifiers
    ///
    /// Might be useful when a downstream user needs the tree structures only.
    #[clap(value_parser, long)]
    tree_structure_only: bool,

    /// Finish training each tree before start training the next
    ///
    /// This limits initial parallelization but saves memory.
    #[clap(value_parser, long)]
    train_trees_1_by_1: bool,

    /// Loss function used by linear classifiers
    #[clap(value_parser, arg_enum, long = "linear.loss", name = "LOSS", default_value_t = TrainHyperParam::DEFAULT.linear.loss_type.into())]
    linear_loss: CliLossType,

    /// Epsilon value for determining linear classifier convergence
    #[clap(value_parser, long = "linear.eps", default_value_t = TrainHyperParam::DEFAULT.linear.eps)]
    linear_eps: f32,

    /// Cost coefficient for regularizing linear classifiers
    #[clap(value_parser, long = "linear.c", name = "C", default_value_t = TrainHyperParam::DEFAULT.linear.c)]
    linear_c: f32,

    /// Threshold for pruning weight vectors of linear classifiers
    #[clap(value_parser, long = "linear.weight_threshold", name = "MIN_WEIGHT", default_value_t = TrainHyperParam::DEFAULT.linear.weight_threshold)]
    linear_weight_threshold: f32,

    /// Max number of iterations for training each linear classifier
    #[clap(value_parser, long = "linear.max_iter", name = "M", default_value_t = TrainHyperParam::DEFAULT.linear.max_iter)]
    linear_max_iter: u32,

    /// Number of clusters
    #[clap(value_parser, long = "cluster.k", name = "K", default_value_t = TrainHyperParam::DEFAULT.cluster.k)]
    cluster_k: usize,

    /// Perform regular k-means clustering instead of balanced k-means clustering
    #[clap(value_parser, long = "cluster.unbalanced")]
    cluster_unbalanced: bool,

    /// Epsilon value for determining linear classifier convergence
    #[clap(value_parser, long = "cluster.eps", default_value_t = TrainHyperParam::DEFAULT.cluster.eps)]
    cluster_eps: f32,

    /// Labels in clusters with sizes smaller than this threshold are reassigned to other
    /// clusters instead
    #[clap(value_parser, long = "cluster.min_size", name = "MIN_SIZE", default_value_t = TrainHyperParam::DEFAULT.cluster.min_size)]
    cluster_min_size: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ArgEnum)]
enum CliLossType {
    Hinge,
    Log,
}

impl From<LossType> for CliLossType {
    fn from(loss: LossType) -> Self {
        match loss {
            LossType::Log => Self::Log,
            LossType::Hinge => Self::Hinge,
        }
    }
}

impl From<CliLossType> for LossType {
    fn from(loss: CliLossType) -> Self {
        match loss {
            CliLossType::Log => LossType::Log,
            CliLossType::Hinge => LossType::Hinge,
        }
    }
}

impl From<&TrainArgs> for TrainHyperParam {
    fn from(args: &TrainArgs) -> Self {
        omikuji::model::train::HyperParam {
            n_trees: args.n_trees,
            min_branch_size: args.min_branch_size,
            max_depth: args.max_depth,
            centroid_threshold: args.centroid_threshold,
            collapse_every_n_layers: args.collapse_every_n_layers,
            tree_structure_only: args.tree_structure_only,
            train_trees_1_by_1: args.train_trees_1_by_1,
            linear: omikuji::model::liblinear::HyperParam {
                loss_type: args.linear_loss.into(),
                eps: args.linear_eps,
                c: args.linear_c,
                weight_threshold: args.linear_weight_threshold,
                max_iter: args.linear_max_iter,
            },
            cluster: omikuji::model::cluster::HyperParam {
                k: args.cluster_k,
                balanced: !args.cluster_unbalanced,
                eps: args.cluster_eps,
                min_size: args.cluster_min_size,
            },
        }
    }
}

#[derive(Args)]
#[clap(rename_all = "snake_case")]
struct TestArgs {
    /// Path of the directory where the trained model is saved
    #[clap(value_parser, required = true)]
    model_path: PathBuf,

    /// Path to test dataset file
    ///
    /// The dataset file is expected to be in the format of the Extreme Classification
    /// Repository.
    #[clap(value_parser, required = true)]
    test_data_path: PathBuf,

    /// Number of worker threads
    ///
    /// If 0, the number is selected automatically.
    #[clap(value_parser, long, default_value_t = 0)]
    n_threads: usize,

    /// Density threshold above which sparse weight vectors are converted to dense format
    ///
    /// Lower values speed up prediction at the cost of more memory usage.
    #[clap(value_parser, long, name = "DENSITY", default_value_t = 0.1)]
    max_sparse_density: f32,

    /// Beam size for beam search
    #[clap(value_parser, long, default_value_t = 10)]
    beam_size: usize,

    /// Number of top predictions to write out for each test example
    #[clap(value_parser, long, name = "K", default_value_t = 5)]
    k_top: usize,

    /// Path to the which predictions will be written, if provided
    #[clap(value_parser, long)]
    out_path: Option<PathBuf>,
}

fn set_num_threads(num_threads: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .stack_size(32 * 1024 * 1024)
        .build_global()
        .unwrap();
}

fn train(args: &TrainArgs) {
    set_num_threads(args.n_threads);
    let train_hyperparam: TrainHyperParam = args.into();

    let training_dataset = {
        omikuji::DataSet::load_xc_repo_data_file(args.training_data_path.as_path())
            .expect("Failed to load training data")
    };

    let model = train_hyperparam.train(training_dataset);
    if let Some(model_path) = args.model_path.as_ref() {
        model.save(model_path).expect("Failed to save model");
    }
}

fn test(args: &TestArgs) {
    set_num_threads(args.n_threads);

    let model = {
        let mut model =
            omikuji::Model::load(args.model_path.as_path()).expect("Failed to load model");
        model.densify_weights(args.max_sparse_density);
        model
    };

    let test_dataset = omikuji::DataSet::load_xc_repo_data_file(args.test_data_path.as_path())
        .expect("Failed to load test data");

    let (predictions, _) =
        { omikuji::model::eval::test_all(&model, &test_dataset, args.beam_size) };
    if let Some(out_path) = args.out_path.as_ref() {
        let mut writer =
            BufWriter::new(File::create(out_path).expect("Failed to create output file"));
        for prediction in predictions {
            for (i, &(ref label, score)) in prediction.iter().take(args.k_top).enumerate() {
                if i > 0 {
                    write!(&mut writer, "\t").unwrap();
                }
                write!(&mut writer, "{} {:.3}", label, score).unwrap();
            }
            writeln!(&mut writer).unwrap();
        }
    }
}

fn main() {
    simple_logger::init().unwrap();
    let cli = Cli::parse();
    match &cli.command {
        Commands::Train(args) => train(args),
        Commands::Test(args) => test(args),
    }
}

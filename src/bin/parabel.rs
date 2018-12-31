extern crate clap;
extern crate parabel;
extern crate rayon;

use clap::{load_yaml, value_t};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

fn set_num_threads(matches: &clap::ArgMatches) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(value_t!(matches, "n_threads", usize).unwrap())
        .build_global()
        .unwrap();
}

fn parse_train_hyper_param(matches: &clap::ArgMatches) -> parabel::model::TrainHyperParam {
    let liblinear_hyperparam = {
        parabel::model::liblinear::HyperParam::builder()
            .loss_type(match matches.value_of("linear.loss").unwrap() {
                "hinge" => parabel::model::liblinear::LossType::Hinge,
                "log" => parabel::model::liblinear::LossType::Log,
                _ => unreachable!(),
            })
            .eps(value_t!(matches, "linear.eps", f32).unwrap())
            .C(value_t!(matches, "linear.c", f32).unwrap())
            .weight_threshold(value_t!(matches, "linear.weight_threshold", f32).unwrap())
            .max_iter(value_t!(matches, "linear.max_iter", u32).unwrap())
            .build()
            .unwrap()
    };

    parabel::model::TrainHyperParam::builder()
        .linear(liblinear_hyperparam)
        .n_trees(value_t!(matches, "n_trees", usize).unwrap())
        .max_leaf_size(value_t!(matches, "max_leaf_size", usize).unwrap())
        .cluster_eps(value_t!(matches, "cluster_eps", f32).unwrap())
        .centroid_threshold(value_t!(matches, "centroid_threshold", f32).unwrap())
        .build()
        .unwrap()
}

fn train(matches: &clap::ArgMatches) {
    set_num_threads(matches);
    let train_hyperparam = parse_train_hyper_param(matches);

    let training_dataset = {
        let path = matches.value_of("training_data").unwrap();
        parabel::DataSet::load_xc_repo_data_file(path).expect("Failed to load training data")
    };

    let model = train_hyperparam.train(training_dataset);
    if let Some(model_path) = matches.value_of("model_path") {
        let model_file = File::create(model_path).expect("Failed to create model file");
        model
            .save(BufWriter::new(model_file))
            .expect("Failed to save model");
    }
}

fn test(matches: &clap::ArgMatches) {
    set_num_threads(matches);

    let model = {
        let model_path = matches.value_of("model_path").unwrap();
        let model_file = File::open(model_path).expect("Failed to open model file");
        parabel::Model::load(BufReader::new(model_file)).expect("Failed to load model")
    };

    if let Some(test_path) = matches.value_of("test_data") {
        let test_dataset =
            parabel::DataSet::load_xc_repo_data_file(test_path).expect("Failed to load test data");

        let (predictions, _) = {
            let beam_size = value_t!(matches, "beam_size", usize).unwrap();
            parabel::model::eval::test_all(&model, &test_dataset, beam_size)
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

    let yaml = load_yaml!("cli.yml");
    let arg_matches = clap::App::from_yaml(yaml).get_matches();

    if let Some(arg_matches) = arg_matches.subcommand_matches("train") {
        train(&arg_matches);
    } else if let Some(arg_matches) = arg_matches.subcommand_matches("test") {
        test(&arg_matches);
    } else {
        println!("{}", arg_matches.usage());
    }
}

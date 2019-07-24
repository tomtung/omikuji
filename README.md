# Parabel-rs

A highly parallelized 🦀Rust implementation of Parabel (Prabhu et al., 2018), a machine learning algorithm for solving extreme classification problems (i.e. multi-label classification problems with extremely large label sets).

## Performance

This Rust implementation has been tested on datasets from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html), and compared against the [C++ implementation](http://manikvarma.org/code/Parabel/download.html) from the paper authors. We measured training time, and calculated precisions at 1, 3, and 5. The results are summarized in the table below.

|     Dataset    | Implementation | Training Time |  P@1  |  P@3  |  P@5  |
| -------------- | -------------- | ------------- | ----- | ----- | ----- |
|    EURLex-4K   |      Rust      |      16.6s    | 81.73 | 69.12 | 57.72 |
|                |      C++       |      22.9s    | 82.25 | 68.70 | 57.53 |
|   Amazon-670K  |      Rust      |     315.1s    | 44.99 | 39.90 | 36.04 |
|                |      C++       |     484.0s    | 44.91 | 39.77 | 35.98 |
| WikiLSHTC-325K |      Rust      |     733.5s    | 65.02 | 43.16 | 32.08 |
|                |      C++       |    1079.0s    | 65.05 | 43.23 | 32.05 |

The tests were run on a quad-core Intel® Core™ i7-6700 CPU. For both implementations, we used the default hyper-parameter settings, and tried to utilize as many CPU cores as possible.

Note that since the C++ implementation trains each tree single-threaded, the number of CPU cores it can utilize is limited to the number of trees (3 by default). In contrast, our Rust implementation is able to utilize **all available CPU cores** whenever possible. On our quad-core machine, this resulted in a **1.3x to 1.5x speed up**; further speed-up is possible with more CPU cores available.

## Build & Install
Parabel-rs can be easily built & installed with [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) as a CLI app:
```
cargo install --git https://github.com/tomtung/parabel-rs.git --features cli
```

The CLI app will be available as `parabel`. For example, to reproduce the results on the EURLex-4K dataset:
```
parabel train eurlex_train.txt --model_path model.bin
parabel test model.bin eurlex_test.txt --out_path predictions.txt
```


### Python Binding

A simple Python binding is also available for training and prediction. It can be install via `pip`:

```
pip install git+https://github.com/tomtung/parabel-rs.git#subdirectory=python -v
```

The following script demonstrates how to use the Python binding to train a model and make predictions:

```python
import parabel

# Train
hyper_param = parabel.Model.default_hyper_param()
hyper_param.n_trees = 5
model = parabel.Model.train_on_data("./eurlex_train.txt", hyper_param)

# Serialize & de-serialize
model.save("model.bin")
model = parabel.Model.load("model.bin")

# Predict
feature_value_pairs = [
    (0, 0.101468),
    (1, 0.554374),
    (2, 0.235760),
    (3, 0.065255),
    (8, 0.152305),
    (10, 0.155051),
    # ...
]
label_score_pairs =  model.predict(feature_value_pairs)
```

## Usage
```
$ parabel train --help
parabel-train
Train a new Parabel model

USAGE:
    parabel train [FLAGS] [OPTIONS] <training_data>

FLAGS:
        --cluster.unbalanced    Perform regular k-means clustering instead of balanced k-means clustering
    -h, --help                  Prints help information
    -V, --version               Prints version information

OPTIONS:
        --centroid_threshold <THRESHOLD>         Threshold for pruning label centroid vectors [default: 0]
        --cluster.eps <EPS>                      Epsilon value for determining clustering convergence [default: 0.0001]
        --cluster.k <K>                          Number of clusters [default: 2]
        --cluster.min_size <SIZE>
            Labels in clusters with sizes smaller than this threshold are reassigned to other clusters instead [default:
            2]
        --linear.c <C>                           Cost co-efficient for regularizing linear classifiers [default: 1]
        --linear.eps <EPS>
            Epsilon value for determining linear classifier convergence [default: 0.1]

        --linear.loss <LOSS>
            Loss function used by linear classifiers [default: hinge]  [possible values: hinge, log]

        --linear.max_iter <M>
            Max number of iterations for training each linear classifier [default: 20]

        --linear.max_sparse_density <DENSITY>
            Density threshold above which weight vectors are stored in dense format. Lower values results in larger
            model but faster prediction [default: 0.15]
        --linear.weight_threshold <THRESHOLD>
            Threshold for pruning weight vectors of linear classifiers [default: 0.1]

        --max_depth <DEPTH>                      Maximum tree depth [default: 20]
        --min_branch_size <SIZE>
            Number of labels below which no futher clustering & branching is done [default: 100]

        --model_path <PATH>                      Path to which the trained model will be saved if provided
        --n_threads <T>
            Number of worker threads. If 0, the number is selected automatically [default: 0]

        --n_trees <N>                            Number of trees [default: 3]

ARGS:
    <training_data>    Path to training dataset file (in the format of the Extreme Classification Repository)
```

```
$ parabel test --help
parabel-test
Test an existing Parabel model

USAGE:
    parabel test [OPTIONS] <model_path> <test_data>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
        --beam_size <beam_size>    Beam size for beam search [default: 10]
        --k_top <K>                Number of top predictions to write out for each test example [default: 5]
        --n_threads <T>            Number of worker threads. If 0, the number is selected automatically [default: 0]
        --out_path <PATH>          Path to the which predictions will be written, if provided

ARGS:
    <model_path>    Path to the trained model
    <test_data>     Path to test dataset file (in the format of the Extreme Classification Repository)
```

### Data format

Our implementation takes dataset files formatted as those provided in the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). A data file starts with a header line with three space-separated integers: total number of examples, number of features, and number of labels. Following the header line, there is one line per each example, starting with comma-separated labels, followed by space-separated feature:value pairs:
```
label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
```

## License
Parabel-rs is licensed under the MIT License.

## References
- Y. Prabhu, A. Kag, S. Harsola, R. Agrawal and M. Varma. Parabel: Partitioned label trees for extreme classification with application to dynamic search advertising. In Proceedings of the International World Wide Web Conference, Lyon, France, April 2018. doi>[10.1145/3178876.3185998](https://doi.org/10.1145/3178876.3185998)

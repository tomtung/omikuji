# Parabel-rs

A highly parallelized 🦀Rust implementation of Partioned Label Trees (Prabhu et al., 2018 & Khandagale et al., 2019) for extreme multi-label classification.

## Performance

This Rust implementation has been tested on datasets from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html), and compared against the [original Parabel C++ implementation](http://manikvarma.org/code/Parabel/download.html) as well as the [original Bonsai C++ implementation](https://github.com/xmc-aalto/bonsai). We measured training time, and calculated precisions at 1, 3, and 5. The tests were run on a quad-core Intel® Core™ i7-6700 CPU. For both implementations, we used the default hyper-parameter settings, and allowed each implementation to utilize as many CPU cores as possible. The results are summarized in the table below.

| Dataset         	| Metric     	| Parabel 	| Parabel-rs<br/>(balanced,<br/>cluster.k=2) 	| Bonsai  	| Parabel-rs<br/>(unbalanced,<br/>cluster.k=100,<br/>max\_depth=3) 	|
|-----------------	|------------	|---------	|-------------------------------------------	|---------	|-----------------------------------------------------------------	|
|  EURLex-4K      	| P@1        	| 82.2    	| 81.9                                      	| 82.8    	| 83.0                                                            	|
|                 	| P@3        	| 68.8    	| 68.8                                      	| 69.4    	| 69.5                                                            	|
|                 	| P@5        	| 57.6    	| 57.4                                      	| 58.1    	| 58.3                                                            	|
|                 	| Train Time 	| 18s     	| 14s                                       	| 87s     	| 19s                                                             	|
| Amazon-670K     	| P@1        	| 44.9    	| 44.8                                      	| 45.5*   	| 45.6                                                            	|
|                 	| P@3        	| 39.8    	| 39.8                                      	| 40.3*   	| 40.4                                                            	|
|                 	| P@5        	| 36.0    	| 36.0                                      	| 36.5*   	| 36.6                                                            	|
|                 	| Train Time 	| 404s    	| 234s                                      	| 5,759s  	| 1,753s                                                          	|
|  WikiLSHTC-325K 	| P@1        	| 65.0    	| 64.8                                      	| 66.6*   	| 66.6                                                            	|
|                 	| P@3        	| 43.2    	| 43.1                                      	| 44.5*   	| 44.4                                                            	|
|                 	| P@5        	| 32.0    	| 32.1                                      	| 33.0*   	| 33.0                                                            	|
|                 	| Train Time 	| 959s    	| 659s                                      	| 11,156s 	| 4,259s                                                          	|

*\*Precision numbers as reported in the paper; our machine doesn't have enough memory to run the full prediction with their implementation.*

Note that since the C++ implementations train each tree single-threadedly, the number of CPU cores they can utilize is limited to the number of trees (3 by default). In contrast, our Rust implementation is able to utilize **all available CPU cores** whenever possible. On our quad-core machine, this resulted in a 1.3x to 1.7x speed up from Parabel, and a 2.6x to 4.6x speed up from Bonsai; **further speed-up is possible if more CPU cores are available**.

## Build & Install
Parabel-rs can be easily built & installed with [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) as a CLI app:
```
cargo install --git https://github.com/tomtung/parabel-rs.git --features cli
```

The CLI app will be available as `parabel`. For example, to reproduce the results on the EURLex-4K dataset:
```
parabel train eurlex_train.txt --model_path ./model
parabel test ./model eurlex_test.txt --out_path predictions.txt
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
model.save("./model")
model = parabel.Model.load("./model")
# Optionally densify model weights to trade off between prediction speed and memory usage
model.densify_weights(0.05)

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
    parabel train [FLAGS] [OPTIONS] <TRAINING_DATA_PATH>

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

        --linear.weight_threshold <THRESHOLD>
            Threshold for pruning weight vectors of linear classifiers [default: 0.1]

        --max_depth <DEPTH>                      Maximum tree depth [default: 20]
        --min_branch_size <SIZE>
            Number of labels below which no futher clustering & branching is done [default: 100]

        --model_path <PATH>                      Path of the directory where the trained model will be saved if provided
        --n_threads <T>
            Number of worker threads. If 0, the number is selected automatically [default: 0]

        --n_trees <N>                            Number of trees [default: 3]

ARGS:
    <TRAINING_DATA_PATH>    Path to training dataset file (in the format of the Extreme Classification Repository)
```

```
$ parabel test --help
parabel-test
Test an existing Parabel model

USAGE:
    parabel test [OPTIONS] <MODEL_PATH> <TEST_DATA_PATH>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
        --beam_size <beam_size>           Beam size for beam search [default: 10]
        --k_top <K>                       Number of top predictions to write out for each test example [default: 5]
        --max_sparse_density <DENSITY>    Density threshold above which sparse weight vectors are converted to dense
                                          format. Lower values speed up prediction at the cost of more memory usage
                                          [default: 0.1]
        --n_threads <T>                   Number of worker threads. If 0, the number is selected automatically [default:
                                          0]
        --out_path <PATH>                 Path to the which predictions will be written, if provided

ARGS:
    <MODEL_PATH>        Path of the directory where the trained model is saved
    <TEST_DATA_PATH>    Path to test dataset file (in the format of the Extreme Classification Repository)
```

### Data format

Our implementation takes dataset files formatted as those provided in the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). A data file starts with a header line with three space-separated integers: total number of examples, number of features, and number of labels. Following the header line, there is one line per each example, starting with comma-separated labels, followed by space-separated feature:value pairs:
```
label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
```

## License
Parabel-rs is licensed under the MIT License.

## References
- Y. Prabhu, A. Kag, S. Harsola, R. Agrawal, and M. Varma, “Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising,” in Proceedings of the 2018 World Wide Web Conference, 2018, pp. 993–1002. doi>[10.1145/3178876.3185998](https://doi.org/10.1145/3178876.3185998)
- S. Khandagale, H. Xiao, and R. Babbar, “Bonsai - Diverse and Shallow Trees for Extreme Multi-label Classification,” CoRR, vol. [abs/1904.08249](http://arxiv.org/abs/1904.08249), 2019.

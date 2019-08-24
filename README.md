# Omikuji
[![Build Status](https://dev.azure.com/yubingdong/omikuji/_apis/build/status/tomtung.omikuji?branchName=master)](https://dev.azure.com/yubingdong/omikuji/_build/latest?definitionId=1&branchName=master) [![Crate version](https://img.shields.io/crates/v/omikuji)](https://crates.io/crates/omikuji) [![PyPI version](https://img.shields.io/pypi/v/omikuji)](https://pypi.org/project/omikuji/)

An efficient implementation of Partitioned Label Trees (Prabhu et al., 2018) and its variations for extreme multi-label classification, written in Rust🦀 with love💖.

## Features & Performance

Omikuji has has been tested on datasets from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). All tests below are run on a quad-core Intel® Core™ i7-6700 CPU, and we allowed as many cores to be utilized as possible. We measured training time, and calculated precisions at 1, 3, and 5. (Note that, due to randomness, results might vary from run to run, especially for smaller datasets.)

### Parabel, better parallelized

Omikuji provides a more parallelized implementation of Parabel (Prabhu et al., 2018) that trains faster when more CPU cores are available. Compared to the [original implementation](http://manikvarma.org/code/Parabel/download.html) written in C++, which can only utilize the same number of CPU cores as the number of trees (3 by default), Omikuji maintains the same level of precision but trains 1.3x to 1.7x faster on our quad-core machine. **Further speed-up is possible if more CPU cores are available**.

| Dataset         	| Metric     	| Parabel 	| Omikuji<br/>(balanced,<br/>cluster.k=2) 	|
|-----------------	|------------	|---------	|------------------------------------------	|
|  EURLex-4K      	| P@1        	| 82.2    	| 82.1                                     	|
|                 	| P@3        	| 68.8    	| 68.8                                     	|
|                 	| P@5        	| 57.6    	| 57.7                                     	|
|                 	| Train Time 	| 18s     	| 14s                                      	|
| Amazon-670K     	| P@1        	| 44.9    	| 44.8                                     	|
|                 	| P@3        	| 39.8    	| 39.8                                     	|
|                 	| P@5        	| 36.0    	| 36.0                                     	|
|                 	| Train Time 	| 404s    	| 234s                                     	|
|  WikiLSHTC-325K 	| P@1        	| 65.0    	| 64.8                                     	|
|                 	| P@3        	| 43.2    	| 43.1                                     	|
|                 	| P@5        	| 32.0    	| 32.1                                     	|
|                 	| Train Time 	| 959s    	| 659s                                     	|

### Regular k-means for shallow trees

Following Bonsai (Khandagale et al., 2019), Omikuji supports using regular k-means instead of balanced 2-means clustering for tree construction, which results in wider, shallower and unbalanced trees that train slower but have better precision. Comparing to the [original Bonsai implementation](https://github.com/xmc-aalto/bonsai), Omikuji also achieves the same precisions while training 2.6x to 4.6x faster on our quad-core machine. (Similarly, further speed-up is possible if more CPU cores are available.)

| Dataset         	| Metric     	| Bonsai  	| Omikuji<br/>(unbalanced,<br/>cluster.k=100,<br/>max\_depth=3)	|
|-----------------	|------------	|---------	|--------------------------------------------------------------	|
|  EURLex-4K      	| P@1        	| 82.8    	| 83.0                                                         	|
|                 	| P@3        	| 69.4    	| 69.5                                                         	|
|                 	| P@5        	| 58.1    	| 58.3                                                         	|
|                 	| Train Time 	| 87s     	| 19s                                                          	|
| Amazon-670K     	| P@1        	| 45.5*   	| 45.6                                                         	|
|                 	| P@3        	| 40.3*   	| 40.4                                                         	|
|                 	| P@5        	| 36.5*   	| 36.6                                                         	|
|                 	| Train Time 	| 5,759s  	| 1,753s                                                       	|
|  WikiLSHTC-325K 	| P@1        	| 66.6*   	| 66.6                                                         	|
|                 	| P@3        	| 44.5*   	| 44.4                                                         	|
|                 	| P@5        	| 33.0*   	| 33.0                                                         	|
|                 	| Train Time 	| 11,156s 	| 4,259s                                                       	|

*\*Precision numbers as reported in the paper; our machine doesn't have enough memory to run the full prediction with their implementation.*

### Balanced k-means for balanced shallow trees

Sometimes it's desirable to have shallow and wide trees that are also balanced, in which case Omikuji supports the balanced k-means algorithm used by HOMER (Tsoumakas et al., 2008) for clustering as well.

| Dataset         	| Metric     	| Omikuji<br/>(balanced,<br/>cluster.k=100)	|
|-----------------	|------------	|------------------------------------------	|
|  EURLex-4K      	| P@1        	| 82.1                                    	|
|                 	| P@3        	| 69.4                                    	|
|                 	| P@5        	| 58.1                                    	|
|                 	| Train Time 	| 19s                                     	|
| Amazon-670K     	| P@1        	| 45.4                                    	|
|                 	| P@3        	| 40.3                                    	|
|                 	| P@5        	| 36.5                                    	|
|                 	| Train Time 	| 1,153s                                  	|
|  WikiLSHTC-325K 	| P@1        	| 65.6                                    	|
|                 	| P@3        	| 43.6                                    	|
|                 	| P@5        	| 32.5                                    	|
|                 	| Train Time 	| 3,028s                                  	|

### Layer collapsing for balanced shallow trees

An alternative way for building balanced, shallow and wide trees is to collapse adjacent layers, similar to the tree compression step used in AttentionXML (You et al., 2019): intermediate layers are removed, and their children replace them as the children of their parents. For example, with balanced 2-means clustering, if we collapse 5 layers after each layer, we can increase the tree arity from 2 to 2⁵⁺¹ = 64.

| Dataset         	| Metric     	| Omikuji<br/>(balanced,<br/>cluster.k=2,<br/>collapse 5 layers)	|
|-----------------	|------------	|---------------------------------------------------------------	|
|  EURLex-4K      	| P@1        	| 82.4                                                          	|
|                 	| P@3        	| 69.3                                                          	|
|                 	| P@5        	| 58.0                                                          	|
|                 	| Train Time 	| 16s                                                           	|
| Amazon-670K     	| P@1        	| 45.3                                                          	|
|                 	| P@3        	| 40.2                                                          	|
|                 	| P@5        	| 36.4                                                          	|
|                 	| Train Time 	| 460s                                                           	|
|  WikiLSHTC-325K 	| P@1        	| 64.9                                                           	|
|                 	| P@3        	| 43.3                                                          	|
|                 	| P@5        	| 32.3                                                          	|
|                 	| Train Time 	| 1,649s                                                        	|

## Build & Install
Omikuji can be easily built & installed with [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) as a CLI app:
```
cargo install omikuji --features cli
```

Or install from the latest source:
```
cargo install --git https://github.com/tomtung/omikuji.git --features cli
```

The CLI app will be available as `omikuji`. For example, to reproduce the results on the EURLex-4K dataset:
```
omikuji train eurlex_train.txt --model_path ./model
omikuji test ./model eurlex_test.txt --out_path predictions.txt
```


### Python Binding

A simple Python binding is also available for training and prediction. It can be install via `pip`:
```
pip install omikuji
```

Note that you might still need to install Cargo should compilation become necessary.

You can also install from the latest source:
```
pip install git+https://github.com/tomtung/omikuji.git -v
```

The following script demonstrates how to use the Python binding to train a model and make predictions:

```python
import omikuji

# Train
hyper_param = omikuji.Model.default_hyper_param()
# Adjust hyper-parameters as needed
hyper_param.n_trees = 5
model = omikuji.Model.train_on_data("./eurlex_train.txt", hyper_param)

# Serialize & de-serialize
model.save("./model")
model = omikuji.Model.load("./model")
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
$ omikuji train --help
omikuji-train
Train a new model

USAGE:
    omikuji train [FLAGS] [OPTIONS] <TRAINING_DATA_PATH>

FLAGS:
        --cluster.unbalanced     Perform regular k-means clustering instead of balanced k-means clustering
    -h, --help                   Prints help information
        --tree_structure_only    Build the trees without training classifiers; useful when a downstream user needs the
                                 tree structures only
    -V, --version                Prints version information

OPTIONS:
        --centroid_threshold <THRESHOLD>         Threshold for pruning label centroid vectors [default: 0]
        --cluster.eps <EPS>                      Epsilon value for determining clustering convergence [default: 0.0001]
        --cluster.k <K>                          Number of clusters [default: 2]
        --cluster.min_size <SIZE>
            Labels in clusters with sizes smaller than this threshold are reassigned to other clusters instead [default:
            2]
        --collapse_every_n_layers <N>
            Number of adjacent layers to collapse, which increases tree arity and decreases tree depth [default: 0]

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
            Number of labels below which no further clustering & branching is done [default: 100]

        --model_path <PATH>
            Optional path of the directory where the trained model will be saved if provided; if an model with
            compatible settings is already saved in the given directory, the newly trained trees will be added to the
            existing model
        --n_threads <T>
            Number of worker threads. If 0, the number is selected automatically [default: 0]

        --n_trees <N>                            Number of trees [default: 3]

ARGS:
    <TRAINING_DATA_PATH>    Path to training dataset file (in the format of the Extreme Classification Repository)
```

```
$ omikuji test --help
omikuji-test
Test an existing model

USAGE:
    omikuji test [OPTIONS] <MODEL_PATH> <TEST_DATA_PATH>

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

## Trivia

The project name comes from [o-mikuji](https://en.wikipedia.org/wiki/O-mikuji) (御神籤), which are predictions about one's future written on strips of paper (labels?) at jinjas and temples in Japan, often tied to branches of pine trees after they are read.

## References
- Y. Prabhu, A. Kag, S. Harsola, R. Agrawal, and M. Varma, “Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising,” in Proceedings of the 2018 World Wide Web Conference, 2018, pp. 993–1002.
- S. Khandagale, H. Xiao, and R. Babbar, “Bonsai - Diverse and Shallow Trees for Extreme Multi-label Classification,” Apr. 2019.
- G. Tsoumakas, I. Katakis, and I. Vlahavas, “Effective and efficient multilabel classification in domains with large number of labels,” ECML, 2008.
- R. You, S. Dai, Z. Zhang, H. Mamitsuka, and S. Zhu, “AttentionXML: Extreme Multi-Label Text Classification with Multi-Label Attention Based Recurrent Neural Networks,” Jun. 2019.

## License
Omikuji is licensed under the MIT License.

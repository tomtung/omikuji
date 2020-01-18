#!/usr/bin/env bash

set -e -x

rm -rf ./model predictions.txt

# Train with some optional parameters
omikuji train --model_path ./model --cluster.k 5 --collapse_every_n_layers --n_trees 2 eurlex_train.txt

ls -lh ./model/

# Test
omikuji test ./model eurlex_test.txt --out_path predictions.txt

head predictions.txt

# Clean up
rm -rf ./model predictions.txt

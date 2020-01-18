import os
import sys
import shutil
import time

import omikuji

if __name__ == "__main__":
    # Adjust hyper-parameters as needed
    hyper_param = omikuji.Model.default_hyper_param()
    hyper_param.n_trees = 2

    # Train
    model = omikuji.Model.train_on_data("./eurlex_train.txt", hyper_param)

    # Serialize & de-serialize
    model.save("./model")
    model = omikuji.Model.load("./model")

    shutil.rmtree("./model", ignore_errors=True)

    # Optionally densify model weights to trade off between
    # prediction speed and memory usage
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
    label_score_pairs = model.predict(feature_value_pairs, top_k=3)
    print("Dummy prediction results: {}".format(label_score_pairs))

    # Also works in forked sub-processes (Unix only)
    # It's important that this works, because the package could be used (intentionally
    # or unintentionally) with multiprocessing, or in a web service that forks worker
    # child processes.
    try:
        pid = os.fork()
    except AttributeError:
        print("OS doesn't support fork, skipping")
    else:
        if pid == 0:
            label_score_pairs = model.predict(feature_value_pairs, top_k=3)
            print("From child process: {}".format(label_score_pairs))
            sys.exit(0)
        else:
            time.sleep(10)  # Should be enough time for child to finish
            child_pid, status = os.waitpid(pid, os.WNOHANG)
            if child_pid == 0:
                print("Child process has hung")
                sys.exit(1)

            sys.exit(status >> 8)

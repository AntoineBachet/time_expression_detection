# Import Libraries
from collections import OrderedDict
import torch
import tensorboardX
import dill as pickle
import pandas as pd
from itertools import product

import sys
sys.path.append("sopa_master/")

from data import read_embeddings, read_docs, read_labels
from soft_patterns import ProbSemiring, MaxPlusSemiring, LogSpaceMaxTimesSemiring, SoftPatternClassifier, train, Batch, evaluate_accuracy
from util import to_cuda
from interpret_classification_results import interpret_documents
from visualize import visualize_patterns
from baselines.cnn import PooledCnnClassifier

train_data_file = "data/time_data_clean/train.data"
train_label_file = "data/time_data_clean/train.labels"
dev_data_file = "data/time_data_clean/dev.data"
dev_label_file = "data/time_data_clean/dev.labels"
test_file = "data/time_data_clean/test.data"
test_label = "data/time_data_clean/test.labels"

print("Loading Embeddings")
vocab = pickle.load(open("data/embeddings/vocab.p", "rb"))
embeddings = pickle.load(open("data/embeddings/embeddings.p", "rb"))
word_dim = pickle.load(open("data/embeddings/word_dim.p", "rb"))

train_input, train_text = read_docs(train_data_file, vocab, num_padding_tokens=3)
train_labels = read_labels(train_label_file)
dev_input, dev_text = read_docs(dev_data_file, vocab, num_padding_tokens=3)
dev_labels = read_labels(dev_label_file)
train_data = list(zip(train_input, train_labels))
dev_data = list(zip(dev_input, dev_labels))
test_input, test_text = read_docs(test_file, vocab, num_padding_tokens=3)
labels = read_labels(test_label)
test_data = list(zip(test_input, labels))

# HyperParameters to change
window_sizes = [4, 5, 6]
cnn_hidden_dims = [50, 100, 200]
mlp_hidden_dims = [10, 25, 50, 100, 300]
num_mlp_layers = [2, 5]
dropouts = [0, 0.05, 0.1, 0.2]
learning_rates = [0.01, 0.05, 0.001, 0.005]
params = list(product(*[window_sizes, cnn_hidden_dims, mlp_hidden_dims, num_mlp_layers, dropouts, learning_rates]))

results = []
i = 0
for p in params:
    i+=1
    param = {
        "w_size": p[0],
        "cnn_h_dim": p[1],
        "mlp_h_dim": p[2],
        "mlp_n": p[3],
        "dropout": p[4],
        "lr": p[5]
    }
    model = PooledCnnClassifier(
        window_size=param["w_size"],
        num_cnn_layers=1,
        cnn_hidden_dim=param["cnn_h_dim"],
        mlp_hidden_dim=param["mlp_h_dim"],
        num_mlp_layers=param["mlp_n"],
        num_classes=2,
        embeddings=embeddings
    )
    train(
        train_data=train_data,
        dev_data=dev_data,
        model=model,
        model_save_dir=f"data/models/gridsearch/cnn/modeltimecnn_{i}",
        num_iterations=250,
        model_file_prefix="traintimecnn",
        learning_rate=param["lr"],
        batch_size=150,
        num_classes=2,
        patience=30,
        gpu=False,
        dropout=param["dropout"]
    )
    acc = evaluate_accuracy(model, test_data, batch_size=150, gpu=False)
    results.append((param, acc))
    i += 1
    pickle.dump(results, open("data/results_GS_cnn.p", "wb"))
    print(param)
    print(f"acc: {acc}")
data = pd.DataFrame(results, columns=["params", "dev_acc"])
pickle.dump(data, open("data/results_GS_cnn.p", "wb"))

# Import Libraries
from collections import OrderedDict
import torch
import pandas as pd
import dill as pickle

import sys
sys.path.append("sopa_master/")

from data import read_embeddings, read_docs, read_labels
from soft_patterns import LogSpaceMaxTimesSemiring, SoftPatternClassifier, train, Batch, evaluate_accuracy
from util import to_cuda
from interpret_classification_results import interpret_documents
from visualize import visualize_patterns

train_data_file = "data/time_data_clean/train.data"
train_label_file ="data/time_data_clean/train.labels"
dev_data_file = "data/time_data_clean/dev.data"
dev_label_file = "data/time_data_clean/dev.labels"
test_file = "data/time_data_clean/test.data"
test_label="data/time_data_clean/test.labels"

print("Loading Embedding")
vocab = pickle.load(open("vocab.p","rb"))
embeddings = pickle.load(open("embeddings.p","rb"))
word_dim = pickle.load(open("word_dim.p","rb"))
print("Embedding Loaded")

train_input, train_text = read_docs(train_data_file, vocab, num_padding_tokens=0)
train_labels = read_labels(train_label_file)
dev_input, dev_text = read_docs(dev_data_file, vocab, num_padding_tokens=0)
dev_labels = read_labels(dev_label_file)

train_data = list(zip(train_input, train_labels))
dev_data = list(zip(dev_input, dev_labels))

pattern1 = "5-10_4-10_3-10_2-10"
pattern2 = "6-10_5-10_4-10"
pattern3 = "6-10_5-10_4-10_3-10_2-10"
pattern4 = "6-20_5-20_4-10_3-10_2-10"
pattern5 = "7-10_6-10_5-10_4-10_3-10_2-10"	



pattern_specs = []
patterns = [pattern1,pattern2,pattern3, pattern4, pattern5]
for pattern in patterns:
    pattern_specs.append(OrderedDict(sorted(([int(y) for y in x.split("-")] for x in pattern.split("_")),
                                key=lambda t: t[0])))

learning_rates = [0.01,0.05,0.001,0.005]
dropouts = [0,0.05,0.1,0.2]
mlp_hidden_dims = [10,25,50,100,300]
num_mlp_layers = [2,5]

params = list(product(*[pattern_specs, learning_rates, dropouts, mlp_hidden_dims, num_mlp_layers]))

print("Launching Grid Search")
results = []
i=0
for p in params:
    i+=1
    param = {
        "pattern": p[0],
        "lr": p[1],
        "dropout": p[2],
        "mlp_hd": p[3],
        "num_mlp_l": p[4]
    }
    caracteristic = f"{p[0]}_{p[1]}_{p[2]}_{p[3]}_{p[4]}"
    model = SoftPatternClassifier(
        pattern_specs=param["pattern"],
        mlp_hidden_dim=param["mlp_hd"],
        num_mlp_layers=param["num_mlp_l"],
        num_classes=2,
        embeddings=embeddings,
        vocab=vocab,
        semiring=LogSpaceMaxTimesSemiring,
        bias_scale_param=1,
        shared_sl=False,
        no_sl=False
    )
    train(
    train_data=train_data,
    dev_data=dev_data,
    model=model,
    model_save_dir=f"models/gridsearch/sopa/sopa_{i}",
    num_iterations=250,
    model_file_prefix="gs_sopa_",
    learning_rate=param["lr"],
    batch_size=150,
    num_classes=2,
    patience=30,
    dropout=param["dropout"]
    ) 

    torch.save(model.state_dict(), f"models/gridsearch/sopa/model_{i}.pth")
    acc = evaluate_accuracy(model, dev_data, batch_size=150, gpu=False)
    results.append((param, acc))
    pickle.dump(
        pd.DataFrame(results, columns=["params", "dev_acc"]),
        open("results_GS_sopa", "wb")
    )
    print(param, acc)

data=pd.DataFrame(results,columns=["params", "dev_acc"])
pickle.dump(data, open("results_GS_sopa.p", "wb"))

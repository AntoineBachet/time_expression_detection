from collections import OrderedDict
import torch

import sys
sys.path.append("sopa_master/")

import tensorboardX
from data import read_embeddings, read_docs, read_labels

from soft_patterns import LogSpaceMaxTimesSemiring, SoftPatternClassifier, train, Batch, evaluate_accuracy
from util import to_cuda
from interpret_classification_results import interpret_documents
from visualize import visualize_patterns

train_data_file = "data_time/train.data"
train_label_file ="data_time/train.labels"
dev_data_file = "data_time/dev.data"
dev_label_file = "data_time/dev.labels"
test_file = "data_time/test.data"
test_label="data_time/test.labels"

import dill as pickle
vocab = pickle.load(open("vocab.p","rb"))
embeddings = pickle.load(open("embeddings.p","rb"))
word_dim = pickle.load(open("word_dim.p","rb"))

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

results = []
i = 0
j=0
k=0
for pattern_spec in pattern_specs:
    i += 1
    for learning_rate in learning_rates:
        j+=1
        for dropout in dropouts:
            k+=1
            for mlp_hidden_dim in mlp_hidden_dims:
                for num_mlp_layer in num_mlp_layers:
                    caracteristic = f"{i}_{j}_{k}_{mlp_hidden_dim}_{num_mlp_layer}"
                    model = SoftPatternClassifier(pattern_specs=pattern_spec,mlp_hidden_dim=mlp_hidden_dim,
                                                  num_mlp_layers=num_mlp_layer,num_classes=2,embeddings=embeddings,
                                                  vocab=vocab,semiring=LogSpaceMaxTimesSemiring, bias_scale_param=1,
                                                  shared_sl=False,no_sl=False)
                    train(train_data=train_data,dev_data=dev_data,model=model,model_save_dir="model_sst/"+caracteristic,
                          num_iterations=250,model_file_prefix="trainMaxTimes_timedata",learning_rate=learning_rate,
                          batch_size=150,num_classes=2,patience=30,dropout=dropout)                    
                    torch.save(model.state_dict(), "modelgridsearch/" + caracteristic + ".pth")
                    acc = evaluate_accuracy(model, dev_data, batch_size=150, gpu=False)
                    results.append((i, learning_rate, dropout, mlp_hidden_dim, num_mlp_layer, acc))
                    print(">>>>>",i, learning_rate, dropout, mlp_hidden_dim, num_mlp_layer, acc,"<<<<<<<<<")
data=pd.DataFrame(results,columns=["pattern", "learning_rate", "dropout", "mlp_hidden_dim", "num_mlp_layer", "dev_acc"])
pickle.dump(data, open("results_GS.p", "wb"))
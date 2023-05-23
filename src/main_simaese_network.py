import os
import pickle
import time
import copy
import numpy as np

import dataset.loader as loader
from embedding.embedding import get_embedding
from tools.tool import parse_args, print_args, set_seed
from train.train import *
from train.test import *
#from train.train import *

import sys




def main():
    args = parse_args()

    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, class_names, vocab = loader.load_dataset(args)

    args.id2word = vocab.itos

    # initialize model
    model = {}
    model["G"] = get_embedding(vocab, args)
    print("-------------------------------------param----------------------------------------------")
    sum = 0
    for name, param in model["G"].named_parameters():
        num = 1
        for size in param.shape:
            num *= size
        sum += num
        print("{:30s} : {}".format(name, param.shape))
    print("total param num {}".format(sum))
    print("-------------------------------------param----------------------------------------------")

    criterion = ContrastiveLoss()


    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        optG = train(train_data, val_data, model, class_names, criterion, args)


    test_acc, test_std = test(test_data, class_names, optG, model, criterion, args, args.test_epochs, False)
    print(("[TEST] {}, {:s} {:s}{:>7.4f} ± {:>6.4f}, "
           ).format(
        datetime.datetime.now(),
        colored("test  ", "cyan"),
        colored("acc:", "blue"), test_acc, test_std,
    ), flush=True)


    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdirs(directory)

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
        }

        for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
    

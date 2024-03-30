# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2024/03/24 18:41:00
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   None
"""

# here put the import lib
import warnings

warnings.filterwarnings("ignore")
import os
import argparse
import warnings
import tensorflow as tf
from utils import (
    load_data,
    load_model,
    get_metrics,
    hyperpara_selection,
    visaul4curves,
    visual4cm,
    visual4tree,
)

warnings.filterwarnings("ignore")

"""
    This is the part for CPU and GPU setting. Notice that part of the project 
    code is run on UCL server with provided GPU resources, especially for NNs 
    and pretrained models.
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# export CUDA_VISIBLE_DEVICES=1  # used for setting specific GPU in terminal
if tf.config.list_physical_devices("GPU"):
    print("Use GPU of UCL server: london.ee.ucl.ac.uk")
    physical_devices = tf.config.list_physical_devices("GPU")
    print(physical_devices)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Use CPU of your PC.")

if __name__ == "__main__":
    """
    Notice that you can specify certain task and model for experiment by passing in
    arguments. Guidelines for running are provided in README.md and Github link.
    """
    # argument processing
    parser = argparse.ArgumentParser(description="Argparse")
    parser.add_argument("--task", type=str, default="speech", help="task A or B")
    parser.add_argument("--method", type=str, default="SVM", help="model chosen")
    parser.add_argument("--dataset", type=str, default="TESS", help="model chosen")
    parser.add_argument(
        "--features", type=str, default="mfcc", help="mfcc, chroma, all, mel"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of NNs like MLP and CNN"
    )
    parser.add_argument("--epochs", type=int, default=10, help="epochs of NNs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of NNs")
    parser.add_argument("--n_mfcc", type=int, default=40, help="epochs of NNs")
    parser.add_argument("--n_mels", type=int, default=128, help="epochs of NNs")
    parser.add_argument("--max_length", type=int, default=160, help="epochs of NNs")
    parser.add_argument(
        "--reverse", type=bool, default=False, help="play the audio in a reverse way"
    )
    parser.add_argument(
        "--noise", type=bool, default=False, help="play the audio in a reverse way"
    )
    parser.add_argument(
        "--denoise", type=bool, default=False, help="play the audio in a reverse way"
    )
    parser.add_argument("--window", nargs="+", type=int, help="An array of integers")
    # python script.py --integers 1 2 3 4 5
    parser.add_argument(
        "--bidirectional",
        type=bool,
        default=False,
        help="whether download and preprocess the dataset",
    )

    parser.add_argument(
        "--cc",
        type=str,
        default="single",
        help="single, cross, finetune",
    )
    parser.add_argument(
        "--scaled",
        type=str,
        default=None,
        help="standard, minmax",
    )
    args = parser.parse_args()
    task = args.task
    method = args.method
    dataset = args.dataset
    features = args.features
    cc = args.cc

    print(
        f"Task: {task} emotion classification, Method: {method}, Cross-corpus: {args.cc}, Dataset: {dataset}, Features: {args.features}. "
    )

    # data processing
    # if pre_data:
    #     data_preprocess4A(raw_path) if task == "A" else data_preprocess4B(raw_path)
    # else:
    #     load_data_log4A(download=False) if task == "A" else load_data_log4B(
    #         download=False
    #     )

    # load data
    print("Start loading data......")
    if task == "speech":
        if method in ["SVM", "DT", "RF", "NB", "KNN", "LSTM", "GMM"]:
            Xtrain, ytrain, Xtest, ytest, Xval, yval, shape, num_classes = load_data(
                task,
                method,
                dataset,
                features,
                args.n_mfcc,
                args.n_mels,
                scaled=args.scaled,
                reverse=args.reverse,
                noise=args.noise,
                denoise=args.denoise,
                window=args.window,
            )
        elif method in ["MLP", "RNN"]:
            train_ds, val_ds, test_ds, shape, num_classes = load_data(
                task,
                method,
                dataset,
                features,
                args.n_mfcc,
                args.n_mels,
                scaled=args.scaled,
                batch_size=args.batch_size,
                reverse=args.reverse,
                noise=args.noise,
                denoise=args.denoise,
                window=args.window,
            )
        elif method in ["CNN", "AlexNet"]:
            Xtrain, ytrain, Xtest, ytest, Xval, yval, shape, num_classes = load_data(
                task,
                method,
                dataset,
                features,
                args.n_mfcc,
                args.n_mels,
                scaled=args.scaled,
                max_length=args.max_length,
                reverse=args.reverse,
                noise=args.noise,
                denoise=args.denoise,
                window=args.window,
            )
        elif method == "wave2vec":
            Xtrain, ytrain, Xtest, ytest, Xval, yval, num_classes, length = load_data(
                task,
                method,
                dataset,
                features,
                args.n_mfcc,
                args.n_mels,
                scaled=args.scaled,
                max_length=args.max_length,
                reverse=args.reverse,
                noise=args.noise,
                denoise=args.denoise,
                window=args.window,
            )
    # elif method in ["CNN", "MLP", "EnsembleNet"]:
    #     train_ds, val_ds, test_ds = load_data(
    #         task, pre_path, method, batch_size=args.batch_size
    #     )
    print("Load data successfully.")

    # model selection
    print("Start loading model......")
    if method in ["SVM", "DT", "RF", "NB", "KNN"]:
        model = load_model(task, method)
    elif method == "MLP":
        model = load_model(
            task,
            method,
            features,
            cc,
            shape,
            num_classes,
            dataset,
            args.epochs,
            args.lr,
        )
    elif method in ["RNN", "LSTM"]:
        model = load_model(
            task,
            method,
            features,
            cc,
            shape,
            num_classes,
            dataset,
            bidirectional=args.bidirectional,
            epochs=args.epochs,
            lr=args.lr,
        )
    elif method in ["CNN", "AlexNet"]:
        model = load_model(
            task,
            method,
            features,
            cc,
            shape,
            num_classes,
            dataset,
            max_length=args.max_length,
            bidirectional=args.bidirectional,
            epochs=args.epochs,
            lr=args.lr,
        )
    elif method == "GMM":
        model = load_model(task, method, features, cc, dataset, num_classes=num_classes)
    elif method == "wave2vec":
        model = load_model(
            task,
            method,
            features,
            cc,
            num_classes,
            dataset,
            length,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
    print("Load model successfully.")

    """
        This part includes all training, validation and testing process with encapsulated functions.
        Detailed process of each method can be seen in corresponding classes.
    """
    if method in ["SVM", "DT", "RF", "NB", "KNN"]:
        if method in ["KNN", "DT", "RF"]:
            cv_results_ = model.train(Xtrain, ytrain, Xval, yval, gridSearch=True)
        else:
            model.train(Xtrain, ytrain, Xval, yval)
        pred_train, pred_val, pred_test = model.test(Xtrain, ytrain, Xval, yval, Xtest)
    elif method in ["MLP", "RNN"]:
        train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(
            model, train_ds, val_ds
        )
        test_res, pred_test, ytest = model.test(model, test_ds)
    elif method in ["LSTM", "CNN", "AlexNet", "wave2vec"]:
        train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(
            Xtrain, ytrain, Xval, yval
        )
        ytest, pred_test = model.test(Xtest, ytest)
    elif method == "GMM":
        pred_train, pred_val, ytrain, yval = model.train(Xtrain, ytrain, Xval, yval)
        pred_test, ytest = model.test(Xtest, ytest)

    # metrics and visualization
    # hyperparameters selection
    if method in ["KNN", "DT", "RF"]:
        hyperpara_selection(
            task, method, features, cc, dataset, cv_results_["mean_test_score"]
        )

    # decision tree
    if "DT" in method:
        (visual4tree(task, method, features, cc, dataset, model.model))

    # confusion matrix, auc roc curve, metrics calculation
    if method != "GMM":
        res = {
            "train_res": get_metrics(task, ytrain, pred_train),
            "val_res": get_metrics(task, yval, pred_val),
            "test_res": get_metrics(task, ytest, pred_test),
        }
        for i in res.items():
            print(i)
        visual4cm(
            task,
            method,
            features,
            cc,
            dataset,
            ytrain,
            yval,
            ytest,
            pred_train,
            pred_val,
            pred_test,
        )
    if method == "LSTM":
        visaul4curves(
            task, method, features, cc, dataset, train_res, val_res, args.epochs
        )

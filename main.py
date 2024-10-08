# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2024/07/24 06:33:52
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :  This file is the main file of the whole program.
"""

# here put the import lib

import os
import argparse
import warnings
import warnings
import tensorflow as tf
from utils import (
    get_imbalance,
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
    code is run on UCL server with provided GPU resources.
"""
# print(os.environ["CUDA_HOME"])
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# export CUDA_VISIBLE_DEVICES=1  # used for setting specific GPU in terminal
if tf.config.list_physical_devices("GPU"):
    print("Use GPU of UCL server: turin.ee.ucl.ac.uk")
    physical_devices = tf.config.list_physical_devices("GPU")
    print(physical_devices)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Use CPU of your PC.")

if __name__ == "__main__":
    """
    Notice that you can specify certain task and method for experiment by passing in
    arguments. Guidelines for running are provided in README.md and Github link.
    """
    # argument processing
    parser = argparse.ArgumentParser(description="Argparse")
    parser.add_argument("--task", type=str, default="speech", help="image or speech")
    parser.add_argument("--method", type=str, default="CNN", help="model chosen")
    parser.add_argument("--dataset", type=str, default="TESS", help="dataset chosen")
    parser.add_argument(
        "--features",
        type=str,
        default="mfcc",
        help="mfcc, all, mel, chroma",  # chroma will have influence on AlexNet for pooling
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of NNs like MLP and CNN"
    )
    parser.add_argument("--epochs", type=int, default=2, help="epochs of NNs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of NNs")
    parser.add_argument(
        "--n_mfcc", type=int, default=40, help="number of mfcc features"
    )
    parser.add_argument(
        "--n_mels", type=int, default=128, help="number of mel features"
    )
    parser.add_argument("--sr", type=int, default=16000, help="sampling rate")
    parser.add_argument(
        "--max_length", type=int, default=150, help="max length for padding in CNN"
    )
    parser.add_argument(
        "--reverse", type=bool, default=False, help="play the audio in a reverse way"
    )
    parser.add_argument(
        "--noise",
        type=str,
        default=None,
        help="play the audio with white noise, white/buzz/bubble/cocktail effect",
    )
    parser.add_argument(
        "--denoise", type=bool, default=False, help="play the audio by denoising"
    )
    parser.add_argument(
        "--landmark",
        type=str,
        default=None,
        help="number of landmark for image emotion detection, 5/68/xyz(468)",
    )
    parser.add_argument(
        "--window",
        nargs="+",
        type=int,
        help="An array of integers as starting and ending point of window",
    )
    # python script.py --integers 1 2 3 4 5
    parser.add_argument(
        "--bidirectional",
        type=bool,
        default=False,
        help="whether construct bidirectional for RNN",
    )

    parser.add_argument(
        "--cv",
        type=bool,
        default=False,
        help="whether cross validation",
    )

    parser.add_argument(
        "--cc",
        type=str,
        default="single",
        help="single, cross, mix, finetune",
    )
    parser.add_argument(
        "--scaled",
        type=str,
        default=None,
        help="standard, minmax",
    )
    parser.add_argument(
        "--corpus",
        nargs="+",
        type=str,
        default=None,
        help="An array of string indicating mixture of corpus in cross-corpus setting",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        help="split of size for cross-corpus setting",
    )  # 0.2, 0.4, 0.5, 0.6, 0.8 (cross-corpus)  # 4, 3, 2.5, 2, 1 (single-corpus)
    parser.add_argument(
        "--process", type=str, default=None, help="blur/noise/filter/sobel/equal/assi"
    )
    args = parser.parse_args()
    task = args.task
    method = args.method
    dataset = args.dataset
    features = args.features
    cc = args.cc
    corpus = args.corpus
    sr = args.sr
    split = args.split
    process = args.process

    if task == "speech":
        print(
            f"Task: {task} emotion classification, Method: {method}, Cross-corpus: {args.cc}, Dataset: {dataset}, Features: {args.features}. "
        )
    elif task == "image":
        print(
            f"Task: {task} emotion classification, Method: {method}, Cross-corpus: {args.cc}, Dataset: {dataset}, Landmark: {args.landmark}. "
        )

    # load data
    print("Start loading data......")
    if task == "speech":  # load speech data
        if method in ["SVM", "DT", "RF", "NB", "KNN", "LSTM", "GMM"]:
            if cc != "finetune":
                Xtrain, ytrain, Xval, yval, Xtest, ytest, shape, num_classes = (
                    load_data(
                        task,
                        method,
                        cc,
                        dataset,
                        features,
                        args.n_mfcc,
                        args.n_mels,
                        scaled=args.scaled,
                        reverse=args.reverse,
                        noise=args.noise,
                        denoise=args.denoise,
                        window=args.window,
                        corpus=corpus,
                        sr=sr,
                        split=split,
                    )
                )
            else:  # finetuning
                (
                    Xtrain,
                    ytrain,
                    Xtest,
                    ytest,
                    Xval,
                    yval,
                    shape,
                    num_classes,
                    Xtune_train,
                    ytune_train,
                    Xtune_val,
                    ytune_val,
                ) = load_data(
                    task,
                    method,
                    cc,
                    dataset,
                    features,
                    args.n_mfcc,
                    args.n_mels,
                    scaled=args.scaled,
                    reverse=args.reverse,
                    noise=args.noise,
                    denoise=args.denoise,
                    window=args.window,
                    corpus=corpus,
                    sr=sr,
                )
        elif method in ["MLP", "RNN"]:
            train_ds, val_ds, test_ds, shape, num_classes = load_data(
                task,
                method,
                cc,
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
                corpus=corpus,
                sr=sr,
                split=split,
            )
        elif method in ["CNN", "AlexNet"]:
            if cc != "finetune":
                Xtrain, ytrain, Xval, yval, Xtest, ytest, shape, num_classes = (
                    load_data(
                        task,
                        method,
                        cc,
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
                        corpus=corpus,
                        sr=sr,
                        split=split,
                    )
                )
            else:  # finetuning
                (
                    Xtrain,
                    ytrain,
                    Xtest,
                    ytest,
                    Xval,
                    yval,
                    shape,
                    num_classes,
                    Xtune_train,
                    ytune_train,
                    Xtune_val,
                    ytune_val,
                ) = load_data(
                    task,
                    method,
                    cc,
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
                    corpus=corpus,
                    sr=sr,
                )
        elif method == "wav2vec":
            train_ds, val_ds, test_ds, num_classes, length = load_data(
                task,
                method,
                cc,
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
                corpus=corpus,
                sr=sr,
            )
    elif task == "image":  # load image data
        if method in ["CNN", "Xception", "MLP"]:
            Xtrain, ytrain, Xval, yval, Xtest, ytest, h, num_classes = load_data(
                task,
                method,
                cc,
                dataset,
                None,
                batch_size=16,
                landmark=args.landmark,
                split=split,
                corpus=corpus,
                process=process,
            )
        elif method in ["ViT"]:
            Xtrain, ytrain, Xval, yval, Xtest, ytest, num_classes = load_data(
                task,
                method,
                cc,
                dataset,
                None,
                batch_size=16,
                landmark=None,
            )
    print("Load data successfully.")

    # model selection
    print("Start loading model......")
    if task == "speech":  # speech emotion detection
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
                epochs=args.epochs,
                lr=args.lr,
            )
        elif method in ["RNN", "LSTM"]:
            cv = False if method == "RNN" else args.cv
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
                cv=cv,
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
                cv=args.cv,
            )
        elif method in ["GMM", "KMeans", "DBSCAN"]:
            model = load_model(
                task, method, features, cc, dataset, num_classes=num_classes
            )
        elif method == "wav2vec":
            model = load_model(
                task,
                method,
                features,
                cc,
                shape=None,
                num_classes=num_classes,
                dataset=dataset,
                max_length=length,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )
    elif task == "image":  # image emotion detection
        if method in ["MLP", "CNN", "Xception"]:
            model = load_model(
                task,
                method,
                cc=cc,
                num_classes=num_classes,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                h=h,
                landmark=args.landmark,
            )
        elif method == "ViT":
            model = load_model(
                task,
                method,
                dataset=dataset,
                cc=cc,
                num_classes=num_classes,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )
    print("Load model successfully.")

    """
        This part includes all training, validation and testing process with encapsulated functions.
        Detailed process of each method can be seen in corresponding classes.
    """
    if task == "speech":
        if method in ["SVM", "DT", "RF", "NB", "KNN"]:
            if method in ["KNN", "DT", "RF"]:
                cv_results_ = model.train(Xtrain, ytrain, Xval, yval, gridSearch=True)
            else:
                model.train(Xtrain, ytrain, Xval, yval)
            pred_train, pred_val, pred_test = model.test(
                Xtrain, ytrain, Xval, yval, Xtest
            )
        elif method in ["MLP", "RNN", "wav2vec"]:
            train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(
                model, train_ds, val_ds
            )
            test_res, pred_test, ytest = model.test(model, test_ds)
        elif method in ["LSTM", "CNN", "AlexNet"]:
            if cc != "finetune":
                train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(
                    Xtrain, ytrain, Xval, yval
                )
            else:  # finetuning
                (
                    train_res,
                    val_res,
                    pred_train,
                    pred_val,
                    ytrain,
                    yval,
                    ytune_train,
                    ytune_val,
                    tune_train_pred,
                    tune_val_pred,
                ) = model.train(
                    Xtrain,
                    ytrain,
                    Xval,
                    yval,
                    Xtune_train,
                    ytune_train,
                    Xtune_val,
                    ytune_val,
                )
            ytest, pred_test = model.test(Xtest, ytest)
        elif method in ["KMeans", "DBSCAN", "GMM"]:
            if method == "KMeans":
                pred_train, pred_val, ytrain, yval, centers = model.train(
                    Xtrain, ytrain, Xval, yval
                )
            else:
                pred_train, pred_val, ytrain, yval = model.train(
                    Xtrain, ytrain, Xval, yval
                )
            pred_test, ytest = model.test(Xtest, ytest)
    elif task == "image":
        if method in ["MLP", "CNN", "Xception"]:
            if cc != "finetune":
                train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(
                    Xtrain, ytrain, Xval, yval
                )
            else:
                (
                    train_res,
                    val_res,
                    pred_train,
                    pred_val,
                    ytrain,
                    yval,
                    ytune_train,
                    ytune_val,
                    tune_train_pred,
                    tune_val_pred,
                ) = model.train(
                    Xtrain,
                    ytrain,
                    Xval,
                    yval,
                    Xtune_train,
                    ytune_train,
                    Xtune_val,
                    ytune_val,
                )
            ytest, pred_test = model.test(Xtest, ytest)
        elif method in ["ViT"]:
            train_res, val_res, pred_train, ytrain, pred_val, yval = model.train(
                Xtrain, ytrain, Xval, yval
            )
            test_res, pred_test, ytest = model.test(Xtest, ytest)

    # metrics and visualization
    # hyperparameters selection
    if method in ["KNN", "DT", "RF"]:
        hyperpara_selection(
            task, method, features, cc, dataset, cv_results_["mean_test_score"]
        )

    # decision tree
    if "DT" in method:
        (visual4tree(task, method, features, cc, dataset, model.model))

    # confusion matrix and metrics calculation
    if method != "GMM":
        if cc != "finetune":
            res = {
                "train_res": get_metrics(task, ytrain, pred_train),
                "val_res": get_metrics(task, yval, pred_val),
                "test_res": get_metrics(task, ytest, pred_test),
            }
        else:
            res = {
                "train_res": get_metrics(task, ytrain, pred_train),
                "val_res": get_metrics(task, yval, pred_val),
                "fine-tune_train_res": get_metrics(task, ytune_train, tune_train_pred),
                "fine-tune_val_res": get_metrics(task, ytune_val, tune_val_pred),
                "test_res": get_metrics(task, ytest, pred_test),
            }
        for i in res.items():
            print(i)

        if split != None:  # imbalanced metrics for cross-corpus
            res_imbalance = {
                "train_res_imbalance": get_imbalance(task, ytrain, pred_train),
                "val_res_imbalance": get_imbalance(task, yval, pred_val),
                "test_res_imbalance": get_imbalance(task, ytest, pred_test),
            }
            for i in res_imbalance.items():
                print(i)

        if cc == "finetune":
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
                ytune_train,
                tune_train_pred,
                ytune_val,
                tune_val_pred,
            )
        else:
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
                corpus=corpus,
                split=split,
                landmark=args.landmark,
                process=process,
            )

    # convergence curves
    if method in ["LSTM", "CNN", "AlexNet"]:
        visaul4curves(
            task, method, features, cc, dataset, train_res, val_res, args.epochs
        )

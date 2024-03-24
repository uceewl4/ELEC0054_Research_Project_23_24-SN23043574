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

import os
import argparse
import warnings
import tensorflow as tf
from utils import (
    load_data,
    load_model,
    get_metrics,
    hyperpara_selection,
    visual4cm,
    visual4auc,
    visual4tree,
    visual4KMeans,
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
    parser.add_argument("--features", type=str, default="mfcc", help="model chosen")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of NNs like MLP and CNN"
    )
    parser.add_argument("--epochs", type=int, default=10, help="epochs of NNs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of NNs")
    parser.add_argument("--mfcc", type=int, default=40, help="epochs of NNs")
    parser.add_argument("--mels", type=int, default=128, help="epochs of NNs")
    parser.add_argument(
        "--pre_data",
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
    if task == "Speech":
        if method in ["SVM", "DT", "RF", "NB", "KNN"]:
            Xtrain, ytrain, Xtest, ytest, Xval, yval = load_data(
                task, method, dataset, features, args.n_mfcc, args.n_mels
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

    # elif method in ["MLP", "CNN"]:
    #     if args.multilabel == False:
    #         train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(
    #             model, train_ds, val_ds, args.epochs
    #         )
    #         test_res, pred_test, ytest = model.test(model, test_ds)
    #     else:  # multilabel
    #         (
    #             train_res,
    #             val_res,
    #             pred_train,
    #             pred_train_multilabel,
    #             pred_val,
    #             pred_val_multilabel,
    #             ytrain,
    #             yval,
    #         ) = model.train(model, train_ds, val_ds, args.epochs)
    #         test_res, pred_test, pred_test_multilabel, ytest = model.test(
    #             model, test_ds
    #         )
    #         print(pred_test_multilabel[:5, :])

    # elif method == "EnsembleNet":
    #     model.train(train_ds, val_ds, args.epochs)
    #     train_res, val_res, pred_train, pred_val, ytrain, yval = model.weight_selection(
    #         train_ds, val_ds
    #     )
    #     test_res, pred_test, ytest = model.test(test_ds)

    # elif (
    #     ("VGG16" in method)
    #     or ("ResNet50" in method)
    #     or ("DenseNet201" in method)
    #     or ("MobileNetV2" in method)
    #     or ("InceptionV3" in method)
    # ):
    #     if (
    #         ("KNN" in method)
    #         or ("DT" in method)
    #         or ("RF" in method)
    #         or ("ABC" in method)
    #     ):
    #         cv_results_ = model.train(
    #             model, Xtrain, ytrain, Xval, yval, Xtest, gridSearch=True
    #         )
    #     else:
    #         model.train(model, Xtrain, ytrain, Xval, yval, Xtest)
    #     pred_train, pred_val, pred_test = model.test(model, ytrain, yval)

    # metrics and visualization
    # hyperparameters selection
    if method in ["KNN", "DT", "RF"]:
        hyperpara_selection(task, method, features, cc, cv_results_["mean_test_score"])

    # decision tree
    if "DT" in method:
        (visual4tree(task, method, features, cc, model.model))

    # confusion matrix, auc roc curve, metrics calculation
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
        ytrain,
        yval,
        ytest,
        pred_train,
        pred_val,
        pred_test,
    )
    # if task == "A":
    #     visual4auc(
    #         task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test
    #     )

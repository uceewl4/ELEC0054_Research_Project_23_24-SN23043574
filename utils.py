# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/03/24 19:09:23
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   None
"""

# here put the import lib

# here put the import lib
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    silhouette_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    auc,
)
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from Speech.models.baselines import Baselines as SpeechBase


def get_features(filename, features="mfcc", n_mfcc=40, n_mels=128):
    with soundfile.SoundFile(filename) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        mfccs = np.mean(
            librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0
        )
        stft = np.abs(librosa.stft(X))  # spectrum
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        mel = np.mean(
            librosa.feature.melspectrogram(
                y=X, sr=sample_rate, n_mels=n_mels, fmax=8000
            ).T,
            axis=0,
        )

        if features == "mfcc":
            result = np.hstack((result, mfccs))
        elif features == "chroma":
            result = np.hstack((result, chroma))
        elif features == "mel":
            result = np.hstack((result, mel))
        elif features == "all":
            result = np.hstack((result, mfccs))
            result = np.hstack((result, chroma))
            result = np.hstack((result, mel))
    return result


def load_RAVDESS(features, n_mfcc, n_mels):
    x, y = [], []
    emotion_map = {
        "01": 0,  # 'neutral'
        "02": 1,  # 'calm'
        "03": 2,  # 'happy'
        "04": 3,  # 'sad'
        "05": 4,  # 'angry'
        "06": 5,  # 'fearful'
        "07": 6,  # 'disgust'
        "08": 7,  # 'surprised'
    }
    for file in glob.glob("datasets/speech/RAVDESS/Actor_*/*.wav"):
        # print(file)
        file_name = os.path.basename(file)
        emotion = emotion_map[file_name.split("-")[2]]
        feature = get_features(file, features, n_mfcc=n_mfcc, n_mels=n_mels)
        x.append(feature)
        y.append(emotion)
    print(np.array(x).shape)  # (864,40), (288,40), (288,40)
    X_train, X_left, ytrain, yleft = train_test_split(
        np.array(x), y, test_size=0.4, random_state=9
    )  # 3:2
    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    return X_train, ytrain, X_val, yval, X_test, ytest


def load_TESS(features, n_mfcc, n_mels):
    x, y = [], []
    emotion_map = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 4,
        "ps": 5,
        "sad": 6,
    }
    for dirname, _, filenames in os.walk("datasets/speech/TESS"):
        for filename in filenames:
            feature = get_features(
                os.path.join(dirname, filename), features, n_mfcc=n_mfcc, n_mels=n_mels
            )
            label = filename.split("_")[-1].split(".")[0]
            emotion = emotion_map[label.lower()]
            x.append(feature)
            y.append(emotion)
        if len(y) == 2800:
            break
    X_train, X_left, ytrain, yleft = train_test_split(  # 2800, 1680, 1120
        np.array(x), y, test_size=0.4, random_state=9
    )  # 3:2
    X_val, X_test, yval, ytest = train_test_split(
        X_left, yleft, test_size=0.5, random_state=9
    )  # 1:1
    # (1680, 40), (560, 40), (560, 40), (1680,)
    return X_train, ytrain, X_val, yval, X_test, ytest


"""
description: This function is used for loading data from preprocessed dataset into model input.
param {*} task: task Aor B
param {*} path: preprocessed dataset path
param {*} method: selected model for experiment
param {*} batch_size: batch size of NNs
return {*}: loaded model input 
"""


def load_data(task, method, dataset, features, n_mfcc=40, n_mels=128):
    if task == "speech":
        if dataset == "RAVDESS":
            X_train, ytrain, X_val, yval, X_test, ytest = load_RAVDESS(
                features=features, n_mfcc=n_mfcc, n_mels=n_mels
            )
        elif dataset == "TESS":
            X_train, ytrain, X_val, yval, X_test, ytest = load_TESS(
                features=features, n_mfcc=n_mfcc, n_mels=n_mels
            )
        return X_train, ytrain, X_val, yval, X_test, ytest

    # file = os.listdir(path)
    # Xtest, ytest, Xtrain, ytrain, Xval, yval = [], [], [], [], [], []

    # # divide into train/validation/test dataset
    # for index, f in enumerate(file):
    #     if not os.path.isfile(os.path.join(path, f)):
    #         continue
    #     else:
    #         img = cv2.imread(os.path.join(path, f))
    #         if task == "A" and method in [
    #             "LR",
    #             "KNN",
    #             "SVM",
    #             "DT",
    #             "NB",
    #             "RF",
    #             "ABC",
    #             "KMeans",
    #         ]:
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         if "test" in f:
    #             Xtest.append(img)
    #             ytest.append(f.split("_")[1][0])
    #         elif "train" in f:
    #             Xtrain.append(img)
    #             ytrain.append(f.split("_")[1][0])
    #         elif "val" in f:
    #             Xval.append(img)
    #             yval.append(f.split("_")[1][0])

    # if method in ["LR", "KNN", "SVM", "DT", "NB", "RF", "ABC", "KMeans"]:  # baselines
    #     if task == "A":
    #         n, h, w = np.array(Xtrain).shape
    #         Xtrain = np.array(Xtrain).reshape(
    #             n, h * w
    #         )  # need to reshape gray picture into two-dimensional ones
    #         Xval = np.array(Xval).reshape(len(Xval), h * w)
    #         Xtest = np.array(Xtest).reshape(len(Xtest), h * w)
    #     elif task == "B":
    #         n, h, w, c = np.array(Xtrain).shape
    #         Xtrain = np.array(Xtrain).reshape(n, h * w * c)
    #         Xval = np.array(Xval).reshape(len(Xval), h * w * c)
    #         Xtest = np.array(Xtest).reshape(len(Xtest), h * w * c)

    #         # shuffle dataset
    #         Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)
    #         Xval, yval = shuffle(Xval, yval, random_state=42)
    #         Xtest, ytest = shuffle(Xtest, ytest, random_state=42)

    #         # use PCA for task B to reduce dimensionality
    #         pca = PCA(n_components=64)
    #         Xtrain = pca.fit_transform(Xtrain)
    #         Xval = pca.fit_transform(Xval)
    #         Xtest = pca.fit_transform(Xtest)

    #     return Xtrain, ytrain, Xtest, ytest, Xval, yval

    # else:  # pretrained or customized
    #     n, h, w, c = np.array(Xtrain).shape
    #     Xtrain = np.array(Xtrain)
    #     Xval = np.array(Xval)
    #     Xtest = np.array(Xtest)

    #     """
    #         Notice that due to large size of task B dataset, part of train and validation data is sampled for
    #         pretrained network. However, all test data are used for performance measurement in testing procedure.
    #     """
    #     if task == "B":
    #         sample_index = random.sample([i for i in range(Xtrain.shape[0])], 40000)
    #         Xtrain = Xtrain[sample_index, :, :, :]
    #         ytrain = np.array(ytrain)[sample_index].tolist()

    #         sample_index_val = random.sample([i for i in range(Xval.shape[0])], 5000)
    #         Xval = Xval[sample_index_val, :]
    #         yval = np.array(yval)[sample_index_val].tolist()

    #         sample_index_test = random.sample([i for i in range(Xtest.shape[0])], 7180)
    #         Xtest = Xtest[sample_index_test, :]
    #         ytest = np.array(ytest)[sample_index_test].tolist()

    #     if method in [
    #         "CNN",
    #         "MLP",
    #         "EnsembleNet",
    #     ]:  # customized, loaded data with batches
    #         train_ds = tf.data.Dataset.from_tensor_slices(
    #             (Xtrain, np.array(ytrain).astype(int))
    #         ).batch(batch_size)
    #         val_ds = tf.data.Dataset.from_tensor_slices(
    #             (Xval, np.array(yval).astype(int))
    #         ).batch(batch_size)
    #         test_ds = tf.data.Dataset.from_tensor_slices(
    #             (Xtest, np.array(ytest).astype(int))
    #         ).batch(batch_size)
    #         normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)  # normalization
    #         train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    #         val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    #         test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    #         return train_ds, val_ds, test_ds
    #     else:
    #         return Xtrain, ytrain, Xtest, ytest, Xval, yval


"""
description: This function is used for loading selected model.
param {*} task: task A or B
param {*} method: selected model
param {*} multilabel: whether configuring multilabels setting (can only be used with MLP/CNN in task B)
param {*} lr: learning rate for adjustment and tuning
return {*}: constructed model
"""


def load_model(task, method, lr=0.001):
    if task == "speech":
        if method in ["SVM", "DT", "RF", "NB", "KNN"]:
            model = SpeechBase(method)

    return model


"""
description: This function is used for visualizing confusion matrix.
param {*} task: task A or B
param {*} method: selected model
param {*} ytrain: train ground truth
param {*} yval: validation ground truth
param {*} ytest: test ground truth
param {*} train_pred: train prediction
param {*} val_pred: validation prediction
param {*} test_pred: test prediction
"""


def visual4cm(
    task,
    method,
    features,
    cc,
    dataset,
    ytrain,
    yval,
    ytest,
    train_pred,
    val_pred,
    test_pred,
):
    # confusion matrix
    cms = {
        "train": confusion_matrix(ytrain, train_pred),
        "val": confusion_matrix(yval, val_pred),
        "test": confusion_matrix(ytest, test_pred),
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey="row")
    for index, mode in enumerate(["train", "val", "test"]):
        disp = ConfusionMatrixDisplay(
            cms[mode], display_labels=sorted(list(set(ytrain)))
        )
        disp.plot(ax=axes[index])
        disp.ax_.set_title(mode)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel("")
        if index != 0:
            disp.ax_.set_ylabel("")

    fig.text(0.45, 0.05, "Predicted label", ha="center")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)

    if not os.path.exists(f"outputs/{task}/confusion_matrix/"):
        os.makedirs(f"outputs/{task}/confusion_matrix/")
    fig.savefig(
        f"outputs/{task}/confusion_matrix/{method}_{features}_{cc}_{dataset}.png"
    )
    plt.close()


"""
description: This function is used for visualizing auc roc curves.
param {*} task: task A or B
param {*} method: selected model
param {*} ytrain: train ground truth
param {*} yval: validation ground truth
param {*} ytest: test ground truth
param {*} train_pred: train prediction
param {*} val_pred: validation prediction
param {*} test_pred: test prediction
"""


# def visual4auc(task, method, ytrain, yval, ytest, train_pred, val_pred, test_pred):
#     # roc curves
#     rocs = {
#         "train": roc_curve(
#             np.array(ytrain).astype(int),
#             train_pred.astype(int),
#             pos_label=1,
#             drop_intermediate=True,
#         ),
#         "val": roc_curve(
#             np.array(yval).astype(int),
#             val_pred.astype(int),
#             pos_label=1,
#             drop_intermediate=True,
#         ),
#         "test": roc_curve(
#             np.array(ytest).astype(int),
#             test_pred.astype(int),
#             pos_label=1,
#             drop_intermediate=True,
#         ),
#     }

#     colors = list(mcolors.TABLEAU_COLORS.keys())

#     plt.figure(figsize=(10, 6))
#     for index, mode in enumerate(["train", "val", "test"]):
#         plt.plot(
#             rocs[mode][0],
#             rocs[mode][1],
#             lw=1,
#             label="{}(AUC={:.3f})".format(mode, auc(rocs[mode][0], rocs[mode][1])),
#             color=mcolors.TABLEAU_COLORS[colors[index]],
#         )
#     plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
#     plt.axis("square")
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.xlabel("False Positive Rate", fontsize=10)
#     plt.ylabel("True Positive Rate", fontsize=10)
#     plt.title(f"ROC Curve for {method}", fontsize=10)
#     plt.legend(loc="lower right", fontsize=5)

#     if not os.path.exists("outputs/images/roc_curve/"):
#         os.makedirs("outputs/images/roc_curve/")
#     plt.savefig(f"outputs/images/roc_curve/{method}_task{task}.png")
#     plt.close()


"""
description: This function is used for visualizing decision trees.
param {*} method: selected model
param {*} model: constructed tree model
"""


def visual4tree(task, method, features, cc, dataset, model):
    plt.figure(figsize=(100, 15))
    class_names = (
        ["pneumonia", "non-pneumonia"]
        if task == "A"
        else ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
    )
    tree.plot_tree(
        model, class_names=class_names, filled=True, rounded=True, fontsize=5
    )
    if not os.path.exists(f"outputs/{task}/trees/"):
        os.makedirs(f"outputs/{task}/trees/")
    plt.savefig(f"outputs/{task}/trees/{method}_{features}_{cc}_{dataset}.png")
    plt.close()


"""
description: This function is used for calculating metrics performance including accuracy, precision, recall, f1-score.
param {*} task: task A or B
param {*} y: ground truth
param {*} pred: predicted labels
"""


def get_metrics(task, y, pred):
    average = "macro"
    result = {
        "acc": round(
            accuracy_score(np.array(y).astype(int), pred.astype(int)) * 100, 4
        ),
        "pre": round(
            precision_score(np.array(y).astype(int), pred.astype(int), average=average)
            * 100,
            4,
        ),
        "rec": round(
            recall_score(np.array(y).astype(int), pred.astype(int), average=average)
            * 100,
            4,
        ),
        "f1": round(
            f1_score(np.array(y).astype(int), pred.astype(int), average=average) * 100,
            4,
        ),
    }
    return result


"""
description: This function is used for visualizing hyperparameter selection for grid search models.
param {*} task: task A or B
param {*} method: selected model
param {*} scores: mean test score for cross validation of different parameter combinations
"""


def hyperpara_selection(task, method, feature, cc, dataset, scores):
    plt.figure(figsize=(8, 5))
    plt.plot(scores, c="g", marker="D", markersize=5)
    plt.xlabel("Params")
    plt.ylabel("Accuracy")
    plt.title(f"Params for {method}")
    if not os.path.exists(f"outputs/{task}/hyperpara_selection/"):
        os.makedirs(f"outputs/{task}/hyperpara_selection/")
    plt.savefig(
        f"outputs/{task}/hyperpara_selection/{method}_{feature}_{cc}_{dataset}.png"
    )
    plt.close()


"""
description: This function is used for visualizing dataset label distribution.
param {*} task: task A or B
param {*} data: npz data
"""


# def visual4label(task, data):
#     fig, ax = plt.subplots(
#         nrows=1, ncols=3, figsize=(6, 3), subplot_kw=dict(aspect="equal"), dpi=600
#     )

#     for index, mode in enumerate(["train", "val", "test"]):
#         pie_data = [
#             np.count_nonzero(data[f"{mode}_labels"].flatten() == i)
#             for i in range(len(set(data[f"{mode}_labels"].flatten().tolist())))
#         ]
#         labels = [
#             f"label {i}"
#             for i in sorted(list(set(data[f"{mode}_labels"].flatten().tolist())))
#         ]
#         wedges, texts, autotexts = ax[index].pie(
#             pie_data,
#             autopct=lambda pct: f"{pct:.2f}%\n({int(np.round(pct/100.*np.sum(pie_data))):d})",
#             textprops=dict(color="w"),
#         )
#         if index == 2:
#             ax[index].legend(
#                 wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
#             )
#         size = 6 if task == "A" else 3
#         plt.setp(autotexts, size=size, weight="bold")
#         ax[index].set_title(mode)
#     plt.tight_layout()

#     if not os.path.exists("outputs/images/"):
#         os.makedirs("outputs/images/")
#     fig.savefig(f"outputs/images/label_distribution_task{task}.png")
#     plt.close()

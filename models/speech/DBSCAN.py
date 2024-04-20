"""
Author: uceewl4 uceewl4@ucl.ac.uk
Date: 2024-03-28 16:56:42
LastEditors: uceewl4 uceewl4@ucl.ac.uk
LastEditTime: 2024-03-28 18:09:52
FilePath: /ELEC0054_Research_Project_23_24-SN23043574/models/speech/GMM.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

# here put the import lib
import numpy as np
from sklearn.cluster import DBSCAN as dbscan
import os
import pickle
import time


class DBSCAN:
    """
    description: This function is used for initialization of Baselines class with instance variables configuration.
    param {*} self
    param {*} method: used for specifying the baseline model of experiment
    """

    def __init__(self, task, method, features, cc, dataset, n_components=None):
        self.features = features
        self.cc = cc
        self.method = method
        self.task = task
        self.dataset = dataset
        self.model = dbscan(eps=3, min_samples=2)

    """
    description: This function includes entire training process and
        the cross-validation procedure for baselines of KNN, DT, RF and ABC.
        Notice that because of the size of dataset, high dimensional features of images and 
        principle of some models, the process of RF, ABC may be extremely slow.
        It can even take several hours for a model to run in task B. 
        Some quick models are recommended on README.md and Github link.
    param {*} self
    param {*} Xtrain: train images
    param {*} ytrain: train ground truth labels
    param {*} Xval: validation images
    param {*} yval: validation ground truth labels
    param {*} gridSearch: whether grid search cross-validation (only for KNN, DT, RF and ABC)
    return {*}: if grid search is performed, the cv results are returned.
    """

    def train(self, Xtrain, ytrain, Xval, yval):
        print(f"Start training for {self.method}......")
        start_time_train = time.time()
        self.model.fit(Xtrain)  # (864,40)
        if not os.path.exists(f"outputs/{self.task}/models/"):
            os.makedirs(f"outputs/{self.task}/models/")
        pickle.dump(
            self.model,
            open(
                f"outputs/{self.task}/models/{self.method}_{self.features}_{self.cc}_{self.dataset}.gmm",
                "wb",
            ),
        )

        # train_score = np.array(self.model.score(Xtrain))
        # pred_train = np.argmax(train_score)
        pred_train = self.model.predict(Xtrain)
        pred_val = self.model.predict(Xval)

        # val_score = np.array(self.model.score(Xval))
        # pred_val = np.argmax(val_score)

        end_time_train = time.time()
        elapsed_time_train = end_time_train - start_time_train
        print(f"Finish training for {self.method}.")
        print(f"Training time: {elapsed_time_train}s")

        return pred_train, pred_val, ytrain, yval

    """
    description: This function is used for the entire process of testing.
    param {*} self
    param {*} Xtrain: train images
    param {*} ytrain: train ground truth labels
    param {*} Xval: validation images
    param {*} yval: validation ground truth labels
    param {*} Xtest: test images
    return {*}: predicted labels for train, validation and test respectively
    """

    def test(self, Xtest, ytest):
        print(f"Start testing for {self.method}......")
        start_time_test = time.time()
        # model = pickle.load(open(filename,'rb'))
        # test_score = np.array(self.model.score(Xtest))
        # pred_test = np.argmax(test_score)
        pred_test = self.model.predict(Xtest)

        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test
        print(f"Finish testing for {self.method}.")
        print(f"Testing time: {elapsed_time_test}s")

        return pred_test, ytest

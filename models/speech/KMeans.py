# -*- encoding: utf-8 -*-
"""
@File    :   KMeans.py
@Time    :   2024/07/24 05:43:54
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file encapsulates all implementation process for K-Means in speech emotion detection.
"""

# here put the import lib
import os
import time
import pickle
import numpy as np
from sklearn.cluster import KMeans as kmeans


class KMeans:
    def __init__(self, task, method, features, cc, dataset, n_components=None):
        self.features = features
        self.cc = cc
        self.method = method
        self.task = task
        self.dataset = dataset

        # model
        self.model = kmeans(n_clusters=n_components, random_state=0, n_init="auto")

    def train(self, Xtrain, ytrain, Xval, yval):
        """
        description: This function is used for the entire process of training.
        param {*} self
        param {*} Xtrain: features of train set
        param {*} ytrain: labels of train set
        param {*} Xval: features of validation set
        param {*} yval: labels of validation set
        return {*}: predicted labels, ground truth labels of train and validation, centers of K-Means clustering
        """

        print(f"Start training for {self.method}......")
        start_time_train = time.time()
        self.model.fit(Xtrain)
        # centers of K-Means
        centers = self.model.cluster_centers_

        # save the model
        if not os.path.exists(f"outputs/{self.task}/models/"):
            os.makedirs(f"outputs/{self.task}/models/")
        pickle.dump(
            self.model,
            open(
                f"outputs/{self.task}/models/{self.method}_{self.features}_{self.cc}_{self.dataset}.gmm",
                "wb",
            ),
        )

        pred_train = self.model.predict(Xtrain)
        pred_val = self.model.predict(Xval)

        end_time_train = time.time()
        elapsed_time_train = end_time_train - start_time_train
        print(f"Finish training for {self.method}.")
        print(f"Training time: {elapsed_time_train}s")

        return pred_train, pred_val, ytrain, yval, centers

    def test(self, Xtest, ytest):
        """
        description: This function is used for the entire process of testing.
        param {*} self
        param {*} Xtest: features of test set
        param {*} ytest: labels of test set
        return {*}: predicted labels and ground truth of test dataset
        """

        print(f"Start testing for {self.method}......")
        start_time_test = time.time()

        pred_test = self.model.predict(Xtest)

        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test
        print(f"Finish testing for {self.method}.")
        print(f"Testing time: {elapsed_time_test}s")

        return pred_test, ytest

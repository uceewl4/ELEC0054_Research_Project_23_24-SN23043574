# -*- encoding: utf-8 -*-
"""
@File    :   baselines.py
@Time    :   2024/07/24 05:32:39
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file encapsulates all implementation process for basic classifiers in speech emotion detection.
"""

# here put the import lib
import time
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB


class Baselines:
    def __init__(self, method=None):
        self.method = method

        if method == "KNN":
            self.model = KNeighborsClassifier()
        elif method == "SVM":
            self.model = svm.SVC(kernel="poly", C=1, gamma="auto")
        elif method == "DT":
            self.model = DecisionTreeClassifier(criterion="entropy")
        elif method == "NB":
            self.model = GaussianNB()
        elif method == "RF":
            self.model = RandomForestClassifier(criterion="entropy", verbose=1)

    def train(self, Xtrain, ytrain, Xval, yval, gridSearch=False):
        """
        description: This function includes entire training process and
            the cross-validation procedure for baselines of KNN, SVM, DT, NB and RF.
        param {*} self
        param {*} Xtrain: features of train set
        param {*} ytrain: labels of train set
        param {*} Xval: features of validation set
        param {*} yval: labels of validation set
        param {*} gridSearch: whether grid search cross-validation (only for KNN, DT, RF)
        return {*}: if grid search is performed, the cv results are returned.
        """
        # train initially
        print(f"Start training for {self.method}......")
        start_time_train = time.time()
        self.model.fit(Xtrain, ytrain)
        print(f"Finish training for {self.method}.")

        # cross-validation
        if gridSearch:
            print(f"Start tuning(cross-validation) for {self.method}......")
            if self.method == "KNN":
                params = [
                    {"n_neighbors": [i for i in range(1, 30, 1)]}
                ]  # parameters for grid search
            if self.method == "DT":
                params = [
                    {"max_leaf_nodes": [i for i in range(100, 200, 2)]}
                ]  # 80, 150
            if self.method == "RF":
                params = [
                    {
                        # "n_estimators": [120, 140, 160, 180, 200],
                        # "max_depth": [8, 10, 12, 14, 16],
                        "n_estimators": [120, 140, 160, 180],
                        "max_depth": [8, 10, 12, 14],
                    }
                ]
            grid = GridSearchCV(self.model, params, cv=10, scoring="accuracy")

            grid.fit(np.concatenate((Xtrain, Xval), axis=0), ytrain + yval)
            print(grid.best_params_)
            self.model = grid.best_estimator_  # best estimator

            end_time_train = time.time()
            elapsed_time_train = end_time_train - start_time_train
            print(f"Finish tuning(cross-validation) for {self.method}.")
            print(f"Training and cross-validation time: {elapsed_time_train}s")

            return grid.cv_results_
        else:
            end_time_train = time.time()
            elapsed_time_train = end_time_train - start_time_train
            print(f"Finish tuning(cross-validation) for {self.method}.")
            print(f"Training and cross-validation time: {elapsed_time_train}s")

    def test(self, Xtrain, ytrain, Xval, yval, Xtest):
        """
        description: This function is used for the entire process of testing.
        param {*} self
        param {*} Xtest: features of test set
        param {*} ytest: labels of test set
        return {*}: predicted labels and ground truth of test dataset
        """

        print(f"Start testing for {self.method}......")
        start_time_test = time.time()

        # predict again on entire train and validation sets
        self.model.fit(np.concatenate((Xtrain, Xval), axis=0), ytrain + yval)
        pred_test = self.model.predict(Xtest)
        pred_train = self.model.predict(Xtrain)
        pred_val = self.model.predict(Xval)

        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test
        print(f"Finish testing for {self.method}.")
        print(f"Testing time: {elapsed_time_test}s")

        # # save the model
        # if not os.path.exists("outputs/speech/models/"):
        #     os.makedirs("outputs/speech/models")
        # self.model.save("outputs/speech/models/NB.h5")

        return pred_train, pred_val, pred_test

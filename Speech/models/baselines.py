# -*- encoding: utf-8 -*-
"""
@File    :   Baselines.py
@Time    :   2024/03/24 20:00:21
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   None
"""

# here put the import lib

# here put the import lib
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


class Baselines:
    """
    description: This function is used for initialization of Baselines class with instance variables configuration.
    param {*} self
    param {*} method: used for specifying the baseline model of experiment
    """

    def __init__(self, method=None):
        self.method = method
        if method == "KNN":
            self.model = KNeighborsClassifier()
        elif method == "SVM":
            self.model = svm.SVC(kernel="poly")
        elif method == "DT":
            self.model = DecisionTreeClassifier(criterion="entropy")
        elif method == "NB":
            self.model = MultinomialNB()
        elif method == "RF":
            self.model = RandomForestClassifier(criterion="entropy", verbose=1)

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

    def train(self, Xtrain, ytrain, Xval, yval, gridSearch=False):
        print(f"Start training for {self.method}......")
        self.model.fit(Xtrain, ytrain)
        print(f"Finish training for {self.method}.")

        # cross-validation
        if gridSearch:
            print(f"Start tuning(cross-validation) for {self.method}......")
            if self.method == "KNN":
                params = [
                    {"n_neighbors": [i for i in range(1, 30, 2)]}
                ]  # parameters for grid search
            if self.method == "DT":
                params = [{"max_leaf_nodes": [i for i in range(20, 100, 5)]}]
            if self.method == "RF":
                params = [
                    {
                        "n_estimators": [120, 140, 160, 180, 200],
                        "max_depth": [8, 10, 12, 14, 16],
                    }
                ]
            grid = GridSearchCV(self.model, params, cv=10, scoring="accuracy")

            grid.fit(np.concatenate((Xtrain, Xval), axis=0), ytrain + yval)
            print(grid.best_params_)
            self.model = grid.best_estimator_  # best estimator

            print(f"Finish tuning(cross-validation) for {self.method}.")
            return grid.cv_results_

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

    def test(self, Xtrain, ytrain, Xval, yval, Xtest):
        print(f"Start testing for {self.method}......")
        self.model.fit(np.concatenate((Xtrain, Xval), axis=0), ytrain + yval)
        pred_test = self.model.predict(Xtest)
        pred_train = self.model.predict(Xtrain)
        pred_val = self.model.predict(Xval)

        print(f"Finish testing for {self.method}.")

        return pred_train, pred_val, pred_test

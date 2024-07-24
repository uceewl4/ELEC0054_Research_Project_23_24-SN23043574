# -*- encoding: utf-8 -*-
"""
@File    :   RNN.py
@Time    :   2024/07/24 05:57:31
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file encapsulates all implementation process for RNN in speech emotion detection.
"""
# here put the import lib

import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorboardX import SummaryWriter
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    Dropout,
    Input,
    SimpleRNN,
    Bidirectional,
)

import warnings

warnings.filterwarnings("ignore")


class RNN(Model):
    def __init__(
        self,
        task,
        method,
        features,
        cc,
        shape,
        num_classes,
        dataset,
        bidirectional=False,
        epochs=10,
        lr=0.001,
    ):
        super(RNN, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.cc = cc
        self.method = method
        self.task = task
        self.bidirectional = bidirectional
        self.dataset = dataset
        self.lr = lr
        self.epochs = epochs

        # network architecture
        if self.bidirectional == True:
            self.r1 = Bidirectional(
                SimpleRNN(256, return_sequences=False, input_shape=(shape, 1))
            )
            self.r2 = Bidirectional(SimpleRNN(256, return_sequences=False))
        else:
            self.r1 = SimpleRNN(256, return_sequences=False, input_shape=(shape, 1))
            self.r2 = SimpleRNN(256, return_sequences=False)
        self.d1 = Dense(128, activation="relu")
        self.do1 = Dropout(0.4)
        self.d2 = Dense(64, activation="relu")
        self.d3 = Dense(32, activation="relu")
        self.do2 = Dropout(0.4)
        self.d4 = Dense(num_classes)

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )  # loss object
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)  # optimizer
        # loss and accuracy
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )
        self.val_loss = tf.keras.metrics.Mean(name="eval_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_accuracy"
        )
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="test_accuracy"
        )

    def call(self, x):
        """
        description: This function is the actual construction process of customized network.
        param {*} self
        param {*} x: input
        return {*}: output logits
        """
        x = self.r1(x)  # (32, 256)
        x = np.expand_dims(x, axis=2)  # (32, 256, 1)
        x = self.r2(x)  # (32, 256)
        x = self.d1(x)
        x = self.do1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.do2(x)
        return self.d4(x)  # (32, 8) RAVDESS

    def train(self, model, train_ds, val_ds):
        """
        description: This function is used for the entire process of training.
        param {*} self
        param {*} model: constructed model
        param {*} train_ds: data loader with batches of training set and labels
        param {*} val_ds: data loader with batches of validation set and labels
        return {*}: accuracy and loss results, predicted labels, ground truth labels of train and validation
        """

        print("Start training......")
        start_time_train = time.time()

        # save changing curves of loss and accuracy
        if not os.path.exists(f"outputs/{self.task}/nn_curves/"):
            os.makedirs(f"outputs/{self.task}/nn_curves/")
        writer = SummaryWriter(
            f"outputs/{self.task}/nn_curves/{self.method}_{self.features}_{self.cc}_{self.dataset}"
        )

        # train
        for epoch in range(self.epochs):
            train_pred = []  # label prediction
            ytrain = []  # ground truth
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            train_progress_bar = tqdm(range(len(train_ds)))

            for step, (train_samples, train_labels) in enumerate(train_ds):
                with tf.GradientTape() as tape:
                    predictions = model(
                        np.expand_dims(train_samples, 2), training=True
                    )  # logits
                    train_prob = tf.nn.softmax(predictions)  # probabilities
                    train_pred += np.argmax(train_prob, axis=1).tolist()
                    ytrain += np.array(train_labels).tolist()  # ground truth
                    loss = self.loss_object(train_labels, predictions)

                # backward propagation
                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(
                    (grad, var)
                    for (grad, var) in zip(gradients, model.trainable_variables)
                    if grad is not None
                )
                train_progress_bar.update(1)
                self.train_loss(loss)
                self.train_accuracy(train_labels, predictions)

                # validation
                if step % 25 == 0:
                    val_pred = []
                    yval = []
                    self.val_loss.reset_state()
                    self.val_accuracy.reset_state()
                    val_progress_bar = tqdm(range(len(val_ds)))

                    for val_samples, val_labels in val_ds:
                        with tf.GradientTape() as tape:
                            predictions = model(
                                np.expand_dims(val_samples, axis=2), training=True
                            )
                            val_prob = tf.nn.softmax(predictions)
                            val_pred += np.argmax(val_prob, axis=1).tolist()
                            yval += np.array(val_labels).tolist()
                            val_loss = self.loss_object(val_labels, predictions)

                            self.val_loss(val_loss)
                            self.val_accuracy(val_labels, predictions)

                        # backward propagation
                        gradients = tape.gradient(val_loss, model.trainable_variables)
                        self.optimizer.apply_gradients(
                            (grad, var)
                            for (grad, var) in zip(gradients, model.trainable_variables)
                            if grad is not None
                        )
                        val_progress_bar.update(1)

                        self.val_loss(val_loss)
                        self.val_accuracy(val_labels, predictions)

                    val_res = {
                        "val_loss": np.array(self.val_loss.result()).tolist(),
                        "val_acc": round(np.array(self.val_accuracy.result()) * 100, 4),
                    }
                    print(f"Epoch: {epoch + 1}, Step: {step} ", val_res)

            train_res = {
                "train_loss": np.array(self.train_loss.result()).tolist(),
                "train_acc": round(np.array(self.train_accuracy.result()) * 100, 4),
            }
            print(f"Epoch: {epoch + 1}", train_res)
            # sketch curves
            writer.add_scalars(
                "loss",
                {
                    "train_loss": np.array(self.train_loss.result()).tolist(),
                    "val_loss": np.array(self.val_loss.result()).tolist(),
                },
                epoch,
            )
            writer.add_scalars(
                "accuracy",
                {
                    "train_accuracy": np.array(self.train_accuracy.result()).tolist(),
                    "val_accuracy": np.array(self.val_accuracy.result()).tolist(),
                },
                epoch,
            )

            train_pred = np.array(train_pred)
            val_pred = np.array(val_pred)

        end_time_train = time.time()
        elapsed_time_train = end_time_train - start_time_train
        print(f"Finish training for {self.method}.")
        print(f"Training time: {elapsed_time_train}s")
        writer.close()

        return train_res, val_res, train_pred, val_pred, ytrain, yval

    def test(self, model, test_ds):
        """
        description: This function is used for the entire process of testing.
        param {*} self
        param {*} model: trained model
        param {*} test_ds: data loader with batches of testing set and labels
        return {*}: predicted labels and ground truth of test dataset
        """

        print("Start testing......")
        start_time_test = time.time()

        test_pred = []  # predicted labels
        ytest = []  # ground truth
        self.test_loss.reset_state()
        self.test_accuracy.reset_state()
        test_progress_bar = tqdm(range(len(test_ds)))

        for test_samples, test_labels in test_ds:
            predictions = model(
                np.expand_dims(test_samples, axis=2), training=False
            )  # logits
            test_prob = tf.nn.softmax(predictions)  # probability
            test_pred += np.argmax(test_prob, axis=1).tolist()
            ytest += np.array(test_labels).tolist()  # ground truth

            t_loss = self.loss_object(test_labels, predictions)
            test_progress_bar.update(1)
            self.test_loss(t_loss)
            self.test_accuracy(test_labels, predictions)

        test_res = {
            "test_loss": np.array(self.test_loss.result()).tolist(),
            "test_acc": round(np.array(self.test_accuracy.result()) * 100, 4),
        }
        test_pred = np.array(test_pred)

        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test
        print("Finish testing.")
        print(f"Testing time: {elapsed_time_test}s")

        return test_res, test_pred, ytest

# -*- encoding: utf-8 -*-
"""
@File    :   wav2vec.py
@Time    :   2024/07/24 06:01:44
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :  This file encapsulates all implementation process for Wav2Vec 2.0 in speech emotion detection.
            The code refers to https://keras.io/examples/audio/wav2vec2_audiocls/.
"""

# here put the import lib

import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from keras.models import Sequential
from tensorboardX import SummaryWriter
from transformers import TFWav2Vec2Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    Dropout,
    Input,
    SimpleRNN,
    Bidirectional,
    GlobalAveragePooling1D,
)


class Wav2Vec(Model):
    def __init__(
        self,
        task,
        method,
        features,
        cc,
        num_classes,
        dataset,
        max_length,
        epochs=10,
        lr=0.001,
        batch_size=16,
    ):
        super(Wav2Vec, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.cc = cc
        self.method = method
        self.task = task
        self.batch_size = batch_size
        self.dataset = dataset
        self.lr = lr
        self.epochs = epochs

        # network architecture
        self.wav2vec2 = TFWav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base", apply_spec_augment=False, from_pt=True
        )  # model
        self.pooling = GlobalAveragePooling1D()
        self.intermediate = Dropout(0.5)
        # print(num_classes)
        self.final = Dense(num_classes, name="outputs")  # output layer

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
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
        description: This function is the actual construction process of Wav2Vec 2.0 model.
        param {*} self
        param {*} x: input
        return {*}: output logits
        """
        model_state = self.wav2vec2(x)[0]
        # print(model_state.shape)  # (None, 453, 768)
        pool_state = self.pooling(model_state)  # (360, 40)
        # print(pool_state.shape)  # (none 768)
        inter_state = self.intermediate(pool_state)
        # print(inter_state.shape)  # (none, 768)
        final_state = self.final(inter_state)
        # print(final_state.shape)
        return final_state  # (32, 8) RAVDESS

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
                with tf.GradientTape() as tape:  # (32, 40)
                    # print(train_samples.shape)  # (16, 145172)
                    # print(np.expand_dims(train_samples, 2).shape)  # (16, 145172, 1)
                    predictions = model(train_samples, training=True)  # logits
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
                            predictions = model(val_samples, training=True)
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
            predictions = model(test_samples, training=False)  # logits
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

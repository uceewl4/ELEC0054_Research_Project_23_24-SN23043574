# -*- encoding: utf-8 -*-
"""
@File    :   MLP.py
@Time    :   2024/03/24 21:41:02
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   None
"""

# here put the import lib


import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm
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


class RNN(Model):
    """
    description: This function includes all initialization of MLP, like layers used for construction,
      loss function object, optimizer, measurement of accuracy and loss.
    param {*} self
    param {*} task: task A or B
    param {*} method: MLP
    param {*} lr: learning rate
    """

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

        # network layers definition
        if self.bidirectional == True:
            self.r1 = Bidirectional(
                SimpleRNN(256, return_sequences=False, input_shape=(shape, 1))
            )
            self.r2 = Bidirectional(SimpleRNN(256, return_sequences=True))
        else:
            # self.r1 = SimpleRNN(256, return_sequences=False, input_shape=(shape, 1))
            self.r1 = SimpleRNN(256, return_sequences=False, input_shape=(shape, 1))
            self.r2 = SimpleRNN(256, return_sequences=True)
        self.d1 = Dense(128, activation="relu")
        self.do1 = Dropout(0.4)
        self.d2 = Dense(64, activation="relu")
        self.d3 = Dense(32, activation="relu")
        self.do2 = Dropout(0.4)
        self.d4 = Dense(num_classes)

        # objective function: binary cross entropy
        # notice that here the loss is calculated from logits, no need to set activation function for the output layer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.lr = lr
        self.epochs = epochs

        # adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

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

    """
  description: This function is the actual construction process of customized network.
  param {*} self
  param {*} x: input 
  return {*}: output logits
  """

    def call(self, x):
        x = self.r1(x)  # (32,256)
        x = self.r2(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.do1(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.do2(x)

        return self.d5(x)

    """
  description: This function is used for the entire process of training. 
    Notice that loss of both train and validation are backward propagated.
  param {*} self
  param {*} model: customized network constructed
  param {*} train_ds: loaded train dataset as batches
  param {*} val_ds: loaded validation dataset as batches
  param {*} EPOCHS: number of epochs
  return {*}: accuracy and loss results, predicted labels, ground truth labels of train and validation
  """

    def train(self, model, train_ds, val_ds):
        print("Start training......")
        start_time_train = time.time()
        if not os.path.exists(f"outputs/{self.task}/nn_curves/"):
            os.makedirs(f"outputs/{self.task}/nn_curves/")
        writer = SummaryWriter(
            f"outputs/{self.task}/nn_curves/{self.method}_{self.features}_{self.cc}_{self.dataset}"
        )

        # train
        for epoch in range(self.epochs):
            train_pred = []  # label prediction
            ytrain = []  # ground truth
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            train_progress_bar = tqdm(range(len(train_ds)))

            for step, (train_images, train_labels) in enumerate(train_ds):
                with tf.GradientTape() as tape:  # 32,40
                    predictions = model(
                        np.expand_dims(train_images, 2), training=True
                    )  # logits
                    train_prob = tf.nn.softmax(predictions)  # probabilities
                    train_pred += np.argmax(train_prob, axis=1).tolist()
                    ytrain += np.array(train_labels).tolist()  # ground truth
                    loss = self.loss_object(train_labels, predictions)

                # backward propagation
                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )
                train_progress_bar.update(1)

                self.train_loss(loss)
                self.train_accuracy(train_labels, predictions)

                # validation
                if step % 50 == 0:
                    val_pred = []
                    yval = []
                    self.val_loss.reset_states()
                    self.val_accuracy.reset_states()
                    val_progress_bar = tqdm(range(len(val_ds)))

                    for val_images, val_labels in val_ds:
                        with tf.GradientTape() as tape:
                            predictions = model(val_images, training=True)
                            val_prob = tf.nn.sigmoid(predictions)
                            val_pred += np.argmax(val_prob, axis=1).tolist()
                            yval += np.array(val_labels).tolist()
                            val_loss = self.loss_object(val_labels, predictions)

                            self.val_loss(val_loss)
                            self.val_accuracy(val_labels, predictions)

                        # backward propagation
                        gradients = tape.gradient(val_loss, model.trainable_variables)
                        self.optimizer.apply_gradients(
                            zip(gradients, model.trainable_variables)
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

    """
  description: This function is used for the entire process of testing. 
    Notice that loss of testing is not backward propagated.
  param {*} self
  param {*} model: customized network constructed
  param {*} test_ds: loaded test dataset as batches
  return {*}: accuracy and loss result, predicted labels and ground truth of test dataset
  """

    def test(self, model, test_ds):
        print("Start testing......")
        start_time_test = time.time()

        test_pred = []  # predicted labels
        ytest = []  # ground truth
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        test_progress_bar = tqdm(range(len(test_ds)))

        for test_images, test_labels in test_ds:
            predictions = model(test_images, training=False)  # logits
            test_prob = tf.nn.sigmoid(predictions)  # probability
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

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
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from keras.models import Sequential
import tqdm
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


class LSTM(Model):
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
        bidirectional=False,
        epochs=10,
        lr=0.001,
        batch_size=16,
    ):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.cc = cc
        self.method = method
        self.task = task
        self.bidirectional = bidirectional
        self.batch_size = batch_size

        # network layers definition
        self.model = Sequential(
            [
                LSTM(
                    256, return_sequences=False, input_shape=(shape, 1)
                ),  # 40 features as 40 timestamp, for each timestamp the input dimension is 1
                # output vector dimension is 256
                Dropout(0.2),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dropout(0.2),
                Dense(num_classes, name="outputs"),
            ]
        )

        self.model.build((None, shape, 1))
        self.model.summary()
        self.output_layer = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.get_layer("outputs").output
        )

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.lr = lr
        self.epoch = epochs
        self.batch_size = batch_size
        self.method = method
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

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

    def train(self, Xtrain, ytrain, Xval, yval):
        print("Start training......")
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_object,
            metrics=["accuracy"],
        )
        history = self.model.fit(
            Xtrain,
            ytrain,
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(Xval, yval),
        )

        # get predictions
        train_predictions = self.output_layer.predict(x=Xtrain)
        train_prob = tf.nn.softmax(train_predictions)
        train_pred += np.argmax(train_prob, axis=1).tolist()
        train_pred = np.array(train_pred)
        train_res = {
            "train_loss": history.history["loss"],
            "train_acc": history.history["accuracy"],
        }

        val_predictions = self.output_layer.predict(x=Xval)
        val_prob = tf.nn.softmax(val_predictions)
        val_pred += np.argmax(val_prob, axis=1).tolist()
        val_pred = np.array(val_pred)
        val_res = {
            "val_loss": history.history["val_loss"],
            "val_acc": history.history["val_accuracy"],
        }

        print(f"Finish training for {self.method}.")
        return train_res, val_res, train_pred, val_pred, ytrain, yval

    """
  description: This function is used for the entire process of testing. 
    Notice that loss of testing is not backward propagated.
  param {*} self
  param {*} model: customized network constructed
  param {*} test_ds: loaded test dataset as batches
  return {*}: accuracy and loss result, predicted labels and ground truth of test dataset
  """

    def test(self, Xtest, ytest):
        print("Start testing......")
        test_pred = []
        test_loss, test_acc = self.model.evaluate(Xtest, ytest, verbose=2)
        test_predictions = self.output_layer.predict(x=Xtest)
        test_prob = tf.nn.softmax(test_predictions)  # probabilities
        test_pred += np.argmax(test_prob, axis=1).tolist()
        test_pred = np.array(test_pred)
        print("Finish training.")
        test_pred = np.array(test_pred)

        return ytest, test_pred

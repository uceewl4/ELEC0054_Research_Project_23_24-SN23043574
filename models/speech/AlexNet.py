# -*- encoding: utf-8 -*-
"""
@File    :   AlexNet.py
@Time    :   2024/03/28 17:52:00
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   None
"""

# here put the import lib

# here put the import lib


import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from keras.models import Sequential
from sklearn.model_selection import KFold
from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    Dropout,
    MaxPooling2D,
)


class AlexNet(Model):
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
        length,
        epochs=10,
        lr=0.001,
        batch_size=16,
        cv=False,
    ):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.cc = cc
        self.method = method
        self.task = task
        self.batch_size = batch_size
        self.dataset = dataset
        self.cv = cv
        self.finetune = True if cc == "finetune" else False
        # network layers definition
        self.model = Sequential(
            [
                Conv2D(
                    64,
                    (3, 3),
                    activation="relu",
                    input_shape=(shape, length, 1),  # 40, 160,1
                ),  # features, length, channel
                MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                Conv2D(192, kernel_size=3, padding="valid", activation="relu"),
                MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                Conv2D(384, kernel_size=3, padding="same", activation="relu"),
                Conv2D(256, kernel_size=3, padding="same", activation="relu"),
                Conv2D(256, kernel_size=3, padding="same", activation="relu"),
                MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
                Dropout(0.5),
                Flatten(),
                Dense(256, activation="relu"),
                Dense(128, activation="relu"),
                Dense(num_classes, name="outputs"),
            ]
        )

        self.model.build((None, shape, length, 1))
        self.model.summary()
        self.output_layer = tf.keras.models.Model(
            inputs=self.model.layers[0].input,
            outputs=self.model.get_layer("outputs").output,
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

    def train(
        self,
        Xtrain,
        ytrain,
        Xval,
        yval,
        Xtune_train=None,
        ytune_train=None,
        Xtune_val=None,
        ytune_val=None,
    ):
        print("Start training......")
        start_time_train = time.time()
        train_pred, val_pred, tune_train_pred, tune_val_pred = [], [], [], []
        if self.cv == True:
            input = np.concatenate((Xtrain, Xval), axis=0)
            target = ytrain + yval
            for kfold, (train, val) in enumerate(
                KFold(n_splits=10, shuffle=True).split(input, target)
            ):
                train_pred, val_pred = [], []
                self.model.compile(
                    optimizer=self.optimizer,
                    loss=self.loss_object,
                    metrics=["accuracy"],
                )
                history = self.model.fit(
                    input[train],
                    target[train],
                    batch_size=self.batch_size,
                    epochs=self.epoch,
                    validation_data=(input[val], target[val]),
                )
        else:
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_object,
                metrics=["accuracy"],
            )
            history = self.model.fit(
                Xtrain,
                np.array(ytrain),
                batch_size=self.batch_size,
                epochs=self.epoch,
                validation_data=(Xval, np.array(yval)),
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

        end_time_train = time.time()
        elapsed_time_train = end_time_train - start_time_train
        print(f"Finish training for {self.method}.")
        print(f"Training time: {elapsed_time_train}s")

        if self.finetune == True:
            print("Start fine-tuning......")
            start_time_tune = time.time()
            for layer in self.model.layers[:-4]:
                layer.trainable = False
            for layer in self.model.layers[-3:]:
                layer.trainable = True
            for layer in self.model.layers:
                print("{}: {}".format(layer, layer.trainable))

            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_object,
                metrics=["accuracy"],
            )
            history = self.model.fit(
                Xtune_train,
                np.array(ytune_train),
                batch_size=self.batch_size,
                epochs=self.epoch,
                validation_data=(Xtune_val, np.array(ytune_val)),
            )

            tune_train_predictions = self.output_layer.predict(x=Xtune_train)
            tune_train_prob = tf.nn.softmax(tune_train_predictions)
            tune_train_pred += np.argmax(tune_train_prob, axis=1).tolist()
            tune_train_pred = np.array(tune_train_pred)

            tune_val_predictions = self.output_layer.predict(x=Xtune_val)
            tune_val_prob = tf.nn.softmax(tune_val_predictions)
            tune_val_pred += np.argmax(tune_val_prob, axis=1).tolist()
            tune_val_pred = np.array(tune_val_pred)

            elapsed_time_tune = start_time_tune - end_time_train
            print(f"Finish fine-tuning for {self.method}.")
            print(f"Fine-tuning time: {elapsed_time_tune}s")

            return (
                train_res,
                val_res,
                train_pred,
                val_pred,
                ytrain,
                yval,
                ytune_train,
                ytune_val,
                tune_train_pred,
                tune_val_pred,
            )

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
        start_time_test = time.time()

        test_pred = []
        test_loss, test_acc = self.model.evaluate(Xtest, np.array(ytest), verbose=2)
        test_predictions = self.output_layer.predict(x=Xtest)
        test_prob = tf.nn.softmax(test_predictions)  # probabilities
        test_pred += np.argmax(test_prob, axis=1).tolist()
        test_pred = np.array(test_pred)
        test_pred = np.array(test_pred)
        end_time_test = time.time()

        elapsed_time_test = end_time_test - start_time_test
        print("Finish testing.")
        print(f"Testing time: {elapsed_time_test}s")

        if not os.path.exists("outputs/speech/models/"):
            os.makedirs("outputs/speech/models")
        model_json = self.model.to_json()
        with open("outputs/speech/models/AlexNet.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("outputs/speech/models/AlexNet.weights.h5")

        return ytest, test_pred

# -*- encoding: utf-8 -*-
"""
@File    :   Xception.py
@Time    :   2024/07/24 05:03:09
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file encapsulates all implementation process for Xception in image emotion detection.
            The code refers to https://github.com/oarriaga/face_classification/tree/master.
"""


# here put the import lib
import os
import time
import numpy as np
import tensorflow as tf
from keras.regularizers import l2
from tensorflow.keras import Model
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import SeparableConv2D
from tensorflow.keras.layers import (
    MaxPooling2D,
    Activation,
    Conv2D,
    GlobalAveragePooling2D,
    BatchNormalization,
    Input,
)


class Xception(Model):
    def __init__(
        self,
        task,
        method,
        cc,
        h,
        num_classes,
        epochs=10,
        lr=0.001,
        batch_size=16,
    ):

        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.cc = cc
        self.method = method
        self.task = task
        self.batch_size = batch_size
        self.finetune = True if cc == "finetune" else False
        self.lr = lr
        self.epoch = epochs

        # network architecture
        self.model = self.mini_XCEPTION((h, h, 1), self.num_classes)
        self.model.build((None, h, h, 1))
        self.model.summary()

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )  # loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # optimizer

    # 48x48 input into mini-xception
    def mini_XCEPTION(self, input_shape, num_classes, l2_regularization=0.01):
        """
        description: This function describes the network architecture of modified Xception.
        param {*} self
        param {*} input_shape: input shape of Xception as 48x48x1 (resolution)
        param {*} num_classes: number of classes for emotion classification
        param {*} l2_regularization: regularization score
        return {*}: constructed model
        """
        regularization = l2(l2_regularization)
        img_input = Input(input_shape)

        # first convolutional layer
        x = Conv2D(
            8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False
        )(img_input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # second convolutional layer
        x = Conv2D(
            8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # four basic builidng blocks
        # module 1
        # right: residual connection
        residual = Conv2D(16, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # left: separable convolution
        x = SeparableConv2D(
            16,
            (3, 3),
            padding="same",
            depthwise_regularizer=regularization,
            use_bias=False,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(
            16,
            (3, 3),
            padding="same",
            depthwise_regularizer=regularization,
            use_bias=False,
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

        # concate
        x = layers.add([x, residual])

        # module 2
        residual = Conv2D(32, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(
            32,
            (3, 3),
            padding="same",
            depthwise_regularizer=regularization,
            use_bias=False,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(
            32,
            (3, 3),
            padding="same",
            depthwise_regularizer=regularization,
            use_bias=False,
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.add([x, residual])

        # module 3
        residual = Conv2D(64, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(
            64,
            (3, 3),
            padding="same",
            depthwise_regularizer=regularization,
            use_bias=False,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(
            64,
            (3, 3),
            padding="same",
            depthwise_regularizer=regularization,
            use_bias=False,
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.add([x, residual])

        # module 4
        residual = Conv2D(128, (1, 1), strides=(2, 2), padding="same", use_bias=False)(
            x
        )
        residual = BatchNormalization()(residual)
        x = SeparableConv2D(
            128,
            (3, 3),
            padding="same",
            depthwise_regularizer=regularization,
            use_bias=False,
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(
            128,
            (3, 3),
            padding="same",
            depthwise_regularizer=regularization,
            use_bias=False,
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        x = layers.add([x, residual])

        # output
        x = Conv2D(
            num_classes,  # take num of classes and do global average pooling
            (3, 3),
            kernel_regularizer=regularization,
            padding="same",
        )(x)
        x = GlobalAveragePooling2D()(x)  # take average of all values
        output = Activation("softmax", name="outputs")(x)

        model = Model(img_input, output)
        return model

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
        """
        description: This function is used for the entire process of training.
        param {*} self
        param {*} Xtrain: features of train set
        param {*} ytrain: labels of train set
        param {*} Xval: features of validation set
        param {*} yval: labels of validation set
        param {*} Xtune_train: features of train set for finetuning
        param {*} ytune_train: labels of train set for finetuning
        param {*} Xtune_val: features of validation set for finetuning
        param {*} ytune_val: labels of validation set for finetuning
        return {*}: accuracy and loss results, predicted labels, ground truth labels of train and validation
        """

        print("Start training......")
        start_time_train = time.time()
        train_pred, val_pred, tune_train_pred, tune_val_pred = [], [], [], []

        # compile and fit
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
        train_predictions = self.model.predict(x=Xtrain)
        train_pred += np.argmax(train_predictions, axis=1).tolist()
        train_pred = np.array(train_pred)
        train_res = {
            "train_loss": history.history["loss"],
            "train_acc": history.history["accuracy"],
        }

        val_predictions = self.model.predict(x=Xval)
        val_pred += np.argmax(val_predictions, axis=1).tolist()
        val_pred = np.array(val_pred)
        val_res = {
            "val_loss": history.history["val_loss"],
            "val_acc": history.history["val_accuracy"],
        }

        end_time_train = time.time()
        elapsed_time_train = end_time_train - start_time_train
        print(f"Finish training for {self.method}.")
        print(f"Training time: {elapsed_time_train}s")

        # finetuning
        if self.finetune == True:
            print("Start fine-tuning......")
            start_time_tune = time.time()

            # freezing
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
            tune_train_pred += np.argmax(tune_train_predictions, axis=1).tolist()
            tune_train_pred = np.array(tune_train_pred)

            tune_val_predictions = self.output_layer.predict(x=Xtune_val)
            tune_val_pred += np.argmax(tune_val_predictions, axis=1).tolist()
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

    def test(self, Xtest, ytest):
        """
        description: This function is used for the entire process of testing.
        param {*} self
        param {*} Xtest: features of test set
        param {*} ytest: labels of test set
        return {*}: predicted labels and ground truth of test dataset
        """

        print("Start testing......")
        start_time_test = time.time()

        test_pred = []
        test_loss, test_acc = self.model.evaluate(
            Xtest, np.array(ytest), verbose=2
        )  # evaluate
        test_predictions = self.model.predict(x=Xtest)
        test_pred += np.argmax(test_predictions, axis=1).tolist()
        test_pred = np.array(test_pred)
        test_pred = np.array(test_pred)

        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test
        print("Finish testing.")
        print(f"Testing time: {elapsed_time_test}s")

        return ytest, test_pred

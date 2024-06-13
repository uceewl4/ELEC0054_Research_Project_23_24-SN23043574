# -*- encoding: utf-8 -*-
"""
@File    :   CNN.py
@Time    :   2024/02/24 21:47:03
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    : This file includes all implementation process of CNN and multilabel setting.
"""

# here put the import lib
import os
import time
import numpy as np
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import Model
from tensorboardX import SummaryWriter  # used for nn curves visualization
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)

# from tensorflow.keras.layers import Activation, Convolution2D, Dropout, Conv2D
# from tensorflow.keras.layers import AveragePooling2D, BatchNormalization
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import SeparableConv2D
# from tensorflow.keras import layers
# from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import Activation, Convolution2D, Dropout, Conv2D
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from tensorflow.keras import layers
from keras.regularizers import l2


class Inception(Model):
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

        super(Inception, self).__init__()
        self.num_classes = num_classes
        self.cc = cc
        self.method = method
        self.task = task
        self.batch_size = batch_size
        self.finetune = True if cc == "finetune" else False

        self.model = self.mini_XCEPTION((h, h, 1), self.num_classes)
        self.model.build((None, h, h, 1))
        self.model.summary()
        # self.model.build((None, shape, length, 1))  # change

        # self.output_layer = tf.keras.models.Model(
        #     inputs=self.model.layers[0].input,
        #     outputs=self.model.get_layer("outputs").output,
        # )

        # objective function: sparse categorical cross entropy for mutliclass classification
        # notice that here the loss is calculated from logits, no need to set activation function for the output layer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.lr = lr
        self.epoch = epochs
        self.batch_size = batch_size
        self.method = method
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # 40x40 input into mini-xception
    def mini_XCEPTION(self, input_shape, num_classes, l2_regularization=0.01):
        regularization = l2(l2_regularization)

        # base, 7 classes
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
        # right
        residual = Conv2D(16, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
        residual = BatchNormalization()(residual)

        # left
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

        # out
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
        print("Start training......")
        start_time_train = time.time()
        train_pred, val_pred, tune_train_pred, tune_val_pred = [], [], [], []

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
        # train_predictions = self.output_layer.predict(x=Xtrain)
        train_predictions = self.model.predict(x=Xtrain)
        # train_prob = tf.nn.softmax(train_predictions)
        train_pred += np.argmax(train_predictions, axis=1).tolist()
        train_pred = np.array(train_pred)
        train_res = {
            "train_loss": history.history["loss"],
            "train_acc": history.history["accuracy"],
        }

        val_predictions = self.model.predict(x=Xval)
        # val_predictions = self.output_layer.predict(x=Xval)
        # val_prob = tf.nn.softmax(val_predictions)
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
            # tune_train_prob = tf.nn.softmax(tune_train_predictions)
            tune_train_pred += np.argmax(tune_train_predictions, axis=1).tolist()
            tune_train_pred = np.array(tune_train_pred)

            tune_val_predictions = self.output_layer.predict(x=Xtune_val)
            # tune_val_prob = tf.nn.softmax(tune_val_predictions)
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
        # test_predictions = self.output_layer.predict(x=Xtest)
        test_predictions = self.model.predict(x=Xtest)
        # test_prob = tf.nn.softmax(test_predictions)  # probabilities
        test_pred += np.argmax(test_predictions, axis=1).tolist()
        test_pred = np.array(test_pred)
        test_pred = np.array(test_pred)

        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test
        print("Finish testing.")
        print(f"Testing time: {elapsed_time_test}s")

        return ytest, test_pred

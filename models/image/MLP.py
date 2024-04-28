import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, Input
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from keras.models import Sequential
from tensorboardX import SummaryWriter  # used for nn curves visualization
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)


class MLP(Model):
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
        epochs=10,
        lr=0.001,
    ):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.features = features
        self.cc = cc
        self.method = method
        self.task = task
        self.dataset = dataset

        self.model = Sequential(
            [
                Dense(256, activation="relu", input_shape=(shape,)),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dropout(0.2),
                Dense(num_classes, name="outputs"),
            ]
        )

        self.model.build((None, 100, 100, 3))

        self.output_layer = tf.keras.models.Model(
            inputs=self.model.layers[0].input,
            outputs=self.model.get_layer("outputs").output,
        )

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

        return ytest, test_pred

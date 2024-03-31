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

from transformers import TFWav2Vec2Model
import time
import tensorflow as tf
from tensorflow.keras import Model
from keras.models import Sequential
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
    GlobalAveragePooling1D,
)


class Wav2Vec(Model):
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
        # self.input = Input(shape=(max_length,), dtype="float32")
        self.wav2vec2 = TFWav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base", apply_spec_augment=False, from_pt=True
        )
        self.pooling = GlobalAveragePooling1D()
        self.intermediate_layer_dropout = Dropout(0.5)
        print(num_classes)
        self.final_layer = Dense(num_classes, name="outputs")

        # self.basenet = BaseNet("facebook/wav2vec2-base", self.num_classes)(self.input)
        # self.basenet = BaseNet("facebook/wav2vec2-base", self.num_classes)
        # # Model
        # self.model = Sequential([self.input, self.basenet])
        # # self.model = tf.keras.Model(self.input, self.basenet)
        # self.model.summary()
        # self.output_layer = tf.keras.models.Model(
        #     inputs=self.input, outputs=self.model.get_layer("outputs").output
        # )

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

    def call(self, x):
        hidden_states = self.wav2vec2(x)[0]
        # print(hidden_states.shape)  # None,453,768
        pooled_state = self.pooling(hidden_states)
        # print(pooled_state.shape)  # none 768
        intermediate_state = self.intermediate_layer_dropout(pooled_state)
        # print(intermediate_state.shape)  # none, 768
        final_state = self.final_layer(intermediate_state)
        # print(final_state.shape)
        return final_state  # (32,8) RAVDESS

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
            # self.train_loss.reset_states()
            # self.train_accuracy.reset_states()
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            train_progress_bar = tqdm(range(len(train_ds)))

            for step, (train_images, train_labels) in enumerate(train_ds):
                with tf.GradientTape() as tape:  # 32,40
                    print(train_images.shape)  # 16,145172
                    print(np.expand_dims(train_images, 2).shape)  # 16,145172,1
                    # predictions = model(
                    #     np.expand_dims(train_images, 2), training=True
                    # )  # logits
                    predictions = model(train_images, training=True)  # logits
                    train_prob = tf.nn.softmax(predictions)  # probabilities  # 32,8
                    train_pred += np.argmax(train_prob, axis=1).tolist()
                    ytrain += np.array(train_labels).tolist()  # ground truth
                    loss = self.loss_object(train_labels, predictions)

                # backward propagation
                gradients = tape.gradient(loss, model.trainable_variables)
                # self.optimizer.apply_gradients(
                #     zip(gradients, model.trainable_variables)
                # )
                self.optimizer.apply_gradients(
                    (grad, var)
                    for (grad, var) in zip(gradients, model.trainable_variables)
                    if grad is not None
                )
                train_progress_bar.update(1)

                self.train_loss(loss)
                self.train_accuracy(train_labels, predictions)

                # validation
                if step % 50 == 0:
                    val_pred = []
                    yval = []
                    # self.val_loss.reset_states()
                    # self.val_accuracy.reset_states()
                    self.val_loss.reset_state()
                    self.val_accuracy.reset_state()
                    val_progress_bar = tqdm(range(len(val_ds)))

                    for val_images, val_labels in val_ds:
                        with tf.GradientTape() as tape:
                            # predictions = model(
                            #     np.expand_dims(val_images, axis=2), training=True
                            # )
                            predictions = model(val_images, training=True)
                            val_prob = tf.nn.sigmoid(predictions)
                            val_pred += np.argmax(val_prob, axis=1).tolist()
                            yval += np.array(val_labels).tolist()
                            val_loss = self.loss_object(val_labels, predictions)

                            self.val_loss(val_loss)
                            self.val_accuracy(val_labels, predictions)

                        # backward propagation
                        gradients = tape.gradient(val_loss, model.trainable_variables)
                        # self.optimizer.apply_gradients(
                        #     zip(gradients, model.trainable_variables)
                        # )
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
        # self.test_loss.reset_states()
        # self.test_accuracy.reset_states()
        self.test_loss.reset_state()
        self.test_accuracy.reset_state()
        test_progress_bar = tqdm(range(len(test_ds)))

        for test_images, test_labels in test_ds:
            # predictions = model(
            #     np.expand_dims(test_images, axis=2), training=False
            # )  # logits
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

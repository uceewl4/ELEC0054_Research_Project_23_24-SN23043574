# -*- encoding: utf-8 -*-
"""
@File    :   ViT.py
@Time    :   2024/07/24 05:18:55
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0054: Research Project
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file encapsulates all implementation process for ViT in image emotion detection.
            The code refers to https://github.com/nikhilroxtomar/Flower-Image-Classification-using-Vision-Transformer/tree/main.
"""

# here put the import lib

import time
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import Model, models
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Concatenate,
    LayerNormalization,
    Add,
    Dropout,
    MultiHeadAttention,
    Layer,
)


# create extra class token as global prediction for all patches
class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls


class ViT(Model):
    def __init__(
        self, task, dataset, method, cc, num_classes, epochs=10, lr=0.001, batch_size=32
    ):
        super(ViT, self).__init__()

        if dataset in ["CK", "FER"]:
            self.num_patches = 36  # 48x48x1 -- 8x8x1 -- 36 patches
            self.patch_size = 8  # 8x8x1
            self.inputs = Input((36, 8 * 8 * 1))
        elif dataset == "RAF":
            self.num_patches = 16  # 100x100x1 -- 25x25x1 -- 16 patches
            self.patch_size = 25  # 25x25x1
            self.inputs = Input((16, 25 * 25 * 1))

        self.channel = 1
        self.hidden_dim = 768
        self.num_layers = num_classes  # num of layers for transformer
        self.mlp_dim = 300  # num of dimension for MLP
        self.num_heads = 12  # multihead
        self.lr = lr
        self.epoch = epochs
        self.batch_size = batch_size
        self.method = method

        # patch + position embedding
        self.patch_embed = Dense(self.hidden_dim)(
            self.inputs
        )  # (None, 36, 768)  dimension for embedding
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)
        self.pos_embed = Embedding(
            input_dim=self.num_patches, output_dim=self.hidden_dim
        )(
            self.positions
        )  # (36, 768)
        self.embed = self.patch_embed + self.pos_embed  # (None, 36, 768)

        # class token
        self.token = ClassToken()(
            self.embed
        )  # the individual one for all patches to reduce bias
        self.hidden = Concatenate(axis=1)([self.token, self.embed])  # (None, 36+1, 768)

        # transformer block
        for _ in range(self.num_layers):  # 37,768
            self.hidden = self.transformer_encoder(self.hidden)

        # classification head
        self.hidden = LayerNormalization(epsilon=1e-7)(self.hidden)  # (None, 37, 768)
        self.hidden = self.hidden[:, 0, :]  # select the class token
        self.outputs = Dense(num_classes, activation="softmax")(self.hidden)

        # build the model
        self.model = Model(self.inputs, self.outputs)
        self.model.build((None, 36, 64))
        self.model.summary()

        self.output_layer = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.layers[-1].output
        )
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()  # loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # optimizer

    def mlp(self, x):
        """
        description: This method defines the MLP used in transformer encoder.
        param {*} self
        param {*} x: input
        return {*}: output
        """
        x = Dense(self.mlp_dim, activation="gelu")(x)
        x = Dropout(0.1)(x)
        x = Dense(self.hidden_dim)(x)
        x = Dropout(0.1)(x)
        return x

    def transformer_encoder(self, x):
        """
        description: This method defines basic transformer encoder block.
        param {*} self
        param {*} x: input
        return {*}: output
        """
        skip_1 = x
        x = LayerNormalization(epsilon=1e-7)(x)
        x = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.hidden_dim)(x, x)
        x = Add()([x, skip_1])  # resiual connection

        skip_2 = x
        x = LayerNormalization(epsilon=1e-7)(x)
        x = self.mlp(x)
        x = Add()([x, skip_2])

        return x

    def train(self, Xtrain, ytrain, Xval, yval):
        """
        description: This function includes entire training and validation process for ViT.
        param {*} self
        param {*} Xtrain: features of train set
        param {*} ytrain: labels of train set
        param {*} Xval: features of validation set
        param {*} yval: labels of validation set
        return {*}: accuracy and loss results, predicted labels, ground truth labels of train and validation
        """
        print(f"Start training for {self.method}......")
        start_time_train = time.time()
        train_pred, val_pred = [], []

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
        train_prob = self.model.predict(x=Xtrain)
        train_pred += np.argmax(train_prob, axis=1).tolist()
        train_pred = np.array(train_pred)
        train_res = {
            "train_loss": history.history["loss"],
            "train_acc": history.history["accuracy"],
        }

        val_prob = self.model.predict(x=Xval)
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

        return train_res, val_res, train_pred, ytrain, val_pred, yval

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

        test_res = self.model.evaluate(Xtest, np.array(ytest), verbose=2)
        test_prob = self.output_layer.predict(x=Xtest)
        test_pred += np.argmax(test_prob, axis=1).tolist()
        test_pred = np.array(test_pred)

        end_time_test = time.time()
        elapsed_time_test = end_time_test - start_time_test
        print("Finish testing.")
        print(f"Testing time: {elapsed_time_test}s")

        return test_res, test_pred, ytest

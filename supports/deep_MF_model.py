from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import argparse
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import numpy as np

class MyDmfModel(Model):

    def __init__(self, args):
        super(MyDmfModel, self).__init__()
        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.du1 = Dense(self.userLayer[0], activation='relu')
        self.du2 = Dense(self.userLayer[1], activation='relu')
        self.it1 = Dense(self.itemLayer[0], activation='relu')
        self.it2 = Dense(self.itemLayer[1], activation='relu')
    def call(self, x):
        user_out = self.du1(x[0])
        user_out = self.du2(user_out)
        item_out = self.it1(x[1])
        item_out = self.it2(item_out)
        norm_user_output = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(item_out), axis=1))
        y_ = tf.reduce_sum(input_tensor=tf.multiply(user_out, item_out), axis=1, keepdims=False) / (
                    norm_item_output * norm_user_output)
        y_ = tf.maximum(1e-6, y_)
        return y_

def main():
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[100, 20])
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[100, 20])
    args = parser.parse_args()
    train_x = np.random.random((100, 2, 100))
    train_x = np.array(train_x, dtype='float32')
    train_y = [np.ones((50, 1)), np.zeros((50, 1))]
    train_y = np.asarray(train_y, dtype='float32')
    train_y = np.reshape(train_y, (100, 1))
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).shuffle(50).batch(32)

    def loss_object(predicted_y, desired_y):
        losses = desired_y * tf.math.log(predicted_y) + (1 - desired_y) * tf.math.log(1 - predicted_y)
        return -tf.reduce_sum(input_tensor=losses)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    optimizer = tf.keras.optimizers.Adam()
    model = MyDmfModel(args)

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = loss_object(predictions, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        # 在下一个epoch开始时，重置评估指标
        train_loss.reset_states()

        for trainx, trainy in train_ds:
            train_step(trainx, trainy)
        template = 'Epoch {}, Loss: {}'
        print(template.format(epoch + 1,
                              train_loss.result()))
    pass


if __name__ == '__main__':
    main()

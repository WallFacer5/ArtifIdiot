from layers.input import Input
from layers.output import Output
from layers.dense import Dense
from losses import softmax_cross_entropy
from session import Session
from activations import relu
from samples.download_mnist import load
import tensorflow as tf


def hw3_a(lr, epochs, batch_size):
    x_train, y_train, x_test, y_test = load()
    x_train = x_train.astype(float)
    y_train = y_train.reshape([-1, 1])
    x_test = x_test.astype(float)
    y_test = y_test.reshape([-1, 1])
    il = Input(784)
    dl1 = Dense([il], 200, use_bias=True, activation=relu)
    dl2 = Dense([dl1], 50, use_bias=True, activation=relu)
    dl3 = Dense([dl2], 10, use_bias=True)
    ol = Output([dl3], 10, loss_function=softmax_cross_entropy, learning_rate=lr)
    sess = Session([ol], x_train, y_train, x_test, y_test)
    sess.train(epochs, batch_size)
    return sess


def hw3_b(lr, epochs, batch_size):
    x_train, y_train, x_test, y_test = load()
    x_train = x_train.astype(float) / 255
    y_train = y_train.reshape([-1, 1])
    x_test = x_test.astype(float) / 255
    y_test = y_test.reshape([-1, 1])
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, use_bias=True, activation='relu'),
        tf.keras.layers.Dense(50, use_bias=True, activation='relu'),
        tf.keras.layers.Dense(10, use_bias=True, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    model.evaluate(x_test, y_test)
    return model

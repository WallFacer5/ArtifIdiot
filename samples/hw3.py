from layers.input import Input
from layers.output import Output
from layers.dense import Dense
from losses import l2_loss, softmax_cross_entropy
from session import Session
from activations import relu, sigmoid, tanh
from samples.download_mnist import load
import numpy as np
import matplotlib.pyplot as plt
import random


def main(lr, epochs, batch_size):
    x_train, y_train, x_test, y_test = load()
    x_train = x_train.astype(float)
    y_train = y_train.reshape([-1, 1])
    il = Input(784)
    dl1 = Dense([il], 200, use_bias=True, activation=relu)
    dl2 = Dense([dl1], 50, use_bias=True, activation=relu)
    dl3 = Dense([dl2], 10, use_bias=True)
    ol = Output([dl3], 10, loss_function=softmax_cross_entropy, learning_rate=lr)
    sess = Session([ol], x_train, y_train, x_test, y_test)
    sess.train(epochs, batch_size)
    return sess

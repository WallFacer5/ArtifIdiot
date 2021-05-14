from layers.input import Input
from layers.output import Output
from layers.dense import Dense
from layers.conv2d import Conv2d
from layers.max_pool2d import MaxPool2d
from layers.flatten import Flatten
from losses import l2_loss, softmax_cross_entropy
from session import Session
from activations import relu, sigmoid, tanh
import numpy as np
import matplotlib.pyplot as plt
import random


def main0(lr):
    il = Input(1)
    dl1 = Dense([il], 8, use_bias=True, activation=relu)
    dl2 = Dense([dl1], 1, use_bias=True, activation=relu)
    ol = Output([dl2], 1, loss_function=l2_loss, learning_rate=lr)
    inputs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    outputs = np.array(inputs) * 1.5 - 0.8
    il.set_cur_input(inputs)
    ol.set_cur_y_true(outputs)
    for epoch in range(200):
        epoch += 1
        print('epoch: {}; loss: {}.'.format(epoch, ol.cur_loss))
        rd = random.randint(0, 7)
        il.set_cur_input(inputs[rd:rd + 1])
        ol.set_cur_y_true(outputs[rd:rd + 1])
        il.forward()
        dl1.forward()
        dl2.forward()
        ol.forward()
        ol.backward()
        dl2.backward()
        dl1.backward()
        il.backward()

    return il, dl1, dl2, ol, inputs, outputs


def main1(lr):
    il = Input(1)
    dl1 = Dense([il], 8, use_bias=True)
    dl2 = Dense([dl1], 1, use_bias=True)
    ol = Output([dl2], 1, loss_function=l2_loss, learning_rate=lr)
    inputs = list(map(np.random.standard_normal, [1] * 8))
    outputs = np.array(inputs) * 1.5 - 0.8 + np.random.standard_normal(1) / 100
    sess = Session([ol], inputs, outputs)
    return sess


def main(lr, epochs, batch_size):
    il = Input(1)
    dl1 = Dense([il], 16, use_bias=True)
    dl2 = Dense([dl1], 16, use_bias=True, activation=tanh)
    dl3 = Dense([dl2], 1, use_bias=True)
    ol = Output([dl3], 1, loss_function=l2_loss, learning_rate=lr)
    inputs = np.linspace(-5, 5, 51).reshape([-1, 1])
    # outputs = 2.3 * np.square(np.array(inputs)) - np.array(inputs) * 1.7 + 0.8
    # outputs = 1.5 * np.square(inputs) - 2.3 * inputs + 0.66
    outputs = np.sin(inputs)
    sess = Session([ol], inputs, outputs)
    sess.train(epochs, batch_size)
    plt.plot(sess.x, sess.y, color='blue', marker='o')
    plt.plot(sess.x, sess.cur_pred, color='red', marker='*')
    plt.show()
    return sess


def mlp_class(lr, epochs, batch_size):
    il = Input(8)
    dl1 = Dense([il], 8, use_bias=True, activation=relu)
    dl2 = Dense([dl1], 2, use_bias=True)
    ol = Output([dl2], 2, loss_function=softmax_cross_entropy, learning_rate=lr)
    inputs = np.random.standard_normal([8, 8])
    outputs = np.matmul(inputs, np.random.standard_normal([8, 1])) + np.random.standard_normal()
    outputs[outputs <= 0] = 0
    outputs[outputs > 0] = 0
    outputs = outputs.astype(int)
    sess = Session([ol], inputs, outputs)
    sess.train(epochs, batch_size)
    return sess


def ass2_p2(lr, epochs):
    il = Input(1)
    dl1 = Dense([il], 3, weights_initializer=np.ones)
    dl1.weights = np.array([[1, -1, 1]], dtype=np.float)
    dl2 = Dense([dl1], 3, weights_initializer=np.zeros, activation=sigmoid)
    dl2.weights = np.array([[1, 1, -1], [-1, 1, 1], [1, -1, -1]], dtype=np.float).transpose()
    ol = Output([dl2], 3, loss_function=l2_loss, learning_rate=lr)
    inputs = np.ones([1, 1])
    outputs = np.zeros([1, 3])
    sess = Session([ol], inputs, outputs, inputs, outputs)
    sess.train(epochs, 1)
    return sess


def conv_test():
    il = Input([5, 5, 1])
    cl = Conv2d([il], 2, kernel_size=[3, 3], strides=[2, 2], use_bias=True)
    v = np.array([[[1, 2, 3, 2, 1], [0, 1, 2, 1, 0], [1, 2, 3, 1, 0], [0, 1, 2, 0, 1], [1, 0, 0, 1, 2]],
                  [[1, 2, 3, 2, 1], [0, 1, 2, 1, 0], [1, 2, 3, 1, 0], [0, 1, 2, 0, 1], [1, 0, 0, 1, 2]]]).reshape(
        [2, 5, 5, 1])
    cl.set_cur_input(il, v)
    cl.forward()
    # print(cl.filters)
    # print(cl.biases)
    ol = Output([cl], 8)
    d = np.ones([2, 2, 2, 2])
    cl.append_cur_delta(ol, d)
    cl.backward()
    # print(cl.filters)
    # print(cl.biases)
    return il, cl, ol, v, d


def max_pool_test():
    il = Input([6, 6, 1])
    pl = MaxPool2d([il], [2, 2])
    v = np.array([[[1, 2, 3, 2, 1, 2], [0, 1, 2, 1, 0, 3], [1, 2, 3, 1, 0, 1], [0, 1, 2, 0, 1, 1], [1, 0, 0, 1, 2, 0],
                   [0, 1, 2, 0, 1, 1]],
                  [[1, 2, 3, 2, 1, 1], [0, 1, 2, 1, 0, 2], [1, 2, 3, 1, 0, 3], [0, 1, 2, 0, 1, 3], [1, 0, 0, 1, 2, 0],
                   [1, 2, 3, 2, 1, 1]]]).reshape([2, 6, 6, 1])
    print(v.reshape([2, 6, 6]))
    pl.set_cur_input(il, v)
    pl.forward()
    ol = Output([pl], 8)
    d = np.ones([2, 3, 3, 1])
    pl.append_cur_delta(ol, d)
    pl.backward()
    return il, pl, ol, v, d


def cnn_test():
    il = Input([6, 6, 1])
    cl = Conv2d([il], 2, kernel_size=[3, 3], strides=[1, 1], use_bias=True, activation=relu)
    pl = MaxPool2d([cl], [2, 2])
    fl = Flatten([pl])
    ol = Output([fl], 8)
    v = np.array([[[1, 2, 3, 2, 1, 2], [0, 1, 2, 1, 0, 3], [1, 2, 3, 1, 0, 1], [0, 1, 2, 0, 1, 1], [1, 0, 0, 1, 2, 0],
                   [0, 1, 2, 0, 1, 1]],
                  [[1, 2, 3, 2, 1, 1], [0, 1, 2, 1, 0, 2], [1, 2, 3, 1, 0, 3], [0, 1, 2, 0, 1, 3], [1, 0, 0, 1, 2, 0],
                   [1, 2, 3, 2, 1, 1]]]).reshape([2, 6, 6, 1])
    cl.set_cur_input(il, v)
    cl.forward()
    pl.forward()
    fl.forward()
    # print(cl.filters)
    # print(cl.biases)
    # print(cl.filters)
    # print(cl.biases)
    d = np.ones([2, 8])
    fl.append_cur_delta(ol, d)
    fl.backward()
    pl.backward()
    cl.backward()
    return il, cl, pl, fl, ol, v, d


if __name__ == '__main__':
    pass

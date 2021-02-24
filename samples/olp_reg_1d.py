from layers.input import Input
from layers.output import Output
from layers.dense import Dense
from losses import l2_loss
from session import Session
import numpy as np
import random


def main0(lr):
    il = Input(1)
    dl1 = Dense([il], 8, use_bias=True)
    dl2 = Dense([dl1], 1, use_bias=True)
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


def main(lr):
    il = Input(2)
    dl1 = Dense([il], 8, use_bias=True)
    dl2 = Dense([dl1], 1, use_bias=True)
    ol = Output([dl2], 1, loss_function=l2_loss, learning_rate=lr)
    inputs1 = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    inputs2 = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    inputs = np.append(inputs1, inputs2, axis=1)
    outputs = np.array(inputs1) * 1.5 - np.array(inputs2) * 2.1 + 0.8
    sess = Session([ol], inputs, outputs)
    return sess


if __name__ == '__main__':
    pass

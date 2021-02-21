from layers.input import Input
from layers.output import Output
from layers.dense import Dense
from losses import l2_loss
import numpy as np
import random


def main(lr):
    il = Input(1)
    dl1 = Dense([il], 8, use_bias=True)
    dl2 = Dense([dl1], 1, use_bias=True)
    ol = Output([dl2], 1, loss_function=l2_loss, learning_rate=lr)
    inputs = list(map(np.random.standard_normal, [1] * 8))
    outputs = np.array(inputs) * 1.5 - 0.8
    il.set_cur_input(inputs)
    ol.set_cur_y_true(outputs)
    epoch = 0
    while not ol.cur_loss or ol.cur_loss > 1e-10:
        epoch += 1
        print('epoch: {}'.format(epoch))
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


if __name__ == '__main__':
    pass

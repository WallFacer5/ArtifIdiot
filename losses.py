import numpy as np


def l2_loss(y_true, y_pred, learning_rate):
    # print('yt: {}, yp: {}'.format(y_true, y_pred))
    loss_value = 0.5 * np.sum(np.square(y_pred-y_true)) / y_true.shape[0]
    grad = (y_true - y_pred) * learning_rate
    # print('lv: {}, g: {}'.format(loss_value, grad))
    return loss_value, grad

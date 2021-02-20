import numpy as np


def l2_loss(y_true, y_pred, learning_rate):
    loss_value = 0.5 * np.mean(np.square(y_pred-y_true))
    grad = np.mean(y_true - y_pred) * learning_rate
    return loss_value, grad

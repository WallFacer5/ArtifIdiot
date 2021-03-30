import numpy as np


def l2_loss(y_true, y_pred, learning_rate):
    # print('yt: {}, yp: {}'.format(y_true, y_pred))
    loss_value = 0.5 * np.sum(np.square(y_pred-y_true)) / y_true.shape[0]
    if learning_rate == 0:
        return loss_value, 0, None  # 0 wait opt
    grad = (y_true - y_pred) * learning_rate
    # print('lv: {}, g: {}'.format(loss_value, grad))
    return loss_value, grad, None


def softmax_cross_entropy(y_true, y_pred, learning_rate):
    # print('yt: {}, yp: {}'.format(y_true, y_pred))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred -= np.max(y_pred)
    exp_scores = np.exp(y_pred)
    sum_scores = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / sum_scores
    # print(probs)
    try:
        cross_entropy = -np.log(probs[range(y_true.shape[0]), y_true.transpose()]).transpose()
        loss_value = np.mean(cross_entropy)
    except Exception:
        loss_value = None
    accuracy = np.mean(np.argmax(probs, axis=1) == y_true.transpose())
    if learning_rate == 0:
        return loss_value, 0, accuracy
    grad = probs
    grad[range(y_true.shape[0]), y_true.transpose()] -= 1
    grad /= -y_true.shape[0]
    grad *= learning_rate

    # print('grad: {}'.format(grad))

    return loss_value, grad, accuracy

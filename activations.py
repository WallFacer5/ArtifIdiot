import numpy as np
from constants import Directions


def relu(z, direction, back_grad=0):
    def forward(z_):
        return np.maximum(z_, 0)

    def backward(z_, back_grad_):
        grad = np.copy(back_grad_)
        grad[z_ > 0] *= 1
        grad[z_ <= 0] *= 0
        return grad

    return forward(z) if direction == Directions.forward else backward(z, back_grad)


def sigmoid(z, direction, back_grad=0):
    def forward(z_):
        return 1 / (1 + np.exp(-z_))

    def backward(z_, back_grad_):
        return back_grad_ * forward(z_) * (1 - forward(z_))

    return forward(z) if direction == Directions.forward else backward(z, back_grad)


def tanh(z, direction, back_grad=0):
    def forward(z_):
        return 2 * sigmoid(2 * z_, Directions.forward) - 1

    def backward(z_, back_grad_):
        return 4 * sigmoid(2 * z_, Directions.backward, back_grad_)

    return forward(z) if direction == Directions.forward else backward(z, back_grad)

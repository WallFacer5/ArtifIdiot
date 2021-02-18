import numpy as np
from layers.base import Layer


class Dense(Layer):
    def __init__(self, input_layers, output_shape, activation=None, use_bias=False,
                 weights_initializer=np.random.standard_normal,
                 bias_initializer=np.random.standard_normal):
        super().__init__(input_layers, output_shape, use_bias, weights_initializer, bias_initializer)

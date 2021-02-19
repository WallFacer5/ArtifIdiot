import numpy as np
from layers.base import Layer


class Dense(Layer):
    def __init__(self, input_layers, output_shape, activation=None, use_bias=False,
                 weights_initializer=np.random.standard_normal,
                 bias_initializer=np.random.standard_normal):
        super().__init__(input_layers, output_shape, use_bias, weights_initializer, bias_initializer)
        self.weights = weights_initializer((self.input_shapes[0], self.output_shape))
        self.delta = np.zeros((self.input_shapes[0], self.output_shape))
        if use_bias:
            self.weights = np.append(self.weights, [bias_initializer(output_shape)], axis=0)
            self.delta = np.append(self.delta, [np.zeros(output_shape)], axis=0)

    def forward(self):
        self.cur_inputs[0] = np.append(self.cur_inputs[0], np.ones((len(self.cur_inputs[0]), 1)), axis=1)
        self.cur_outputs = np.matmul(self.cur_inputs, self.weights)
        list(map(lambda ol: ol.set_cur_input(self, self.cur_outputs), self.output_layers.keys()))

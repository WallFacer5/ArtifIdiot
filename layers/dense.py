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
        self.cur_outputs = np.matmul(self.cur_inputs[0], self.weights)
        list(map(lambda ol: ol.set_cur_input(self, self.cur_outputs), self.output_layers.keys()))
        self.clear_cur_inputs_flags()

    def backward(self):
        # todo: filter inputs which are actually no need to go
        mean_inputs = np.mean(self.cur_inputs[0], axis=1)
        self.delta = np.sum(self.cur_deltas, axis=0)
        backward_delta = np.matmul(self.delta, np.transpose(self.weights))[:, :-1]
        list(map(lambda layer: layer.append_cur_delta(self, backward_delta), self.input_layers))
        self.weights += np.matmul(np.transpose(mean_inputs), self.delta)
        self.clear_cur_deltas_flags()

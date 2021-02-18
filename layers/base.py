import numpy as np


class Layer:
    def __init__(self, input_layers=[], output_shape=0, use_bias=False, weights_initializer=np.zeros,
                 bias_initializer=np.zeros):
        self.input_shapes = list(map(lambda l: l.get_output_shape(), input_layers))
        self.input_layers = input_layers
        self.output_shape = output_shape
        self.output_layers = []
        self.weights = list(map(weights_initializer, zip(self.input_shapes, [output_shape] * len(input_layers))))
        if use_bias:
            self.bias = [bias_initializer(output_shape)] * len(self.weights)
        self.delta = list(map(weights_initializer, zip(self.input_shapes, [output_shape] * len(input_layers))))
        self.cur_inputs = list(map(np.zeros, self.input_shapes))
        # pend
        self.starts = set()
        if input_layers:
            list(map(lambda layer: self.starts.update(layer.get_starts()), input_layers))
        else:
            self.starts.add(self)
        list(map(lambda layer: layer.append_output_layer(self), input_layers))

    def get_output_shape(self):
        return self.output_shape

    def get_output_layers(self):
        return self.output_layers

    def get_input_shape(self):
        return self.input_shapes

    def get_input_layers(self):
        return self.input_layers

    def get_starts(self):
        return self.starts

    def append_output_layer(self, _out_layer):
        self.output_layers.append(_out_layer)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

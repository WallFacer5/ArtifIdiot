import numpy as np


class Layer:
    def __init__(self, input_layers=[]):
        self.input_shapes = list(map(lambda l: l.get_output_shape(), input_layers))
        self.input_layers = input_layers
        self.output_shape = 0
        self.output_layers = []
        self.weights = np.array([])
        self.delta = np.zeros(shape=[0])
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


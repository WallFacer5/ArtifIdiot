import numpy as np


class Layer:
    def __init__(self, input_layers=[], output_shape=0, use_bias=False, weights_initializer=np.zeros,
                 bias_initializer=np.zeros):
        self.input_shapes = list(map(lambda l: l.get_output_shape(), input_layers))
        # self.input_layers = input_layers
        self.input_layers = dict(map(lambda t: (t[1], t[0]), enumerate(input_layers)))
        self.output_shape = output_shape
        self.output_layers = {}
        self.cur_inputs = [[]] * len(input_layers)
        self.cur_inputs_ready_flags = set()
        self.cur_deltas_ready_flags = set()
        self.cur_outputs = []
        self.delta = None
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

    def set_cur_input(self, _layer_ref, values):
        self.cur_inputs[self.input_layers[_layer_ref]] = values
        self.cur_inputs_ready_flags.add(self.input_layers[_layer_ref])

    def append_cur_delta(self, _layer_ref, values):
        self.delta += values
        self.cur_deltas_ready_flags.add(self.output_layers[_layer_ref])

    def append_output_layer(self, _out_layer):
        # self.output_layers.append(_out_layer)
        self.output_layers[_out_layer] = len(self.output_layers)

    def clear_cur_inputs_flags(self):
        self.cur_inputs_ready_flags = set()

    def clear_cur_deltas_flags(self):
        self.cur_deltas_ready_flags = set()

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    @property
    def can_forward(self):
        if len(self.cur_inputs_ready_flags) == len(self.input_layers):
            return True
        return False

    def can_backward(self):
        if len(self.cur_deltas_ready_flags) == len(self.output_layers):
            return True
        return False

import math


class Sin:
    def __init__(self):
        self.type = 'sin'

    def forward(self, value):
        self.value = value
        result = math.sin(value)
        print('Input: {}; Type: {}; Forward result: {}.'.format(value, self.type, result))
        return result

    def backward(self, grad):
        result = math.cos(self.value) * grad
        print('Input: {}; Gradient: {}; Type: {}; Backward result: {}.'.format(self.value, grad, self.type, result))
        return result


class Cos:
    def __init__(self):
        self.type = 'cos'

    def forward(self, value):
        self.value = value
        result = math.cos(value)
        print('Input: {}; Type: {}; Forward result: {}.'.format(value, self.type, result))
        return result

    def backward(self, grad):
        result = -math.sin(self.value) * grad
        print('Input: {}; Gradient: {}; Type: {}; Backward result: {}.'.format(self.value, grad, self.type, result))
        return result


class Mul:
    def __init__(self):
        self.type = 'multiply'

    def forward(self, v1, v2):
        self.values = [v1, v2]
        result = v1 * v2
        print('Inputs: {}; Type: {}; Forward result: {}.'.format(self.values, self.type, result))
        return result

    def backward(self, grad):
        result = self.values[1] * grad, self.values[0] * grad
        print('Inputs: {}; Gradient: {}; Type: {}; Backward result: {}.'.format(self.values, grad, self.type, result))
        return result


class Square:
    def __init__(self):
        self.type = 'square'

    def forward(self, value):
        self.value = value
        result = value * value
        print('Input: {}; Type: {}; Forward result: {}.'.format(value, self.type, result))
        return result

    def backward(self, grad):
        result = 2 * self.value * grad
        print('Input: {}; Gradient: {}; Type: {}; Backward result: {}.'.format(self.value, grad, self.type, result))
        return result


class Add:
    def __init__(self):
        self.type = 'add'

    def forward(self, v1, v2):
        self.values = (v1, v2)
        result = v1 + v2
        print('Inputs: {}; Type: {}; Forward result: {}.'.format(self.values, self.type, result))
        return result

    def backward(self, grad):
        result = (grad, grad)
        print('Inputs: {}; Gradient: {}; Type: {}; Backward result: {}.'.format(self.values, grad, self.type, result))
        return result


class AddC:
    def __init__(self, c):
        self.c = c
        self.type = 'add{}'.format(c)

    def forward(self, value):
        self.value = value
        result = value + self.c
        print('Input: {}; Type: {}; Forward result: {}.'.format(value, self.type, result))
        return result

    def backward(self, grad):
        print('Input: {}; Gradient: {}; Type: {}; Backward result: {}.'.format(self.value, grad, self.type, grad))
        return grad


class reciprocal:
    def __init__(self):
        self.type = 'reciprocal'

    def forward(self, value):
        self.value = value
        result = 1 / value
        print('Input: {}; Type: {}; Forward result: {}.'.format(value, self.type, result))
        return result

    def backward(self, grad):
        result = -grad / (self.value * self.value)
        print('Input: {}; Gradient: {}; Type: {}; Backward result: {}.'.format(self.value, grad, self.type, result))
        return result


if __name__ == '__main__':
    w1 = math.pi / 4
    x1 = 1
    w2 = -math.pi / 4
    x2 = 1
    print('w1={}, x1={}, w2={}, x2={}'.format(w1, x1, w2, x2))

    m1 = Mul()
    m2 = Mul()
    sin = Sin()
    cos = Cos()
    sqr = Square()
    add = Add()
    add_c = AddC(2)
    rec_x = reciprocal()

    print('Forward:')
    m1_fr = m1.forward(w1, x1)
    m2_fr = m2.forward(w2, x2)
    sin_fr = sin.forward(m1_fr)
    cos_fr = cos.forward(m2_fr)
    sqr_fr = sqr.forward(sin_fr)
    add_fr = add.forward(sqr_fr, cos_fr)
    add_c_fr = add_c.forward(add_fr)
    rec_fr = rec_x.forward(add_c_fr)

    print()

    print('Backward:')
    rec_br = rec_x.backward(1)
    add_c_br = add_c.backward(rec_br)
    add_br = add.backward(add_c_br)
    sqr_br = sqr.backward(add_br[0])
    sin_br = sin.backward(sqr_br)
    cos_br = cos.backward(add_br[1])
    w1_grad, x1_grad = m1.backward(sin_br)
    w2_grad, x2_grad = m2.backward(cos_br)

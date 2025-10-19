from functools import partialmethod
import numpy as np


# Start with 3 base classes
class Context:

    def __init__(self):

        self.saved_tensors = []

    def save_for_backward(self, *x):

        self.saved_tensors.extend(x)


class Tensor:

    def __init__(self, data, _children=()):

        self.data = data
        self.grad = np.zeros_like(data)
        # Internal variables used for autograd graph construction
        self._prev = set(_children)


class Function:

    def apply(self, arg, *x):

        ctx = Context()
        x = [self] + list(x)
        ret = Tensor(arg.forward(ctx, *[t.data for t in x]))

        return ret


def register(name, function):

    setattr(Tensor, name, partialmethod(function.apply, function))


class Dot(Function):

    @staticmethod
    def forward(ctx, input, weight):

        ctx.save_for_backward(input, weight)
        return input.dot(weight)

    @staticmethod
    def backward(ctx, grad_output):

        input, weight = ctx.saved_tensors
        grad_input = grad_output.dot(weight.T)
        grad_weight = grad_output.dot(input)

        return grad_input, grad_weight


register("dot", Dot)


# Maybe wrong
class Sum(Function):

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        return input.sum()

    @staticmethod
    def backward(ctx, grad_output):

        input = ctx.saved_tensors
        return grad_output * input


register("sum", Sum)

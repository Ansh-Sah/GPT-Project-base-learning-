
import numpy as np


def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.children = children
        self.backward_fn = lambda: None

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,
                     self.requires_grad or other.requires_grad,
                     (self, other))

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.shape)

        out.backward_fn = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,))
        out.backward_fn = lambda: (
            self.grad.__isub__(out.grad) if self.requires_grad else None
        )
        return out

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -Tensor(other))

    def __rsub__(self, other):
        return (other if isinstance(other, Tensor) else Tensor(other)) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                     self.requires_grad or other.requires_grad,
                     (self, other))

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(other.data * out.grad, self.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data * out.grad, other.shape)

        out.backward_fn = backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __pow__(self, power):
        out = Tensor(self.data ** power, self.requires_grad, (self,))

        def backward():
            if self.requires_grad:
                self.grad += power * self.data ** (power - 1) * out.grad

        out.backward_fn = backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad,
                     (self, other))

        def backward():
            if self.requires_grad:
                grad = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad += unbroadcast(grad, self.shape)

            if other.requires_grad:
                grad = np.swapaxes(self.data, -1, -2) @ out.grad
                other.grad += unbroadcast(grad, other.shape)

        out.backward_fn = backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,))

        def backward():
            if not self.requires_grad:
                return
            grad = out.grad

            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                for ax in sorted(a % self.data.ndim for a in axes):
                    grad = np.expand_dims(grad, ax)

            self.grad += np.broadcast_to(grad, self.shape)

        out.backward_fn = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            n = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis, keepdims) / n

    def exp(self):
        out = Tensor(np.exp(self.data), self.requires_grad, (self,))
        out.backward_fn = lambda: (
            self.grad.__iadd__(out.data * out.grad)
            if self.requires_grad else None
        )
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-12),
                     self.requires_grad, (self,))
        out.backward_fn = lambda: (
            self.grad.__iadd__(out.grad / (self.data + 1e-12))
            if self.requires_grad else None
        )
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0),
                     self.requires_grad, (self,))
        out.backward_fn = lambda: (
            self.grad.__iadd__((self.data > 0) * out.grad)
            if self.requires_grad else None
        )
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape),
                     self.requires_grad, (self,))
        out.backward_fn = lambda: (
            self.grad.__iadd__(out.grad.reshape(self.shape))
            if self.requires_grad else None
        )
        return out

    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))

        out = Tensor(self.data.transpose(*axes),
                     self.requires_grad, (self,))
        inverse = np.argsort(axes)

        out.backward_fn = lambda: (
            self.grad.__iadd__(out.grad.transpose(*inverse))
            if self.requires_grad else None
        )
        return out

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, index):
        out = Tensor(self.data[index], self.requires_grad, (self,))

        def backward():
            if self.requires_grad:
                np.add.at(self.grad, index, out.grad)

        out.backward_fn = backward
        return out

    def backward(self):
        if self.data.size != 1:
            raise ValueError("backward() requires a scalar loss")

        order = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build(child)
                order.append(node)

        build(self)
        self.grad = np.ones_like(self.data)

        for node in reversed(order):
            node.backward_fn()

    def item(self):
        return float(self.data.item())


def softmax(x, axis=-1):
    shifted = x - Tensor(
        np.max(x.data, axis=axis, keepdims=True)
    )
    exp_x = shifted.exp()
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    shifted = x - Tensor(
        np.max(x.data, axis=axis, keepdims=True)
    )
    return shifted - shifted.exp().sum(
        axis=axis, keepdims=True
    ).log()

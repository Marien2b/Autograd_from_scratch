import numpy as np
import random as rd


class Value:
    def __init__(self, data, parents = (), op = ''):
        self.data = data
        self.parents = set(parents)
        self.op = op
        self.grad = 0.0
        self.visited = 0
        self._backward = lambda : None

    @staticmethod
    def _coerce(other):
        if isinstance(other, Value):
            return other
        return Value(other)

    def __repr__(self):
        return f"Value(data = {self.data})"

    def __add__(self, other):
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), op = '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    #ça permet de rendre la classe plus robuste
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __neg__(self):
        out = Value(-self.data, (self,), op = '-')

        def _backward():
            self.grad += (-1) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = self._coerce(other)
        out = Value(self.data / other.data, (self, other), op = '/')

        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __mul__(self, other):
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), op = '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def tanh(self):
        out = Value(np.tanh(self.data), (self,), op = 'tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out


    def backward(self):
        tri_topos = []
        self.grad = 1
        

        def tri_topo(self):
            parents = self.parents
            for parent in parents:
                if parent.visited == 0:
                    parent.visited = 1
                    tri_topo(parent)
                    tri_topos.append(parent)
                    
                    
        tri_topo(self)
        tri_topos.append(self)
        for el in reversed(tri_topos):
            el._backward()
        for el in tri_topos:
            el.visited = 0

class Linear:
    def __init__(self, n_in, n_out):
        self.W = [[Value(rd.uniform(-1,1)) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [Value(rd.uniform(-1,1)) for _ in range(n_out)]

    def __call__(self, x):
        out = []
        for i in range(len(self.W)):
            acc = Value(0.0)
            row = self.W[i]
            for j in range(len(x)):
                acc = acc + row[j]*x[j]
            out.append(acc + self.b[i])
        return out

    def params(self):
        return ([p for row in self.W for p in row] + self.b)


class Optimizer:
    def __init__(self, params, lr):
        if callable(params):
            params = params()
        self.params = list(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            grad = param.grad
            param.data -= self.lr * grad

    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0


class MLP:
    def __init__(self, n_in, n_hidden1, n_hidden2, n_out):
        self.layers = []
        self.layers.append(Linear(n_in, n_hidden1))
        self.layers.append(Linear(n_hidden1, n_hidden2))
        self.layers.append(Linear(n_hidden2, n_out))

    def __call__(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            y = []
            for xi in x:
                y.append(xi.tanh())
            x = y
        x = self.layers[-1](x)
        return x

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

class Loss:
    def __init__(self, n):
        self.n = n

    def MSE(self, y_pred, y_train):
        diffs = [y_pred[i] - Value(y_train[i]) for i in range(self.n)]
        squared = [diff * diff for diff in diffs]
        return sum(squared, Value(0.0)) / self.n



    

        













    
    

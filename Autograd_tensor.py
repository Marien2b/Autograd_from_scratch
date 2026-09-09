from multiprocessing.dummy import Value

import numpy as np
import random as rd

class tensor:
    def __init__(self, data, parents=(), op=' ', requires_grad=True):
        self.data = np.asarray(data)
        self.shape = self.data.shape
        self.parents = set(parents)
        self.op = op
        self.grad = np.zeros(self.shape)
        self.requires_grad = requires_grad
        self.visited = 0
        self._backward = lambda : None

    def __repr__(self):
            return f"Value(data = {self.data})"

    @staticmethod
    def _coerce(other):
        if isinstance(other, tensor):
            return other
        return tensor(other)

    def _unbroadcast_grad(self, grad):
        #Premier cas: grad a plus de dimensions que self.data
        while grad.ndim > self.data.ndim:
            grad = grad.sum(axis=0)
        
        # Deuxième cas: grad a le même nombre de dimensions que self.data mais certaines dimensions sont égales à 1
        for i in range(self.data.ndim):
            if self.data.shape[i] == 1 and grad.shape[i] > 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad

    def unsqueeze(self, axis):
        requires_grad = self.requires_grad
        out = tensor(np.expand_dims(self.data, axis=axis), (self,), op='unsqueeze', requires_grad=requires_grad)

        if requires_grad:
            def _backward():
                self.grad += np.squeeze(out.grad, axis=axis)

            out._backward = _backward

        return out
    

    

    def __add__(self, other):
        other = self._coerce(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data + other.data, (self, other), op = '+', requires_grad = requires_grad)
        if requires_grad:   
            def _backward():
                self.grad += self._unbroadcast_grad(out.grad)
                other.grad += other._unbroadcast_grad(out.grad)
    
            out._backward = _backward

        return out

    def sum(self):
        requires_grad = self.requires_grad
        out = tensor(np.sum(self.data), (self,), op='sum',
                     requires_grad=requires_grad)

        if requires_grad:
            def _backward():
                self.grad += np.ones_like(self.data) * out.grad

            out._backward = _backward

        return out

    def __neg__(self):
        out = tensor(-self.data, (self,), op = '-')
        requires_grad = self.requires_grad
        if requires_grad:
        
            def _backward():
                self.grad += self._unbroadcast_grad((-1) * out.grad)
        
            out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-self._coerce(other))
    
    
    
    def __truediv__(self, other):
        other = self._coerce(other)
        out = tensor(self.data / other.data, (self, other), op = '/')
        requires_grad = self.requires_grad or other.requires_grad
        if requires_grad:

            def _backward():
                self.grad += self._unbroadcast_grad((1 / other.data) * out.grad)
                other.grad += other._unbroadcast_grad(
                    (-self.data / (other.data ** 2)) * out.grad
                )
    
            out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        return self._coerce(other) / self
    
    def __mul__(self, other):
        other = self._coerce(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data * other.data, (self, other), op = '*')
        if requires_grad:
            def _backward():
                self.grad += self._unbroadcast_grad(other.data * out.grad)
                other.grad += other._unbroadcast_grad(self.data * out.grad)
    
            out._backward = _backward

        return out

    def T(self):
        requires_grad = self.requires_grad
        out = tensor(self.data.T, (self,), op='T', requires_grad=requires_grad)

        if requires_grad:
            def _backward():
                self.grad += out.grad.T

            out._backward = _backward

        return out


    def tanh(self):
        requires_grad = self.requires_grad
        out = tensor(np.tanh(self.data), (self,), op = 'tanh')

        if requires_grad:
    
            def _backward():
                self.grad += (1 - out.data ** 2) * out.grad
    
            out._backward = _backward
        return out

    def __matmul__(self, other):
        other = self._coerce(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data @ other.data, (self, other), op = '@', requires_grad = requires_grad)

        if requires_grad:
            def _backward():
                self.grad += self._unbroadcast_grad(out.grad @ other.data.T)
                other.grad += other._unbroadcast_grad(self.data.T @ out.grad)

            out._backward = _backward

        return out

    def backward(self, grad = None):
        if grad is None:
            grad = np.ones_like(self.data)
        else:
            grad = np.array(grad)
            if grad.shape != self.data.shape:
                raise ValueError(f"Gradient shape {grad.shape} does not match tensor shape {self.data.shape}")
        tri_topos = []
        self.grad = grad
            
    
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
        #ligne:  n_out  colonne: n_in
        self.W = tensor(np.random.uniform(-1, 1, size=(n_out, n_in)), requires_grad=True)
        self.b = tensor(np.random.uniform(-1, 1, size=(n_out,)), requires_grad=True)

    def __call__(self, x):
        out = x @ self.W.T() + self.b
        return out

    def params(self):
        return [self.W, self.b]


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
            param.grad = np.zeros(param.shape)

class MLP:
    def __init__(self, n_in, n_hidden1, n_hidden2, n_out):
        self.layers = []
        self.layers.append(Linear(n_in, n_hidden1))
        self.layers.append(Linear(n_hidden1, n_hidden2))
        self.layers.append(Linear(n_hidden2, n_out))

    def __call__(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            y = x.tanh()
            x = y
        x = self.layers[-1](x)
        return x

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

class Loss:
    def __init__(self, n):
        self.n = n

    def MSE(self, y_pred, y_train):
        diff = y_pred - y_train
        squared = diff * diff
        return squared.sum() / self.n


    
    
        
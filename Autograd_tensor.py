import numpy as np
import random as rd

class tensor:
    def _init__(self, data, parents, op = ' ' , requires_grad = True):
        self.data = data
        self.shape = self.data.shape
        self.parents = parents
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
    

    

    def __add__(self, other):
        other = self._coerce(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = tensor(self.data + other.data, (self, other), op = '+', requires_grad = requires_grad)
        if requires_grad:   
            def _backward():
                self.grad += out.grad
                other.grad += out.grad
    
            out._backward = _backward

        return out

    def __neg__(self):
        out = tensor(-self.data, (self,), op = '-')
        
        def _backward():
            self.grad += (-1) * out.grad
        
        out._backward = _backward
        return out
    
    
    def __sub__(self, other):
        return self + (-self._coerce(other))
    
    def __rsub__(self, other):
        return self._coerce(other) - self
    
    
    
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
    

        
        
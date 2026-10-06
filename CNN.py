from multiprocessing.dummy import Value
import numpy as np
import random as rd

from Autograd_from_scratch.Autograd_tensor import *

class Conv:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weights = tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1, requires_grad=True)
        self.bias = tensor(np.random.randn(out_channels) * 0.1, requires_grad=True)

    

    def __call__(self, x):
        batch_size, in_channels, height, width = x.data.shape
        if self.padding > 0:
            x_padded = x.pad()
        else:
            x_padded = x
        out = tensor(np.zeros((batch_size, self.out_channels, height, width)), requires_grad=x.requires_grad)
        
        for i in range(self.out_channels): 
            for j in range(self.in_channels):
                kernel = self.weights.index(i,j)
                bias = self.bias.index(i)
                for a in range(0, height - self.kernel_size + 1, self.stride):
                    for b in range(0, width - self.kernel_size + 1, self.stride):
                        patch = x_padded.index(slice(None), j, slice(a, a + self.kernel_size), slice(b, b + self.kernel_size))
                        out.data[:, i, a//self.stride, b//self.stride] += (patch * kernel).sum() + bias
                        #implémenter stack
                        #implémenter un sum(axis)

                pass
        pass
from multiprocessing.dummy import Value
import numpy as np
import random as rd

from Autograd_from_scratch.Autograd_tensor import *

class Conv:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def pad(self, x): #on utilise une sytaxe numpy (merci numpy)
        x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode="constant"
        )
        return x_padded

    def __call__(self, x):
        return (self.pad(x))
from multiprocessing.dummy import Value
import numpy as np
import random as rd

from Autograd_tensor import *

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
        
        if self.padding > 0:
            x_padded = x.pad()
        else:
            x_padded = x
        self.batch_size, self.in_channels, height, width = x.data.shape

        image_out = []

        for i in range(self.out_channels):
            image = []
            for a in range(0, height - self.kernel_size + 1, self.stride):
                ligne = []
                for b in range(0, width - self.kernel_size + 1, self.stride):
                    patch = x_padded.index(slice(None), slice(None), slice(a, a + self.kernel_size), slice(b, b + self.kernel_size))
                    value = (patch * self.weights.index(i)).sum(axis = (1, 2, 3)) + self.bias.index(i) #Patch est de dimension (batch, in_channels, kernel_size, kernel_size) et weights.index(i) est de dimension (in_channels, kernel_size, kernel_size)
                    ligne.append(value)
                valeur_ligne = tensor.stack(ligne, axis=1) # On stack les valeurs de la ligne selon l'axe des colonnes
                image.append(valeur_ligne)
            valeur_image = tensor.stack(image, axis=1) # On stack les lignes selon l'axe des lignes
            image_out.append(valeur_image)
        return tensor.stack(image_out, axis=1) # On stack les images selon l'axe des canaux de sortie



    def params(self):
        return [self.weights, self.bias]

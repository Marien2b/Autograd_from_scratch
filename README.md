# Autograd_from_scratch

## Mircrograd scalaire

J'ai commencé par suivre l'architecture "classique" du micrograd, puis m'en suis un peu écarté. Globalement très peu perfomant, je ne peux pas décemment faire une classification MNIST.

Néanmoins le micrograd apprend: un MLP(2, 4, 4, 1) apprend très facilement l'opération XOR.

## Autograd Vectoriel

On reprend le principe du micrograd mais la classe Value est remplacée par la classe tensor qui sont des vecteurs numpys. Toute la difficulté réside alors surtout dans la gestion des shape numpy, les règles de la chaine restent globalement les mêmes (petite subtilité par rapport à la multiplication matricielle).

La grosse partie a été de comprendre vraiment l'unbroadcasting et la gestion des gradients dans ce cas.



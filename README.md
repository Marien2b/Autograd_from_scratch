# Autograd_from_scratch

Le but de ce projet est de réaliser un mini-framework de deep-learning que j'améliore au fur et à mesure (pour se rapprocher le plus possible de Pytorch)
J'ai commencé par un Mircrograd scalaire pour comprendre les idées d'un autograd, puis j'ai modifié ce Micrograd pour obtenir un AutoGrad vectoriel.

## Mircrograd scalaire

J'ai commencé par suivre l'architecture "classique" du micrograd, puis m'en suis un peu écarté. Globalement très peu perfomant, je ne peux pas décemment faire une classification MNIST.

Néanmoins le micrograd apprend: un MLP(2, 4, 4, 1) apprend très facilement l'opération XOR.


## Autograd Vectoriel

On reprend le principe du micrograd mais la classe Value est remplacée par la classe tensor qui sont des vecteurs numpys. Toute la difficulté réside alors surtout dans la gestion des shape numpy, les règles de la chaine restent globalement les mêmes (petite subtilité par rapport à la multiplication matricielle).

La partie vraiment difficile par rapport au Micrograd a été de comprendre vraiment l'unbroadcasting et la gestion des gradients dans ce cas. 

## CNN

Encore une fois la partie vraiment technique est la gestion des dimensions. Je dois raffiner la classe Tensor pour qu'elle supporte le padding (ajouter des 0 autour de mon image), l'extraction par indices et la concaténation/somme par rapport à des axes. Une fois les fonctions stack, padding et index ajoutées à la classe tensor ça se déroule assez bien (voir le fichier Maths.md pour toutes les précisions techniques)






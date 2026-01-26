# CNN Models

The sources of the architecture is recorded in each file, but they are also listed here for clarity. All the code grew out of this helpful article:

https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef

but diverged as I learnt more about Keras and early CNN architectures.

As recorded elsewhere, all these models were pulled together (I hesitate to say "developed", since so much of them was drawn from the work of others; I just tried to clarify and provide some uniformity to, hopefully, aid understanding) for teaching purposes, and so err on the side of ease of reading with more comments than are probably comfortable for most.

## VGGs

All models drawn from:
K. Simonyan & A, Zisserman (2015) Very Deep Convolutional Networks for Large-scale Image Recognition, 3rd International Conference on Learning Representations.

### VGG11

The lightest model, included here to overlap with the models in basic-vision. We have a 64 filter layer, a 128 filter layer, 2 x 256 filter layers, 2 x 512 filter layers and another 2 x 512 filter layers. There is max-pooling between each block of layers.

The original model had two 4096 fully-connected layers followed by a 100 unit FC layer for handling ImageNet. The models in basic-vision, dealing with 10-class problems, used a 120 unit FC layer, an 84 unit FC layer and then the 10 unit output layer (taken from the description of LeNet5 in https://pabloinsente.github.io/the-convolutional-network).

### VGG16

The paper contains 11, 13, 16 and 19 layer models. This is the 16-layer model with only 3 x 3 convolutions (there is a version with some 1 x 1 covolution layers). There are the same number of blocks as in the 11 model, but with more filters in each (and less than in the 19 layer model). In partiocular, we have: 2 x 64 filter layers, 2 x 128 filter layers, 3 x 256 filter layers, 3 x 512 filter layers and another 3 x 512 filter layers, followed by a 120 unit FC layer, an 84 unit FC layer and then the 10 unit output layer.

### VGG19

The deepest model from the paper. We have 2 x 64 filter layers, 2 x 128 filter layers, 4 x 256 filter layers, 4 x 512 filter layers and another 4 x 512 filter layers, followed by the usual 120 unit FC layer, an 84 unit FC layer and then the 10 unit output layer.

## ResNets

As introduced by:
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770–778).

Note that this paper shows (again) the ambiguuity over what counts as "deep" in deep learning. The VGG paper refers to a 19 weight-layer network as "very deep", while a year later this paper includes a 152-layer network and doesn't apparently consider that "very" deep (perhaps because, as they report, it has lower complexity than VGG19)

### ResNet

A simple ResNet to illustrate how residuals can be handled. Has only 6 convolutional layers: 16 filters, 2 x 32 filters, 2 x 64 filters and 128 filters, followed by a 10 unit FC layer. Despite being in the "needs CUDA" collection, this actually runs ok on a laptop CPU.

### Resnet18
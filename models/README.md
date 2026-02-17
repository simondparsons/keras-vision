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

The paper contains 11, 13, 16 and 19 layer models. This is the 16-layer model with only 3 x 3 convolutions (there is a version with some 1 x 1 covolution layers). There are the same number of blocks as in the 11 model, but with more filters in each (and less than in the 19 layer model). In particular, we have: 2 x 64 filter layers, 2 x 128 filter layers, 3 x 256 filter layers, 3 x 512 filter layers and another 3 x 512 filter layers, followed by a 120 unit FC layer, an 84 unit FC layer and then the 10 unit output layer.

### VGG19

The deepest model from the paper. We have 2 x 64 filter layers, 2 x 128 filter layers, 4 x 256 filter layers, 4 x 512 filter layers and another 4 x 512 filter layers, followed by the usual 120 unit FC layer, an 84 unit FC layer and then the 10 unit output layer.


## ResNets

As introduced by:
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 770–778).

Note that this paper shows (again) the ambiguity over what counts as "deep" in deep learning. The VGG paper refers to a 19 weight-layer network as "very deep", while a year later this paper includes a 152-layer network and doesn't apparently consider that "very" deep (perhaps because, as they report, it has lower complexity than VGG19).

From an implementation perspective we have to move away from the Sequential class definition approach to defining networks (see https://machinelearningmastery.com/three-ways-to-build-machine-learning-models-in-keras/) because the residual pass-through creates a branching network. The follwoing networks show how to do this sing first the functional approach to defining networks and then the class-based approach (which I think is neatest fo rlarg enetworks but less clear).

### SimpleResNet

As the name suggests, a simple ResNet to illustrate how residuals can be handled. A good place to start understanding the code. Has only 6 convolutional layers: 16 filters, 2 x 32 filters, 2 x 64 filters and 128 filters, followed by a 10 unit FC layer. Despite being in the "needs CUDA" collection, this actually runs ok on a laptop CPU.

Note that it will only handle grey scale images. The issue is easy to fix (as in ResNet18), but I couldn't be bothered to backpropagate the fix once I figured it out.

### ResNet18

The simplest model from He et al (see Table 1) with 18 weight layers: a 64 filter layer (with 7x7 filters) then 4 x 64 filters, 4 x 128 filters, 4 x 256 filters, 4 x 512 filters, followed by a 10 unit FC layer. As described in the code, this downsamples less than the original ResNet18 so that it works on smaller images.

### SimpleClassResNet

My model of ResNet18 is as complex a network as I want to implement without some kind of hierarchical structure. (Arguably it is already too complex, though I think it is instructive to have to deal with the complexity, especially in terms of the different cases of residual combination.)  A cleaner structure provides a Residual class that implements a residual block as a Keras Layer, and this code gives a simple version that fits the structure of ResNets from He at al. while remaining relatively small. It would be a ResNet10 if we chose to classify it that way, with 2 x 64 filters (1 residual layer), 2 x 128 filters, 2 x 256 filters, 2 x 512 filters plus the initial 64 filter layer and the FC output layer. Several blocks downsample just to show we can do that.

### ResNet34

The prototypical model from He et al (see Table 1 and Figure 2) with 34 weight layers, built using the Residual class. The structure has a 64 filter layer (with 7x7 filters) then 6 x 64 filters (3 residual layers), 8 x 128 filters, 12 x 256 filters, 6 x 512 filters, followed by a 10 unit FC layer. As described in the code, this downsamples less than the original ResNet34 so that it works on smaller images.

### Further ResNets

This is as much as I needed for teaching, so I stopped here. Deeper ResNets need the bottleneck building block from (He et al. 2016, Figure 5) which I may get to at some point.
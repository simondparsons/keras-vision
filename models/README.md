# CNN Models

The sources of the architecture is recorded in each file, but they are also listed here for clarity. All the code grew out of this helpful article:

https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef

but diverged as I learnt more about Keras and early CNN architectures.

As recorded elsewhere, all these models were pulled together (I hesitate to say "developed", since so much of them was drawn from the work of others; I just tried to clarify and provide some uniformity to, hopefully, aid understanding) for teahcing purposes, and so err on the side of ease of reading with more comments than are probably comfortable for most.

## VGGs

## VGG11

The lightest model from:
K. Simonyan & A, Zisserman Very Deep Convolutional Networks for Large-scale Image Recognition, 3rd International Conference on Learning Representations, 2015.

and included here to overlap with the models in basic-vision. We have a 64 filter layer, a 128 filter layer, 2 x 256 filter layers, 2 x 512 filter layers and another 2 x 512 filter layers, followed by the usual 120 unit FC layer, an 84 unit FC layer and then the 10 unit output layer.

## VGG16

## VGG19
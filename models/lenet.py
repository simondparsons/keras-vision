# lenet.py
#
# Simon Parsons
# 25-04-25
#
# This holds the LeNet reconstruction from:
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# and the structure is confirmed here:
#
# https://pabloinsente.github.io/the-convolutional-network
#
# but re-worked as a funcntional model
#
# though the latter has many fewer filters at each convolution, but an additional
# dense layer at the end.

from models.backbone import Backbone 
from tensorflow.keras import layers, models

class LeNet(Backbone):
    # Here we set up some constants that we will use to do this across
    # the various layers.
    kernel_shape = 3, 3  # train 3x3 kernels across all Conv layers
    activation = 'relu'  # use Rectified Linear Unit activiation functions
    pool_shape = 2, 2    # reduce dimensionality by 2 x 2 pooling
    dropout_rate = 0.5   # drop 50% of neurons
    padding = 'same'     # maintain the shape of feature maps per layer
    strides = 1          # do not downsample via stride

    nfilters_hidden1 = 32  # Start with 32 convolution filters to train
    nfilters_hidden2 = 64  # end with twice as many filters to train next

    # In functional style. I don't like the repeated use of the same
    # variable name, but it seems to be the standard.
    def buildModel(self):
        # Create the input layer to understand the shape of each image
        # and batch-size
        input = layers.Input(shape=self.img_shape)

        # Add the first convolution layer. This has 32 filters
        x = layers.Conv2D(
            filters=self.nfilters_hidden1,
            kernel_size=self.kernel_shape,
            activation=self.activation,
            padding=self.padding,
            strides=self.strides)(input)
        
        # Reduce the dimensionality after the first Conv-layer w/ MaxPool2D
        x = layers.MaxPooling2D(
            pool_size=self.pool_shape)(x)

        # Add the next convolution layer. This has 64 filters
        x = layers.Conv2D(
            filters=self.nfilters_hidden2,
            kernel_size=self.kernel_shape,
            activation=self.activation,
            padding=self.padding,
            strides=self.strides)(x)

        # Reduce the dimensionality after the second Conv-layer w/ MaxPool2D
        x = layers.MaxPooling2D(
            pool_size=self.pool_shape)(x)

        # Convert the 2D outputs to a 1-D vector in preparation for
        # label prediction
        x = layers.Flatten()(x)

        # Dropout 50% of the neurons from the Conv+Flatten layers to regulate
        x = layers.Dropout(
            rate=self.dropout_rate)(x)

        # Compute the weighted-logistic for each possible label in
        # one-hot encoding
        output = layers.Dense(
            units=self.num_classes,
            activation="softmax")(x)

        self.model = models.Model(input, output ,name='LeNet_Reconstruction')

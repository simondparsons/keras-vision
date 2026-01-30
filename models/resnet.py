# resnet.py
#
# Simon Parsons
# 26-01-20
#
# A merge of ideas from
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
# and
# https://gist.github.com/FirefoxMetzger/6b6ccf4f7c344459507e73bbd13ec541rting from:
#
# to create a simple ResNet, as defined in
#
# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
# for image recognition. In Proceedings of the IEEE conference on
# computer vision and pattern recognition (pp. 770–778).
#
# Note that this structure is that of FirefoxMetzger, not one from He
# at al. and does not run because of the issues reported in
# Residual.py

from models.backbone import Backbone 
from models.residual import Residual
from tensorflow.keras import layers, models

class ResNet(Backbone):

    # Set up some constants that we will use to do this across the various
    # layers.
    kernel_shape = (3, 3)   # use 3 x 3 kernels throughout.
    activation = 'relu'     # use Rectified Linear Unit activiation functions
    pool_shape = (2, 2)     # reduce dimensionality by 4 = 2 x 2 in pooling layers
    dropout_rate = 0.5      # drop 50% of neurons
    padding = 'same'        # maintain the shape of feature maps per layer
    strides = 1             # don't downsample via stride

    # Define how we will build the model
    model = models.Sequential(name='Simple_ResNet_Classes')

    # By my count this is a 13 layer model: one intro convolutional
    # layer, 5 residual blocks for another 10 convolutional layers,
    # and then two FC layers at the end.
    def buildModel(self):
        # Pre-processing
        #
        # Create the input layer to understand the shape of each image and batch-size 
        self.model.add(layers.Input(shape=self.img_shape,
                                    name='Input_Layer'))
        
        # Add a rescaling layer to convert the inputs to fall in the range (-1, 1).
        # https://machinelearningmastery.com/image-augmentation-with-keras-preprocessing-layers-and-tf-image/
        self.model.add(layers.Rescaling(1/127.5,
                                        offset=-1))
        
        # Now the network proper.
        #
        # Initial convolutional layer. Keeping convolution and
        # activation separate in ResNet style.
        self.model.add(layers.Conv2D(32, (3, 3),
                                     padding='same'))
        self.model.add(layers.Activation('relu'))

        # A stack of residual blocks
        self.model.add(Residual(32,(3,3)))
        self.model.add(Residual(32,(3,3)))
        self.model.add(Residual(32,(3,3)))
        self.model.add(Residual(32,(3,3)))
        self.model.add(Residual(32,(3,3)))

        # Output layers
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512))
        self.model.add(layers.Activation(self.activation))
        self.model.add(layers.Dropout(self.dropout_rate))
        self.model.add(layers.Dense(self.num_classes))
        self.model.add(layers.Activation('softmax'))

        self.model.build()#(None, 32, 32, 3))

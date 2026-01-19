# vgg11.py
#
# Simon Parsons
# 25-04-28
#
# This starts from the original VGG paper:
#
# K. Simonyan & A, Zisserman Very Deep Convolutional Networks for
# Large-scale Image Recognition, 3rd International Conference on
# Learning Representations, 2015
#
# where VGG11 is the least complex/deep of the models. The 11 comes from
# the 8 convolutional layers and the 3 fully connected (Dense in Keras
# terms) layers, making 11 in all.
#
# Note that there is no batch normalization in the original.
#
# The last three layers in the original were fully conncted with 4096,
# 4096 and 1000 units (the last feeding into softmax to do the
# classification). I have trimmed this down to the output stage that:
#
# https://pabloinsente.github.io/the-convolutional-network
#
# gives for LeNet, given we are working on the same
# classifications. This has the same number of FC layers, but has 120,
# 84 and 10 units in them.

from models.backbone import Backbone 
from tensorflow.keras import layers, models

class VGG11(Backbone):
    # First we set up some constants that we will use to do this
    # across the various layers.
    kernel_shape = (3, 3)  # train 3x3 kernels across all Conv layers
    activation = 'relu'    # use Rectified Linear Unit activiation functions
    pool_shape = (2, 2)    # reduce dimensionality by 2 x 2 pooling
    dropout_rate = 0.5     # drop 50% of neurons
    padding = 'same'       # maintain the shape of feature maps per layer
    strides = 1            # do not downsample via stride

    # Filters in the convolution layers.
    nfilters_hidden1 = 64   # Start with 64 convolution filters to train
    nfilters_hidden2 = 128  # Then twice as many filters to train
    nfilters_hidden3 = 256  # The doubling the number of filters once more.
    nfilters_hidden4 = 512  # The final layers all have 512 filters
    nfilters_hidden5 = 512  

    # Define how we will build the model
    model = models.Sequential(name='VGG11')

    def buildModel(self):
        # Create the input layer to understand the shape of each image and batch-size 
        self.model.add(
            layers.Input(
                shape=self.img_shape,
                # batch_size=batch_size,
                name='Image_Batch_Input_Layer',
            )
        )

        # Add a rescaling layer to convert the inputs to fall in the range (-1, 1).
        # https://machinelearningmastery.com/image-augmentation-with-keras-preprocessing-layers-and-tf-image/
        self.model.add(
            layers.Rescaling(
                1/127.5,
                offset=-1
            )
        )

        # Add the first convolution layer with 64 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden1,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_11'
            )
        )
        # A batch normalization layer
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_1'
            )
        )
        # Reduce the dimensionality after the first Conv-layer w/
        # MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="MaxPool2D_Layer_1"
            )
        )

        # Add the next convolution block, again 1 layer this time with
        # 128 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden2,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_21'
            )
        )
        # Another batch normalization layer. 
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_2'
            )
        )
        # Reduce the dimensionality after the second Conv-layer w/
        # MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="MaxPool2D_Layer_2"
            )
        )

        # Add the third convolution block. This has 2 convolution
        # layers, each with 256 filters.
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_31'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_32'
            )
        )
        # Another batch normalization layer. 
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_3'
            )
        )
        # Reduce the dimensionality after the third Conv-layer w/
        # MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
               name="MaxPool2D_Layer_3"
            )
        )

        # Add the fourth convolution block. This has 2 convolution
        # layers, each with 512 filters.
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden4,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_41'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden4,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_42'
            )
        )
        # Another batch normalization layer. 
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_4'
            )
        )
        # The size after 4 layers is just 2x2, so don't pool any
        # more.withthis layer in place, the final feature map is just
        # 1x1 if we start with 32x32 images.
        #
        # Need to remove for 28x28 images
        #
        # Reduce the dimensionality after the fourth Conv-layer w/
        # MaxPool2D
        #self.model.add(
        #    layers.MaxPooling2D(
        #        pool_size=self.pool_shape,
        #       name="MaxPool2D_Layer_4"
        #    )
        #)

        # Add the fifth convolution block. This has 2 convolution
        # layers, each with 512 filters.
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden5,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_51'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden5,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Conv2D_Layer_52'
            )
        )
        # Another batch normalization layer. 
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_5'
            )
        )
        # Reduce the dimensionality after the fourth Conv-layer w/
        # MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
               name="MaxPool2D_Layer_5"
            )
        )

        # Convert the 2D outputs to a 1-D vector in preparation for
        # label prediction
        self.model.add(
            layers.Flatten(
                name="Flatten_from_Conv2D_to_Dense"
            )
        )
        # Dropout 50% of the neurons from the Conv+Flatten layers to
        # regulate
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Flatten_to_Dense"
            )
        )

        # The output stage in all the VGG models had 4096, 4096 and
        # 1000 units to predict 1000 classes. We use the same output
        # stage as in the "Dense" LeNet networks.
        self.model.add(
            layers.Dense(
                units=120,
                activation=self.activation,
                name="Dense_Layer_1"
            )
        )
        # Dropout 50% between Dense layers
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Dense_to_Dense_1"
            )
        )
        self.model.add(
            layers.Dense(
                units=84,
                activation=self.activation,
                name="Dense_Layer_2"
            )
        )
        # Dropout 50% between Dense layers
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Dense_to_Dense_2"
            )
        )
        # Compute the weighted-logistic for each possible label in
        # one-hot encoding
        self.model.add(
            layers.Dense(
                units=self.num_classes, #10 classes in MNIST etc
                activation="softmax",
                name="n-Dimensional_Logistic_Output_Layer"
            )
        )

# vgg19.py
#
# Starting from:
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# tries to implement the VGG19 classifier of ImageNet fame, based on:
#
# K. Simonyan & A, Zisserman Very Deep Convolutional Networks for
# Large-scale Image Recognition, 3rd International Conference on
# Learning Representations, 2015
#

from models.backbone import Backbone 
from tensorflow.keras import layers, models

class VGG19(Backbone):

    # Set up some constants that we will use to do this across the various
    # layers.
    kernel_shape = (3, 3)   # use 3 x 3 kernels throughout.
    activation = 'relu'     # use Rectified Linear Unit activiation functions
    pool_shape = (2, 2)     # reduce dimensionality by 4 = 2 x 2 in pooling layers
    dropout_rate = 0.5      # drop 50% of neurons
    padding = 'same'        # maintain the shape of feature maps per layer
    strides = 1             # don't downsample via stride

    nfilters_hidden1 = 64   # Increase filters as we go further through the 
    nfilters_hidden2 = 128  # backbone.
    nfilters_hidden3 = 256  
    nfilters_hidden4 = 512
    nfilters_hidden5 = 512

    # Define how we will build the model
    model = models.Sequential(name='VGG19')

    def buildModel(self):
        # Pre-processing
        #
        # Create the input layer to understand the shape of each image and batch-size 
        self.model.add(
            layers.Input(
                shape=self.img_shape,
                # batch_size=batch_size,
                name='Image_Batch_Input_Layer',
            )
        )
        # Resize images to 224 x 224
        #self.model.add(
        #    layers.Resizing(
        #        height = 224,
        #        width = 224
        #    )
        #)
        # Add a rescaling layer to convert the inputs to fall in the range (-1, 1).
        # https://machinelearningmastery.com/image-augmentation-with-keras-preprocessing-layers-and-tf-image/
        self.model.add(
            layers.Rescaling(
                1/127.5,
                offset=-1
            )
        )

        # Now the network proper.
        #
        # Add the first pair of convolution layers. These have 64 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden1,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_11'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden1,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_12'
            )
        )
        # Batch normalization. Not in the original VGG19 because batch
        # norm came later.
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_1'
            )
        )
        # Reduce the dimensionality w/ MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="Max_Pool_Layer_1"
            )
        )

        # Add the next pair of convolution layers, these have 128 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden2,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_21'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden2,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_22'
            )
        )
        # Batch normalization
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_2'
            )
        )
        # Reduce the dimensionality w/ MaxPool2D
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="Max_Pool_Layer_2"
            )
        )

        # Now four convolution layers with 256 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_31'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_32'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_33'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden3,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_34'
            )
        )
        # Batch normalization
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_3'
            )
        )
        # And now max pooling for the third time.
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="Max_Pool_Layer_3"
            )
        )

        # Now four convolution layers with 512 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden4,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_41'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden4,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_42'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden4,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_43'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden4,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_44'
            )
        )
        # Batch normalization
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_4'
            )
        )
        # And now max pooling for the fourth time.
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="Max_Pool_Layer_4"
            )
        )
        
        # Another four convolution layers with 512 filters
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden5,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_51'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden5,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_52'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden5,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_53'
            )
        )
        self.model.add(
            layers.Conv2D(
                filters=self.nfilters_hidden5,
                kernel_size=self.kernel_shape,
                activation=self.activation,
                padding=self.padding,
                strides=self.strides,
                name='Convolution_Layer_54'
            )
        )
        # Batch normalization
        self.model.add(
            layers.BatchNormalization(
                name='Batch_Norm_Layer_5'
            )
        )
        # And now max pooling for the fifth time.
        self.model.add(
            layers.MaxPooling2D(
                pool_size=self.pool_shape,
                name="Max_Pool_Layer_5"
            )
        )
        
        # Feed through fully connected layers.
        self.model.add(
            layers.Dense(
                units= 120, # 4096
                activation=self.activation,
                name="First_Dense_layer"
            )
        )
        # VGG19 does not use Dropout, but we will :-) Dropout 50% of the
        # neurons between the FC layers.
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Dense_to_Dense"
            )
        )
        self.model.add(
            layers.Dense(
                units= 84, # 4096
                activation=self.activation,
                name="Second_Dense_layer"
            )
        )
        # Dropout 50% of the neurons between the FC layer and output.
        self.model.add(
            layers.Dropout(
                rate=self.dropout_rate,
                name="Dropout_from_Dense_to_Output"
            )
        )

        # Convert the 2D outputs to a 1-D vector in preparation for label prediction
        self.model.add(
            layers.Flatten(
                name="Flatten_from_Conv2D_to_Dense"
            )
        )
        # Compute the weighted-logistic for each possible label
        self.model.add(
            layers.Dense(
                units=self.num_classes,
                activation="softmax",
                name="n-Dimensional_Logistic_Output_Layer"
            )
        )
        

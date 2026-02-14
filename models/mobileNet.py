# mobileNet.py
#
# Simon Parsons
# 26-02-05
#
# Merging the approach I used for simple CNN models, which was based on:
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# and drawing heavily on:
# https://github.com/arthurdouillard/keras-mobilenet/blob/master/mobilenet.py
#
# we have an implmentation of MobileNetv1as defined in:
#
# Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand,
# T., Andreetto, M. and Adam, H., (2017). Mobilenets: Efficient
# convolutional neural networks for mobile vision applications. arXiv
# preprint arXiv:1704.04861.
#
# arthurdouillard's code is worth looking at for the neat way that it
# builds the structural blocks of the architecture as functions, but
# as with other models in this set, I have opted for (what I think is)
# the clearest way of doing things.

# For comparison with other models, this is equivalent to an N layer
# model built with standard n x n (n > 1) convolutions.

from models.backbone import Backbone
from tensorflow.keras import layers, models

class MobileNet(Backbone):
    # Set up some constants that we will use to do this across the
    # various layers. These are broadly the same as for other models
    # in this series.
    kernel_shape1 = (1, 1)  # Use 3 x 3 and (1 x 1) kernels throughout.
    kernel_shape3 = (1, 1)
    dropout_rate = 0.5      # Drop 50% of neurons
    padding = 'same'        # Maintain the shape of feature maps per layer
    strides = (1, 1)        # Two stride values for downsampling (_ds)
    strides_ds = (2, 2)     # and not.
    nfilters_1 = 32         # Increase filters as we go further through the 
    nfilters_2 = 64         # backbone. A wider range of sizes than we
    nfilters_3 = 128        # have seen before.
    nfilters_4 = 256
    nfilters_5 = 512
    nfilters_6 = 1024

    # Define how we will build the model
    model = models.Sequential(name='MobileNet')

    # The structure of the model is taken from Howard et al. (2017),
    # Table 1, with blocks structured as in Figure 3, right.
    def buildModel(self):
        
        # Standard keras stuff. Create an input layer and rescale.
        self.model.add(layers.Input(shape=(self.img_shape)))
        self.model.add(layers.Rescaling(1/127.5,offset=-1))

        # Standard convolution layer
        self.model.add(layers.Conv2D(filters=self.nfilters_1,
                              kernel_size=self.kernel_shape3,
                              strides=self.strides_ds,
                              padding=self.padding,
                              name='cnv2d'))
        self.model.add(layers.BatchNormalization(name='bn_1'))
        self.model.add(layers.ReLU(name='relu_1'))

        # There are now 13 blocks of depthwise+pontwise convolution,
        # each equivalent in some sense to a single conventional
        # convolutional layer. 6 of these scale from 32 to 512
        # filters, then there are 5 with 512 fliters, then 2 scalling
        # up to 1024.
        
        # Block1
        #
        # Depthwise convolution followed by pointwise convolution with
        # BN an activitation after each (Howard et al 2016, Figure 3).
        #
        # Depthwise. Note we don't specify the number of filters since
        # this is fixed by the number of input channels.
        self.model.add(layers.DepthwiseConv2D(kernel_size=self.kernel_shape3,
                                              strides=self.strides,
                                              padding='same',
                                              name='b1_dw'))
        self.model.add(layers.BatchNormalization(name='b1_bn1'))
        self.model.add(layers.Activation('relu', name='b1_act1'))
        #
        # Pointwise, 64 filters.
        self.model.add(layers.Conv2D(filters=self.nfilters_2,
                                     kernel_size=self.kernel_shape1,
                                     strides=self.strides,
                                     padding=self.padding,
                                     name='b1_cnv'))
        self.model.add(layers.BatchNormalization(name='b1_bn2'))
        self.model.add(layers.Activation('relu', name='b1_act2'))

        # Block2
        #
        # Original model downsamples in the depthwise convolution
        self.model.add(layers.DepthwiseConv2D(kernel_size=self.kernel_shape3,
                                              strides=self.strides_ds,
                                              padding='same',
                                              name='b2_dw'))
        self.model.add(layers.BatchNormalization(name='b2_bn1'))
        self.model.add(layers.Activation('relu', name='b2_act1'))
        #
        # 128 filters.
        self.model.add(layers.Conv2D(filters=self.nfilters_3,
                                     kernel_size=self.kernel_shape1,
                                     strides=self.strides,
                                     padding=self.padding,
                                     name='b2_cnv'))
        self.model.add(layers.BatchNormalization(name='b2_bn2'))
        self.model.add(layers.Activation('relu', name='b2_act2'))

        # Block3
        #
        self.model.add(layers.DepthwiseConv2D(kernel_size=self.kernel_shape3,
                                              strides=self.strides,
                                              padding='same',
                                              name='b3_dw'))
        self.model.add(layers.BatchNormalization(name='b3_bn1'))
        self.model.add(layers.Activation('relu', name='b3_act1'))
        #
        # 128 filters.
        self.model.add(layers.Conv2D(filters=self.nfilters_3,
                                     kernel_size=self.kernel_shape1,
                                     strides=self.strides,
                                     padding=self.padding,
                                     name='b3_cnv'))
        self.model.add(layers.BatchNormalization(name='b3_bn2'))
        self.model.add(layers.Activation('relu', name='b3_act2'))

        # Block4
        #
        # The original downsamples here, but that is not possible for
        # our little imagaes.
        self.model.add(layers.DepthwiseConv2D(kernel_size=self.kernel_shape3,
                                              strides=self.strides,
                                              padding='same',
                                              name='b4_dw'))
        self.model.add(layers.BatchNormalization(name='b4_bn1'))
        self.model.add(layers.Activation('relu', name='b4_act1'))
        #
        # 256 filters.
        self.model.add(layers.Conv2D(filters=self.nfilters_4,
                                     kernel_size=self.kernel_shape1,
                                     strides=self.strides,
                                     padding=self.padding,
                                     name='b4_cnv'))
        self.model.add(layers.BatchNormalization(name='b4_bn2'))
        self.model.add(layers.Activation('relu', name='b4_act2'))

        # Block5
        #
        # No downsampling in the original.
        self.model.add(layers.DepthwiseConv2D(kernel_size=self.kernel_shape3,
                                              strides=self.strides,
                                              padding='same',
                                              name='b5_dw'))
        self.model.add(layers.BatchNormalization(name='b5_bn1'))
        self.model.add(layers.Activation('relu', name='b5_act1'))
        #
        # 256 filters.
        self.model.add(layers.Conv2D(filters=self.nfilters_4,
                                     kernel_size=self.kernel_shape1,
                                     strides=self.strides,
                                     padding=self.padding,
                                     name='b5_cnv'))
        self.model.add(layers.BatchNormalization(name='b5_bn2'))
        self.model.add(layers.Activation('relu', name='b5_act2'))

        # Block6
        #
        # The original downsamples here.
        self.model.add(layers.DepthwiseConv2D(kernel_size=self.kernel_shape3,
                                              strides=self.strides,
                                              padding='same',
                                              name='b6_dw'))
        self.model.add(layers.BatchNormalization(name='b6_bn1'))
        self.model.add(layers.Activation('relu', name='b6_act1'))
        #
        # 512 filters.
        self.model.add(layers.Conv2D(filters=self.nfilters_5,
                                     kernel_size=self.kernel_shape1,
                                     strides=self.strides,
                                     padding=self.padding,
                                     name='b6_cnv'))
        self.model.add(layers.BatchNormalization(name='b6_bn2'))
        self.model.add(layers.Activation('relu', name='b6_act2'))

        #
        # More layers in here
        #
        
        # Output layers.
        # Average pooling with a 7x7 pool is in the original
        self.model.add(layers.AveragePooling2D(name="avpool",
                                               pool_size=(7, 7)))
        self.model.add(layers.Flatten(name="flatten_to_dense"))   
        # Howard at al. do not use Dropout, but I do for continuity with
        # the other models.
        self.model.add(layers.Dropout(rate=self.dropout_rate,
                                      name="dropout_from_dense_to_output"))
        self.model.add(layers.Dense(self.num_classes,
                                    name='output_fc_layer',
                                    activation='softmax'))

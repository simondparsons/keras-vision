# simple-resnet.py
#
# Simon Parsons
# 26-01-26
#
# Merging the approach I used for simple CNN models, which was based on:
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# and the implementation of ResNet provided by:
# https://github.com/aliprf/tutorials_1_residual_network/blob/master/network_model.py
#
# to give a version of ResNet as defined in:
#
# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
# for image recognition. In Proceedings of the IEEE Conference on
# Computer Vision and Pattern Recognition (pp. 770–778).
#
# Note that this structure is that of that of aliprf, not one from He
# at al. (it is much shallower), but works as a demo. Some of the
# detail differs from aliprf since I think he got some of it
# wrong. Also, unlike Aliprf's code, this copes with residuals x that
# have a different shape to F(x).
#
# This version also only works with gray-scale images. It relies on
# the fact that the Add layer can combine a 1 channel tensor with a 32
# channel tensor(apparently) in a way that it can't combine a 3
# channel tensor with a 32 channel one. Applying a suitable
# convolution layer (as ResNet18 et al. do) will make this problem go
# away.

from models.backbone import Backbone
from models.padLayer import PadLayer
from tensorflow.keras import layers, models

class SimpleResNet(Backbone):
    # Set up some constants that we will use to do this across the
    # various layers. These are broadly the same as for other models
    # in this series.
    kernel_shape = (3, 3)   # use 3 x 3 kernels throughout.
    pool_shape = (2, 2)     # reduce dimensionality by 4 = 2 x 2 in pooling layers
    dropout_rate = 0.5      # drop 50% of neurons
    padding = 'same'        # maintain the shape of feature maps per layer
    strides = 1             # don't downsample via stride. Here this
                            # is essential to ensure that input and
                            # b1_bn_2 (He et al's x and F(x)) have the
                            # same dimensions.

    nfilters_1 = 16         # Increase filters as we go further through the 
    nfilters_2 = 32         # backbone.
    nfilters_3 = 64  
    nfilters_4 = 128
    
    # This defines how we will build the model. Unlike the VGG models here,
    # and others in basic-vision, we don't start with a sequential
    # model and add layers. Instead we create each layer explicitly,
    # and name its input. Then the model is built using the
    # models.Model constructor.
    #
    # Other points of comparison are that we define a separate
    # activation layer since it is separated from the convolution (see
    # below). Other features of the convolution are kept the same as
    # in previous models in this series: we don't, for example, use He
    # initialization from:
    #
    # He, K., Zhang, X., Ren, S. and Sun, J., 2015. Delving deep into
    # rectifiers: Surpassing human-level performance on ImageNet
    # classification. In Proceedings of the IEEE International
    # Conference on Computer Vision (pp. 1026-1034).
    #
    # We also keep the number of filters the same within a given block,
    # but generally deepen these as we add blocks.
    def buildModel(self):
        
        input = layers.Input(shape=(self.img_shape))

        # Block 1
        #
        # Defines a single residual block as in He et al's Figure
        # 2. This is conv2D (weight layer in He et al's words), BN,
        # activation (relu), conv2D, BN, add identity, activation.
        b1_cnv2d_1 = layers.Conv2D(filters=self.nfilters_1,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b1_cnv2d_1')(input)
        # "We adopt Batch normalization right after each convolution
        # and before activation" (He at al. 2016, Section 3.4). This
        # detail is not included in Figure 2.
        b1_bn_1 = layers.BatchNormalization(name='b1_bn_1')(b1_cnv2d_1)
        b1_relu_1 = layers.ReLU(name='b1_relu_1')(b1_bn_1)
        b1_cnv2d_2 = layers.Conv2D(filters=self.nfilters_2,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b1_cnv2d_2')(b1_relu_1)
        b1_bn_2 = layers.BatchNormalization(name='b1_bn_2')(b1_cnv2d_2)
        # Here is the residual bit
        b1_add = layers.add([input, b1_bn_2])
        b1_relu_2 = layers.ReLU(name='b1_relu_2')(b1_add)

        # Block 2
        b2_cnv2d_1 = layers.Conv2D(filters=self.nfilters_2,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b2_cnv2d_1')(b1_relu_2)
        b2_bn_1 = layers.BatchNormalization(name='b2_bn_1')(b2_cnv2d_1)
        b2_relu_1 = layers.ReLU(name='b2_relu_1')(b2_bn_1)
        b2_cnv2d_2 = layers.Conv2D(filters=self.nfilters_3,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b2_cnv2d_2')(b2_relu_1)
        b2_bn_2 = layers.BatchNormalization(name='b2_bn_2')(b2_cnv2d_2)
        # Here we need to pad our identity tensor because we have
        # expanded the number of channels by changing the number of
        # filters in the second Conv layer.
        identity2 = PadLayer()(b1_relu_2, channels1=self.nfilters_2, channels2=self.nfilters_3)
        b2_add = layers.add([identity2, b2_bn_2])
        b2_relu_2 = layers.ReLU(name='b2_relu_2')(b2_add)

        # Block 3
        b3_cnv2d_1 = layers.Conv2D(filters=self.nfilters_3,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b3_cnv2d_1')(b2_relu_2)
        b3_bn_1 = layers.BatchNormalization(name='b3_bn_1')(b3_cnv2d_1)
        b3_relu_1 = layers.ReLU(name='b3_relu_1')(b3_bn_1)
        b3_cnv2d_2 = layers.Conv2D(filters=self.nfilters_4,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b3_cnv2d_2')(b3_relu_1)
        b3_bn_2 = layers.BatchNormalization(name='b3_bn_2')(b3_cnv2d_2)
        identity3 = PadLayer()(b2_relu_2, channels1=self.nfilters_3, channels2=self.nfilters_4)
        b3_add = layers.add([identity3, b3_bn_2])
        b3_relu_2 = layers.ReLU(name='b3_relu_2')(b3_add)
        
        # Output stage: Average pooling then flatten and put through a
        # Dense layer with the same number of output units as classes.
        avg_pool = layers.GlobalAveragePooling2D()(b3_relu_2)
        flatten = layers.Flatten(name="flatten_to_dense")(avg_pool)
        # He at al. do not use Dropout, but I do for continuity with
        # the other models.
        dropout = layers.Dropout(rate=self.dropout_rate,
                                 name="Dropout_from_Dense_to_Output")(flatten)
        output = layers.Dense(self.num_classes,
                              name='output_fc_layer',
                              activation='softmax')(dropout)

        # This is where we explicitly build the model.
        self.model = models.Model(input, output)#(name='Simple_ResNet')




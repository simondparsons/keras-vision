# resnet18.py
#
# Simon Parsons
# 27-01-26
#
# Merging the approach I used for simple CNN models, which was based on:
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# and the implementation of ResNet provided by:
# https://github.com/aliprf/tutorials_1_residual_network/blob/master/network_model.py
#
# to give a version of ResNet18 as defined in:
#
# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
# for image recognition. In Proceedings of the IEEE Conference on
# Computer Vision and Pattern Recognition (pp. 770–778).
#
# This was written with teahcing in mind, so always aims for clarity
# over efficiency. I'm sure that there are many ways it can be optimized.
#
# Note that this general approach started from aliprf but uses the
# ResNet18 architecture from He at al, (2016, Table 1). Some of the
# detail differs from aliprf since I think they got some of it wrong
# (particularly around the handling of residuals). As a result, unlike
# Aliprf's code, this includes residuals x that have a different shape
# to F(x).
#
# Using functions for each layer would undoubtedly be neater (and
# nicely hierarchical) but would hide some of the complexity which I
# think is instructive, particularly around handing the three
# different cases of residuals. These are: 1) x and F(x) have the same
# shape; 2) we increased the number of filters in the new block so
# that F(x) has more channels than x; 3) we started the block with a
# convolutional layer with a stride > 1, so F(x) is a smaller "image"
# than x.
#
# Having said that, I would not fancy creating a deep ResNet than this
# without building in some form of abstraction.
#
# Note that in the networks in He at al., cases 2 and 3 are always
# combined. In Figure 3 each group of same-coloured blocks starts with
# a convolutional layer that downsamples by setting strides=2, and
# also increases the number of filters. (The downsampling is implicit
# in Table 1 where ResNet18 is specified.). Since I was working with
# smaller images, this version of ResNet18 doesn't downsample at every
# block. It would normally make sense to do the down-sampling earlier
# in the network, but since this code was developed for teaching, I
# wanted to introduce the cases one by one.

# TODO: Build a version with residual blocks defined using a Residual
# class as in:
# https://gist.github.com/FirefoxMetzger/6b6ccf4f7c344459507e73bbd13ec541rting from:
#
# A starter version is in residual.py, but it doesn't yet work.
#

from models.backbone import Backbone
from models.padLayer import PadLayer
from models.downSampleLayer import DSLayer
from tensorflow.keras import layers, models#, initializers

class ResNet18(Backbone):
    # Set up some constants that we will use to do this across the
    # various layers. These are broadly the same as for other models
    # in this series.
    kernel_shape = (3, 3)   # Use 3 x 3 kernels throughout.
    pool_shape = (2, 2)     # Reduce dimensionality by 4 = 2 x 2 in pooling layers
    dropout_rate = 0.5      # Drop 50% of neurons
    padding = 'same'        # Maintain the shape of feature maps per layer
    strides = (1, 1)        # Two stride values for downsampling (_ds)
    strides_ds = (2, 2)     # and not since orignal ResNet downsampled
                            # at the start of block from the second
                            # block onwards.
    nfilters_1 = 64         # Increase filters as we go further through the 
    nfilters_2 = 128        # backbone.
    nfilters_3 = 256  
    nfilters_4 = 512
    
    # This defines how we will build the model. Unlike the VGG models
    # here, and others in basic-vision, we don't start with a
    # sequential model and add layers. Instead we create each layer
    # explicitly, using the KERAS functional API and name its
    # input. Then the model is built using the models.Model
    # constructor. Keras figures out which elements are needed and
    # connects them together.
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
    def buildModel(self):

        # Standard keras stuff. Create an input layer and rescale.
        input = layers.Input(shape=(self.img_shape))
        rescale = layers.Rescaling(1/127.5,offset=-1)(input)
        
        # conv1 (see He el al. (2016), Table 1)
        #
        # Initial Conv2D and maxPooling
        #
        # As in He et al. this is a 7x7 filter with stride 2.
        cnv2d = layers.Conv2D(filters=self.nfilters_1,
                              kernel_size=(7, 7),
                              strides=self.strides_ds,
                              padding=self.padding,
                              name='cnv2d')(rescale)
        maxpool = layers.MaxPooling2D(pool_size=self.pool_shape,
                                      name="maxpool")(cnv2d)

        # conv2_x
        #
        # Block 1
        #
        # Defines a single residual block as in He et al's Figure
        # 2. This is conv2D (weight layer in He et al's words), BN,
        # activation (relu), conv2D, BN, add identity, activation.
        b1_cnv2d_1 = layers.Conv2D(filters=self.nfilters_1,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b1_cnv2d_1')(maxpool)
        # "We adopt Batch normalization right after each convolution
        # and before activation" (He at al. 2016, Section 3.4). This
        # detail is not included in Figure 2.
        b1_bn_1 = layers.BatchNormalization(name='b1_bn_1')(b1_cnv2d_1)
        b1_relu_1 = layers.ReLU(name='b1_relu_1')(b1_bn_1)
        b1_cnv2d_2 = layers.Conv2D(filters=self.nfilters_1,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b1_cnv2d_2')(b1_relu_1)
        b1_bn_2 = layers.BatchNormalization(name='b1_bn_2')(b1_cnv2d_2)
        # Residual layer, case 1). The simple case because we have not
        # changed the number of filters yet. maxpool is x, b1_bn_2 is
        # F(x).
        b1_add = layers.add([maxpool, b1_bn_2], name='b1_residual')
        b1_relu_2 = layers.ReLU(name='b1_relu_2')(b1_add)

        # Block 2
        b2_cnv2d_1 = layers.Conv2D(filters=self.nfilters_1,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b2_cnv2d_1')(b1_relu_2)
        b2_bn_1 = layers.BatchNormalization(name='b2_bn_1')(b2_cnv2d_1)
        b2_relu_1 = layers.ReLU(name='b2_relu_1')(b2_bn_1)
        b2_cnv2d_2 = layers.Conv2D(filters=self.nfilters_1,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b2_cnv2d_2')(b2_relu_1)
        b2_bn_2 = layers.BatchNormalization(name='b2_bn_2')(b2_cnv2d_2)
        # Residual layer, case 1).
        b2_add = layers.add([b1_relu_2, b2_bn_2], name='b2_residual')
        b2_relu_2 = layers.ReLU(name='b2_relu_2')(b2_add)

        # conv3_x
        #
        # Block 3
        #
        # Number of filters increases
        b3_cnv2d_1 = layers.Conv2D(filters=self.nfilters_2,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b3_cnv2d_1')(b2_relu_2)
        b3_bn_1 = layers.BatchNormalization(name='b3_bn_1')(b3_cnv2d_1)
        b3_relu_1 = layers.ReLU(name='b3_relu_1')(b3_bn_1)
        b3_cnv2d_2 = layers.Conv2D(filters=self.nfilters_2,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b3_cnv2d_2')(b3_relu_1)
        b3_bn_2 = layers.BatchNormalization(name='b3_bn_2')(b3_cnv2d_2)
        # Residual layer case 2). Here we need to pad our identity tensor because we have
        # expanded the number of channels by changing the number of
        # filters in the second Conv layer.
        identity3 = PadLayer()(b2_relu_2, channels1=self.nfilters_1, channels2=self.nfilters_2)
        b3_add = layers.add([identity3, b3_bn_2], name='b3_residual')
        b3_relu_2 = layers.ReLU(name='b3_relu_2')(b3_add)
        #
        # Block 4
        b4_cnv2d_1 = layers.Conv2D(filters=self.nfilters_2,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b4_cnv2d_1')(b3_relu_2)
        b4_bn_1 = layers.BatchNormalization(name='b4_bn_1')(b4_cnv2d_1)
        b4_relu_1 = layers.ReLU(name='b4_relu_1')(b4_bn_1)
        b4_cnv2d_2 = layers.Conv2D(filters=self.nfilters_2,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b4_cnv2d_2')(b4_relu_1)
        b4_bn_2 = layers.BatchNormalization(name='b4_bn_2')(b4_cnv2d_2)
        # Residual layer, case 1).
        b4_add = layers.add([b3_relu_2, b4_bn_2], name='b4_residual')
        b4_relu_2 = layers.ReLU(name='b4_relu_2')(b4_add)

        # conv4_x
        #
        # Block 5
        #
        # Number of filters increases, and we downsample in the convolution.
        b5_cnv2d_1 = layers.Conv2D(filters=self.nfilters_3,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides_ds,
                                   padding=self.padding,
                                   name='b5_cnv2d_1')(b4_relu_2)
        b5_bn_1 = layers.BatchNormalization(name='b5_bn_1')(b5_cnv2d_1)
        b5_relu_1 = layers.ReLU(name='b5_relu_1')(b5_bn_1)
        b5_cnv2d_2 = layers.Conv2D(filters=self.nfilters_3,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b5_cnv2d_2')(b5_relu_1)
        b5_bn_2 = layers.BatchNormalization(name='b5_bn_2')(b5_cnv2d_2)
        # Residual layer, cases 2 and 3. First we do the usual padding
        padded5 = PadLayer(name='b5_pd_1')(b4_relu_2, channels1=self.nfilters_2, channels2=self.nfilters_3)
        # This leaves us with the right number of channels but too
        # large an image. Solve this by convolution with an identity
        # kernel with stride > 1. This is wrapped in a DownSample
        # layer:
        identity5 = DSLayer(filters=self.nfilters_3,
                            kernel_size=(1, 1),
                            strides=self.strides_ds)(padded5)
        b5_add = layers.add([identity5, b5_bn_2], name='b5_residual')
        b5_relu_2 = layers.ReLU(name='b5_relu_2')(b5_add)
        #
        # Block 6
        b6_cnv2d_1 = layers.Conv2D(filters=self.nfilters_3,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b6_cnv2d_1')(b5_relu_2)
        b6_bn_1 = layers.BatchNormalization(name='b6_bn_1')(b6_cnv2d_1)
        b6_relu_1 = layers.ReLU(name='b6_relu_1')(b6_bn_1)
        b6_cnv2d_2 = layers.Conv2D(filters=self.nfilters_3,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b6_cnv2d_2')(b6_relu_1)
        b6_bn_2 = layers.BatchNormalization(name='b6_bn_2')(b6_cnv2d_2)
        # Residual layer, case 1).
        b6_add = layers.add([b5_relu_2, b6_bn_2], name='b6_residual')
        b6_relu_2 = layers.ReLU(name='b6_relu_2')(b6_add)

        # conv5_x
        #
        # Block 7
        #
        # Number of filters increases, and we downsample in the convolution.
        b7_cnv2d_1 = layers.Conv2D(filters=self.nfilters_4,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides_ds,
                                   padding=self.padding,
                                   name='b7_cnv2d_1')(b6_relu_2)
        b7_bn_1 = layers.BatchNormalization(name='b7_bn_1')(b7_cnv2d_1)
        b7_relu_1 = layers.ReLU(name='b7_relu_1')(b7_bn_1)
        b7_cnv2d_2 = layers.Conv2D(filters=self.nfilters_4,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b7_cnv2d_2')(b7_relu_1)
        b7_bn_2 = layers.BatchNormalization(name='b7_bn_2')(b7_cnv2d_2)
        # Residual layer, cases 2 and 3. Padding and then downsampling
        padded7 = PadLayer(name='b7_pd_1')(b6_relu_2, channels1=self.nfilters_3, channels2=self.nfilters_4)
        identity7 = DSLayer(filters=self.nfilters_4,
                            kernel_size=(1, 1),
                            strides=self.strides_ds)(padded7)
        b7_add = layers.add([identity7, b7_bn_2], name='b7_residual')
        b7_relu_2 = layers.ReLU(name='b7_relu_2')(b7_add)
        #
        # Block 8
        b8_cnv2d_1 = layers.Conv2D(filters=self.nfilters_4,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b8_cnv2d_1')(b7_relu_2)
        b8_bn_1 = layers.BatchNormalization(name='b8_bn_1')(b8_cnv2d_1)
        b8_relu_1 = layers.ReLU(name='b8_relu_1')(b8_bn_1)
        b8_cnv2d_2 = layers.Conv2D(filters=self.nfilters_4,
                                   kernel_size=self.kernel_shape,
                                   strides=self.strides,
                                   padding=self.padding,
                                   name='b8_cnv2d_2')(b8_relu_1)
        b8_bn_2 = layers.BatchNormalization(name='b8_bn_2')(b8_cnv2d_2)
        # Residual layer, case 1).
        b8_add = layers.add([b7_relu_2, b8_bn_2], name='b8_residual')
        b8_relu_2 = layers.ReLU(name='b8_relu_2')(b8_add)
               
        # Output stage: Average pooling then flatten and put through a
        # Dense layer with the same number of output units as classes.
        avg_pool = layers.GlobalAveragePooling2D(name="avpool")(b8_relu_2)
        flatten = layers.Flatten(name="flatten_to_dense")(avg_pool)        
        # He at al. do not use Dropout, but I do for continuity with
        # the other models.
        dropout = layers.Dropout(rate=self.dropout_rate,
                                 name="dropout_from_dense_to_output")(flatten)
        output = layers.Dense(self.num_classes,
                              name='output_fc_layer',
                              activation='softmax')(dropout)

        # This is where we explicitly build the model.
        self.model = models.Model(input, output)#(name='ResNet18')




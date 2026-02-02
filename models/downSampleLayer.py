# downSampleLayer.py
#
# Simon Parsons
# 27-01-26
#
# As in the rest of the code, the key reference is:
# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
# for image recognition. In Proceedings of the IEEE Conference on
# Computer Vision and Pattern Recognition (pp. 770–778).
# 
# Often in a ResNet we need to combine the output of layers of
# different sizes. This class makes that happen.
#
# As explained in He et al., Section 3.3, we can do this by doing a
# convolution with a 1x filter, and if we set this to a constant
# (non-trainable) value of 1, we get an identity operation. Then all
# we need to do is have strides > 1 to downsample. This doesn't need
# to be a layer, but doing so wraps it nicely like the padding
# operation.
from keras import layers, initializers
from keras.layers import Layer
import tensorflow as tf

# Since we want is a 2D convolution with fixed filter, we just
# sub-class the 2DCov layer
class DSLayer(layers.Conv2D):

    # Everything is done in initialization. Ideally we wouldn't have
    # to pass kernel_size, but this seems to be obligatory. So we
    # ignore whatever value is passed.
    def __init__(self, filters, kernel_size, strides):
        super(DSLayer, self).__init__(
            filters= filters, 
            strides = strides,
            #name = name,
            kernel_size=(1, 1),
            # Set the kernel to be [1]
            kernel_initializer=initializers.Ones(),
            # Don't allow the kernel to update
            trainable=False, 
            padding='same')

                             

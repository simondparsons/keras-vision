# residual.py
#
# Simon Parsons
# 26-02-02
#
# Starting from:
# https://gist.github.com/FirefoxMetzger/6b6ccf4f7c344459507e73bbd13ec541rting from:
# 
# Provides an implementation of a residual block from:
#
# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
# for image recognition. In Proceedings of the IEEE conference on
# computer vision and pattern recognition (pp. 770–778).
#
# though only with some help from:
# https://machinelearningmastery.com/three-ways-to-build-machine-learning-models-in-keras/
#
# This gives us a way to build residual networks with different
# numbers of filters and downsaming (so all the complex cases that
# elude FirefoxMetzger), making it pretty general. I make no claims of
# efficiency, but hopefully it is clear.

import math
import tensorflow as tf
from keras import layers
from keras.layers import Layer
from models.downSampleLayer import DSLayer

# Initialise with the number of filters and the strides for both
# convolutional layers, and include kernel so that this can be
# specified at the network level.
class Residual(Layer):
    def __init__(self, filters1, filters2, strides1, strides2, kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        # Some parameters
        self.filters1 = filters1
        self.filters2 = filters2
        self.strides1 = strides1
        self.strides2 = strides2
        self.kernel   = kernel
        # Here we define the layers within the residual block, mainly
        # self-explanatory if you know the standard parts of a ResNet.
        self.w1_x =      layers.Conv2D(self.filters1,
                                       self.kernel,
                                       self.strides1,
                                       padding="same")
        # He at al. added BN betwen each convolution layer and the
        # subsequent activation layer
        self.bn_w1_x =    layers.BatchNormalization()   
        self.sigma_w1_x = layers.Activation("relu")
        self.Fx =         layers.Conv2D(self.filters2,
                                        self.kernel,
                                        self.strides2,
                                        padding="same")
        self.bn_Fx =      layers.BatchNormalization()
        # Needed for downsampling the input before combination.
        self.ds  =        DSLayer(filters=self.filters2,
                                  kernel_size=(1, 1),
                                  strides=self.strides1)
        self.Fx_plus_x =  layers.Add()
        self.out =        layers.Activation("relu")
        
    # The business part of the layer, which implements the structure
    # from Figure 2 of He et al. (2016). The notation for the
    # intermediate outputs is taken from He et al. (2016, Section 3.2)
    def call(self, input):
        # In Figure 2, x feeds directly into the convolutional layers,
        # First convolution in the residual layer
        x = input
        w1_x = self.w1_x(x)
        # Then BN
        bn_w1_x = self.bn_w1_x(w1_x)
        # Then through a RELU
        sigma_w1_x = self.sigma_w1_x(bn_w1_x)
        # Second convolutional layer gives us F(x) (also w2_sigma_w1_x
        # in my version of the notation from Section 3.2).
        Fx = self.Fx(sigma_w1_x)
        # Followed by BN
        bn_Fx =  self.bn_Fx(Fx)
        # Now add in our residual, which is the unprocessed x (called
        # the identity in Figure 2), except that we have to align its
        # shape with bn_Fx. First we downsample where necessary
        identity = self.ds(x)
        # Then we pad the dimensions if needed
        identity = self.pad(identity, bn_Fx)
        # Combine x/identity with Fx/bn_Fx
        Fx_plus_x = self.Fx_plus_x([bn_Fx, identity])
        # One more RELU and we are done
        out = self.out(Fx_plus_x)
        return out

    # Pad the input tensor have the right number of channels. Easier
    # to get this working than using the padLayer class.
    def pad(self, x, y):
        channels1 = tf.keras.backend.int_shape(x)[3]
        channels2 = tf.keras.backend.int_shape(y)[3]
        # This was cribbed from:
        # https://github.com/tflearn/tflearn/blob/master/tflearn/layers/conv.py#L1590
        ch = (channels2 - channels1)//2
        return tf.pad(x, [[0, 0], [0, 0], [0, 0], [ch, ch]])

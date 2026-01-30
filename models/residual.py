# residual.py
#
# Simon Parsons
# 26-01-20
#
# Starting from:
# https://gist.github.com/FirefoxMetzger/6b6ccf4f7c344459507e73bbd13ec541rting from:
# 
# Provides a simple implementation of a residual block from:
#
# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
# for image recognition. In Proceedings of the IEEE conference on
# computer vision and pattern recognition (pp. 770–778).
#
# though only with some help from:
# https://machinelearningmastery.com/three-ways-to-build-machine-learning-models-in-keras/
#
# This gives us a way to build residual networks where the number of
# filters is constant. If we want to have the usual stack of deepening
# filters and downsampling blocks, we will have to do the necessary
# work to modify x before adding to Fx.

from keras import layers
from keras.layers import Layer

# Based on the Keras base layer:
# https://keras.io/api/layers/base_layer/
class Residual(Layer):
    def __init__(self, filters, kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        # Here we define the layers within the residual block
        self.w1_x =      layers.Conv2D(self.filters,
                                       self.kernel,
                                       padding="same")
        # He at al. added BN betwen each convolution layer and the
        # subsequent activation layer
        self.bn_w1_x =    layers.BatchNormalization()   
        self.sigma_w1_x = layers.Activation("relu")
        self.Fx =         layers.Conv2D( self.filters,
                                         self.kernel,
                                         padding="same")
        self.bn_Fx =      layers.BatchNormalization() 
        self.Fx_plus_x =  layers.Add()
        
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
        # the identity in Figure 2)
        Fx_plus_x = self.Fx_plus_x([bn_Fx, x])
        # One more RELU and we are done
        return  layers.Activation("relu")(Fx_plus_x)

    def compute_output_shape(self, input_shape):
        return input_shape

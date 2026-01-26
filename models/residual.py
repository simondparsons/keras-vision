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

# This currently does not work because of the way that Keras handles
# the class variables when we create multiple instances of the class
# in parallel.
#
# 
from keras import layers
from keras.layers import Layer

# Based on the Keras base layer:
# https://keras.io/api/layers/base_layer/
class Residual(Layer):
    def __init__(self, filters, kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.x = None
        self.w1_x = None
        self.sigma_w1_x = None
        self.Fx = None
        self.Fx_plus_x = None
        
    # The business part of the layer, which implements the structure
    # from Figure 2 of He et al. (2016). The notation for the
    # intermediate outputs is taken from He et al. (2016) Section 3.2
    def call(self, input):
        # In Figure 2, x feeds directly into the convolutional layers,
        # but here we pass the input through an activation layer
        # first.
        self.x = layers.Activation("linear", trainable=False)(input)
        # First convolution in the residual layer
        if self.w1_x == None:
            self.w1_x = layers.Conv2D(self.filters,
                                      self.kernel,
                                      padding="same")(self.x)
        else:
            self.w1_x = layers.Conv2D(self.filters,
                                      self.kernel,
                                      padding="same")(self.x)
        # Then through a RELU
        self.sigma_w1_x =  layers.Activation("relu")(self.w1_x)
        # Second convolutional layer gives us F(x) (also w2_sigma_w1_x
        # in my version of the notation from Section 3.2).
        self.Fx =  layers.Conv2D( self.filters,
                      self.kernel,
                      padding="same")(self.sigma_w1_x)
        # Now add in our residual, which is the unprocessed x (called
        # the identity in Figure 2)
        self.Fx_plus_x = layers.Add()([self.Fx, self.x])
        # One more RELU and we are done
        return  layers.Activation("relu")(self.Fx_plus_x)

    def compute_output_shape(self, input_shape):
        return input_shape

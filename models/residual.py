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

from keras.layers import Layer

# Based on the Keras base layer:
# https://keras.io/api/layers/base_layer/
class Residual(Layer):
    def __init__(self, filters, kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel

    # The business part of the layer, which implements the structure
    # from Figure 2 of He et al. (2016). The notation for the
    # intermediate outputs is taken from He et al. (2016) Section 3.2
    def call(self, input):
        # In Figure 2, x feeds directly into the convolutional layers,
        # but here we pass the input through an activation layer
        # first.
        x = Activation("linear", trainable=False)(input)
        # First convolution in the residual leyer
        w1_x = Conv2D(self.filters,
                      self.kernel,
                      padding="same")(x)
        # Then through a RELU
        sigma_w1_x =   Activation("relu")(w1_x)
        # Second convolutional layer gives us F(x) (also w2_sigma_w1_x
        # in my version of the nottation from Section 3.2).
        Fx =  Conv2D( self.filters,
                      self.kernel,
                      padding="same")(sigma_w1_x)
        # Now add in our residual, which is the unprocessed x (called
        # the identity in Figure 2)
        Fx_plus_x = Add()([Fx, x])
        # One more RELU and we are done
        output = Activation("relu")(Fx_plus_x)
        
        return output

    #def call(self, x):
    #    # the residual block using Keras functional API
    #    first_layer =   Activation("linear", trainable=False)(x)
    #    x =             Conv2D( self.channels_in,
    #                            self.kernel,
    #                            padding="same")(first_layer)
    #    x =             Activation("relu")(x)
    #    x =             Conv2D( self.channels_in,
    #                            self.kernel,
    #                            padding="same")(x)
    #    residual =      Add()([x, first_layer])
    #    x =             Activation("relu")(residual)
    #    return x

    def compute_output_shape(self, input_shape):
        return input_shape

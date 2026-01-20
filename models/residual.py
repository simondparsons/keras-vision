# residual.py
#
# Starting from:
# https://gist.github.com/FirefoxMetzger/6b6ccf4f7c344459507e73bbd13ec541rting from:
# 
# Provides a simple implementation of a residual block from:
#
# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
# for image recognition. In Proceedings of the IEEE conference on
# computer vision and pattern recognition (pp. 770–778).

# Based on the Keras base layer:
# 
class Residual(Layer):
    def __init__(self, channels_in,kernel,**kwargs):
        super(Residual, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel

    def call(self, x):
        # the residual block using Keras functional API
        first_layer =   Activation("linear", trainable=False)(x)
        x =             Conv2D( self.channels_in,
                                self.kernel,
                                padding="same")(first_layer)
        x =             Activation("relu")(x)
        x =             Conv2D( self.channels_in,
                                self.kernel,
                                padding="same")(x)
        residual =      Add()([x, first_layer])
        x =             Activation("relu")(residual)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

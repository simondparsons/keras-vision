# PadLayer.py
#
# Simon Parsons
# 26-01-26
#
# As in the rest of the code, the key reference is:
# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
# for image recognition. In Proceedings of the IEEE Conference on
# Computer Vision and Pattern Recognition (pp. 770–778).
# 
# Often in a ResNet we need to combine the output of layers with
# different numbers of channels (because they use different numbers of
# filters). This class makes that happen.
#
# As explained in He et al., Section 3.2, we need to create a linear
# projection of x to make this happen, and as:
#
# https://stackoverflow.com/questions/46121283/what-is-linear-projection-in-convolutional-neural-network
#
# explains, this is done by padding the tensor, and:
# https://github.com/tflearn/tflearn/blob/master/tflearn/layers/conv.py#L1590
#
# provides code for that (see below). Since we use tf.pad() and this
# is a tf function, we can't call it direct from Keras (says the
# interpreter) and so we have to wrap it in a Keras layer.
#
# Note also the comments under "Residual Network" in He et al. about zero padding.

from keras import layers
from keras.layers import Layer
import tensorflow as tf

# Based on the Keras base layer:
# https://keras.io/api/layers/base_layer/
class PadLayer(Layer):
    
    def call(self, x, channels1, channels2):
        # This was cribbed from:
        # https://github.com/tflearn/tflearn/blob/master/tflearn/layers/conv.py#L1590
        ch = (channels2 - channels1)//2
        return tf.pad(x, [[0, 0], [0, 0], [0, 0], [ch, ch]])

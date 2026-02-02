# resnet34.py
#
# Simon Parsons
# 26-01-20
#
# A merge of ideas from
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
# and
# https://gist.github.com/FirefoxMetzger/6b6ccf4f7c344459507e73bbd13ec541rting from:
#
# to create a deep ResNet, as defined in
#
# He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
# for image recognition. In Proceedings of the IEEE conference on
# computer vision and pattern recognition (pp. 770–778).
#
# Note that this structure started with that of FirefoxMetzger, but
# goes further to handle blocks with different numbers of filters and
# downsampling, getting to something that implements the kinds of
# smaller networks from He at al.

from models.backbone import Backbone 
from models.residual import Residual
from tensorflow.keras import layers, models

class ResNet34(Backbone):

    # Set up some constants that we will use to do this across the various
    # layers. The first constants apply to the residual layers
    kernel = (3, 3)        # Use 3 x 3 kernels throughout the residual
                           # layers.
    pool_shape = (2, 2)    # Reduce dimensionality by 4 = 2 x 2 in pooling layers
    strides = (1, 1)       # Two stride values for downsampling (_ds)
    strides_ds = (2, 2)    # and not since orignal ResNet downsampled
                           # at the start of block from the second
                           # block onwards.
    nfilters_1 = 64        # Increase filters as we go further through the 
    nfilters_2 = 128       # backbone.
    nfilters_3 = 256  
    nfilters_4 = 512

    # These apply to the layers defined here, basically the blocks
    # before and after the residual layers
    activation = 'relu'    # use Rectified Linear Unit activiation functions
    dropout_rate = 0.5     # Drop 50% of neurons
    padding = 'same'       # Maintain the shape of feature maps per layer

    # Define how we will build the model
    model = models.Sequential(name='ResNet34')

    # Resnet34, as per He et al, (2016, Figure 3). This is the second
    # smallest network defined in He at al, but the largest where the
    # residual layers are the two convolutional block layer of our
    # Residual class. The 50, 101 and 152 layer models use a 3 block
    # "bottleneck" design as in (He et al, 2016, Figure 5).
    def buildModel(self):
        # Pre-processing
        #
        # Create the input layer to understand the shape of each image and batch-size 
        self.model.add(layers.Input(shape=self.img_shape,
                                    name='input'))
        # Add a rescaling layer to convert the inputs to fall in the range (-1, 1).
        # https://machinelearningmastery.com/image-augmentation-with-keras-preprocessing-layers-and-tf-image/
        self.model.add(layers.Rescaling(1/127.5, offset=-1,
                                        name='rescale'))        
        # Now the network proper.
        #
        # Initial Conv2D and maxPooling
        #
        # As in He et al. this is a 7x7 filter with stride 2.
        self.model.add(layers.Conv2D(filters=self.nfilters_1,
                                     kernel_size=(7, 7),
                                     strides=self.strides_ds,
                                     padding=self.padding,
                                     name='cnv2d'))
        self.model.add(layers.Activation(self.activation))
        self.model.add(layers.MaxPooling2D(pool_size=self.pool_shape,
                                      name="maxpool"))

        # Three residual blocks with 64 filters each
        self.model.add(Residual(self.nfilters_1, self.nfilters_1, self.strides, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_1, self.nfilters_1, self.strides, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_1, self.nfilters_1, self.strides, self.strides, self.kernel))

        # Four blocks with 128 filters. We downsample at the first block.
        self.model.add(Residual(self.nfilters_2, self.nfilters_2, self.strides_ds, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_2, self.nfilters_2, self.strides, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_2, self.nfilters_2, self.strides, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_2, self.nfilters_2, self.strides, self.strides, self.kernel))

        # Six blocks with 256 filters. We downsample at the first block
        self.model.add(Residual(self.nfilters_3, self.nfilters_3, self.strides_ds, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_3, self.nfilters_3, self.strides, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_3, self.nfilters_3, self.strides, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_3, self.nfilters_3, self.strides, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_3, self.nfilters_3, self.strides, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_3, self.nfilters_3, self.strides, self.strides, self.kernel))
        
        # Three blocks with 512 filters. In He et al, (2016, Figure 3)
        # the first block downsamples, but with the smaller images
        # that I was using we don't really want to go smaller.
        self.model.add(Residual(self.nfilters_4, self.nfilters_4, self.strides, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_4, self.nfilters_4, self.strides, self.strides, self.kernel))
        self.model.add(Residual(self.nfilters_4, self.nfilters_4, self.strides, self.strides, self.kernel))

        # Output layers
        #
        # Average Pooling, then through an FC layer with one neuron per class.
        self.model.add(layers.GlobalAveragePooling2D(name="avpool"))
        self.model.add(layers.Flatten(name="flatten_to_dense"))
        self.model.add(layers.Dropout(rate=self.dropout_rate,
                                      name="dropout_from_dense_to_output"))
        self.model.add(layers.Dense(self.num_classes,
                                    name='output_fc_layer',
                                    activation='softmax'))

        self.model.build()

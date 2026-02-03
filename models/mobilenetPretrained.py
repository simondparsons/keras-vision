# mobilenet-pretrained.py
#
# Simon Parsons
# 26-02-03

# An illustration of how to take a pre-trained Keras model and
# fine-tune it with a new dataset.
#
# Drawn from the tutorial at:
# https://keras.io/guides/transfer_learning/
#
# but using MobileNet as a much lighter model.

from models.backbone import Backbone
from tensorflow.keras import layers, models, applications

# Note that we will build this exactly as we have previous models, as
# a subclass of Backbone.
class MobileNetPretrained(Backbone):

    # Many fewer parameters than usual since we are mainly using a
    # pre-specified network.
    
    dropout_rate = 0.5      # drop 50% of neurons

    def buildModel(self):
        # We start with MobileNet. We load the weights from training on
        # ImageNet, but we don't include the "top", the final classifier
        # part.
        base_model = applications.MobileNet(weights='imagenet',  
                                            input_shape=self.img_shape,
                                            include_top=False)

        # Don't train the base model
        base_model.trainable = False
        
        # Now build a model that includes the base_model. We will do this
        # in functional style.
        
        # As always we start with an input level
        inputs = layers.Input(shape=self.img_shape)
        # Run the base model (setting training to false means it is doing inference).
        mobile = base_model(inputs, training=False)
        # Put the result of the MobileNet inference through a standardish output stage
        pooled =  layers.GlobalAveragePooling2D()(mobile)
        flatten = layers.Flatten()(pooled) 
        dropout = layers.Dropout(rate=self.dropout_rate)(flatten)
        outputs = layers.Dense(self.num_classes,
                              activation='softmax')(dropout)
        self.model = models.Model(inputs, outputs)

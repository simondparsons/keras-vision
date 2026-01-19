# backbone.py
#
# Badly named, because it is really the whole network, rather than
# just the backbone.
#
# But what it does is to build a keras network object based on some
# parameters that it is passed. The options are some standard ones
# from the literature, and some variants thereof.

class Backbone:
    def __init__(self, imageShape, numberOfClasses):
        self.img_shape = imageShape
        self.num_classes = numberOfClasses

    modelType = ""
    model = []
 

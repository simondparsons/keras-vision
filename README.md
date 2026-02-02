# keras-vision

Using Keras/Tensorflow to re-create some classic vision models.

This is an extension of the work in basic-vision. That contains models, developed for teaching purposes, that I am confident can be run just on a CPU. These are, basically, LeNet up to VGG11. Going further in architecture complexity caused my laptop to lock up so I called a halt to that.

With CUDA working locally, I picked up the project of pulling together code for classic vision models that could be run on a personal device. I opted to keep these separate from those in basic-vision so as to not compromise the CPU-purity of that exercise. However all those models should run on a GPU and will be much much faster if you do.

More details on the models can be found in the README in the models folder, but they curerntly include the deeper VGG models (VGG16 and VGG19), and ResNets (currently some simple demos along with ResNet18 and ResNet34, illustrating different ways to construct the residual blocks).
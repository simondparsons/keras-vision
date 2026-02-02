# classify-images-vgg-etal.py
#
# Simon Parsons
# 26-01-19
#
# Code to explore the performance of some of the deeper early CNN
# architectures using data from some of the TensorFlow datasets.
#
# This started with
# https://exowanderer.medium.com/what-is-this-keras-thing-anyways-fe7aa00158ef
#
# but modified to call different networks and to use command line
# arguments.
#
# This is a pretty minor edit of classify-images-keras.py from my
# basic-vision repo.

import string
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, callbacks, utils, datasets, models
from operator import itemgetter
# These are the architectures available roughly in order of complexity.
from models.vgg11 import VGG11
from models.vgg16 import VGG16
from models.vgg19 import VGG19
from models.simpleResnet import SimpleResNet
from models.resnet18 import ResNet18
from models.simpleClassResnet import SimpleClassResNet
from models.resnet34 import ResNet34

def main():
    # Generalise the code by allowing the model, dataset and some of the
    # hyperparameters to be picked on the command line. 
    parser = argparse.ArgumentParser(description='Keras/TensorFlow for image classification')
    # Possible values: mnist, fashion_mnist, cifar10
    parser.add_argument('--dataset', help='Which datset to use.', default='mnist')
    # Possible values: many, see the README.md in the models folder
    parser.add_argument('--model', help='Which model to use.', default='VGG11')
    # Possible values: y or yes for display, anothing else for no display
    parser.add_argument('--display', help='Display training data?', default='n')
    # Use epochs to specify a number to run without early stopping. If
    # you don't specify the script will run 50 epochs with early
    # stopping (which for the 3 simple datsets has rarely been more
    # than 20).
    parser.add_argument('--epochs', help='Specify number of epochs')
    # Batch size, in case we need to adjust this
    parser.add_argument('--batch_size', help='Specify batch size', default=64)
    # Patience, in case we need to adjust this
    parser.add_argument('--patience', help='How many epochs to wait before invoking early stopping ', default=3)

    args = parser.parse_args()

    # Load the data from TensorFlow. 
    #
    dataset = args.dataset
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    else:
        print("I don't know the dataset:", args.dataset)
        exit(0)
    
    # Conv2D, the main Keras model layer we will use, requires 4D
    # inputs: batch, row, col, color. If we have no color dimension
    # (as in MNIST), add the color dimensions to represent greyscale.
    if np.ndim(X_train) == 3: 
        COLOR_DIM = -1
        X_train = np.expand_dims(X_train, axis=COLOR_DIM)
        X_test = np.expand_dims(X_test, axis=COLOR_DIM)

    # Pull out key features of the data. This assumes that the first image
    # is the same size as the rest.
    num_classes = np.unique(y_train).__len__()  
    img_shape = X_train[0].shape
    print("Classes:", num_classes)
    print("Image dimensions:", img_shape)

    # One-hot encode the output
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)

    # For flexibility, we will separate out the NN definition so that we
    # can experiment with a range of backbones.

    # Input the model description
    arch = args.model

    if arch == 'VGG11':
        network = VGG11(img_shape, num_classes)
    elif arch == 'VGG16':
        network = VGG16(img_shape, num_classes)
    elif arch ==  'VGG19':
        network = VGG19(img_shape, num_classes)
    elif arch ==  'ResNet':
        network = ResNet(img_shape, num_classes)
    elif arch ==  'SimpleResNet':
        network = SimpleResNet(img_shape, num_classes)
    elif arch ==  'SimpleClassResNet':
        network = SimpleClassResNet(img_shape, num_classes)
    elif arch ==  'ResNet18':
        network = ResNet18(img_shape, num_classes)        
    elif arch ==  'ResNet34':
        network = ResNet34(img_shape, num_classes)        
    else:
        print("I don't know the model:", args.model)
        exit(0)
    
    network.buildModel()
    print(network.model.name)
    model = network.model
    # Print a summary of the model
    model.summary()

    # Now compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # Train the model.
    #
    # epochs:           How many iterations should we cycle over
    #                   the entire MNIST dataset
    # validation_split: How many images to hold out per epoch
    # batch size:       Could be 32, 64, 128, 256, 512
    # early_stopping:   When to stop training if performance plateaus.
    
    validation_split = 0.1  
    batch_size = 64 # The larger the batch size, the more memory a
                    # given dataset uses.

    # If we have specified the number of epochs, then run for that
    # number irrespective of the way that training goes. Otherwise do
    # early stopping after validation error hadn't improved for
    # patience=3 epochs.
    if args.epochs:
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=int(args.epochs),
            # The alternative is to explicitly set validation_data 
            validation_split = validation_split,
        )
    else:
        early_stopping = callbacks.EarlyStopping(patience=int(args.patience))
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=50,
            validation_split = validation_split,
            callbacks=[early_stopping]
        )

    # Show the change in accuracy and loss over training.
    if args.display == 'y' or args.display == 'yes':
        epochs = np.arange(len(history.history['loss']))

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15,10))
        ax1.plot(epochs, history.history['loss'], 'o-', label='Training Loss')
        ax1.plot(epochs, history.history['val_loss'], 'o-', label='Validation Loss')
        ax1.legend()
    
        ax2.plot(epochs, history.history['accuracy'], 'o-', label='Training Accuracy')
        ax2.plot(epochs, history.history['val_accuracy'], 'o-', label='Validation Accuracy')
        ax2.legend()

        plt.show()

    test_score = model.evaluate(X_test, y_test, verbose=0)
    train_score = model.evaluate(X_train, y_train, verbose=0)

    print("Train loss     :", train_score[0])
    print("Train accuracy :", train_score[1])
    print()
    print("Test loss      :", test_score[0])
    print("Test accuracy  :", test_score[1])

    return 0

if __name__ == "__main__":
    main()

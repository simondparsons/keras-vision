# dataLoader.py
#
# Simon Parsons
# 26-02-04

# A utility to load datasets from the larger public dataset in TensorFlow Datasets
#
# Draws on:
# https://www.tensorflow.org/datasets/overview
# https://www.tensorflow.org/datasets/keras_example

import tensorflow as tf
import tensorflow_datasets as tfds

# Parameters that could be command line arguments.
TRAIN_RATIO = 0.8 # What fraction of the data to train on when no separate test split
SPLIT = "train[:15%]" # How much of the dataset to use
# A utility to load datasets from the larger public dataset in TensorFlow Datasets
#
# Draws on:
# https://www.tensorflow.org/datasets/overview
# https://www.tensorflow.org/datasets/keras_example
#
# testTrain is a binary. If true then the dataset already has a test
# split. If false, it does not.
def loadData(dataset, testTrain):

    if testTrain:
        # The heavy lifing is done by tfds.load.
        (ds_train, ds_test), ds_info = tfds.load(dataset,
                                                 # Only works where we have both test
                                                 # and train splits
                                                 split=['train', 'test'],
                                                 # Shuffle on loading
                                                 shuffle_files=True,
                                                 # Set the right format for the data
                                                 as_supervised=True,
                                                 with_info=True,)
    else:
        # From CoPilot, a different way to get tfds.load to do that
        # lifting
        #
        # First load the split that is available
        ds_full, ds_info = tfds.load(dataset,
                                     split=SPLIT,
                                     shuffle_files=True,
                                     as_supervised=True,
                                     with_info=True,)

        # Then split this according to th TRAIN_RATIO parameter
        num_examples = ds_info.splits[SPLIT].num_examples
        train_size = int(num_examples * TRAIN_RATIO)
        test_size  = num_examples - train_size
        ds_train = ds_full.take(train_size)
        ds_test  = ds_full.skip(train_size)
    
    # Normalize training data, cache, shuffle and prefetch for performance
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # As suggested here:
    # https://stackoverflow.com/questions/62436302/extract-target-from-tensorflow-prefetchdataset
    # we use map to extract the X and y, into a list, and then we
    # convert the list into a tensor
    X_train = tf.convert_to_tensor(list(map(lambda x: x[0], ds_train)))
    y_train = tf.convert_to_tensor(list(map(lambda x: x[1], ds_train)))
    
    # Then test data
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    X_test =  tf.convert_to_tensor(list(map(lambda x: x[0], ds_test)))
    y_test =  tf.convert_to_tensor(list(map(lambda x: x[1], ds_test)))

    print("  X_train:", X_train.shape, " y_train:", y_train.shape)
    print("  X_test: ", X_test.shape,  " y_test: ", y_test.shape)

    return (X_train, y_train), (X_test, y_test)

#
# converts uint8 to float32, and resizes
#
# The resizing is to 224 x 224 to fit with pre-trained MobileNet
# provided by Keras. This seems to be key to getting good test
# results.
def normalize_img(image, label):
  img = tf.cast(image, tf.float32) / 255.
  img = tf.image.resize(img, [224, 224])
  return img, label

    
    

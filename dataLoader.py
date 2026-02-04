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

# A utility to load datasets from the larger public dataset in TensorFlow Datasets
#
# Draws on:
# https://www.tensorflow.org/datasets/overview
# https://www.tensorflow.org/datasets/keras_example
def loadData(dataset):

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

    # The heavy lifing is done by tfds.load.
    #(ds_train), ds_info = tfds.load(dataset,
    #                                         # Only works where we have both test
    #                                         # and train splits
    #                                         split=['train'],#, 'test'],
    #                                         # Shuffle on loading
    #                                         shuffle_files=True,
    #                                         # Set the right format for the data
    #                                         as_supervised=True,
    #                                         with_info=True,)

    # Normalize training data, cache and shuffle
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    #ds_train = map(normalize_img, ds_train)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    # Prefetch for performance
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    # As suggested here:
    # https://stackoverflow.com/questions/62436302/extract-target-from-tensorflow-prefetchdataset
    # we use map to extract the X and y, into a list, and then we
    # convert the list into a tensor
    X_train = tf.convert_to_tensor(list(map(lambda x: x[0], ds_train)))
    y_train = tf.convert_to_tensor(list(map(lambda x: x[1], ds_train)))

    # Then training data
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    X_test =  tf.convert_to_tensor(list(map(lambda x: x[0], ds_test)))
    y_test =  tf.convert_to_tensor(list(map(lambda x: x[1], ds_test)))

    return (X_train, y_train), (X_test, y_test)

def normalize_img(image, label):
  # converts uint8 to float32, and resizes
  img = tf.cast(image, tf.float32) / 255.
  img = tf.image.resize(img, [244, 244])
  return img, label

    
    

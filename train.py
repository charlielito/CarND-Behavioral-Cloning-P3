import os
import csv

import cv2
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf

from model import model, deep_model, nvidia_net


### Model Parameters
conv_dropout = 0.4
dense_dropout = 0.5
l2_regularization = 0.001

# l1_regularizer = tf.keras.regularizers.l1(0.01)
l2_regularizer = None #tf.keras.regularizers.l2(l2_regularization)

CROP_UP = 50
CROP_DOWN = 30
final_height = 160 - (CROP_UP + CROP_DOWN)


mu = 0
sigma = 0.1
init = None #tf.initializers.truncated_normal(mean = mu, stddev = sigma)

params = dict(
    conv_dropout=conv_dropout,
    dense_dropout=dense_dropout,
    kernel_init=init,
    height=final_height
)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    predictions = nvidia_net(features, mode, params)
    preds = predictions["steering"]


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    with tf.name_scope('loss'):
            loss = tf.losses.mean_squared_error(
                labels=labels, predictions=tf.squeeze(preds, axis = 1), scope='loss')
            tf.summary.scalar('loss', loss)
    
    # Accuracy    
    with tf.name_scope('mae'):
            mae = tf.metrics.mean_absolute_error(
                labels=labels, predictions=tf.squeeze(preds, axis = 1), name='mae')
            tf.summary.scalar('mae', mae[1])

    # Create a hook to print acc, loss & global step every 100 iter.   
    train_hook_list= []
    train_tensors_log = {'mae': mae[1],
                         'loss': loss}
    train_hook_list.append(tf.train.LoggingTensorHook(
        tensors=train_tensors_log, every_n_iter=100))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
#             optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#             optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
    
        return tf.estimator.EstimatorSpec(
            mode=mode, 
            loss=loss, 
            train_op=train_op, 
            training_hooks=train_hook_list)

    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops={'mae/mae': mae}, evaluation_hooks=None)


# Create the Estimator
classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/pilotnetVpaper6")


# ## Dataset creation

### folder where the data is
DATASET = "data_mine"



def read_img(name):
    img = name.split('/')[-1]
    return cv2.imread(os.path.join(DATASET,"IMG", img))[...,::-1]



### Load csv as a pandas dataframe
header = ["center", "left", "right", "steering", "throttle", "break", "speed"]
df = pd.read_csv(os.path.join(DATASET, "driving_log.csv"), header=None, names=header)
df.head()


def get_image_path(filename):
    return os.path.join(DATASET,"IMG", filename.split("/")[-1])


### Make paths relative to this jupyter notebook
df["center"] = df.center.apply(get_image_path)
df["left"] = df.left.apply(get_image_path)
df["right"] = df.right.apply(get_image_path)
df["steering"] = df.steering.apply(float)

### Split train and dev set 
train_samples, validation_samples = train_test_split(df, test_size=0.2)


# Get filenames and labels (steerings)
c_filenames = train_samples.center.values
c_labels = train_samples.steering.values

val_filenames = validation_samples.center.values
val_labels = validation_samples.steering.values


# Add side cameras to train set
steer_correction = 0.2

l_filenames = train_samples.left.values
l_labels = train_samples.steering.values + steer_correction

r_filenames = train_samples.right.values
r_labels = train_samples.steering.values - steer_correction

filenames = np.concatenate([c_filenames, l_filenames, r_filenames])
labels = np.concatenate([c_labels, l_labels, r_labels])


# Aux function to crop the images
def get_crop_window(crop_up, crop_down):
    final_height = 160 - (crop_up + crop_down)
    final_width = 320

    return [
        crop_up,
        0,
        final_height,
        final_width,
    ]


# Function that returns the image decoded from jpg and label
def parse_function(filename, label):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    image = tf.image.decode_and_crop_jpeg(
            image_string,
            crop_window = get_crop_window(CROP_UP, CROP_DOWN),
            channels = 3
        )

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


# ### Use of TF DataSet API

# Define parameters
BATCH_SIZE = 32
EPOCHS = 50
steps_per_epoch = len(train_samples)*3//BATCH_SIZE


# Create training dataset
def create_train_dataset():
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    dataset = dataset.repeat(EPOCHS)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)
    return dataset


# Create dev dataset
def create_eval_dataset():
    val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(parse_function, num_parallel_calls=4)
    val_dataset = val_dataset.repeat(EPOCHS)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(1)
    return val_dataset


# ## Training

def serving_input_fn():

    input_image = tf.placeholder(
        dtype=tf.float32,
        shape=[None, None, None, 3],
        name="input_image"
    )

#     images = tf.image.resize_images(input_image, [params.image_height, params.image_width])
    images = input_image
    images = tf.image.crop_to_bounding_box(images, *get_crop_window(CROP_UP, CROP_DOWN))

    images = tf.cast(images, tf.float32)

    return tf.estimator.export.TensorServingInputReceiver(
        features = images,
        receiver_tensors = input_image
    )



exporter = tf.estimator.LatestExporter(
    "test_exporter",
    lambda: serving_input_fn(),
)



# Specs
train_spec = tf.estimator.TrainSpec(
    input_fn=create_train_dataset,
    max_steps=EPOCHS*steps_per_epoch)

eval_spec = tf.estimator.EvalSpec(
    input_fn=create_eval_dataset,
    steps=None,
    exporters=[exporter],
    start_delay_secs=10,  # Start evaluating after 10 sec.
    throttle_secs=30  # Evaluate only every 30 sec
)


tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


classifier.export_savedmodel(
    "export/nvidia_dataset",
    lambda: serving_input_fn()
)


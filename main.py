import tensorflow as tf
print(tf.__version__)

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from keras import layers
import time

from IPython import display

BATCH_SIZE = 32
IMG_HEIGHT = 650
IMG_WIDTH = 1250

# Data loading
datadir = "C:\\Users\\masua\\Downloads\\Cimat\\oil-spill-dataset\\images"
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    datadir,
    label_mode=None,
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH)
)


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import get_model
import tensorflow.keras.utils as tf_utils
from tensorflow.keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
import os
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

IMG_SAVE_PATH = 'image_data'

CLASS_MAP = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
    "none": 3
}

NUM_CLASSES = len(CLASS_MAP)

def mapper(val):
    return CLASS_MAP[val]

# load images from the directory
dataset = []
for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (77, 77))
        dataset.append([img, directory])

data, labels = zip(*dataset)
labels = np.asarray(list(map(mapper, labels)))
data = np.asarray(data)
print(labels.shape)
print(data.shape)
model = get_model()

model.compile(
    optimizer=Adam(lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(data,labels, epochs=10)
model.save("rock-paper-scissors-model.h5")

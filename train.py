import cv2
import os
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.optimizers import Adam

IMG_SAVE_PATH = 'image_data'

#load images

dataset = []

for directory in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        if item.startswith('.'):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300,300))
        dataset.append([img, directory])
                  
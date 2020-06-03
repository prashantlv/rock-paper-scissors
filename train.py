import cv2
import os
import numpy as np 
import pandas as pd 
from tensorflow.keras.utils import to_categorical
from model import get_model
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

data, labels = zip(*dataset)
np.array(data).reshape(-1, 300,300,3)
CLASS_MAP = {
    'rock': 0,
    'paper': 1,
    'scissors': 2,
    'none': 3
}
NUM_CLASSES = len(CLASS_MAP)

def maper(temp):
    return CLASS_MAP[temp]
labels = list(map(maper, labels))
labels = to_categorical(labels, num_classes=None )
input_size = np.array(data).shape
model = get_model(input_size)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['Accuracy'])
model.fit(x=data, y=labels, epochs=100, batch_size=50, validation_split=0.25)
#model.save('trained_model.h5')
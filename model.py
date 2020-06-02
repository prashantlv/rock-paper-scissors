import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.optimizers import Adam

def get_model():
    
    model = Sequential()
    model.add(Dense(128, input_shape=()))
    model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))

    model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))

    model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(Conv2D(128, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(3))
    model.add(Activation('softmax'))

return model
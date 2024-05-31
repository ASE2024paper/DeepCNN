import numpy as np
import os
import cv2
import keras
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from kerastuner.tuners import GridSearch
import time
import tensorflow as tf

train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)
img_folder = r'/notebook/CNN_Evalu/data/DataSet/Model#9/xray_dataset_covid19'

train = train_datagen.flow_from_directory(
            directory=img_folder + r'/train/',
            target_size=(200,200),
            color_mode='rgb',
            batch_size=20,
            class_mode='binary')

test = val_datagen.flow_from_directory(
            directory=img_folder + r'/test/',
            target_size=(200,200),
            color_mode='rgb',
            batch_size=20,
            class_mode='binary')


# Build model with hyperparameters
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', input_shape=(200,200,3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))
    model.add(Dropout(0.30))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(
        units=hp.Choice('dense_1_units', [32,64, 256, 512, 1024]),
        activation='relu'
    ))
    model.add(Dense(
        units=hp.Choice('dense_2_units', [64,128, 256, 512, 1024]),
        activation='relu'
    ))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Create a GridSearch tuner
tuner = GridSearch(
    build_model,
    objective='accuracy',
    max_trials=3,
    executions_per_trial=3,
    directory='/notebook/CNN_Evalu/data/Test Model/DU',
    project_name='Model#9_2'
)
tuner.search_space_summary()

# Search for the best parameters
tuner.search(
    train,
    epochs=30,
     batch_size=32,
    validation_data=(test)
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the hyperparameters
print(f"""
The hyperparameter search is complete. 
the optimal number of units in the dense layer 1 is {best_hps.get('dense_1_units')}
the optimal number of units in the dense layer 2 is {best_hps.get('dense_2_units')}


""")
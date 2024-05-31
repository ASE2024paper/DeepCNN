import numpy as np
import os
import cv2
import keras
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
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
print(test)
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', input_shape=(200,200,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))
model.add(Dropout(0.30))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), padding="same"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train, epochs=30, validation_data=test, callbacks=[es])


# Importing the necessary libraries, which we may or may not use. Its always good idea to import them befor (if you remember) else we can do it at any point of time no problem.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input, AveragePooling2D, Activation,Conv2D, MaxPooling2D, BatchNormalization,Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from kerastuner.tuners import GridSearch, RandomSearch
import time
import tensorflow as tf

(x_train, y_train), (x_test, y_test)=cifar10.load_data()

print('Shape of x_train is {}'.format(x_train.shape))
print('Shape of x_test is {}'.format(x_test.shape)) 
print('Shape of y_train is {}'.format(y_train.shape))
print('Shape of y_test is {}'.format(y_test.shape))

from tensorflow.keras.utils import to_categorical

# Normalizing
x_train=x_train/255
x_test=x_test/255

#One hot encoding
y_train_cat=to_categorical(y_train,10)
y_test_cat=to_categorical(y_test,10)



# Build model with hyperparameters
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(4, 4), strides=(1, 1), padding="same", input_shape=(32,32,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(32, kernel_size=(4, 4), strides=(1, 1), padding="same", input_shape=(32,32,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(
        units=hp.Choice('dense_1_units', [32,128, 256, 512, 1024]),
        activation='relu'
    ))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model

# Create a GridSearch tuner
tuner = RandomSearch(
    build_model,
    objective='accuracy',
    max_trials=3,
    executions_per_trial=3,
    directory='./Model#2/',
    project_name='Model#2_1_KerasTuner'
)
tuner.search_space_summary()

# Search for the best parameters
tuner.search(
    x_train,
    y_train_cat,
    epochs=20,
     batch_size=32,
    validation_data=(x_test, y_test_cat)
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the hyperparameters
print(f"""
The hyperparameter search is complete. 
the optimal number of units in the dense layer is {best_hps.get('dense_1_units')}

""")
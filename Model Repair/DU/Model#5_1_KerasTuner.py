import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, BatchNormalization, Dropout
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from datetime import datetime

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from kerastuner.tuners import GridSearch
import time
import tensorflow as tf

warnings.filterwarnings("ignore")
sns.set_style ('darkgrid')
#%matplotlib inline
np.random.seed(42)
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))
#Found GPU at: /device:GPU:0
train = pd.read_csv('/notebook/CNN_Evalu/data/DataSet/Model#5/input/train.csv')
train.head()


test = pd.read_csv('/notebook/CNN_Evalu/data/DataSet/Model#5/input/test.csv')
test.head()

features = train.drop('label', axis=1)
target = train['label']
X_ = np.array(features)
X_test = np.array(test)
X_train = X_.reshape(X_.shape[0], 28, 28)
X_train.shape

target.value_counts(normalize=True)

len(target.value_counts())
X_train = X_.reshape(X_.shape[0], 28, 28, 1)
X_test_reshape = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, target, test_size=0.25, 
                                            random_state=42)



# Build model with hyperparameters
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation ='elu', input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation ='elu', input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(AvgPool2D(pool_size=(2, 2), padding="same"))
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation ='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='glorot_uniform', activation ='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(AvgPool2D(pool_size=(2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(units=hp.Choice('dense_1_units', [128, 256, 512, 1024]),
     activation = "elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=hp.Choice('dense_2_units', [64,128, 256, 512, 1024]),
     activation = "elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=hp.Choice('dense_3_units', [64,128, 256, 512, 1024]),
     activation = "elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(10, activation = "softmax"))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    return model

# Create a GridSearch tuner
tuner = GridSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=3,
    directory='/notebook/CNN_Evalu/data/Test Model/DU',
    project_name='Model#5_1'
)
tuner.search_space_summary()

# Search for the best parameters
tuner.search(
    X_tr, 
    y_tr,
    epochs=50,
     batch_size=16,
    validation_data=(X_val, y_val)
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the hyperparameters
print(f"""
The hyperparameter search is complete. 
the optimal number of units in the dense layer 1 is {best_hps.get('dense_1_units')}
the optimal number of units in the dense layer 2 is {best_hps.get('dense_2_units')}
the optimal number of units in the dense layer 3 is {best_hps.get('dense_3_units')}

""")
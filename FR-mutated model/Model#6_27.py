# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import confusion_matrix

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image

np.random.seed(42)
data= pd.read_csv('/work/LAS/hridesh-lab/Experiment/Mohammad/input/hmnist_28_28_RGB.csv')
data.head(5)
Label = data["label"]
data.drop('label', axis=1 , inplace= True)

Label.unique()
#array([2, 4, 3, 6, 5, 1, 0])
Label.value_counts()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, Label, test_size=0.2, random_state=42)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_smote, y_smote = sm.fit_resample(X_train, y_train)
#Checking for number of instances in each class after running SMOTE algorithm.

y_smote.value_counts()


X_tr= np.array(X_smote).reshape(len(X_smote),28, 28, 3)
model= Sequential()
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', input_shape = (28, 28, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation= 'relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation= 'relu'))
model.add(Flatten())
model.add(Dense(24, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(Label), activation='softmax'))
model.compile(optimizer= 'adam', loss= keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.summary()
history= model.fit(X_tr, y_smote, epochs=50, batch_size= 32, validation_split=0.2,shuffle= True)

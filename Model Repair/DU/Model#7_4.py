#Follow Kaggle's way to load datasets.
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

#import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as kr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#change the dataset into Pandas dataframe
import pandas as pd
dataset = pd.read_csv("/notebook/CNN_Evalu/data/DataSet/Model#7/chineseMNIST.csv")

#Let's start labelencoding!
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded = le.fit_transform(dataset['character'].values)
decoded = le.inverse_transform(encoded)
dataset['character2'] = encoded
print('This dataset has following true labels, ', le.classes_)


#Then, make a dictionary. 
dic={0:'一',1:'七',2:'万',3:'三',4:'九',5:'二',6:'五',7:'亿',8:'八',9:'六',10:'十',11:'千',12:'四',13:'百',14:'零'}
#Shuffle data :-)
dataset_shuffled = dataset.sample(frac=1, random_state=0)
#split the shuffled dataset into explanation data and target data.
x =dataset_shuffled.iloc[:,0:4096]
y =dataset_shuffled.iloc[:,[4098]]
#Let's use sklearn's train_test_split function
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#normalizes data from 1 to 0. 
x_train = x_train.astype('float32')/255
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')/255
y_test = y_test.astype('float32')
#5. Check the shape of dataset' images
#Check the shape of dataset' images
import numpy as np
np.sqrt(4096)

#change data into numpy array
x_train= x_train.to_numpy()
y_train= y_train.to_numpy()
x_test= x_test.to_numpy()
y_test= y_test.to_numpy()

#reshape train data and test data into 64 * 64 * 1channel
x_train = x_train.reshape(x_train.shape[0], 64, 64, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 64, 64, 1).astype('float32')
#convert y_train and y_test into 15 categories
y_train = kr.utils.to_categorical(y_train, 15)
y_test = kr.utils.to_categorical(y_test, 15)
num_classes = y_train.shape[1]
num_classes

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', input_shape=(64, 64, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())
model1 = model.fit(x_train, y_train,batch_size=128, epochs=20)
 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import os
import cv2
import glob as gb
from skimage.filters import gaussian
from skimage.morphology import dilation,erosion
from skimage.feature import canny
from skimage.measure import find_contours
import imutils
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from kerastuner.tuners import GridSearch
import time
import tensorflow as tf

def findedges(image):
    gray = cv2.GaussianBlur(image, (1, 1), 0)
    edged = cv2.Canny(gray, 100, 400)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged


def getimageconturs(edged):
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    return contours


def getboxes(contours,orig):
    boxes = []
    centers = []
    for contour in contours:
        box = cv2.minAreaRect(contour)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box
        if (dist.euclidean(tl, bl)) > 0 and (dist.euclidean(tl, tr)) > 0:
            boxes.append(box)
    return boxes


image_size=(120,120)
code={"EOSINOPHIL":0,"LYMPHOCYTE":1,"MONOCYTE":2,"NEUTROPHIL":3}
def getcode(n):
    if type(n)==str:
        for x,y in code.items():
            if n==x:
                return y 
    else:
        for x,y in code.items():
            if n==y:
                return x
            


def loaddata():
    datasets=["/notebook/CNN_Evalu/data/DataSet/Model#15/dataset2-master/dataset2-master/images/TRAIN/",
          "/notebook/CNN_Evalu/data/DataSet/Model#15/dataset2-master/dataset2-master/images/TEST/",]



    
    images=[]
    labels=[]
    count=0
    for dataset in datasets:
        for folder in os.listdir(dataset):
            files=gb.glob(pathname=str(dataset+folder+"/*.jpeg"))
            label=getcode(folder)
            for file in files:
                image = cv2.imread(file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # add padding to the image to better detect cell at the edge
                image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[198, 203, 208])
                
                #thresholding the image to get the target cell
                image1 = cv2.inRange(image,(80, 80, 180),(180, 170, 245))
                
                # openning errosion then dilation
                kernel = np.ones((3, 3), np.uint8)
                kernel1 = np.ones((5, 5), np.uint8)
                img_erosion = cv2.erode(image1, kernel, iterations=2)
                image1 = cv2.dilate(img_erosion, kernel1, iterations=5)
                
                #detecting the blood cell
                edgedImage = findedges(image1)
                edgedContours = getimageconturs(edgedImage)
                edgedBoxes =  getboxes(edgedContours, image.copy())
                if len(edgedBoxes)==0:
                    count +=1
                    continue
                # get the large box and get its cordinate
                last = edgedBoxes[-1]
                max_x = int(max(last[:,0]))
                min_x = int( min(last[:,0]))
                max_y = int(max(last[:,1]))
                min_y = int(min(last[:,1]))
                
                # draw the contour and fill it 
                mask = np.zeros_like(image)
                cv2.drawContours(mask, edgedContours, len(edgedContours)-1, (255,255,255), -1) 
                
                # any pixel but the pixels inside the contour is zero
                image[mask==0] = 0
                
                # extract th blood cell
                image = image[min_y:max_y, min_x:max_x]

                if (np.size(image)==0):
                    count +=1
                    continue
                # resize th image
                image = cv2.resize(image, image_size)

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

    images = np.array(images, dtype = 'float32')
    labels = np.array(labels, dtype = 'int32')

    return images,labels



images,labels=loaddata()
images,labels=shuffle(images,labels,random_state=10)
images=images/255
train_image,test_image,train_label,test_label=train_test_split(images,labels,test_size=.2)
test_image,val_image,test_label,val_label=train_test_split(test_image,test_label,test_size=.5)
def displayrandomimage(image,label,typeofimage):
    plt.figure(figsize=(15,15))
    plt.suptitle("some random image of "+typeofimage,fontsize=17)
    for n,i in  enumerate(list(np.random.randint(0,len(image),36))):
        plt.subplot(6,6,n+1)
        plt.imshow(image[i])
        plt.axis("off")
        plt.title(getcode(label[i]))
#Here we call displayrandomimage function to display some image of train image
displayrandomimage(train_image,train_label,"train image")

# Build model with hyperparameters
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation='relu',input_shape=(120,120,3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Choice('dense_1_units', [32, 128, 512, 1024]),
        activation='relu'
    ))

    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])

    return model

# Create a GridSearch tuner
tuner = GridSearch(
    build_model,
    objective='accuracy',
    max_trials=3,
    executions_per_trial=3,
    directory='/home/o/oananbeh/notebook/CNN_Evalu/data/Test Model/DU',
    project_name='Model#15_1'
)
tuner.search_space_summary()

# Search for the best parameters
tuner.search(
    train_image,
    train_label,
    epochs=10,
     batch_size=32,
    validation_data=(val_image,val_label)
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the hyperparameters
print(f"""
The hyperparameter search is complete. 
the optimal number of units in the dense layer 1 is {best_hps.get('dense_1_units')}

""")
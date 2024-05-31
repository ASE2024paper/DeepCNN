#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:26:04 2023

@author: wardat
"""


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model("/Users/wardat/Downloads/CNN/model/transformer_model")
tokenizer_path = '/Users/wardat/Downloads/CNN/model/transformer_tokenizer.pickle'
datasetPath='/Users/wardat/Downloads/CNN/result/Evaluate_transformer_Result.csv'


import pickle

# Load the tokenizer
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the dataset
data = pd.read_csv(datasetPath)

# remove rows with empty 'Commit Message' values
data.dropna(subset=['code'], inplace=True)
# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(data["label"])


# Create a dictionary to map each class label to a corresponding number
label_to_number = {
    "KL": 1,
    "KI": 2,
    "PL": 3,
    "FR": 4,
    "DU": 5,
    "SE": 6,
}


def predict(text):
    # Tokenize and pad the input text
    input_sequence = tokenizer.texts_to_sequences([text])
    input_sequence = pad_sequences(input_sequence, maxlen=200, padding="post", truncating="post")
    
    # Make a prediction
    predictions = model.predict(input_sequence)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=-1)[0]
    
    # Get the probability of the predicted class
    predicted_class_probability = predictions[0][predicted_class]

    #get the label back as a string, use the inverse_transform method of the LabelEncoder
    predicted_label = label_encoder.inverse_transform([predicted_class])

    print("Predicted class:", predicted_class)
    print("Predicted label:", predicted_label[0])
    print("Predicted class probability:", predicted_class_probability)
    # Get the corresponding number for the predicted label
    predicted_number = label_to_number[predicted_label[0]]
    print("Predicted number:", predicted_number)
    
# Test with a sample text statement
#sample_text = "SequentialConv2D 64 kernel_size 3 3 strides 1 1 padding same activation relu kernel_initializer glorot_uniform input_shape 28 28 1MaxPooling2D pool_size 2 2 padding sameConv2D 512 kernel_size 3 3 strides 1 1 padding same activation relu kernel_initializer glorot_uniformMaxPooling2D pool_size 2 2 padding sameConv2D 64 kernel_size 3 3 strides 1 1 padding same activation relu kernel_initializer glorot_uniformMaxPooling2D pool_size 2 2 padding sameFlattenDense 6 activation reluDropout 0 20Dense num_classes activation softmaxcompile loss keras losses categorical_crossentropy optimizer keras optimizers Adam metrics accuracyhistory fit x_train y_train validation_data x_test y_test epochs epochs batch_size batch_size"
sample_text = "sequentialconv2d 64 kernel_size 3 3 strides 1 1 padding same activation relu kernel_initializer glorot_uniform input_shape 28 28 1maxpooling2d pool_size 2 2 padding sameconv2d 64 kernel_size 1 1 strides 1 1 padding same activation relu kernel_initializer glorot_uniformmaxpooling2d pool_size 2 2 padding sameconv2d 64 kernel_size 3 3 strides 1 1 padding same activation relu kernel_initializer glorot_uniformmaxpooling2d pool_size 2 2 padding sameflattendense 6 activation reludropout 0 20dense num_classes activation softmaxcompile loss keras losses categorical_crossentropy optimizer keras optimizers adam metrics accuracyhistory fit x_train y_train validation_data x_test y_test epochs epochs batch_size batch_size"
predict(sample_text)
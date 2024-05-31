#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:27:24 2023

@author: wardat
"""

# import library
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import re

path='/Users/wardat/Desktop/DeepCNN/' 

data_df=pd.read_csv(path+'DataSetCNN.csv')

# convert  to lowercase, and remove any special characters

# Define a function to preprocess the text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text

# Apply the preprocessing to the 'code' column
data_df['code'] = data_df['code'].apply(preprocess_text)
data_df.head()

other=['label','code'] 
data_df=data_df[other]

#Create a final dataframe with all numerical variables in the first columns.
final_df = data_df.drop('label', axis=1)
target = data_df.label
print(f"Original class counts: {Counter(target)}")


sampler = RandomOverSampler(random_state=1)
X_trainres, y_trainres = sampler.fit_resample(final_df, target)
print(f"Class counts after resampling {Counter(y_trainres)}")




#Connect final with target
RandomOverSampler_dataSet = pd.concat([X_trainres, y_trainres],axis=1)

RandomOverSampler_dataSet['label'].value_counts()


RandomOverSampler_dataSet.to_csv(path+'balancedDataset.csv',index=False) 
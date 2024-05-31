#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:26:04 2023

@author: wardat
"""

import os
import pandas as pd
# Specify the base path where your folders are located
base_path = '/Users/wardat/Downloads/CNN/src/DataPrepare/LargeDataset'

# Create a list to store the data
data_list = []

# Loop through each folder
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    # Check if the current item is a folder
    if os.path.isdir(folder_path):
        # Loop through each txt file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                # Open the txt file and read the contents
                with open(file_path, 'r') as file:
                    # Create a dictionary with the folder name as the "label" and the text as the "code"
                    data_dict = {'label': folder, 'code': file.read()}
                    # Add the dictionary to the data list
                    data_list.append(data_dict)

# Convert the data list to a DataFrame
df = pd.DataFrame(data_list)

# Save the DataFrame as a CSV file
df.to_csv('DataSetCNN.csv', index=False)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from collections import Counter

# Define path
modelPath='/Users/wardat/Downloads/CNN/model/transformer_model'
tokenizer_path = '/Users/wardat/Downloads/CNN/model/ransformer_tokenizer.pickle'

resultPath='/Users/wardat/Downloads/CNN/output'
lossPlotPath=resultPath+'loss_plot.png'
confusionMatrixPath=resultPath+'confusion_matrix.png'
datasetPath='/Users/wardat/Downloads/CNN/data/balancedDataset.csv'
datasetPathTest='/Users/wardat/Downloads/CNN/result/Evaluate_transformer_Result.csv'
transformerMetricsPath=resultPath+'transformer_metrics.csv'
classResultsPath=resultPath+"transformer_class_results.csv"
# Define hyperparameters

batch_size=32
epochs=200
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

# Implement a Transformer block as a layer

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

# Implement embedding layer
#Two seperate embedding layers, one for tokens, one for token index (positions).

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
# Load the dataset
data = pd.read_csv(datasetPath)

dataTest = pd.read_csv(datasetPathTest)

# remove rows with empty 'code' values
data.dropna(subset=['code'], inplace=True)
dataTest.dropna(subset=['code'], inplace=True)
#The maximum length sentance to use it in maxlen
max_length = data['code'].apply(len).max()

print(f"The maximum length is: {max_length}")


# get al the uniq words to use it in  vocab_size
all_text = " ".join(data['code'])
words = all_text.split()
word_counts = Counter(words)

print(f"Total number of unique words: {len(word_counts)}")


vocab_size = 150  # Only consider the top 20k words
maxlen = 100  # Only consider the first 200 words of each code 

# Determine the number of classes
num_classes = len(data["label"].unique())

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenize the text data
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data["code"])

# Convert text to sequences
x_train = tokenizer.texts_to_sequences(train_data["code"])
x_val = tokenizer.texts_to_sequences(val_data["code"])
y_test = tokenizer.texts_to_sequences(dataTest["code"])

# Pad sequences
x_train = pad_sequences(x_train, maxlen=maxlen, padding="post", truncating="post")
x_val = pad_sequences(x_val, maxlen=maxlen, padding="post", truncating="post")
x_test = pad_sequences(y_test, maxlen=maxlen, padding="post", truncating="post")

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(data["label"])
y_train = label_encoder.transform(train_data["label"])
y_val = label_encoder.transform(val_data["label"])
y_test = label_encoder.transform(dataTest["label"])

print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")


import pickle

# Save the tokenizer
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
num_classes = len(data["label"].unique())
print(f"Number of class: {num_classes}")

#Create classifier model using transformer layer
inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)


class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)
        

#Train and Evaluat

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# model.compile(
#     optimizer="adam",
#     loss=keras.losses.SparseCategoricalCrossentropy(), 
#     metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy"),get_f1,precision,recall], 
# )

history = model.fit(
    x_train, y_train, batch_size, epochs, validation_data=(x_val, y_val)
)

# loss, accuracy, f1, prec, rec = model.evaluate(x_val, y_val, verbose=0)

# print(f"Validation f1: {round(f1,6)}")
# print(f"Validation accuracy: {round(accuracy,6)}")
# print(f"Validation Precision: {round(prec,6 )}")
# print(f"Validation Recall: {round(rec,6)}")

# y_pred = model.predict(x_val)


# Get training and validation loss history
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(loss) + 1)

# # Visualize loss history
# plt.plot(epoch_count, loss, 'r--')
# plt.plot(epoch_count, val_loss, 'b-')
# plt.legend(['Training Loss', 'Validation Loss'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()
# plt.savefig(lossPlotPath)  # Save the plot as a PNG file



from sklearn.metrics import classification_report, accuracy_score
import csv
# Get predictions for validation data
y_pred = model.predict(x_val)
y_pred_classes = y_pred.argmax(axis=-1)

y_test_predict = model.predict(x_test)
y_pred_classes_test = y_test_predict.argmax(axis=-1)
print(y_pred_classes_test)
accuracy_test = accuracy_score(y_test, y_pred_classes_test)
print(y_test)
print("{0}".format(accuracy_test))
# Calculate metrics
report = classification_report(y_val, y_pred_classes, output_dict=True)
accuracy = accuracy_score(y_val, y_pred_classes)

# Print metrics
print(f"Accuracy: {round(accuracy, 2)}")
print(f"Precision: {round(report['weighted avg']['precision'], 2)}")
print(f"Recall: {round(report['weighted avg']['recall'], 2)}")
print(f"F1 Score: {round(report['weighted avg']['f1-score'], 2)}")

with open(transformerMetricsPath, mode='w', newline='') as metrics_file:
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(['Metric', 'Value'])
    metrics_writer.writerow(['Accuracy', round(accuracy, 2)])
    metrics_writer.writerow(['Precision', round(report['weighted avg']['precision'], 2)])
    metrics_writer.writerow(['Recall', round(report['weighted avg']['recall'], 2)])
    metrics_writer.writerow(['F1 Score', round(report['weighted avg']['f1-score'], 2)])
import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import os
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

# Hyperparameters
max_length = 256
batch_size = 64
epochs = 10
learning_rate = 1e-5
test_size = 0.2
validation_split = 0.1

# Set paths and labels
data_folder = "/Users/Desktop/DeepCNN/LargeDataSet/ModelCode"
classes = ["FR", "KI", "KL", "PL", "SE", "DU"]

# Load tokenizer and model
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

#tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
#model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")

# Read .txt files and preprocess
classes =["FR", "KI", "KL", "PL", "SE"]
#texts, labels = [], []

texts, labels = [], []

for i, class_name in enumerate(classes):
    class_folder = os.path.join(data_folder, class_name)
    for filename in os.listdir(class_folder):
        with open(os.path.join(class_folder, filename), mode='r', encoding='utf-8') as file:
            texts.append(file.read())
            labels.append(i)


## Encode labels
#le = LabelEncoder()
#y_encoded = le.fit_transform(y)
#y_encoded = to_categorical(y_encoded, num_classes=len(labels))
#
## Tokenize and create input tensors
#input_ids, attention_masks = [], []
#
#for text in texts:
#    encoding = tokenizer.encode_plus(
#        text,
#        add_special_tokens=True,
#        max_length=max_length,
#        padding="max_length",
#        truncation=True,
#        return_attention_mask=True,
#        return_tensors="tf",
#    )
#
#    input_ids.append(encoding["input_ids"])
#    attention_masks.append(encoding["attention_mask"])
#
#input_ids = tf.concat(input_ids, axis=0)
#attention_masks = tf.concat(attention_masks, axis=0)
# Tokenize the texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="tf")

# Split the dataset into training and testing sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs["input_ids"], labels, test_size=0.3, random_state=42, stratify=labels)
train_attention_mask, test_attention_mask = train_test_split(inputs["attention_mask"], test_size=0.3, random_state=42, stratify=labels)

# Load the model
model = TFAutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=len(classes))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# Train the model
model.fit({"input_ids": train_inputs, "attention_mask": train_attention_mask}, np.array(train_labels), batch_size=8, epochs=3)

# Make predictions
predictions = model.predict({"input_ids": test_inputs, "attention_mask": test_attention_mask})

# Convert logits to class labels
predicted_labels = np.argmax(predictions.logits, axis=1)

# Calculate the scores
f1 = f1_score(test_labels, predicted_labels, average="weighted")
accuracy = accuracy_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels, average="weighted")
precision = precision_score(test_labels, predicted_labels, average="weighted")

print("F1 Score:", f1)
print("Accuracy Score:", accuracy)
print("Recall Score:", recall)
print("Precision Score:", precision)
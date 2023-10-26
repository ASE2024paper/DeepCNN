import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Hyperparameters
max_length = 64
batch_size = 64
epochs = 10
learning_rate = 2e-5
test_size = 0.2
validation_split = 0.1

# Set paths and labels
data_folder = "/Users/Desktop/DeepCNN/LargeDataSet/ModelCode"
labels = ["DU", "FR", "KI", "KL", "PL", "SE", "DU"]

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))
#tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
#model = AutoModelForMaskedLM.from_pretrained("bert-large-uncased")

# Read .txt files and preprocess
texts, y = [], []

for label in labels:
    label_folder = os.path.join(data_folder, label)
    for file in os.listdir(label_folder):
        if file.endswith(".py"):
            with open(os.path.join(label_folder, file), "r") as f:
                texts.append(f.read())
                y.append(label)


# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_encoded = to_categorical(y_encoded, num_classes=len(labels))

# Tokenize and create input tensors
input_ids, attention_masks = [], []

for text in texts:
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="tf",
    )

    input_ids.append(encoding["input_ids"])
    attention_masks.append(encoding["attention_mask"])

input_ids = tf.concat(input_ids, axis=0)
attention_masks = tf.concat(attention_masks, axis=0)

# Split into train and test sets
X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(input_ids.numpy(), y_encoded, attention_masks.numpy(), test_size=test_size, random_state=42)

# Convert NumPy arrays to TensorFlow tensors
X_train, X_test, y_train, y_test, mask_train, mask_test = map(tf.convert_to_tensor, [X_train, X_test, y_train, y_test, mask_train, mask_test])


# Fine-tune model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.fit([X_train, mask_train], y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)


# Predict and evaluate
y_pred = model.predict([X_test, mask_test])[0].argmax(axis=-1)
y_true = y_test.numpy().argmax(axis=-1)

f1 = f1_score(y_true, y_pred, average="weighted")
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, average="weighted")
precision = precision_score(y_true, y_pred, average="weighted")

print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
# CONFUSION: 0.0
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# --------------------- Load data & model ---------------------
data = tf.data.Dataset.load('../Models/newDataValidation') # Load the data from a file
data = data.shuffle(4000)
data = data.batch(16)
data = data.prefetch(16)
#model = keras.models.load_model('../Models/modelNew.h5')
model = keras.models.load_model('../Models/model2.h5')

# --------------------- Confusion Matrix ---------------------
LANGUAGES = ['english', 'french', 'german', 'spanish']

# Define a function to get the label and prediction values for a batch of data
def get_labels_and_predictions(model, data):
    labels = []
    predictions = []
    for batch in data:
        batch_labels = batch[1]
        batch_predictions = model.predict(batch[0],verbose=0)
        labels.append(batch_labels)
        predictions.append(batch_predictions)
    labels = tf.concat(labels, axis=0)
    predictions = tf.concat(predictions, axis=0)
    return labels, predictions

# Get the labels and predictions for the validation dataset
val_labels, val_predictions = get_labels_and_predictions(model, data)
val_labels = tf.squeeze(val_labels, axis=1)
val_predictions = tf.squeeze(val_predictions, axis=1)

confusion = tf.math.confusion_matrix(tf.argmax(val_labels, axis=1),
                                     tf.argmax(val_predictions, axis=1),
                                     num_classes=4)
print(confusion)

# --------------------- Plot the confusion matrix---------------------
confusion = confusion.numpy()

# Calculate precision and recall
precision = np.diag(confusion) / np.sum(confusion, axis=0)
recall = np.diag(confusion) / np.sum(confusion, axis=1)

# Create plot
fig, ax = plt.subplots()
im = ax.imshow(confusion, cmap='Blues')

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# Set axis labels and tick labels
ax.set_xticks(np.arange(len(LANGUAGES)))
ax.set_yticks(np.arange(len(LANGUAGES)))
ax.set_xticklabels(LANGUAGES)
ax.set_yticklabels(LANGUAGES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

# Add text annotations for precision and recall
text_colors = ["black", "white"]
thresh = im.norm(confusion.max()) / 2.
for i in range(len(LANGUAGES)):
    for j in range(len(LANGUAGES)):
        color = text_colors[int(im.norm(confusion[i, j]) > thresh)]
        ax.text(j, i, f"{confusion[i, j]:d}\n(p={precision[j]:.2f}, r={recall[i]:.2f})", ha="center", va="center", color=color)

# Show plot
plt.show()







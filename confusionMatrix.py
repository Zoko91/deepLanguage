# --------------------- Model Testing ---------------------
# CONFUSION: 0.0
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# --------------------- Load data & model ---------------------
data = tf.data.Dataset.load('./Data/Augmented/test_dataset').batch(32).prefetch(16)
# CHANGE THE MODEL AND DATA TO TRY DIFFERENT CONFIGURATIONS AS WANTED:
# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-
# data = tf.data.Dataset.load('./Data/test_dataset').batch(32).prefetch(16)
# data  = tf.data.Dataset.load('./Data/self_validation').batch(32).prefetch(16)
model = keras.models.load_model('./Models/__largeModels/model3.h5')

# --------------------- Confusion Matrix ---------------------
LANGUAGES = ['french', 'spanish', 'german', 'english']

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

confusion = tf.math.confusion_matrix(tf.argmax(val_labels, axis=1),
                                     tf.argmax(val_predictions, axis=1),
                                     num_classes=4)
print(confusion)

# --------------------- Plot the confusion matrix---------------------
confusion_matrix = confusion.numpy()
# Normalize the confusion matrix
confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

# Create a figure and axis
fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix, cmap='Blues')

# Add axis labels and a title
ax.set_xticks(np.arange(len(LANGUAGES)))
ax.set_yticks(np.arange(len(LANGUAGES)))
ax.set_xticklabels(LANGUAGES)
ax.set_yticklabels(LANGUAGES)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')

# Add text annotations for each cell
thresh = confusion_matrix.max() / 2.
for i in range(len(LANGUAGES)):
    for j in range(len(LANGUAGES)):
        ax.text(j, i, format(confusion_matrix[i, j], '.2f'),
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")

# Add a colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Normalized Frequency", rotation=-90, va="bottom")

# Show the plot
plt.show()





import matplotlib.pyplot as plt
import numpy as np

LANGUAGES = ['english', 'french', 'german', 'spanish']

# Define the confusion matrix as a numpy array
confusion_matrix = np.array([[17,  7,  1,  1],
                             [27, 51, 15, 34],
                             [ 0,  1,  0,  0],
                             [ 0,  3,  1,  1]])

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

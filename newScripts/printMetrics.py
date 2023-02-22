# Load Dependencies
from tensorflow import keras
import matplotlib.pyplot as plt


model = keras.models.load_model('../Models/modelNew.h5')


# Load the training history
history = model.history

# Plot the training and validation loss over time
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

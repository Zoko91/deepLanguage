# Load Dependencies
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('Models/model3.h5')
tf.keras.utils.plot_model(model, show_shapes=True)

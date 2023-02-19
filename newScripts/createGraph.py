# Load Dependencies
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('../Models/modelNew.h5')
tf.keras.utils.plot_model(model, show_shapes=True)

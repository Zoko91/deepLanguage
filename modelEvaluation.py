# VALIDATION: 0.0
import tensorflow as tf
from tensorflow import keras

LANGUAGES = ['english', 'french', 'german', 'spanish']


# --------------------- Load data & model ---------------------
data = tf.data.Dataset.load('Models/newDataValidation') # Load the data from a file
data = data.shuffle(4000)
data = data.batch(16)
data = data.prefetch(16)
model = keras.models.load_model('Models/model3.h5')

# --------------------- Evaluate ---------------------
val_loss, val_accuracy = model.evaluate(data)
print("Validation Loss: ", val_loss)
print("Validation Accuracy: ", val_accuracy)

# --------------------- Recap ---------------------

# --------------------- Validation ----------------
# data : newDataValidation
# model : modelNew
# accuracy : 0.86
# loss : 0.39

# data : newDataValidation
# model : model3
# accuracy : 0.88
# loss : 0.58



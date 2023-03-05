# Machine learning model for language recognition

import tensorflow as tf

# --------------------- Load the data ---------------------
data = tf.data.Dataset.load('../Models/data')

# --------------------- Prepare the data ---------------------
# Shuffle the data
data = data.shuffle(40000)
data = data.batch(32)
data = data.prefetch(32)
# Split the data into training and validation sets
# Train has 34000 samples, test has 6000 samples
train = data.take(34000)
test = data.skip(34000).take(6000)
# samples, labels = train.as_numpy_iterator().next()
# print(samples.shape)  # result:  (32, 1, 153, 13)


# --------------------- Build the model ---------------------
# Model N°1
# -----------------------------------------------------------
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Reshape((153, 13, 1), input_shape=(1, 153, 13)))
# model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Model N°2
# -----------------------------------------------------------
model = tf.keras.Sequential()
# Creates an empty sequential model in TensorFlow using the Keras API.
# A sequential model is a linear stack of layers, where you use the large majority of the layers in practice.

# Input reshape
model.add(tf.keras.layers.Reshape((153, 13, 1), input_shape=(1, 153, 13)))


# Convolutional layers
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1)))

# Fully connected layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
# Output layer
model.add(tf.keras.layers.Dense(units=4, activation='softmax'))


#  Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

# --------------------- Train the model ---------------------
history = model.fit(train, epochs=10, validation_data=test, verbose=1)
model.save('../Models/modelNew.h5')

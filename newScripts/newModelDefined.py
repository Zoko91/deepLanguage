# Machine learning model for language recognition

import tensorflow as tf

# --------------------- Load the data ---------------------
data = tf.data.Dataset.load('../Models/newData')

# --------------------- Prepare the data ---------------------
# Shuffle the data
data = data.shuffle(60000)
data = data.batch(64)
data = data.prefetch(32)
# Split the data into training and validation sets
# Train has 34000 samples, test has 6000 samples
train = data.take(40000)

# Model N°2
# -----------------------------------------------------------
model = tf.keras.Sequential()
# Input reshape
model.add(tf.keras.layers.Reshape((153, 13, 1), input_shape=(1, 153, 13)))
# Convolutional layers
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# Fully connected layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
# Output layer
model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
model.add(tf.keras.layers.Reshape((1, 4)))

#  Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --------------------- Train the model ---------------------
history = model.fit(train, epochs=10, verbose=1)
model.save('../Models/modelNew.h5')

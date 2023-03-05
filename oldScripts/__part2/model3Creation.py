# # # # # Playground # # # #
############################

import tensorflow as tf

# --------------------- Load the data ---------------------
data = tf.data.Dataset.load('Models/newData')


# --------------------- Prepare the data ---------------------
# Shuffle the data
data = data.shuffle(50000)
data = data.batch(64)
data = data.prefetch(32)
# Split the data into training and validation sets
# Train has 34000 samples, test has 6000 samples
train = data.take(35000)
test = data.skip(35000).take(8000)


# Model
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
model.add(tf.keras.layers.BatchNormalization())

# Fully connected layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.2))
# Output layer
model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
model.add(tf.keras.layers.Reshape((1, 4))) # match the labels which are [[0,0,0,1]] distribution over 4 languages

#  Compile the model
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(train,validation_data=test, epochs=20,verbose=1,callbacks=[early_stopping_callback])
#model.save('Models/model3.h5')
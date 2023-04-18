# --------------------- Model Creation 3 ---------------------
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------- Load the data ---------------------
train_dataset = tf.data.Dataset.load('../../../Data/train_dataset').batch(128).prefetch(16)
val_dataset = tf.data.Dataset.load('../../../Data/validation_dataset').batch(128).prefetch(16)

def create_model(input_shape=(13, 157, 1), num_classes=4):
    # --------------------- Model --------------------------------
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Convolutional layers
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(4, 4), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(2, 2), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))

    # Recurrent layers
    model.add(tf.keras.layers.Permute((2, 1, 3)))
    model.add(tf.keras.layers.Reshape((-1, 512)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True)))

    # Fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.3))

    # Output layer
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))
    return model


# Create the model
model = create_model()
model.summary()
# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=[early_stopping_callback],verbose=1)
model.save('../../../Models/__largeModels/RNN.h5')


# Plot the loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot the accuracy
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

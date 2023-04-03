# --------------------- Model Creation 3 ---------------------
import matplotlib.pyplot as plt
import tensorflow as tf

# --------------------- Load the data ---------------------
train_dataset = tf.data.Dataset.load('../../Data/Augmented/train_dataset').batch(64).prefetch(1)
val_dataset = tf.data.Dataset.load('../../Data/Augmented/val_dataset').batch(64).prefetch(1)


def create_model(input_shape=(13, 157, 1), num_classes=4):

    # Define model
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Convolutional layers
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(0.10))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(0.10))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(0.10))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(0.1))
    # Fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.10))
    # Output layer
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    return model

# Create the model
model = create_model()
model.summary()

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Early stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train the model
history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[early_stopping_callback],verbose=1)
model.save('../../Models/__largeModels/augmented.h5')


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

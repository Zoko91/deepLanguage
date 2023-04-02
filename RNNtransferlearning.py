# --------------------- Model Creation 3 ---------------------
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------- Load the data ---------------------
train_dataset = tf.data.Dataset.load('./Data/train_dataset').batch(128).prefetch(16)
val_dataset = tf.data.Dataset.load('./Data/validation_dataset').batch(128).prefetch(16)


# Load the previous model
model = tf.keras.models.load_model('./Models/__largeModels/RNN.h5')
# Freeze the weights of the convolutional layers
for layer in model.layers[:14]:
    layer.trainable = False

# Remove dense layers
model.pop()
model.pop()
# Add LSTM layers
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True),name='ltsm_added1'))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True),name='ltsm_added2'))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256),name='ltsm_added3'))
# Add new dense layer
model.add(tf.keras.layers.Dense(units=4, activation='softmax'))
# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[early_stopping_callback],verbose=1)
model.save('./Models/__largeModels/RNNtransfer.h5')


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

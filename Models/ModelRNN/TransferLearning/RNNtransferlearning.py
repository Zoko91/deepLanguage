import tensorflow as tf

def create_new_model(num_classes=4, model_path='../../__largeModels/RNN.h5'):
    # Load the model and get the layers
    model = tf.keras.models.load_model(model_path)
    layers = model.layers[:17]  # Get the first 14 layers from the loaded model

    # Create a new model with the loaded layers and add a dense layer for classification
    # Define the new model with the conv layers
    new_model = tf.keras.Sequential()
    new_model.add(tf.keras.layers.Input(shape=(13, 157, 1)))
    for layer in layers:
        new_model.add(layer)

    # Recurrent layers
    new_model.add(tf.keras.layers.Dropout(0.3,name='dropout_10'))
    new_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True,name='ltsm_11'),name="bidi"))
    new_model.add(tf.keras.layers.Dropout(0.3,name='dropout_11'))
    new_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256,name='ltsm_13'),name='bidi2'))
    new_model.add(tf.keras.layers.Dropout(0.4,name='dropout_13'))

    # Output layer
    new_model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax',name='dense_classifier'))
    return new_model

new_model = create_new_model()
for layer in new_model.layers[:14]:
    layer.trainable = False

new_model.summary()
# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
new_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# --------------------- Load the data ---------------------
train_dataset = tf.data.Dataset.load('../../../Data/train_dataset').batch(128).prefetch(16)
val_dataset = tf.data.Dataset.load('../../../Data/validation_dataset').batch(128).prefetch(16)


# Train the model
history = new_model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[early_stopping_callback],verbose=1)
new_model.save('../../__largeModels/RNNtransfer.h5')

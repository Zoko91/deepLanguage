# Machine learning model for language recognition

import tensorflow as tf

# --------------------- Load the data ---------------------
data = tf.data.Dataset.load('Models/data')

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


# --------------------- Train the model ---------------------
# history = model.fit(train, epochs=10, validation_data=test, verbose=1)


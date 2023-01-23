# README - deepLanguage
*Computer science project*

*Goal: Initation to deep learning*

*Project: Recognize a language from an audio file*
```html
Recognize language of an audio file using deep learning models

The code is an implementation of a convolutional neural network for language recognition using PyTorch. It first loads the audio files and converts them to spectrograms, then splits the data into training, validation, and test sets. The data is then converted to PyTorch tensors, which are used as inputs to the model.

The LanguageRecognitionModel class is a subclass of torch.nn.Module and defines the layers of the model in its __init__ method. The forward pass of the model is defined in the forward method.

The model is trained using stochastic gradient descent (SGD) with a learning rate of 0.01, and the cross-entropy loss is used as the loss function. The model is evaluated on the validation set after each epoch, and the final test loss is printed after training is complete.
```

# How to improve

1.
Use a larger training dataset: Typically, the more training data a model has, the better it will perform. Using a larger and more diverse dataset of audio files and their corresponding labels could potentially improve the performance of the model.

2. 
Use a more powerful model architecture: The model architecture used in this code appears to be relatively simple, with only two convolutional layers and two fully-connected layers. Using a more powerful model architecture, such as a deeper convolutional neural network or a transformer model, could potentially improve the performance of the model.

3. 
Use a different optimizer and/or loss function: The code uses stochastic gradient descent (SGD) with a learning rate of 0.01 as the optimizer, and the cross-entropy loss as the loss function. Depending on the specific problem, using a different optimizer and/or loss function could potentially improve the performance of the model. For example, using the Adam optimizer and the mean squared error (MSE) loss function could be a good choice for regression problems, while using the RMSProp optimizer and the binary cross-entropy loss function could be a good choice for binary classification problems.

4. 
Perform hyperparameter tuning: The model's hyperparameters, such as the learning rate and the number of epochs, can have a significant impact on its performance. Performing a hyperparameter search or using a hyperparameter optimization algorithm to find the best combination of hyperparameters for the specific problem could potentially improve the performance of the model.

# Dataset - Audio files 
```html
The Common Voice dataset: This dataset, which was released by Mozilla in 2017, contains over 500 hours of voice data from more than 20,000 contributors, in a variety of languages. The dataset can be downloaded from the Common Voice website: 
<a href="https://voice.mozilla.org/en/datasets"/>
```
# Sources
```html
https://www.tensorflow.org/tutorials/audio/simple_audio
https://www.connectedpapers.com/main/85dc10fe4b8ad2761a81e9dce7755f87d822428e/Multiclass-Language-Identification-using-Deep-Learning-on-Spectral-Images-of-Audio-Signals/graph
https://paperswithcode.com/task/spoken-language-identification
https://github.com/HPI-DeepLearning/crnn-lid
```

# Help

To join the **environment**, run the following command in your terminal:

```bash
source path/to/myenv/bin/activate
```
To leave the **environment**, run the following command in your terminal:

```bash
deactivate
```

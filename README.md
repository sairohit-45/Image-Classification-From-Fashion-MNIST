# Classifying Images from Fashion MNIST

This repository contains code for training a neural network model to classify FashionMNIST images using PyTorch. FashionMNIST is a dataset of Zalando's article images, consisting of 60,000 training examples and 10,000 test examples, each of which is a 28x28 grayscale image belonging to one of 10 classes.

![d3defbdcc02d7e8f00899237079e8f7f.png](https://imgtr.ee/images/2023/09/23/d3defbdcc02d7e8f00899237079e8f7f.png)

## Table of Contents

- [Dataset](#dataset)
- [Labels](#labels)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Results](#results)


## Dataset

The FashionMNIST dataset is used for this project. It is loaded using torchvision and consists of the following parts:

- *Training dataset*: 60,000 images for training the model.
- *Testing dataset*: 10,000 images for evaluating the model.

## Labels

Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Model Architecture

The neural network used for this classification task is a simple feedforward neural network with two hidden layers:

1. *Input layer*: 784 neurons (28x28 pixels)
2. *Hidden layer 1*: 16 neurons with ReLU activation
3. *Hidden layer 2*: 32 neurons with ReLU activation
4. *Output layer*: 10 neurons (corresponding to the 10 classes)

## Training

The training of the model is done in two phases, each consisting of 5 epochs. The chosen optimizer is Stochastic Gradient Descent (SGD), and the learning rate is set to 0.1. During training, both training and validation loss and accuracy are monitored.
![bf3faa1cf149a98d4a9cb61742447643.png](https://imgtr.ee/images/2023/09/23/bf3faa1cf149a98d4a9cb61742447643.png)

## Evaluation

The evaluation of the model is performed on the validation dataset. It calculates the loss and accuracy for the validation dataset to track the model's performance during training.

## Prediction

You can also use the trained model to make predictions on new images. The predict_image function takes a PIL image as input, converts it to a PyTorch tensor, and uses the model to make predictions.

## Results

After training the model for 10 epochs, it achieves an accuracy of approximately 85% on the validation dataset.
![27e45efd0bb6361055b9246e6ba6e4a1.png](https://imgtr.ee/images/2023/09/23/27e45efd0bb6361055b9246e6ba6e4a1.png)

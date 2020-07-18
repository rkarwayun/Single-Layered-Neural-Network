# Single-Layered-Neural-Network

Single Layered Neural Network from scratch in Python. This code uses only numpy and pandas (to load data).
Sigmoid activation function was used in the hidden layer and the output layer used Softmax activation function along with Cross Entropy loss.




#### Some notes:

This model takes in data from "train_data.csv" and labels from "train_labels.csv". This is done in the takeInput() function. The labels are expected to be one hot encoded.

It then divides the data into 80%/20% train/test subsets. This division is done in the main function.

Number of input nodes, number of hidden nodes and output nodes can also be modified in the main function. So can the learning rate and number of epochs.

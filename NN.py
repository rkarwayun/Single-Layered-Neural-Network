# References:
# http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
# https://peterroelants.github.io/posts/cross-entropy-logistic/
# https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60
# https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
# https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/?

import pandas as pd
import numpy as np
import pickle


# Function to calculate sigmoid.
def sigmoid(A):
    return 1 / (1 + np.exp(-A))


# Function to calculate softmax.
def softmax(A):
    # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    # Softmax can give NaN value due to floating point limitation in Numpy. Hence it is not a stable function.
    # To make it stable, a constant is multiplied to both numerator and denominator. A popular choice is max(A)
    # denotes maximum value in Array "A".
    e = np.exp(A - np.max(A))
    return e / e.sum(axis=0, keepdims=True)


# Function to calculate derivative of a sigmoid. Here, the value that we take as input is the value sigmoid(x).
def sigmoid_d(A):
    sig = sigmoid(A)
    return sig * (1 - sig)


# This function reads the input files.
def takeInput():
    df_in = pd.read_csv("train_data.csv")
    df_lab = pd.read_csv("train_labels.csv")
    return np.array(df_in), np.array(df_lab)


# This function executes the forward propagation.
def feedforward(X_in):
    db['Zh'] = db['Wh'].dot(X_in.T) + db['bh']      # Zh is the value derived from multiplying weights with input and
    # adding bias for hidden layer.
    db['Ah'] = sigmoid(db['Zh'])                    # Ah is the output from the hidden layer.
    db['Zo'] = db['Wo'].dot(db['Ah']) + db['bo']    # Zo is the value derived from multiplying weights with output of
    # hidden layer and adding bias for output layer.
    db['Ao'] = softmax(db['Zo'])                    # Ao is the output from the output layer. It is basically the output
    # from our network.

    # 'db' is a database where we are storing all our weights, and output and interim results.
    return db['Ao']


# This function executes the back propagation.
def backprop(X, y):
    # dJ / dWo = (dJ / dAo) * (dAo / dZo) * (dZo / dWo)
    # dJ / dbo = (dJ / dAo) * (dAo / dZo) * (dZo / dbo)
    # dJ / dWh = (dJ / dAo) * (dAo / dZo) * (dZo / dAh) * (dAh / dZh) * (dZh / dWh)
    # dJ / dbh = (dJ / dAo) * (dAo / dZo) * (dZo / dAh) * (dAh / dZh) * (dZh / dbh)
    d_Z = db['Ao'] - y.T  # = (dJ / dZo).
    d_Wo = d_Z.dot(db['Ah'].T) / m
    d_bo = np.sum(d_Z, axis=1, keepdims=True) / m

    # t1, t2 and t3 are just interim results for calculating dJ / dWh and dJ / dbh
    t1 = db['Wo'].T.dot(d_Z)
    t2 = sigmoid_d(db['Ah'])
    t3 = t1 * t2

    d_Wh = t3.dot(X) / m
    d_bh = np.sum(t3, axis=1, keepdims=True) / m

    # Returning all the weight changes.
    return d_Wo, d_Wh, d_bo, d_bh


# Function used for prediction using MLP for determining Training and Test Accuracies during the training process.
def predict(X, y):
    y_out = feedforward(X)

    # Checking if the maximum accuracy is reported for the same values as it is in the actual labels.
    y_hat = np.argmax(y_out, axis=0)
    Y = np.argmax(y, axis=1)

    # Calculating accuracy.
    accuracy = (y_hat == Y).mean()
    return accuracy * 100


# This functions trains the MLP.
def fit(X, y, lr=0.1, epochs=200):

    # Full batch gradient descent.
    for i in range(1, epochs + 1):
        # Fist we do forward propagation to get the output from that network.
        feedforward(X)
        A = db['Ao']

        # Calculating the cost function (Cross Entropy).
        cost = (-1) * np.mean(y * np.log(A.T + 1e-8))   # Adding a small constant to handle case when A might be zero.

        # Printing the training accuracy every 10 epochs.
        if i % 10 == 0:
            val = predict(X, y)
            print("For epoch ", str(i), ", cost is: ", str(cost),
                  ", and the Training Accuracy is: ", str(val))

        # Back-propagating to update weights and biases.
        d_Wo, d_Wh, d_bo, d_bh = backprop(X, y)
        db['Wo'] = db['Wo'] - lr * d_Wo
        db['Wh'] = db['Wh'] - lr * d_Wh
        db['bo'] = db['bo'] - lr * d_bo
        db['bh'] = db['bh'] - lr * d_bh


# This is a database where we store all the weights, biases, interim results and output for latest epoch.
db = {}


# Setup function. Initializes the weights and biases to random values at the beginning of training.
def setup():
    db['Wh'] = np.random.randn(hidden_size, inp_size)
    db['bh'] = np.zeros((hidden_size, 1))
    db['Wo'] = np.random.randn(out_size, hidden_size)
    db['bo'] = np.zeros((out_size, 1))


# Encoding the output of the MLP into One Hot Encoding. Used in test_mlp to encode the results.
def encode(opt):
    # Here opt is the transpose of the output of feedforward() function.
    ans = np.zeros((opt.shape[0], opt.shape[1]))
    for i in range(len(opt)):
        maxi = np.argmax(opt[i])
        ans[i][maxi] = 1
    return ans


# This is the main function
if __name__ == '__main__':

    # Reading the input.
    data, lab = takeInput()

    # Train-test split (80%-20%).
    X_train = data[:19802, :]
    X_test = data[19802:, :]
    y_train = lab[:19802, :]
    y_test = lab[19802:, :]

    inp_size = 784
    hidden_size = 90
    out_size = 4
    l_rate = 0.3
    total_epochs = 1000

    # Initializing the network.
    setup()
    m = X_train.shape[0]

    # Training the network.
    fit(X_train, y_train, l_rate, total_epochs)

    # Dumping the Weights and Biases of the trained model so that they can be used later.
    pickle.dump(db, open("weights.p", "wb"))

    # Determining the test accuracy of the model.
    test1 = predict(X_train, y_train)
    test2 = predict(X_test, y_test)
    print("Train accuracy is: ", str(test1), " and Test accuracy is: ", str(test2))

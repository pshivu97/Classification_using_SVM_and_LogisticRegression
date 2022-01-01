import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt


def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Initializing bias term
    biases = np.full((n_data, 1), 1)

    # Appending bias term to training data
    X = np.concatenate((biases, train_data), axis=1)

    w = initialWeights.reshape((n_features + 1), 1)

    # Calculating theta using Logistic Regression sigmoid activation function
    theta = sigmoid(np.dot(X, w))

    # Error function for Logistic Regression
    error_func = labeli * np.log(theta) + (1.0 - labeli) * np.log(1.0 - theta)

    # Error for Logistic Regression
    error = (-1.0 / n_data) * (np.sum(error_func))

    # Error gradiance for Logistic Regression
    error_grad = np.sum((theta - labeli) * X, axis=0) / n_data

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    # label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    n_data = data.shape[0]

    # Initializing bias term
    biases = np.full((n_data, 1), 1)

    # Appending bias term to data
    X = np.concatenate((biases, data), axis=1)

    # Predicted Probability for each class by Logistic Regression
    predicted_probability = sigmoid(np.dot(X, W))

    # Selecting the labels which have highest probability
    predicted_labels = np.argmax(predicted_probability, axis=1).reshape(-1, 1)

    return predicted_labels


def calculate_theta_mlr(X, W):
    """
    Computes theta for MultiClass Logistic Regression using Multinoulli Formula
    Input:
        X - the data matrix of size N x D
        W - the weight vector of size D x 10

    Output:
        Theta - Theta calculated using Multinoulli Formula of size N x 10
    """
    numerator = np.exp(np.dot(X, W))
    denominator = np.sum(numerator, axis=1).reshape(-1, 1)
    return numerator / denominator


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Initializing bias term
    bias = np.full((n_data, 1), 1)

    # Appending bias term to training data
    X = np.concatenate((bias, train_data), axis=1)

    W = params.reshape((n_feature + 1, n_class))

    # Using Multinoulli Logistic Regression Formula
    theta = calculate_theta_mlr(X, W)

    # Error for MultiClass Logistic Regression
    error = - (np.sum(np.sum(Y * np.log(theta))))

    # Error gradiance for MultiClass Logistic Regression
    error_grad = np.dot(X.T, (theta - labeli)).flatten()

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    n_data = data.shape[0]

    # Initializing bias term
    bias = np.full((n_data, 1), 1)

    # Appending bias term to data
    X = np.concatenate((bias, data), axis=1)

    # Predicted Probability/Theta for each class by MultiClass Logistic Regression
    predicted_probability = calculate_theta_mlr(X, W)

    predicted_labels = np.argmax(predicted_probability, axis=1).reshape(n_data, 1)

    return predicted_labels


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

for i in range(n_class):
    current_predicted_label = predicted_label[np.where(train_label==np.array([i]))]
    current_train_label = train_label[np.where(train_label==np.array([i]))]
    float_comparison = (current_predicted_label == current_train_label).astype(float)
    print('\n Training set Accuracy for class:' + str(i) + '  ' + str(100 * np.mean(float_comparison)) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

for i in range(n_class):
    current_predicted_label = predicted_label[np.where(validation_label==np.array([i]))]
    current_validation_label = validation_label[np.where(validation_label==np.array([i]))]
    float_comparison = (current_predicted_label == current_validation_label).astype(float)
    print('\n Validation set Accuracy for class:' + str(i) + '  ' + str(100 * np.mean(float_comparison)) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

for i in range(n_class):
    current_predicted_label = predicted_label[np.where(test_label==np.array([i]))]
    current_test_label = test_label[np.where(test_label==np.array([i]))]
    float_comparison = (current_predicted_label == current_test_label).astype(float)
    print('\n Testing set Accuracy for class:' + str(i) + '  ' + str(100 * np.mean(float_comparison)) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

# Randomly Selecting 1000 samples for training SVM
index = np.random.randint(50000, size=10000)
X_svm = train_data[index, :]
y_svm = train_label[index, :]

# Initializing Linear SVM
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_svm, y_svm.ravel())

# Printing the results for linear SVM Model
print('\n-----------------------------------------SVM with Linear Kernel-----------------------------------------\n')
print('\n SVM Training Data Accuracy : ' + str(100 * svm_linear.score(X_svm, y_svm)) + '%')
print('\n Training Data Accuracy : ' + str(100 * svm_linear.score(train_data, train_label)) + '%')
print('\n Validation Data Accuracy : ' + str(100 * svm_linear.score(validation_data, validation_label)) + '%')
print('\n Testing Data Accuracy : ' + str(100 * svm_linear.score(test_data, test_label)) + '%')

# RBF with Gamma = 1

svm_rbf = svm.SVC(kernel='rbf', gamma=1.0)
svm_rbf.fit(X_svm, y_svm.ravel())

print('\n-----------------------------------------SVM RBF ; Gamma = 1-----------------------------------------\n')
print('\n SVM Training Data Accuracy : ' + str(100 * svm_rbf.score(X_svm, y_svm)) + '%')
print('\n Training Accuracy : ' + str(100 * svm_rbf.score(train_data, train_label)) + '%')
print('\n Validation Accuracy : ' + str(100 * svm_rbf.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy : ' + str(100 * svm_rbf.score(test_data, test_label)) + '%')

# RBF with gamma = default(0.0)

svm_rbf_default_gamma = svm.SVC(kernel='rbf', gamma='auto')
svm_rbf_default_gamma.fit(X_svm, y_svm.ravel())

print(
    '\n-----------------------------------------SVM RBF ; Gamma = default(0.0)-----------------------------------------\n')
print('\n SVM Training Data Accuracy : ' + str(100 * svm_rbf_default_gamma.score(X_svm, y_svm)) + '%')
print('\n Training Accuracy : ' + str(100 * svm_rbf_default_gamma.score(train_data, train_label)) + '%')
print('\n Validation Accuracy : ' + str(100 * svm_rbf_default_gamma.score(validation_data, validation_label)) + '%')
print('\n Testing Accuracy : ' + str(100 * svm_rbf_default_gamma.score(test_data, test_label)) + '%')

# RBF with gamma = default(0.0) and Changing values of C to find the optimal one

svm_Xtrain_accuracy = np.zeros(11, float)
svm_train_accuracy = np.zeros(11, float)
svm_validation_accuracy = np.zeros(11, float)
svm_test_accuracy = np.zeros(11, float)
C_values = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

for i, C in enumerate(C_values):
    svm_rbf_C = svm.SVC(kernel='rbf', C=C)
    svm_rbf_C.fit(X_svm, y_svm.ravel())

    svm_Xtrain_accuracy[i] = 100 * svm_rbf_C.score(X_svm, y_svm)
    svm_train_accuracy[i] = 100 * svm_rbf_C.score(train_data, train_label)
    svm_validation_accuracy[i] = 100 * svm_rbf_C.score(validation_data, validation_label)
    svm_test_accuracy[i] = 100 * svm_rbf_C.score(test_data, test_label)

    print('\n-----------------------------------------SVM RBF ; Gamma = default(0.0) ; C  = ' + str(
        C) + '-----------------------------------------\n')
    print('\n SVM Training Data Accuracy : ' + str(svm_Xtrain_accuracy[i]) + '%')
    print('\n Training Accuracy : ' + str(svm_train_accuracy[i]) + '%')
    print('\n Validation Accuracy : ' + str(svm_validation_accuracy[i]) + '%')
    print('\n Testing Accuracy : ' + str(svm_test_accuracy[i]) + '%')

# Accuracy vs C Plot
plt.plot(C_values, svm_Xtrain_accuracy, color='b')
plt.plot(C_values, svm_train_accuracy, color='g')
plt.plot(C_values, svm_validation_accuracy, color='r')
plt.plot(C_values, svm_test_accuracy, color='y')

plt.figure(figsize=(16, 16))
plt.title('Accuracy vs C', fontweight='bold')
plt.xlabel('C', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.xticks(np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
plt.yticks(np.arange(95, 100, step=0.1))
plt.grid(True)

plt.legend(['SVM Training Data Accuracy', 'Training Accuracy', 'Validation Accuracy', 'Test Accuracy'])
plt.show()
"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
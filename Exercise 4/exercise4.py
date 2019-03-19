# -*- coding: utf-8 -*-
# ------------------------------#
# Arvid Edenheim                #
# ID 106502907                  #
# Introduction to Data Science  #
# Exercise 4                    #
# ------------------------------#

import time
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.preprocessing


def load_data(train_ratio=0.5):
    feature_col = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    label_col = ['Y']
    data = pd.read_csv('./HTRU2/HTRU_2.csv', names=feature_col + label_col)
    X = data[feature_col]
    y = data[label_col]

    return sklearn.model_selection.train_test_split(X, y, test_size=1 - train_ratio)


def scale_features(X_train, X_test, low, upp):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(
        feature_range=(low, upp)).fit(np.vstack((X_train, X_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)

    return X_train_scale, X_test_scale


def sgd(data, y, learn_rate, total_epochs, reg):
    theta = theta_ = np.zeros(len(data[0]))

    for epoch in range(total_epochs):
        k = 0

        for row in data:
            yhat = predict(row, theta)
            grad = (y[k] - yhat) * row - reg * theta
            theta = theta + learn_rate * grad
            k += 1

        if has_converged(theta, theta_, len(data[0])):
            break

        theta_ = np.copy(theta)
        epochs = epoch + 1

    return theta, epochs


def has_converged(thet1, thet2, length):
    eps = 1e-10
    diff = 0

    for i in range(0, length):

        if (np.abs(thet2[i] - thet1[i]) > diff):
            diff = np.abs(thet2[i] - thet1[i])

    return diff < eps


def predict(row, theta):

    return 1.0 / (1.0 + np.exp(- (np.dot(theta, row))))


def log_reg_test(test, theta):
    predictions = []
    rounded_predictions = []

    for row in test:
        predictions.append(predict(row, theta))

    for threshold in predictions:
        predictions_ = []
        for val in predictions:
            predictions_.append(threshold_round(val, threshold))
        rounded_predictions.append(predictions_)

    return rounded_predictions


def threshold_round(prediction, threshold):

    return 0 if prediction < threshold else 1


def calculate_tpr_fpr(prediction, y):
    tp = fp = fn = tn = 0

    for i in range(len(prediction)):
        predval = prediction[i]

        if predval == y[i]:
            if predval == 1:
                tp += 1
            else:
                tn += 1
        elif predval == 1:
            fp += 1
        else:
            fn += 1

    return [tp / (tp + fn + 1), fp / (fp + tn + 1)]


def calculate_points(predictions, y):
    points = [[], []]

    for prediction in predictions:
        tpr, fpr = calculate_tpr_fpr(prediction, y)
        points[0].append(tpr)
        points[1].append(fpr)

    return points


def plot_roc_curve(tpr, fpr):
    auc = metrics.auc(fpr, tpr)
    print('Accuracy: ', auc)
    plt.plot(fpr, tpr, 'b', lw=1, label='AUC = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def main():
    start_time = time.time()

    X_train, X_test, y_train, y_test = load_data(train_ratio=.5)

    X_train_scale, X_test_scale = scale_features(X_train, X_test, -1, 1) #not sure why I scaled -1, 1, it just seems more reasonable
                                                                         #since x contains negative data, but i guess 0,1 should be equally "good"
    X_train_scale = np.concatenate(
        (np.ones((len(X_train_scale), 1)), X_train_scale), axis=1)
    X_test_scale = np.concatenate(
        (np.ones((len(X_test_scale), 1)), X_test_scale), axis=1)

    theta, epochs = sgd(X_train_scale, y_train.as_matrix(),
                learn_rate=0.01, total_epochs=100, reg=0.0001)
    print('SGD-algorithm elapsed time: ', time.time() - start_time, 's')

    predictions = log_reg_test(X_test_scale, theta)
    points = calculate_points(predictions, y_test.as_matrix())

    print('Number of epochs: ', epochs)
    print('Total elapsed time: ', int((time.time() - start_time) / 60) , 'min, ', (time.time() - start_time) % 60, 's')

    plot_roc_curve(np.sort(points[0]), np.sort(points[1]))


main()

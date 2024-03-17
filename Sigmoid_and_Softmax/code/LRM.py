#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        import pandas as pd
        n_samples, n_features = X.shape
        y = pd.get_dummies(labels).values
        self.W = np.array([[0] * self.k] * n_features)
        for i in range(0, self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for j in range(0, n_samples, batch_size):
                batch_indices = indices[j:j + batch_size]
                temp_gradient = 0
                batch_len = 0
                for k in batch_indices:
                    batch_len = batch_len + 1
                    temp_gradient = temp_gradient + self._gradient(X[k], y[k])
                self.W = self.W - self.learning_rate*(temp_gradient/batch_len)
            # print('Weight for epoch:',i,'is: ', '\n', self.W)

        return self
    
		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        _g = _x.reshape(-1,1) * (self.softmax(np.dot(_x, self.W)) - _y.reshape(1,-1))
        return _g

		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    
		### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        pred = np.argmax(np.dot(X,self.W), axis = 1)
        return pred

		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		# ### YOUR CODE HERE
        predicted_values = self.predict(X)
        count = 0 
        for i in range(0, labels.shape[0]):
            if(predicted_values[i] == labels[i]):
                count = count + 1
        accuracy_score = count/labels.shape[0]

        return accuracy_score

		# ### END YOUR CODE


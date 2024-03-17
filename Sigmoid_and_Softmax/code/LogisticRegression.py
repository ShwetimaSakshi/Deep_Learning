import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        self.W = np.array([0]*(n_features))
        for i in range(0, self.max_iter):
            temp_gradient = 0 
            for j in range(0, n_samples):
                temp_gradient = temp_gradient + self._gradient(X[j], y[j])
            self.W = self.W - self.learning_rate*(temp_gradient/n_samples)
		### END YOUR CODE
        return self  

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		# ## YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.array([0]*(n_features))
        for i in range(0, self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for j in range(0, n_samples, batch_size):
                batch_indices = indices[j:j + batch_size]
                temp_gradient = 0 
                batch_len = 0 
                for k in batch_indices:
                    batch_len = batch_len + 1
                    temp_gradient += self._gradient(X[k], y[k])
                self.W = self.W - self.learning_rate*(temp_gradient/batch_len)
            # print('Weight for epoch:',i, 'is: ', '\n', self.W)

		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		## YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for i in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            batch_indices = indices
            for j in batch_indices:
                temp_gradient = self._gradient(X[j], y[j])
                self.W = self.W - (self.learning_rate * temp_gradient)
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		# ### YOUR CODE HERE
        _g = -( _y * _x)/(1 + np.exp(_y * np.dot(_x, self.W)))
        return _g
        ### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        pred_proba = []
        for x in X:
            logits = 1 / (1 + np.exp(-np.dot(x,self.W)))
            pred_proba.append(logits)

        pred_proba = np.stack((np.array(pred_proba), 1 - np.array(pred_proba)), axis=-1)
        return pred_proba

		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        sigmoid_func = 1 /(1 + np.exp(-np.dot(X,self.W)))
        pred = sigmoid_func
        
        pred[pred>=0.5] = 1
        pred[pred<0.5] = -1
        
        return pred
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        predicted_values = self.predict(X)
        count = 0 
        for i in range(0, y.shape[0]):
            if(predicted_values[i] == y[i]):
                count = count + 1
        accuracy_score = count/y.shape[0]
    
        return accuracy_score
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self


import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features.     
    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.    
    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure()
    plt.scatter(X[y==1,0],X[y==1,1],  marker='o', c='skyblue', label='1')
    plt.scatter(X[y==-1,0],X[y==-1,1],  marker='o', c='lightgreen', label='-1')
    plt.legend(['1','-1'],loc="lower left", title="Classes")
    plt.xlabel("Feature 1: Symmetry")
    plt.ylabel("Feature 2: Intensity")
    plt.title("Fig: train_features")
    plt.savefig("train_features.png")

    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.     
    Args:
    	X: An array of shape [n_samples, 2].
    	y: An array of shape [n_samples,]. Only contains 1 or -1.
    	W: An array of shape [n_features,].
    
    Returns:
    	No return. Save the plot to 'train_result_sigmoid.*' and include it
    	in submission.
    '''
    # ### YOUR CODE HERE
    plt.figure()
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', c='skyblue', label='1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], marker='o', c='lightgreen', label='-1')
    x_points = np.linspace(-0.27, -0.25, 10)
    y_points = - (W[0] / ( W[2])) - x_points * (W[1] / (W[2]))
    plt.plot(x_points, y_points, c='red')
    plt.legend(['1','-1'], loc="lower left", title="Classes")
    plt.xlabel("Feature 1: Symmetry")
    plt.ylabel("Feature 2: Intensity")
    plt.title("Fig: train_result_sigmoid")
    plt.savefig("train_result_sigmoid.png")

    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.     
    Args:
    	X: An array of shape [n_samples, 2].
    	y: An array of shape [n_samples,]. Only contains 0,1,2.
    	W: An array of shape [n_features, 3].
    
    Returns:
    	No return. Save the plot to 'train_result_softmax.*' and include it
    	in submission.
    '''
    ### YOUR CODE HERE
    W = W.T
    plt.figure()
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='orange', label='Class 0', s=30)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='skyblue', label='Class 1', s=30)
    plt.scatter(X[y == 2, 0], X[y == 2, 1], marker='o', color='lightgreen', label='Class 2', s=30)
    x_points0 = np.linspace(-0.45, 0, 10)
    y_points0 = - (W[0, 0] / W[0, 2]) - x_points0 * (W[0, 1] / W[0, 2])
    plt.plot(x_points0, y_points0, color='black')
    x_points1 = np.linspace(-0.40, -0.1, 10)
    y_points1 = - (W[1, 0] / W[1, 2]) - x_points1 * (W[1, 1] / W[1, 2])
    plt.plot(x_points1, y_points1, color='blue')
    x_points2 = np.linspace(-0.28, -0.19, 10)
    y_points2 = - (W[2, 0] / W[2, 2]) - x_points2 * (W[2, 1] / W[2, 2])
    plt.plot(x_points2, y_points2, color='red')
    plt.legend(loc="lower left", title="Classes")
    plt.xlabel("Feature 1: Symmetry")
    plt.ylabel("Feature 2: Intensity")
    plt.title("Fig: train_result_softmax")
    plt.savefig('train_result_softmax.png')

    ### END YOUR CODE

def main():
    # ------------Data Preprocessing------------
    # Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)
    
    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  
    
    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 
    
     # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


    # ------------Logistic Regression Sigmoid Case------------

    ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)
    
    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    
    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    
    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    
    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    learningrate = [0.001, 0.01, 0.05, 0.03, 0.1]

    best_model_accuracy = 0
    best_parameters = np.zeros(train_X.shape[0])
    best_model = None
    best_learning_rate = 0

    for lr in learningrate:
        logisticR_classifier = logistic_regression(learning_rate=lr, max_iter=1000)

        logisticR_classifier.fit_BGD(train_X, train_y)
        if logisticR_classifier.score(valid_X, valid_y) >= best_model_accuracy:
            best_learning_rate = lr
            best_model_accuracy = logisticR_classifier.score(valid_X, valid_y)
            best_param = logisticR_classifier.get_params()
            best_model = logisticR_classifier
        
        logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
        if logisticR_classifier.score(valid_X, valid_y) >= best_model_accuracy:
            best_learning_rate = lr
            best_model_accuracy = logisticR_classifier.score(valid_X, valid_y)
            best_param = logisticR_classifier.get_params()
            best_model = logisticR_classifier

        logisticR_classifier.fit_miniBGD(train_X, train_y, 50)
        if logisticR_classifier.score(valid_X, valid_y) >= best_model_accuracy:
            best_learning_rate = lr
            best_model_accuracy = logisticR_classifier.score(valid_X, valid_y)
            best_param = logisticR_classifier.get_params()
            best_model = logisticR_classifier

        logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
        if logisticR_classifier.score(valid_X, valid_y) >= best_model_accuracy:
            best_learning_rate = lr
            best_model_accuracy = logisticR_classifier.score(valid_X, valid_y)
            best_param = logisticR_classifier.get_params()
            best_model = logisticR_classifier

        logisticR_classifier.fit_SGD(train_X, train_y)
        if logisticR_classifier.score(valid_X, valid_y) >= best_model_accuracy:
            best_learning_rate = lr
            best_model_accuracy = logisticR_classifier.score(valid_X, valid_y)
            best_param = logisticR_classifier.get_params()
            best_model = logisticR_classifier

    print("----------------------------------------------")
    print("Logistic Regression Sigmoid case")
    print("Accuracy of the best model selected: ", best_model_accuracy)
    print("Learning rate of the best model selected: ", best_learning_rate)
    print("Parameters of the best model selected: ", best_param)
   
    
    ### END YOUR CODE
    
    # Visualize the your 'best' model after training.
    ### YOUR CODE HERE
    visualize_result(train_X[:, 1:3], train_y, best_param)
    ### END YOUR CODE
    
    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    test_raw_data, test_labels = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(test_raw_data)
    test_y_all, test_idx = prepare_y(test_labels)
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = -1
    data_shape= test_y.shape[0] 
    
    print("----------------------------------------------")
    print("Logistic Regression Sigmoid case")
    print("Best model accuracy for test data: ", best_model.score(test_X, test_y))
    print("----------------------------------------------")
    
    
    ### END YOUR CODE
    
    
    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all
    
    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))
    
    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    lr_multiclass = [0.001, 0.01, 0.05, 0.03, 0.1]
    
    best_model_accuracy_multiclass = 0
    best_parameters_multiclass = np.zeros(train_X.shape[0])
    best_model_multiclass = None
    best_learning_rate_multi = 0

    for lr in lr_multiclass:
        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=lr, max_iter=1000,  k= 3)
    
        logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 50)
        if logisticR_classifier_multiclass.score(valid_X, valid_y) >= best_model_accuracy_multiclass:
            best_learning_rate_multi = lr
            best_model_accuracy_multiclass = logisticR_classifier_multiclass.score(valid_X, valid_y)
            best_param_multiclass = logisticR_classifier_multiclass.get_params()
            best_model_multiclass = logisticR_classifier_multiclass
    
        logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
        if logisticR_classifier_multiclass.score(valid_X, valid_y) >= best_model_accuracy_multiclass:
            best_learning_rate_multi = lr
            best_model_accuracy_multiclass = logisticR_classifier_multiclass.score(valid_X, valid_y)
            best_param_multiclass = logisticR_classifier_multiclass.get_params()
            best_model_multiclass = logisticR_classifier_multiclass
    
    print("----------------------------------------------")
    print("Logistic Regression Multiple-class case")
    print("Accuracy of the best model selected: ", best_model_accuracy_multiclass)
    print("Learning rate of the best model selected: ", best_learning_rate_multi)
    print("Parameters of the best model selected: ", best_param_multiclass)
    
    
    ### END YOUR CODE
    
    # Visualize the your 'best' model after training.
    # visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())
    visualize_result_multi(train_X[:, 1:3], train_y, best_param_multiclass)
    
    
    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    test_X = test_X_all
    test_y = test_y_all
    print("----------------------------------------------")
    print("Logistic Regression Multiple-class case")
    print("Best model accuracy for test data: ", best_model_multiclass.score(test_X, test_y))
    
    
    ### END YOUR CODE
    
    
    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 
    
    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = 0
    
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.01, max_iter=10000, k=2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print("----------------------------------------------")
    print("Logistic Regression Softmax Binary-class case")
    print("weights:")
    print(logisticR_classifier_multiclass.get_params())
    print("Training Accuracy:", logisticR_classifier_multiclass.score(train_X, train_y))
    print("Validation Accuracy:", logisticR_classifier_multiclass.score(valid_X, valid_y))
    print("Testing Accuracy:", logisticR_classifier_multiclass.score(test_X, test_y))
    
    ### END YOUR CODE
    
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   
    
    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = -1
    
    logisticR_classifier = logistic_regression(learning_rate=0.02, max_iter=10000)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print("----------------------------------------------")
    print("Logistic Regression Sigmoid Binary-class case")
    print("weights:")
    print(logisticR_classifier.get_params())
    print("Training Accuracy:", logisticR_classifier.score(train_X, train_y))
    print("Validation Accuracy:", logisticR_classifier.score(valid_X, valid_y))
    print("Testing Accuracy:", logisticR_classifier.score(test_X, test_y))
    
    
    ### END YOUR CODE
    
    
    ###############Compare and report the observations/prediction accuracy
    

    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE

    # setting the learning rates as same.

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  

    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = 0
    
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.01, max_iter=3, k=2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, train_X.shape[0])
    print("----------------------------------------------")
    print("Exploring Logistic Regression Softmax Binary-class case")
    print("weights:")
    print(logisticR_classifier_multiclass.get_params())
    print("Training Accuracy:", logisticR_classifier_multiclass.score(train_X, train_y))
    print("Validation Accuracy:", logisticR_classifier_multiclass.score(valid_X, valid_y))
    print("Testing Accuracy:", logisticR_classifier_multiclass.score(test_X, test_y))
    
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   
    
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = -1

    logisticR_classifier = logistic_regression(learning_rate=0.01, max_iter=3)
    logisticR_classifier.fit_miniBGD(train_X, train_y, train_X.shape[0])
    print("----------------------------------------------")
    print("Exploring Logistic Regression Sigmoid Binary-class case")
    print("weights:")
    print(logisticR_classifier.get_params())
    print("Training Accuracy:", logisticR_classifier.score(train_X, train_y))
    print("Validation Accuracy:", logisticR_classifier.score(valid_X, valid_y))
    print("Testing Accuracy:", logisticR_classifier.score(test_X, test_y))



    # setting the learning rates so that w_1 -w_2 = w

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  

    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = 0
    
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=10, k=2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print("----------------------------------------------")
    print("Exploring Logistic Regression Softmax Binary-class case")
    print("weights:")
    print(logisticR_classifier_multiclass.get_params())
    print("Training Accuracy:", logisticR_classifier_multiclass.score(train_X, train_y))
    print("Validation Accuracy:", logisticR_classifier_multiclass.score(valid_X, valid_y))
    print("Testing Accuracy:", logisticR_classifier_multiclass.score(test_X, test_y))
    
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   
    
    test_X = test_X_all[test_idx]
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y == 2)] = -1

    logisticR_classifier = logistic_regression(learning_rate=1, max_iter=10)
    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print("----------------------------------------------")
    print("Exploring Regression Sigmoid Binary-class case")
    print("weights:")
    print(logisticR_classifier.get_params())
    print("Training Accuracy:", logisticR_classifier.score(train_X, train_y))
    print("Validation Accuracy:", logisticR_classifier.score(valid_X, valid_y))
    print("Testing Accuracy:", logisticR_classifier.score(test_X, test_y))


    # ### END YOUR CODE

    # ------------End------------
    

if __name__ == '__main__':
	main()

import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    training_data_x = []
    training_data_y = []
    test_data_x = []
    test_data_y = []
    
    # there are 5 batches for test data
    for i in range(1, 6):
        training_batch_file_path = os.path.join(data_dir, f"data_batch_{i}")
        with open(training_batch_file_path, 'rb') as file:
            dict = pickle.load(file, encoding='bytes')
            training_data_x.extend(dict[b'data'])
            training_data_y.extend(dict[b'labels'])

    test_batch_file_path = os.path.join(data_dir, "test_batch")
    with open(test_batch_file_path, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
        test_data_x.extend(dict[b'data'])
        test_data_y.extend(dict[b'labels'])

    x_train = np.array(training_data_x)
    y_train = np.array(training_data_y)
    x_test = np.array(test_data_x)
    y_test = np.array(test_data_y)

    # print shapes 
    print('Shape of training data and label:', x_train.shape, y_train.shape)
    print('Shape of testing data and label:', x_test.shape, y_test.shape)
    print("------------------------------------")
    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid

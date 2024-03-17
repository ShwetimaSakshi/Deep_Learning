import numpy as np

""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        # padding with 4 extra pixels on all sides and set the constant value to 0
        pad_width = ((4, 4), (4, 4), (0, 0))
        image = np.pad(image, pad_width, mode='constant', constant_values=0)
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        # the size of crop section
        cropping_size = (32, 32)
        #randomly selecting a crop section in upper left
        upper_left_x = np.random.randint(0, image.shape[1] - cropping_size[1])
        upper_left_y = np.random.randint(0, image.shape[0] - cropping_size[0])
        image = image[upper_left_y:upper_left_y+cropping_size[0], upper_left_x:upper_left_x+cropping_size[1]]
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        threshold = 0.5
        flip_image_prob = np.random.rand()
        if threshold > 0.5:
            imgage = np.fliplr(imgage)
        ### YOUR CODE HERE

    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean_pixel_value = np.mean(image, axis=(0, 1))
    std_dev_pixel_value = np.std(image, axis=(0, 1))
    image = (image - mean_pixel_value) / std_dev_pixel_value
    ### YOUR CODE HERE

    return image
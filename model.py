import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import utils
from PIL import Image
import math
import keras_model
from keras_model import KerasModel
from random import shuffle
import random


def augment_data(images, measurements, augment=False, multivariant=False):
    """ This method apply augmentation on the given images and measurements.
            images: the input images array to augment
            mesurements: the input mesurements array to augments
            augment: a bool to apply all augmentation mainly shadow, False by default
            multivariant: a bool to specify if the data is multivariant, False by default
    """
    # Initializing empty augmented arrays
    augmented_images = []  # + images
    augmented_measurements = []  # + measurements
    # Temp iterator to hold both inputs
    c = list(zip(images, measurements))
    # Shuffling the inputs to get a random representation of the data
    shuffle(c)
    # passing back the items to their own lists
    images, measurements = zip(*c)
    # Applying augmentation on a fraction of the images
    n_images_to_augment = int(len(images) * 0.30)
    # For every image and its steering angle flip the image and apply augmentation on shuffled
    # fraction of the images
    for image, steering_angle in zip(images, measurements):
        # augment is only true in training
        if augment and n_images_to_augment > 0:
            shadow_image = add_random_shadow(image)
            augmented_images.append(shadow_image)
            augmented_measurements.append(steering_angle)
            n_images_to_augment -= 1
            # Add a flipped version of the augmented image to eliminate any bias
            flipped_shadow_image, flipped_shadow_steering_angle = flip_image_steering(
                shadow_image, steering_angle, multivariant)
            augmented_images.append(flipped_shadow_image)
            augmented_measurements.append(flipped_shadow_steering_angle)
        # Create a flipped version of the image and steering angle
        flipped_image, flipped_steering_angle = flip_image_steering(image,
                                                                    steering_angle,
                                                                    multivariant=multivariant)
        augmented_images.append(flipped_image)
        augmented_images.append(image)
        augmented_measurements.append(flipped_steering_angle)
        augmented_measurements.append(steering_angle)
    return augmented_images, augmented_measurements


def flip_image_steering(image, measurements, multivariant=False):
    """ This method creates a flipped version of the input image and steering angle
            image: input image to flip
            steering_angle: either a steering angle or a steering angel and a throttle value
                            if multivariant is true
            multivariant: bool that specifies if the steering angle has multiple values, False by
                          default
    """
    flipped_image = np.fliplr(image)
    if multivariant:
        # flip the steering angle
        flipped_steering_angle = measurements[0] * -1.0
        # Return the throttle value measurements[1] as it is
        return flipped_image, (flipped_steering_angle, measurements[1])
    else:
        # If not multivariant then measurements is the steering angle so flip it
        flipped_steering_angle = measurements * -1.0
        return flipped_image, flipped_steering_angle


def add_random_shadow(img):
    """This method adds random shadow to the sides of the road for the selected passed image
        img: The image to add random shadow
    """
    # Getting the image dimentions
    height = img.shape[0]
    width = img.shape[1]
    # print('{}x{}'.format(width, height))
    # Shadow reectangle dimentions
    rectangle_height = 160
    rectangle_width = 130

    # Random Rectangle.
    # Get a random point in the cropped image
    y1, x1 = random.randint(70, height - 25), random.randint(0, width)
    # Then calculate a closing rectangle point of the shadow rectangle dimentions and within the
    # cropped image
    y2 = y1 + rectangle_height if (y1 + rectangle_height) < height else y1 - rectangle_height
    x2 = x1 + rectangle_width if (x1 + rectangle_width) < width else x1 - rectangle_width
    # print('({}, {}), ({}, {})'.format(x1, y1, x2, y2))

    # Ordering the points
    y_from, y_to = (y1, y2) if y1 < y2 else (y2, y1)
    x_from, x_to = (x1, x2) if x1 < x2 else (x2, x1)

    # Loop through points between the (y1,x1) and (y2,x2) points
    for i in range(y_from, y_to):
        for j in range(x_from, x_to):
            # Set the numpy array to enable writing
            img.setflags(write=1)
            # Create a dark shadowy array of pixel accross the image channels
            dark_pixel_value = tuple([int(x * 0.5) for x in img[i][j]])
            # Set the image pixels to the dark shadow value
            img[i][j] = dark_pixel_value
    # Reset numpy array to read-only
    img.setflags(write=0)
    return img


def load_csv_data(file_path):
    """ This method reads a csv file from file_path
            file_path: a str of the location of the csv file to open
    """
    # Reading the recorded data from the .csv file
    lines = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def get_images_and_measurements(lines, multivariant=False):
    """ This method loads images and gets mesurement values from a str array input line
            lines: array of str arrays of data which are a line in a csv document
            multivariant: a bool to indicate wether or not its a multivariant model, False by default
        returns
    """

    # Empty arrays to load values into
    images = []
    measurements = []

    # Iterate through all input lines
    for line in lines:
        # Get steering angle value and cast it from str to float
        steering_center_angel = float(line[3])
        if multivariant:
            # Add the throttle value if multivariant is True
            throttle = float(line[4])
        # Correction factor for steering angles
        correction_factor = 0.2  # should change this to a computed parameter in the future
        # Adjusted Steering angels for side cameras images
        steering_left_angel = steering_center_angel + correction_factor
        steering_right_angel = steering_center_angel - correction_factor

        # Read in the images form center, left, and right cameras
        center_image = np.asarray(Image.open(line[0]))
        left_image = np.asarray(Image.open(line[1]))
        right_image = np.asarray(Image.open(line[2]))

        # Add images to the images array
        images.extend([center_image, left_image, right_image])
        # If multivariant add both the steering angles and throttle values to the mesurements array
        if multivariant:
            measurements.extend([(steering_center_angel, throttle),
                                 (steering_left_angel, throttle),
                                 (steering_right_angel, throttle)])
        else:
            # Add steering angles to the mesurements array
            measurements.extend([steering_center_angel, steering_left_angel, steering_right_angel])
    return images, measurements

#  Keras Data generato


def data_generator1(samples, batch_size, validation=False, multivariant=False):
    """ This method is a Python Generator that takes in a number of samples (lines) and yeilds
        loaded augmented batch sized images and mesurements.
            samples: Array of str arrays (lines) each contain csv line of data
            batch_size: int number of data points to yield for one batch
            validation: Bool to specify whether or not the generator is for trainig or validation, False by default
            multivariant: Bool to specify whether or not the data should be multivariant, False by default
        yields batch_size images and mesurements
    """
    while True:  # Forever loop to keep the generator up till the termination of the program
             # (end of training and validation)
        # shuffling input samples for good measure
        shuffle(samples)
        # Empty arrays for data collection
        X_data = []
        y_data = []
        # Iterates for every sample
        for i, sample in enumerate(samples):
            # Get the samples images, which will return 3 images (center, left, right)
            # and their angles
            sample_images, sample_measurements = get_images_and_measurements([sample], multivariant)
            # Augment sample images flip, but adds shadow to only training data
            augmented_sample_images, augmented_sample_measurements = augment_data(sample_images,
                                                                                  sample_measurements,
                                                                                  augment=not validation,
                                                                                  multivariant=multivariant)
            # Adding our generated sample data into our yield arrays
            X_data.extend(augmented_sample_images)
            y_data.extend(augmented_sample_measurements)
            # print('X_data length: {}'.format(len(X_data)))
            # Check if X is of batch_size or if its the last element
            # Yield if we have collected a batch_size or more (due to concurrent loading) or if its
            # the last batch which will usually be less than batch_size
            if len(X_data) > batch_size or i == len(samples) - 1:
                # print('==================Batch====================')
                # Putting our augmented data into numpy arrays cause Keras require numpy arrays
                # yield the batch
                # Shuffle the batch data for good measure
                yield sklearn.utils.shuffle(np.array(X_data[:batch_size]), np.array(y_data[:batch_size]))
                # Keep any extra data that was loaded but exceded the batch_size for next batch
                X_data = X_data[batch_size:]
                y_data = y_data[batch_size:]


def get_data_generator_and_steps_per_epoch(samples, batch_size, validation=False, multivariant=False):
    """ This method creates a generator and calculates the steps_per_epoch for the generator based
        on images loaded and augmentation applied.
            samples: Array of str arrays (lines) each contain csv line of data
            batch_size: int number of data points to yield for one batch
            validation: Bool to specify whether or not the generator is for trainig or validation, False by default
            multivariant: Bool to specify whether or not the data should be multivariant, False by default
        returns a generator and steps_per_epoch
        """
    # Constants of number of images being loaded
    N_CAMERA_IMAGES = 3
    # Constant of number of augmentation
    N_AUGMENTATION = 1 + 1
    # A generator for the samples given be it a training, validation, or test samples
    generator = data_generator1(
        samples, batch_size, validation=validation, multivariant=multivariant)
    # Calculates the number of images shadow augmentation adds to the data
    shadow_augmentation = int(len(samples) * 3 * 0.3) if not validation else 0
    # Number of batches that the fit_generator() method will accept before declaring on epoch
    print((((len(samples) * N_CAMERA_IMAGES + shadow_augmentation) * N_AUGMENTATION)))
    steps_per_epoch = math.ceil(
        (((len(samples) * N_CAMERA_IMAGES + shadow_augmentation) * N_AUGMENTATION)) / BATCHSIZE)
    return generator, steps_per_epoch


# Hyperparameters that
EPOCHS = 40
BATCHSIZE = 512
DROPOUT = 0.3
# KerasModel training flags
TRACK = 3
MULTIVARIANT = True
GRAY = False
BATCH_NORM = True
LOAD = False
# Load data from csv files (for each track)
lines_1 = load_csv_data('./DrivingData/driving_log.csv')
lines_2 = load_csv_data('./DrivingData_track2/driving_log.csv')
# Empty lines array to be feed for training
lines = []

# Only add track data specified by the KerasModel training flag TRACK
if TRACK == 1:
    # Add track 1 data
    lines.extend(lines_1)
elif TRACK == 2:
    # Add track 2 data
    lines.extend(lines_2)
else:
    # Add the combined track 1 & 2 data
    lines.extend(lines_1)
    lines.extend(lines_2)

print('Total numnber of samples {}'.format(len(lines)))
# Shuffling the samples to get a more random representation before Splitting
shuffle(lines)
# Splitting the data to 80% training and 20% validation
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Training and validation generators
print('Training generator')
train_generator, train_steps_per_epoch = get_data_generator_and_steps_per_epoch(
    train_samples, BATCHSIZE, multivariant=MULTIVARIANT)

print('Validation generator')
validation_generator, validation_steps_per_epoch = get_data_generator_and_steps_per_epoch(
    validation_samples, BATCHSIZE, validation=True, multivariant=MULTIVARIANT)

print('Training steps per epoch {}'.format(train_steps_per_epoch))
print('Validation steps per epoch {}'.format(validation_steps_per_epoch))

# Model file to save to or load from
model_file = 'model_combined_multivariant_50_cropping.h5'
# Initializing a KerasMoel instance
k_model = KerasModel(1,
                     keras_model.NVIDIA_ARCHITECTURE,
                     dropout=DROPOUT,
                     batch_norm=BATCH_NORM,
                     model_file=model_file,
                     multivariant=MULTIVARIANT,
                     gray=GRAY,
                     load=LOAD)
# Training the KerasModel model and getting the metrics
model_history = k_model.train_model_with_generator(train_generator,
                                                   train_steps_per_epoch,
                                                   EPOCHS,
                                                   validation_generator,
                                                   validation_steps_per_epoch,
                                                   save_model_filepath=model_file)

# Training KerasModel for pretrained models
# model_history = k_model.train_learned_model_with_generator(train_generator,
#                                                            train_steps_per_epoch,
#                                                            EPOCHS,
#                                                            validation_generator,
#                                                            validation_steps_per_epoch,
#                                                            save_model_filepath='model_transfer_Inceptionv3.h5')

# Plotting the model Loss
utils.plot_loss(model_history=model_history)


# Uncomment these lines to visualize the model's first and second Convolutions for track 1 and
# track 2 test images
# Track 1 layers visualization
# k_model = KerasModel(load=True, model_file='model.h5')
# test_image = np.asarray(Image.open(
#     './assets/Layer_visualization/Track1/center_2018_05_07_18_39_19_350.jpg'))
# print(np.array(test_image).shape)
# k_model.visualize_layer(test_image, 'Track1 Model First Convolution')
# k_model.visualize_layer(test_image, 'Track1 Model Second Convolution', layer_num=4)


# Track 2 layers visualization
# test_image = np.asarray(Image.open(
#     './assets/Layer_visualization/Track2/center_2018_05_12_06_03_51_508.jpg'))
# print(np.array(test_image).shape)
# k_model.visualize_layer(test_image, 'Track2 Model First Convolution')
# k_model.visualize_layer(test_image, 'Track2 Model Second Convolution', layer_num=4)

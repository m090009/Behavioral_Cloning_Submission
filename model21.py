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
    """ This method is  """
    augmented_images = []  # + images
    augmented_measurements = []  # + measurements
    c = list(zip(images, measurements))
    shuffle(c)
    images, measurements = zip(*c)
    # Applying augmentation on a fraction of the images and their flipped counterparts to prevent
    # steering bias
    n_images_to_augment = int(len(images) * 0.30)
    # For every image and its steering angle flip the image and apply augmentation on shuffled
    # fraction of the images
    for image, steering_angle in zip(images, measurements):
        flipped_image, flipped_steering_angle = flip_image_steering(image,
                                                                    steering_angle,
                                                                    multivariant=multivariant)
        augmented_images.append(flipped_image)
        augmented_images.append(image)
        augmented_measurements.append(flipped_steering_angle)
        augmented_measurements.append(steering_angle)
        if augment and n_images_to_augment > 0:
            shadow_image = add_random_shadow(image)
            flipped_shadow_image = add_random_shadow(flipped_image)
            augmented_images.append(shadow_image)
            augmented_images.append(flipped_shadow_image)
            augmented_measurements.append(steering_angle)
            augmented_measurements.append(flipped_steering_angle)
            n_images_to_augment -= 1
    return augmented_images, augmented_measurements


def flip_image_steering(image, steering_angle, multivariant=False):
    """ This method is  """
    flipped_image = np.fliplr(image)
    if multivariant:
        flipped_steering_angle = steering_angle[0] * -1.0
        return flipped_image, (flipped_steering_angle, steering_angle[1])
    else:
        flipped_steering_angle = steering_angle * -1.0
        return flipped_image, flipped_steering_angle


def bright_dim_image(img):  # This augments lighting conditions
    """ This method is  """
    bright_img = adjust_contrast_brightness(img, alpha=1.1, beta=30)
    dim_img = adjust_contrast_brightness(img, alpha=0.9, beta=-80)
    if img.shape[2] == 1:
        return [np.expand_dims(bright_img, axis=-1),
                np.expand_dims(dim_img, axis=-1)]
    else:
        return [bright_img, dim_img]


def adjust_contrast_brightness(img, alpha=1.1, beta=30):
    """ This class is  """
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)


def add_random_shadow(img):
    """This method adds random shadow to the sides of the road for the selected passed image"""
    height = img.shape[0]
    width = img.shape[1]
    print('{}x{}'.format(width, height))
    rectangle_height = 160
    rectangle_width = 130

    # Random Rectangle.
    y1, x1 = random.randint(70, height - 25), random.randint(0, width)
    y2 = y1 + rectangle_height if (y1 + rectangle_height) < height else y1 - rectangle_height
    x2 = x1 + rectangle_width if (x1 + rectangle_width) < width else x1 - rectangle_width
    print('({}, {}), ({}, {})'.format(x1, y1, x2, y2))
    #

    y_from, y_to = (y1, y2) if y1 < y2 else (y2, y1)
    x_from, x_to = (x1, x2) if x1 < x2 else (x2, x1)
    for i in range(y_from, y_to):
        for j in range(x_from, x_to):
            img.setflags(write=1)
            dark_pixel_value = tuple([int(x * 0.5) for x in img[i][j]])
            img[i][j] = dark_pixel_value
    img.setflags(write=0)
    return img


def load_csv_data(file_path):
    """ This class is  """
    # Reading the recorded data from the .csv file
    lines = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def get_images_and_measurements(lines, multivariant=False):
    """ This class is  """
    images = []
    measurements = []
    # For every line or samples (center, left, right, steering, throttle)
    for line in lines:
        steering_center_angel = float(line[3])
        if multivariant:
            speed = float(line[4])
        # Adjusted Steering angels for side camera images
        correction_factor = 0.2  # should change this to a computed parameter
        steering_left_angel = steering_center_angel + correction_factor
        steering_right_angel = steering_center_angel - correction_factor

        # Read in the images form center, left, and right cameras
        center_image = np.asarray(Image.open(line[0]))
        left_image = np.asarray(Image.open(line[1]))
        right_image = np.asarray(Image.open(line[2]))
        # np.asarray(Image.open(path + row[0]))
        # image = cv2.imread(source_path)

        # Add images and angels to the dataset
        images.extend([center_image, left_image, right_image])
        if multivariant:
            measurements.extend([(steering_center_angel, speed),
                                 (steering_left_angel, speed),
                                 (steering_right_angel, speed)])
        else:
            measurements.extend([steering_center_angel, steering_left_angel, steering_right_angel])
    return images, measurements

#  Keras Data generator


def data_generator1(samples, batch_size, validation=False, multivariant=False):
    """ This class is  """
    # print('here')
    while True:  # Forever loop to keep the generator up till the termination of the program
             # (end of training and inference)
        shuffle(samples)
        X_data = []
        y_data = []
        for i, sample in enumerate(samples):
            # Get the samples images, which will return 3 images (center, left, right)
            # and their angles
            sample_images, sample_measurements = get_images_and_measurements([sample], multivariant)
            # Augment sample images (flip)
            augmented_sample_images, augmented_sample_measurements = augment_data(sample_images,
                                                                                  sample_measurements,
                                                                                  augment=not validation,
                                                                                  multivariant=multivariant)
            # Adding our generated sample data into our yield arrays
            X_data.extend(augmented_sample_images)
            y_data.extend(augmented_sample_measurements)
            # print('X_data length: {}'.format(len(X_data)))
            # Check if X is of batch_size or if its the last element
            if len(X_data) > batch_size or i == len(samples) - 1:
                # print('==================Batch====================')
                # Putting our augmented data into numpy arrays cause Keras require numpy arrays
                # yield the batch
                # Shuffle the batch data for good measure
                yield sklearn.utils.shuffle(np.array(X_data[:batch_size]), np.array(y_data[:batch_size]))
                X_data = X_data[batch_size:]
                y_data = y_data[batch_size:]


N_CAMERA_IMAGES = 3
N_AUGMENTATION = 1 + 1


def get_data_generator_and_steps_per_epoch(samples, batch_size, validation=False, multivariant=False):
    """ This class is  """
    # A generator for the samples given be it a training, validation, or test samples
    generator = data_generator1(
        samples, batch_size, validation=validation, multivariant=multivariant)
    shadow_augmentation = int(len(samples) * 3 * 0.3) if not validation else 0
    # Number of batches that the fit_generator() method will accept before declaring on epoch
    print((((len(samples) * N_CAMERA_IMAGES + shadow_augmentation) * N_AUGMENTATION)))
    steps_per_epoch = math.ceil(
        (((len(samples) * N_CAMERA_IMAGES + shadow_augmentation) * N_AUGMENTATION)) / BATCHSIZE)
    return generator, steps_per_epoch


# Hyperparameters
EPOCHS = 30
BATCHSIZE = 512
TRACK = 3
MULTIVARIANT = False
GRAY = False
BATCH_NORM = True
# Load data from csv file
lines_1 = load_csv_data('./DrivingData/driving_log.csv')
lines_2 = load_csv_data('./DrivingData_track2/driving_log.csv')
lines = []
if TRACK == 1:
    lines.extend(lines_1)
elif TRACK == 2:
    lines.extend(lines_2)
else:
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

model_file = 'model_combined_last_0.2_drop_batch_new_augmentation.h5'
# Initializing a KerasMoel instance
k_model = KerasModel(1,
                     keras_model.NVIDIA_ARCHITECTURE,
                     dropout=0.2,
                     batch_norm=BATCH_NORM,
                     model_file=model_file,
                     multivariant=MULTIVARIANT,
                     gray=GRAY,
                     load=False)
# Training the KerasModel model and getting the metrics
model_history = k_model.train_model_with_generator(train_generator,
                                                   train_steps_per_epoch,
                                                   EPOCHS,
                                                   validation_generator,
                                                   validation_steps_per_epoch,
                                                   save_model_filepath=model_file)
# model_history = k_model.train_learned_model_with_generator(train_generator,
#                                                            train_steps_per_epoch,
#                                                            EPOCHS,
#                                                            validation_generator,
#                                                            validation_steps_per_epoch,
#                                                            save_model_filepath='model_transfer_Inceptionv3.h5')
# Plotting the model Loss
utils.plot_loss(model_history=model_history)

# Track 1 layers visualization
# k_model = KerasModel(load=True, model_file='./models/model_modular_nvidia.h5')
# test_image = np.asarray(Image.open(
#     './assets/Layer_visualization/Track1/center_2018_05_07_18_39_19_350.jpg'))
# print(np.array(test_image).shape)
# k_model.visualize_layer(test_image, 'Track1 Model First Convolution')
# k_model.visualize_layer(test_image, 'Track1 Model Second Convolution', layer_num=4)


# Track 2 layers visualization
# test_image = np.asarray(Image.open(
#     './assets/Layer_visualization/Track2/center_2018_05_12_06_03_51_508.jpg'))
# print(np.array(test_image).shape)
# k_model.load_model('./models/model_track2_modular_nvidia_relu_all.h5')
# k_model.visualize_layer(test_image, 'Track2 Model First Convolution')
# k_model.visualize_layer(test_image, 'Track2 Model Second Convolution', layer_num=4)

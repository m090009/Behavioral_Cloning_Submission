import winsound
import matplotlib.pyplot as plt
import os
import csv


def beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)


def plot_loss(model_history):
    print(model_history.history.keys())
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def load_csv_data(file_path):
    # Reading the recorded data from the .csv file
    lines = []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


def get_imagepaths_and_measurements(lines):
    images = []
    measurements = []

    for line in lines:
        steering_center_angel = float(line[3])

        # Adjusted Steering angels for side camera images
        correction_factor = 0.2  # should change this to a computed parameter
#         steering_left_angel = steering_center_angel + correction_factor
#         steering_right_angel = steering_center_angel - correction_factor

        # Read in the images form center, left, and right cameras
        center_image = line[0]
#         left_image = line[1]
#         right_image = line[2]

        # Add images and angels to the dataset
        images.extend([center_image])  # eft_image, right_image])
        measurements.extend([steering_center_angel])  # teering_left_angel, steering_right_angel])
    return images, measurements


def get_data_count(X_data, y_data, mapping=None):
    # Create a counter to count the occurences of a sign (label)
    values = mapping.values() if mapping else list(set(y_data))
    data_counter = Counter(values)

    # We count each label occurence and store it in our label_counter
    for label in y_data:
        if mapping:
            data_counter[mapping[str(label)]] += 1
        else:
            data_counter[label] += 1
    return data_counter


def draw_labels(labels, mapping=None):
    # Drawing labels and their frequencies
    fig, ax = plt.subplots(figsize=(15, 10))

    print('max: {}, min: {}'.format(np.max(labels), np.min(labels)))
    mu, sigma = 0, 0.1

    # the histogram of the data
    n, bins, patches = plt.hist(labels, 25, facecolor='green', alpha=0.75)

    plt.xlabel('Angles')
    plt.ylabel('Frequency(Counts)')
    plt.title(r'$\mathrm{Histogram\ of\ Cameras\ Angels:}\ \mu=0,\ \sigma=0.1$')
    plt.axis([-1.1, 1.1, 0, 10000])
    plt.grid(True)
    plt.show()

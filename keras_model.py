import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
########## Keras imports ###########
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Input
from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras import backend as K
from keras.callbacks import ModelCheckpoint
# Model architecture constants
# ===Base architectures===
LENET_ARCHITECTURE = 1
NVIDIA_ARCHITECTURE = 2
# ===Pretrained architectures===
VGG_NET = 3
INCEPTION_V3 = 4
RESNET = 5
INCEPTIONRESNET = 6


class KerasModel:
    """ This class deals with creating, loading, and accessing Keras models.
        Here we'll deal with any operation that deals with keras this so we can have a
        clean and extensible code.
    """

    def __init__(self,
                 track=1,
                 architecture=1,
                 dropout=0.2,
                 load=False,
                 model_file=None,
                 batch_norm=False,
                 multivariant=False,
                 gray=False):
        """ This method initializes the KerasModel object with the given flags
                track: int that specifies the track number, 1 by default
                architecture: constant int that specifies the model architecture, 1 by default
                dropout: float that specifies the dropout value, 0.2 by default
                load: bool that specifies if we should load model_file, False by default
                model_file: file to load model from, save trained model to, or both , None by default
                batch_norm: bool which specifies whether or not to use BatchNormalization, False by default
                multivariant: bool which specifies if the model is multivariant, False by default
                gray: bool to indicate the use of Grayscaled images, False by default
        """
        # Setting instance (object) variables
        self.batch_norm = batch_norm
        self.multivariant = multivariant
        self.gray = gray
        self.dropout = dropout

        if not load:
            # Create a Keras Sequential model as self.model
            self.model = Sequential()
            # Apply track preprocessing
            if track == 1:
                self.track1_model_preproceccing()
            else:
                self.track2_model_preproceccing()
            # Apply architecture
            if architecture == NVIDIA_ARCHITECTURE:
                self.use_Nvidia_architecture()
            elif architecture > 2:
                self.use_transfer_learning(architecture)
            else:
                self.use_LeNet_architecture()
        else:
            # Load the model
            if model_file:
                self.model = load_model(model_file)
                print('Succesfully loaded {}'.format(model_file))
            else:
                # Create a new model
                self.model = Sequential()
                print('No model to load, please specify a model_file')

    def load_model(model_file):
        """ This method loads a model of path model_file
                model_file: str of the model file path
        """
        # Loads a keras model
        self.model = load_model(model_file)
        print('Succesfully loaded {}'.format(model_file))

    def track1_model_preproceccing(self):
        """ This method applies track 1 preprocessing by adding preprocessing layers to self.model
        """
        # Grayscale the images if gray is true or don't otherwise
        if self.gray:
            # Adding a Lambda layer with tf.image,rgb_to_grayscale to Grayscale images
            self.model.add(Lambda(lambda x: K.tf.image.rgb_to_grayscale(x),
                                  input_shape=(160, 320, 3)))
            # Normalizing and standardizing our images
            self.model.add(Lambda(lambda x: x / 255.0 - 0.5))
            print("Gray")
        else:
            # Normalizing and standardizing our images
            self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

        # Cropping our images using Cropping2D
        self.model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    def track2_model_preproceccing(self):
        """ This method applies preprocessing to track 2 images by adding layers to the model """
        # Normalizing and standardizing our images
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
        # Cropping our images using Cropping2D
        self.model.add(Cropping2D(cropping=((50, 25), (0, 0))))

    def use_LeNet_architecture(self):
        """
        Keras LeNet Model
        This method applies the LeNet model implementation in Keras
        """
        # First Convolution2D layer with
        self.model.add(Conv2D(6, (5, 5), activation='relu'))
        # MaxPooling2D layer
        self.model.add(MaxPooling2D())
        # Second Convolution2D layer with
        self.model.add(Conv2D(6, (5, 5), activation='relu'))
        # MaxPooling2D layer
        self.model.add(MaxPooling2D())
        # Flattening the Images after the convolutional steps
        self.model.add(Flatten())
        # Fist dense layer
        self.model.add(Dense(120))
        # Second dense layer
        self.model.add(Dense(84))
        if self.multivariant:
            # One regression layer with 2 outputs to predict steering angle and speed
            self.model.add(Dense(2))
        else:
            # One Regression layer with one output for predicting steering angle
            self.model.add(Dense(1))

    def use_Nvidia_architecture(self):
        """ This method applies the modified Nvidia End to End Learning for Self-Driving Cars Model in Keras
            you can find the original Nvidia architecture here
            in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
        """
        # First Convolution2D layer with
        self.model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
        if self.batch_norm:
            # Batch normalization layer to speed up
            self.model.add(BatchNormalization())
        # Second Convolution2D layer with
        self.model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
        if self.batch_norm:
            # Batch normalization layer to speed up
            self.model.add(BatchNormalization())
        # Third Convolution2D layer with
        self.model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
        if self.batch_norm:
            # Batch normalization layer to speed up
            self.model.add(BatchNormalization())
        # Fourth Convolution2D layer with
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        if self.batch_norm:
            # Batch normalization layer to speed up
            self.model.add(BatchNormalization())
        # Fifth Convolution2D layer with
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        if self.batch_norm:
            # Batch normalization layer to speed up
            self.model.add(BatchNormalization())
        # Flattening the Images after the convolutional steps
        self.model.add(Flatten())
        if self.batch_norm:
            # Batch normalization layer to speed up
            self.model.add(BatchNormalization())
        # Fist dense layer fc_1
        self.model.add(Dense(100, activation='relu'))
        # # Dropout layer of drop fraction self.dropout
        # self.model.add(Dropout(self.dropout))
        if self.batch_norm:
            # Batch normalization layer to speed up
            self.model.add(BatchNormalization())
        # Second dense layer fc_2
        self.model.add(Dense(50, activation='relu'))
        if self.batch_norm:
            # Batch normalization layer to speed up
            self.model.add(BatchNormalization())
        # Dropout layer of drop fraction self.dropout
        self.model.add(Dropout(self.dropout))
        # Third dense layer fc_3
        self.model.add(Dense(10, activation='relu'))
        if self.multivariant:
            # One regression layer with 2 outputs to predict steering angle and speed
            self.model.add(Dense(2))
        else:
            # One Regression layer with one output for predicting steering angle
            self.model.add(Dense(1))

    # WIP
    def use_transfer_learning(self, pretrained_networks=VGG_NET):
        """ This method applies transfer learning to the model
            , its still a work in progress but works
                pretrained_networks: int flag specifying the pretrained network to choose from
        """
        # Data input
        # input = Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3))
        # input_tensor = Cropping2D(input_shape=(160, 320, 3), cropping=((70, 25), (0, 0)))
        input_tensor = Input(shape=(160, 320, 3))
        # if pretrained_networks == VGG_NET:
        #     base_model = keras.applications.vgg16.VGG16(
        #         include_top=False, weights='imagenet', input_shape=shape, pooling=True)
        # elif pretrained_networks == INCEPTION_V3:
        base_model = keras.applications.vgg16.VGG16(input_tensor=input_tensor,
                                                    weights='imagenet',
                                                    include_top=False,
                                                    pooling='max')

        # add a global spatial average pooling layer
        x = base_model.output
        # Fully Connected layer
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(10, activation='relu')(x)
        # Regression one output layer
        predictions = Dense(1)(x)

        # this is the model we will train
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze all pretrained Layers weights and biases
        # self.freeze_model_layers(base_model)
        for layer in base_model.layers:
            layer.trainable = False

    def freeze_model_layers(self, model):
        """ This method  freezes a given model layers weights for feature extraction
                model: the model to freeze its weights
        """
        for layer in model.layers:
            # Sets the layer's weights to be un-trainable
            layer.trainable = False

    def train_model_with_generator(self,
                                   train_generator,
                                   steps_per_epoch,
                                   epochs,
                                   validation_generator=None,
                                   validation_steps=None,
                                   save_model_filepath='model.h5'):
        """ This method defines the model training configuration
            via calling the Keras model.compile() method
            that takes the loss function and optimizer type, then
            it calls model.fit_generator() to train the network on the
            given generators. The method also keeps track of the model training metrics
            in model_history then returns it for further analysis
                train_generator: Training data Python generator
                steps_per_epoch: int number of batches that the fit_generator() method will accept before declaring on epoch
                epochs: int number of epochs to train
                validation_generator: Validation data Python generator, with a default argument of None
                validation_steps: int Number of batches that the fit_generator() method will accept before declaring on epoch
                                  , with a default argument of None
                save_model_filepath: h5 model file path to save to , with a default argument of 'model.h5'
        """
        # Defining the loss function and optimizer
        self.model.compile(loss='mse', optimizer='adam',  metrics=['accuracy'])
        # Early stopping callback
        earlyStoppingCallBack = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=3,
                                                              verbose=0,
                                                              mode='auto')
        # Creates a checkpoint and saves it if the val_loss decreased
        checkpointer = ModelCheckpoint(filepath='tmp/best_model.h5',
                                       verbose=1, save_best_only=True)
        # Tensorboard callback
        tbCallBack = keras.callbacks.TensorBoard(
            log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        # Training summary
        print('Started training on ')
        if self.gray:
            print('Gray')
        if self.dropout > 0:
            print('With dropout {}'.format(self.dropout))
        if self.batch_norm:
            print('Batch normalized')
        print('Data')
        # Training the model and getting the model history for future visualization
        model_history = self.model.fit_generator(train_generator,
                                                 steps_per_epoch=steps_per_epoch,
                                                 validation_data=validation_generator,
                                                 validation_steps=validation_steps,
                                                 epochs=epochs,
                                                 verbose=1,
                                                 callbacks=[tbCallBack,
                                                            checkpointer,
                                                            earlyStoppingCallBack])
        # Saving the model to the save file
        self.model.save(save_model_filepath)
        print('Saved model to {}'.format(save_model_filepath))
        # Returning the model history for showing loss graph
        return model_history

    # WIP
    def train_learned_model_with_generator(self,
                                           train_generator,
                                           steps_per_epoch,
                                           epochs,
                                           validation_generator=None,
                                           validation_steps=None,
                                           save_model_filepath='model.h5'):
        """ This method defines the model training configuration
            via calling the Keras model.compile() method
            that takes the loss function and optimizer type, then
            it calls model.fit_generator() to train the network on the
            given generators. The method also keeps track of the model training metrics
            in model_history then returns it for further analysis. This method is only used for applying
            transfer learning its the transfer variant from the original train_model_with_generator method
                train_generator: Training data Python generator
                steps_per_epoch: int number of batches that the fit_generator() method will accept before declaring on epoch
                epochs: int number of epochs to train
                validation_generator: Validation data Python generator, with a default argument of None
                validation_steps: int Number of batches that the fit_generator() method will accept before declaring on epoch
                                  , with a default argument of None
                save_model_filepath: h5 model file path to save to , with a default argument of 'model.h5'
        """
        # Defining the loss function and optimizer
        self.model.compile(loss='mse', optimizer='adam')
        # Tensorboard callback
        tbCallBack = keras.callbacks.TensorBoard(
            log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        model_history = self.model.fit_generator(train_generator,
                                                 steps_per_epoch=steps_per_epoch,
                                                 validation_data=validation_generator,
                                                 validation_steps=validation_steps,
                                                 epochs=epochs,
                                                 verbose=1,
                                                 callbacks=[tbCallBack])
        # Fine tuning which is were we train
        # for layer in self.model.layers[:249]:
        #     layer.trainable = False
        # for layer in self.model.layers[249:]:
        #     layer.trainable = True
        # self.model.compile(loss='mse', optimizer='adam')
        # model_history = self.model.fit_generator(train_generator,
        #                                          steps_per_epoch=steps_per_epoch,
        #                                          validation_data=validation_generator,
        #                                          validation_steps=validation_steps,
        #                                          epochs=epochs,
        #                                          verbose=1,
        #                                          callbacks=[tbCallBack])
        # Saving the model to the save file
        self.model.save(save_model_filepath)
        print('Saved model to {}'.format(save_model_filepath))
        # Returning the model history for showing loss graph
        return model_history

    def get_layer_output(self, activation_image, layer_num):
        """ This method  gets a keras model that cuts the original model and returns
                a specific layer activation output
                    activation_image: numpy array image that activates the model
                    layer_num: int index of the layer to get its activation value
        """
        # Create a new modle with the its output being the layer_num layer output
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.layers[layer_num].output)
        # Get the predicted output
        intermediate_output = intermediate_layer_model.predict([activation_image])
        return intermediate_output

    def visualize_layer(self,
                        input_image,
                        layer_name='Convolution',
                        layer_num=2,
                        activation_min=-1,
                        activation_max=-1,
                        plt_num=1):
        """ This method Visualizes the model layers
                input_image: The input image to run though the models
                layer_name: lyer namem, 'Convolution' by default
                layer_num: Layer index, 2 by default
                activation_min:, -1 by default
                activation_max: , -1 by default
                plt_num: plt number, 1 by default
            The method shows an image of the conv layer's grid of filters actuvation
        """
        # Adding an extra dimetion to make the image array  for the keras model
        input_image = np.expand_dims(input_image, axis=0)
        # Getting the output of layer with model index layer_num
        activation = self.get_layer_output(np.array(input_image), layer_num=layer_num)
        featuremaps = activation.shape[3]
        print(activation.shape)
        plt.figure(plt_num, figsize=(10, 10))
        plt.suptitle(layer_name, fontsize=16)
        gs1 = gridspec.GridSpec(10, 10)
        gs1.update(wspace=0.005, hspace=0.005)
        for featuremap in range(featuremaps):
            cols = 4
            rows = featuremaps / cols
            # sets the number of feature maps to show on each row and column
            plt.subplot(rows, cols, featuremap+1)
            plt.title('' + str(featuremap))  # displays the feature map number
            plt.axis('off')
            if activation_min != -1 & activation_max != -1:
                plt.imshow(activation[0, :, :, featuremap], interpolation="nearest",
                           vmin=activation_min, vmax=activation_max, cmap="gray")
            elif activation_max != -1:
                plt.imshow(activation[0, :, :, featuremap],
                           interpolation="nearest", vmax=activation_max, cmap="gray")
            elif activation_min != -1:
                plt.imshow(activation[0, :, :, featuremap],
                           interpolation="nearest", vmin=activation_min, cmap="gray")
            else:
                plt.imshow(activation[0, :, :, featuremap],
                           interpolation="nearest",
                           cmap="gray",
                           aspect='auto')
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

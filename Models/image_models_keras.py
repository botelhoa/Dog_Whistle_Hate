# -*- coding: utf-8 -*-
"""Image_Models_Keras.ipynb

This py script runs the unimodal image classifier including image resizing. 
It tests InceptionResNet-V2, NASNetLarge, and Xception and is compatible with the Keras pre-trained
CovNet library.

"""

import random
import cv2
import numpy as np
import pandas as pd
import glob
import tensorflow as tf


from PIL import Image
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import SGD, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.applications import InceptionV3, VGG16, DenseNet201, MobileNetV2, ResNet152V2, InceptionResNetV2, NASNetLarge, Xception
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from helperfunctions import *

#Move train data to colab virtual machine to increase speed
!cp -r "/content/drive/My Drive/Dog_Whistle_Code/Data/Train/Train_images" "/content"

#Move dev data to colab virtual machine to increase speed
!cp -r "/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/Validation_images" "/content"

#Move test data to colab virtual machine to increase speed
!cp -r "/content/drive/My Drive/Dog_Whistle_Code/Data/Test/Test_images" "/content"

 

def fine_tune(base_model, train_directory, dev_directory, optimzer, scheduler, image_size: int, num_epochs: int=10, min_delta= 0, patience: int=10, batch_size: int=64, dropout: float=0.5):
    """
    Function for fine tuning pretrained CovNets

    base_model: Type of pre-trained model

    train_directory: Path to parent folder containing train images sub-divided into folders by label names

    dev_directory: Path to parent folder containing dev images sub-divided into folders by label names

    #x_train: Matrix of image train data

    #y_train: One hot encoded vectors of train labels

    #x_dev: Matrix of image dev data

    #y_dev: One hot encoded vectors of dev labels

    optimizer: Algorithim according to which values from the loss function should be back propogated

    scheduler: Callback determining learning rate decay

    image_size (int): Number of pixels in input dimensions (assumes square shape) 

    num_epochs (int): Number of passes through train data. Defaulted to 10.

    min_delta: Minimum reduction in val loss for early stopping

    patience (int): Number of epochs at min_delta before early stopping. Defaulted to 10.

    batch_size (int): Size of mini batches used in training. Defaulted to 64.

    dropout (int): Percentage of nodes turned off in final softmax layer. Defaulted to 0.5.

    """

    #Load data
    train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)

    
    train_generator = train_datagen.flow_from_directory(train_directory,
      target_size=(image_size, image_size),
      classes=["None", "Hateful", "Counter-speech", "Reclaimed"],
      class_mode="categorical",
      batch_size=batch_size,
      shuffle=True,
      seed=22)


    validation_generator = validation_datagen.flow_from_directory(dev_directory,
      target_size=(image_size, image_size),
      classes=["None", "Hateful", "Counter-speech", "Reclaimed"],
      class_mode="categorical",
      batch_size=batch_size,
      shuffle=True,
      seed=22)
    
    #Add Classifier atop pre-trained model

    # global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(dropout)(x)
    # logistic layer
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Train top 5% of pre-trained model layers with top layers
    for layer in model.layers[:int(len(model.layers) * 0.95)]:
        layer.trainable = False

    for layer in model.layers[int(len(model.layers) * 0.95):]:
        layer.trainable = True

    early_stop = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1, mode='auto')
    model.compile(optimizer=optimzer, loss='categorical_crossentropy', metrics=['accuracy']) 
    history = model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator, callbacks=[early_stop, scheduler])
    
    train_dict = {"Train Accuracy": history.history['accuracy'], "Train Loss": history.history['loss'], "Val Accuracy": history.history['val_accuracy'], "Val Loss": history.history['val_loss'] }
    
    return model, train_dict


def test(trained_model, test_directory, image_size: int, batch_size: int=64):
    """
    Outputs metrics, predictions, and labels from image model

    trained_model: Trained image model

    test_directory: Path to parent folder containing test images sub-divided into folders by label names

    image_size (int): Number of pixels in input dimensions (assumes square shape)

    batch_size (int): Size of mini batches used in training. Defaulted to 64.
    """

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(test_directory,
        target_size=(image_size, image_size),
        classes=["None", "Hateful", "Counter-speech", "Reclaimed"],
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=22)

    label_list = []

    for i in range(len(test_generator)):
        for j in range(len(test_generator[i][1])):
            value = np.argmax(test_generator[i][1][j])
            label_list.append(value)
    
    preds = trained_model.predict(test_generator)
    pred_values = tf.argmax(preds, axis=1)

    #label_list = label_list.numpy().tolist()

    results, predictions, labels = metrics(label_list, pred_values)

    return results, predictions, labels



"""
InceptionResNet-V2 paper specifies...
optimizer = RMSProp
epsilon = 1.0
learning_rate =  0.045
exponential decay every 2 epochs at rate of 0.94
image size = (299, 299)

NASNet paper specifies... 
optimizer = RMSProp
epsilon = 1.0
learning_rate =  0.045
drop_out = 0.5
epochs = 20
exponential decay every 2 epochs at rate of 0.9999
image size = (331, 331)

Xception paper specifies... 
optimizer = SGD
learning_rate =  0.045
drop_out = 0.5
decay every 2 epochs at rate of 0.94
Momentum: 0.9
image size = (299, 299)
"""

#Define shared hyperparameters
MIN_DELTA = 0.001
PATIENCE = 3
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.045
DROPOUT = 0.5
EPSILON = 1.0 
TRAIN_LEN = 3998

OUTPUT_DIR = "/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Image"
TRAIN_PATH = "/content/Train_images"
DEV_PATH = "/content/Validation_images"
TEST_PATH_UNIMODAL = "/content/Test_images/Unimodal Labels"
TEST_PATH_MULTIMODAL = "/content/Test_images/Multimodal Labels"


#InceptionResNet-V2 Hyperparameters
IRN_IMAGE_SIZE = 299
IRN_OPTIMIZER = RMSprop(learning_rate = LEARNING_RATE, epsilon=EPSILON)

def IRN_decay(epoch, lr):
    epochs_drop = 2.0
    IRN_DECAY_RATE = 0.94
    lrate = lr * (IRN_DECAY_RATE**((1+epoch)/epochs_drop))
    return lrate

IRN_SCHEDULER = LearningRateScheduler(IRN_decay)


#NASNet Hyperparameters
NN_IMAGE_SIZE = 331
NN_OPTIMIZER = RMSprop(learning_rate= LEARNING_RATE, epsilon=EPSILON)

def NN_decay(epoch, lr):
    epochs_drop = 2.0
    NN_DECAY_RATE = 0.9999
    lrate = lr * (NN_DECAY_RATE**((1+epoch)/epochs_drop))
    return lrate

NN_SCHEDULER = LearningRateScheduler(NN_decay)


#Xception Hyperparametes
X_IMAGE_SIZE = 299
X_OPTIMIZER = SGD(lr= LEARNING_RATE, momentum=0.9)

def X_decay(epoch, lr):
    epochs_drop = 2.0
    X_DECAY_RATE = 0.94
    lrate = lr * (X_DECAY_RATE**((1+epoch)/epochs_drop))
    return lrate

X_SCHEDULER = LearningRateScheduler(X_decay)


#Visualize learning rate decay
# import matplotlib.pyplot as plt

# vals = []
# lr = 0.045
# for i in range(20):
#     val = step_decay(i, lr)
#     vals.append(val)
#     lr = val

# plt.rcParams["figure.figsize"] = (12,6)

# plt.plot(vals)

# plt.xlabel("Epochs")
# plt.ylabel("Learning Rates")
# plt.xticks(np.arange(20))
# plt.legend()

# #plt.savefig("learning_rate_decay.png",bbox_inches='tight')


#Train and Test InceptionResNet-V2
model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(IRN_IMAGE_SIZE, IRN_IMAGE_SIZE, 3))
trained_model, train_dict = fine_tune(model, TRAIN_PATH, DEV_PATH, IRN_OPTIMIZER, IRN_SCHEDULER, IRN_IMAGE_SIZE, NUM_EPOCHS, MIN_DELTA, PATIENCE, BATCH_SIZE, DROPOUT)   
results1, predictions1, labels1 = test(trained_model, TEST_PATH_MULTIMODAL, IRN_IMAGE_SIZE, BATCH_SIZE) 
results2, _, labels2 = test(trained_model, TEST_PATH_UNIMODAL, IRN_IMAGE_SIZE, BATCH_SIZE)
image_model_saver(trained_model, "InceptionResNetV2", OUTPUT_DIR, train_dict, labels1, labels2, predictions1, results1, results2)


#Train and Test NASNet
model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(NN_IMAGE_SIZE, NN_IMAGE_SIZE, 3))
trained_model, train_dict = fine_tune(model, TRAIN_PATH, DEV_PATH, NN_OPTIMIZER, NN_SCHEDULER, NN_IMAGE_SIZE, NUM_EPOCHS, MIN_DELTA, PATIENCE, BATCH_SIZE, DROPOUT)   
results1, predictions1, labels1 = test(trained_model, TEST_PATH_MULTIMODAL, NN_IMAGE_SIZE, BATCH_SIZE) 
results2, _, labels2 = test(trained_model, TEST_PATH_UNIMODAL, NN_IMAGE_SIZE, BATCH_SIZE) 
image_model_saver(trained_model, "NASNet", OUTPUT_DIR, train_dict, labels1, labels2, predictions1, results1, results2)


#Train and Test Xception
model = Xception(weights='imagenet', include_top=False, input_shape=(X_IMAGE_SIZE, X_IMAGE_SIZE, 3))
trained_model, train_dict = fine_tune(model, TRAIN_PATH, DEV_PATH, X_OPTIMIZER, X_SCHEDULER, X_IMAGE_SIZE, NUM_EPOCHS, MIN_DELTA, PATIENCE, BATCH_SIZE, DROPOUT)   
results1, predictions1, labels1 = test(trained_model, TEST_PATH_MULTIMODAL, X_IMAGE_SIZE, BATCH_SIZE) 
results2, _, labels2 = test(trained_model, TEST_PATH_UNIMODAL, X_IMAGE_SIZE, BATCH_SIZE)
image_model_saver(trained_model, "Xception", OUTPUT_DIR, train_dict, labels1, labels2, predictions1, results1, results2)



# -*- coding: utf-8 -*-
"""Image_Models_Keras.ipynb

This py script runs the unimodal image classifier including image resizing. It is compatible with 
InceptionV3, VGG16, DenseNet201, MobileNetV2, ResNET152V2, InceptionResNetV2, NASNetLarge, and Xception.

Credit to https://marubon-ds.blogspot.com/2018/03/some-fine-tuning-models-with-keras.html for some inspiration

"""

pip install import-ipynb

import random
import cv2
import numpy as np

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.applications import InceptionV3, VGG16, DenseNet201, MobileNetV2, ResNet152V2, InceptionResNetV2, NASNetLarge, Xception
from matplotlib import pyplot

# %cd "/content/drive/My Drive/Dog_Whistle_Code"
from HelperFunctions import *


# read data
x_train = #TODO: load data
y_train = #TODO: load data
x_test = #TODO: load data
y_test = #TODO: load data


def resize_data(data):
    """
    Resize data to be compatible with expected model inputs

    data: image data
    """
    data_upscaled = np.zeros((data.shape[0], 320, 320, 3))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled

# resize train and  test data
x_train_resized = resize_data(x_train)
x_test_resized = resize_data(x_test)

#make explained variable hot-encoded
y_train_hot_encoded = to_categorical(y_train)
y_test_hot_encoded = to_categorical(y_test)

def resize_data_NASNet(data): #TODO: check 
    """
    Resizes data to be compatible with NASNet input requirements

    data: image data
    """
    data_upscaled = np.zeros((data.shape[0], 331, 331, 3))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(331, 331), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled

# resize train and  test data
x_train_resized_NASNet = resize_data_NASNet(x_train)
x_test_resized_NASNet = resize_data_NASNet(x_test)

def model(x_train, y_train, base_model, num_epochs: int=10, learning_rate: int=0.0001, min_delta= 0, patience: int=10, batch_size: int=128, val_split: float=0.2):
    """
    Model for extracting image features

    x_train: Matrix of image data

    y_train: One hot encoded vectors of labels

    base_model: Type of pre-trained model

    num_epochs (int): Number of passes through train data. Defaulted to 10.

    learning_rate (int): Step size of loss function. Defaulted to 0.0001

    min_delta: Minimum reduction in val loss for early stopping

    patience (int): Number of epochs at min_delta before early stopping. Defaulted to 10.

    batch_size (int): Size of mini batches used in training. Defaulted to 128.

    val_split (float): Proportion of train data saved for validation. Defaulted to 0.2

    """

    # get layers and add average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # add fully-connected layer
    x = Dense(512, activation='relu')(x)

    # add output layer
    predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze pre-trained model area's layer
    for layer in base_model.layers:
        layer.trainable = False

    # update the weight that are added
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.fit(x_train, y_train)

    # choose the layers which are updated by training
    layer_num = len(model.layers)
    for layer in model.layers[:int(layer_num * 0.9)]:
        layer.trainable = False

    for layer in model.layers[int(layer_num * 0.9):]:
        layer.trainable = True
    
    # update the weights
    early_stop = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1, mode='auto')
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch_size, validation_split= val_split, callbacks=[early_stop])
    train_acc, val_acc, train_loss, val_loss = history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss']  
    train_dict = {"Train Accuracy": train_acc, "Train Loss": train_loss, "Val Accuracy": val_acc, "Val Loss": val_loss}
    return history, train_dict

#Define hyperparameters
MODEL_DICT = {"InceptionV3": InceptionV3, "VGG16": VGG16, "DenseNet201": DenseNet201, "MobileNetV2": MobileNetV2, 
              "ResNet15V2": ResNet152V2, "InceptionResNetV2": InceptionResNetV2, "NASNetLarge": NASNetLarge, "Xception": Xception}
LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
MIN_DELTA = 0.0001
PATIENCE = 5
VAL_SPLIT = 0.2
BATCH_SIZES = [64, 128] #TODO: set batch sizes
NUM_EPOCHS = 50
OUTPUT_DIR = "/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Image"

best_models = {}

for model_selection in MODEL_DICT.keys():
    print("Running {}...".format(model_selection))

    top_f1 = 0

    for bs in BATCH_SIZE:
        for lr in LEARNING_RATE:
            temp_model = MODEL_DICT[model_selection](weights='imagenet', include_top=False)
    
            if model_selection != "NASNetLarge":
                history_temp_model, train_dict = model(x_train_resized, y_train_hot_encoded, temp_model, NUM_EPOCHS, lr, MIN_DELTA, PATIENCE, bs, VAL_SPLIT)
                #evaluation_temp_model = history_temp_model.model.evaluate(x_test_resized, y_test_hot_encoded)
                preds = history_temp_model.model.predict_classes(x_test_resized, verbose=1)
                results, predictions, labels = metrics(y_test, preds)
            else:
                history_temp_model, train_dict = model(x_train_resized_NASNet, y_train_hot_encoded, temp_model,  NUM_EPOCHS, lr, MIN_DELTA, PATIENCE, bs, VAL_SPLIT)
                #evaluation_temp_model = history_temp_model.model.evaluate(x_test_resized_NASNet, y_test_hot_encoded)
                preds = history_temp_model.model.predict_classes(x_test_resized_NASNet, verbose=1)
                results, predictions, labels = metrics(y_test, preds)

            if results["f1"] > top_f1:
                top_f1 = learning_rate_dict[lr]["f1"]
                print("The new top F1 score is: {}. Saving model...".format(top_f1))
                image_save_model(history_temp_model, model_selection, OUTPUT_DIR , train_dict, y_test, preds, results)
                best_models[model_selection] = {"Learning Rate": lr, "Batch_Size": bs, "Results": results} #only save results for best model

#save complete training results
np.save(os.path.join(OUTPUT_DIR, "dogwhistle_total_image_training_results.npy"), results_dict)

#Practice with one iteration
temp_model = NASNetLarge(weights='imagenet', include_top=False)
history_temp_model = model(x_train_resized_NASNet, y_train_hot_encoded, temp_model, NUM_EPOCHS, 0.001, MIN_DELTA, PATIENCE, 64, VAL_SPLIT)
preds = history_temp_model.model.evaluate(x_test_resized_NASNet, y_test_hot_encoded)
results, predictions, labels = metrics(y_test, preds)
image_save_model(history_temp_model, model_selection, OUTPUT_DIR , train_dict, y_test, preds, results)

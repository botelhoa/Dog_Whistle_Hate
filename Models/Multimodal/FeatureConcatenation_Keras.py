#!/usr/bin/env python
# coding: utf-8

"""
This script trains and evaluates an MLP implemeneted in Keras. The model accepts a pre-extracted, concatenated mutlimodal feature vector as its input and outputs a classification decision.
"""


import tensorflow as tf
import pandas as pd
import numpy as np
import functools
import operator

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.models import load_model, Model


#Hyperparamters
DROPOUT = 0.2
HIDDEN_SIZE = 100
BATCH_SIZES = [8, 16, 32]
NUM_LABELS = 4
NUM_EPOCHS = 100
EARLY_STOPPING = {"patience": 3, "delta": 0.005}
LEARNING_RATES = [0.0001, 0.001, 0.01, 0.1]
WEIGHT_DECAY = 0.1 
WARMUP = 0.06 
OUTPUT_DIR = "/content/drive/My Drive/Dog_Whistle_Code/Fine_Tuned_Models/Multimodal/Feature Concatenation"

def decay(epoch, lr):
    epochs_drop = 5
    DECAY_RATE = 0.94
    lrate = lr * (DECAY_RATE**((1+epoch)/epochs_drop))
    return lrate

SCHEDULER = LearningRateScheduler(decay)


# Load data
# train = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/combined_features.csv")
# dev = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/combined_features.csv")
# test = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/combined_features.csv")

train = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Train/combined_features_sabat.csv")
dev = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Validation/combined_features_sabat.csv")
test = pd.read_csv("/content/drive/My Drive/Dog_Whistle_Code/Data/Test/combined_features_sabat.csv")

# Divide labels and features
x_train = train.iloc[:, 1:-2]
y_train = pd.get_dummies(train.loc[:, "labels"])
x_dev = dev.iloc[:, 1:-2]
y_dev = pd.get_dummies(dev.loc[:, "labels"])
x_test = test.iloc[:, 1:-2]
y_test = test.loc[:, "labels"].values.tolist()

def construct_model(MLP_type, hidden_size: int=50, dropout: float=0.2, num_labels: int=4, input_len: int = 5120):
        """Builds the network structure
        image_model: CovNet from Keras library to use as image feature extractor

        text_model: Transformer Model from Hugging Face to use as the text feature extractor

        hidden_size (int): Number of nodes in the hidden layer. Defaulted to 50. 

        dropout (float): Rate at which nodes are deactivated. Defaulted to 0.2. 
        
        num_labels (int): Number of labels to predict. Defaulted to 4.

        input_len (int): Length of input vector. Defaulted to 5120 (Text feature length (4096) + image feature length (1024)).
        """

        if MLP_type == "Sabat":
            model = Sequential()
            model.add(Dense(units=hidden_size, activation='relu',input_dim=input_len))
            model.add(Dropout(0.2))
            model.add(Dense(units=hidden_size, activation='relu',input_dim=hidden_size))
            #model.add(Dropout(0.2))
            model.add(Dense(units=num_labels, activation='softmax', input_dim=hidden_size))

        if MLP_type == "Gomez":
            model = Sequential()
            model.add(Dense(units=input_len, activation='relu',input_dim=input_len))
            model.add(BatchNormalization())
            #model.add(Dropout(0.2))
            model.add(Dense(units=1024, activation='relu',input_dim=input_len))
            model.add(BatchNormalization())
            #model.add(Dropout(0.2))
            model.add(Dense(units=512, activation='relu',input_dim=1024))
            model.add(BatchNormalization())
            #model.add(Dropout(0.2))
            model.add(Dense(units=num_labels, activation='softmax', input_dim=512))
            
        return model


def model_trainer(input_model, x_train, x_test, x_dev, y_dev, early_stop_vals: dict, scheduler, epochs: int = 25, learning_rate: float = 1e-5, batch_size: int=8):   
    """
    Trains multimodal model

    input_model: Instatiation of model

    x_train: Dataframe containing train features

    y_train: Pandas series containing train labels

    x_dev: Dataframe containing validation features

    y_dev: Pandas series containing validation labels

    early_stopping: Dictionary containing patience value (int) and delta value (float). The patience determines the number of epochs to wait to achieve the given delta

    epochs (int): Number of times to run through all batches. Default value is 25.

    learning_rate (float): Default value is 1e-5.

    batch_size (int): Number of examples to be passed through the model at a given time. Defaulted to 8.
    """


    Early_Stop = EarlyStopping(monitor='val_loss', min_delta=early_stop_vals["delta"], patience=early_stop_vals["patience"], verbose=1, mode='auto')
    opt = Adam(learning_rate=learning_rate)
    input_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    history = input_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_dev, y_dev), callbacks=[Early_Stop, scheduler])

    train_dict = {"Train Accuracy": history.history['accuracy'], "Train Loss": history.history['loss'], "Val Accuracy": history.history['val_accuracy'], "Val Loss": history.history['val_loss'] }

    return input_model, train_dict


def model_tester(input_model, x_test, y_test):
    """
    Tests the model's performance based on a several metrics

    input_model: Instatiation of model

    x_test: Dataframe containing test features

    y_test: Pandas series containing test labels
    """

    print('Predicting labels for {} sentences...'.format(len(x_test)))

    preds = input_model.predict(x_test)
    results, labels, predictions = metrics(y_test, preds, argmax_needed=True)

    return results, labels, predictions

# Run Gomez
Keras_Classifier = construct_model("Gomez", HIDDEN_SIZE, DROPOUT, NUM_LABELS, 2048)
trained_model, train_dict = model_trainer(Keras_Classifier, x_train, x_test, x_dev, y_dev, EARLY_STOPPING, SCHEDULER, NUM_EPOCHS, LEARNING_RATES[0], 8)   
results, labels, predictions = model_tester(trained_model, x_test, y_test) 
print(results)

# Run Sabat
Keras_Classifier = construct_model("Sabat", HIDDEN_SIZE, DROPOUT, NUM_LABELS, 2048)
trained_model, train_dict = model_trainer(Keras_Classifier, x_train, x_test, x_dev, y_dev, EARLY_STOPPING, SCHEDULER, NUM_EPOCHS, LEARNING_RATES[0], 25) #Note: lr from paper was LEARNING_RATES[-1]   
results, labels, predictions = model_tester(trained_model, x_test, y_test) 
print(results)

results_dict = {}
max_f1_value = 0

for i in BATCH_SIZES:
    learning_rate_dict = {}
    for j in LEARNING_RATES: 
        Keras_Classifier = construct_model("Sabat", HIDDEN_SIZE, DROPOUT, NUM_LABELS, 2048)
        trained_model, train_dict = model_trainer(Keras_Classifier, x_train, x_test, x_dev, y_dev, EARLY_STOPPING, SCHEDULER, NUM_EPOCHS, j, i) 
        learning_rate_dict[j], labels, predictions = model_tester(trained_model, x_test, y_test) 

    if learning_rate_dict[j]["f1"] >= max_f1_value: #only save best model
        max_f1_value = learning_rate_dict[j]["f1"]
        print("The new top F1 score is: {}. Saving model...".format(max_f1_value))
        model_saver(trained_model, "Sabat", "Keras", OUTPUT_DIR, train_dict, labels, predictions, test.loc[:, "ids"].values.tolist(), learning_rate_dict[j])

    results_dict[i] = learning_rate_dict 

#save complete training results
np.save(os.path.join(os.path.join(OUTPUT_DIR, "Sabat"), "dogwhistle_total_training_results_sabat.npy"), results_dict)

results_dict = {}
max_f1_value = 0

for i in BATCH_SIZES:
    learning_rate_dict = {}
    for j in LEARNING_RATES: 
        Keras_Classifier = construct_model("Gomez", HIDDEN_SIZE, DROPOUT, NUM_LABELS, 2048)
        trained_model, train_dict = model_trainer(Keras_Classifier, x_train, x_test, x_dev, y_dev, EARLY_STOPPING, SCHEDULER, NUM_EPOCHS, j, i) 
        learning_rate_dict[j], labels, predictions = model_tester(trained_model, x_test, y_test) 

    if learning_rate_dict[j]["f1"] >= max_f1_value: #only save best model
        max_f1_value = learning_rate_dict[j]["f1"]
        print("The new top F1 score is: {}. Saving model...".format(max_f1_value))
        model_saver(trained_model, "Gomez", "Keras", OUTPUT_DIR, train_dict, labels, predictions, test.loc[:, "ids"].values.tolist(), learning_rate_dict[j])

    results_dict[i] = learning_rate_dict 

#save complete training results
np.save(os.path.join(os.path.join(OUTPUT_DIR, "Gomez"), "dogwhistle_total_training_results_gomez.npy"), results_dict)


# -*- coding: utf-8 -*-
"""HelperFunctions.ipynb

This is a collection of helper functions used throughout the other scripts
in this repo.

"""

pip install ekphrasis

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# % matplotlib inline
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import matthews_corrcoef, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from tensorflow import keras

def stratify(data, strata_field: str, split_sizes, **kwargs):
    """Stratify and split the data.
    :data (base.DataType): dataset to split.
    :split_sizes (int | base.List[int]): The number of documents in each split.
    :strata_field (str): Name of label field.
    :returns train, dev, test (base.Tuple[list, base.Union[list, None], list]): Return stratified splits.
    """

    train_size = split_sizes[0]

    num_splits = len(split_sizes)
    if num_splits == 1:
        test_size = 1 - split_sizes[0]
    elif num_splits == 2:
        test_size = split_sizes[-1]
    elif num_splits == 3:
        dev_size = split_sizes[1]
        test_size = split_sizes[2]

    stratified_train_list = []
    stratified_dev_list = []
    stratified_test_list = []

    for i in set(data[strata_field]):
        temp_df = data[data[strata_field] == i]

        if dev_size is not None:
            train, temp_test = train_test_split(temp_df, test_size=test_size+dev_size, random_state=42)
            dev, test = train_test_split(temp_test, test_size= (test_size/(test_size+dev_size)), random_state=42)

            stratified_train_list.append(train)
            stratified_dev_list.append(dev)
            stratified_test_list.append(test)

        else:
            train, test = train_test_split()
            stratified_train_list.append(train)
            stratified_test_list.append(test)


    train = pd.concat(stratified_train_list)
    dev = pd.concat(stratified_dev_list)
    test = pd.concat(stratified_test_list)

    train = shuffle(train).reset_index(drop=True)
    dev = shuffle(dev).reset_index(drop=True)
    test = shuffle(test).reset_index(drop=True)

    return train, dev, test

# df = pd.read_csv("", encoding='utf-8')
# train, dev, test = stratify(df, "", [0.8, 0.1, 0.1])


def clean_text(data, normalize_list, annotate_list):
        """
        This function preprocesses the text using the Ekphrasis library
        
        data: Pandas series object containing strings of text

        normalize_list: list of data features to clean

        annotate_list: list of data features to annotate
        """

        text_processor = TextPreProcessor(
            normalize= normalize_list,
            annotate= annotate_list,
            fix_html=True,
            segmenter="twitter", 
            unpack_hashtags=True,  
            unpack_contractions=True,  
            spell_correct_elong=True,  
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        )

        clean_data = data.map(lambda x: " ".join(text_processor.pre_process_doc(x)))

        return clean_data


def early_stopping(val_loss_values, early_stop_vals):
    """
    Determines whether or not the model will keep running based on the patience and delta given relative to the val loss
    """
    if len(val_loss_values) > early_stop_vals["patience"]:
        if val_loss_values[-1] <= np.mean(np.array(val_loss_values[-1-self.early_stop_vals["patience"]:-1])) - early_stop_vals["delta"]:
        return False
      else:
        return True
    else:
        return False


def training_plot(train_loss_values, val_loss_values):
    """
    Plots loss after each epoch

    training_loss_values: list of floats; output from fine_tune function

    val_loss_values: list of floats; output from fine_tune function
    """
    sns.set(style='darkgrid')
    plt.rcParams["figure.figsize"] = (12,6)

    plt.plot(train_loss_values, 'b-o', label="train")
    plt.plot(val_loss_values, 'g-o', label="valid")

    #plt.title("Training and Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    #plt.savefig("dogwhistle_train_plot.png",bbox_inches='tight')

    return plt.show()


def metrics(labels, preds, argmax_needed: bool = False):
    """
    Returns the Matthew's correlation coefficient, accuracy rate, true positive rate, true negative rate, false positive rate, false negative rate, precission, recall, and f1 score
    
    labels: list of correct labels

    pred: list of model predictions
    """
    labels = labels
    preds = preds

    if argmax_needed == True:
        preds = np.argmax(preds, axis=1).flatten()

    mcc = matthews_corrcoef(labels, preds)
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    f1 = f1_score(labels, preds, average= "weighted")
    precision = precision_score(labels, preds, average= "weighted")
    recall = f1_score(labels, preds, average= "weighted")

    results = {
        "mcc": mcc,
        "acc": acc,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    return results, labels, preds


def model_saver(model, model_type, output_directory, training_dict, labels, preds, results, tokenizer= None):
    """
    Saves PyTorch Model and other outputs

    model: Model to be saved
    
    model_type (string): Name of model
    
    output_directory: Directory to folder to save file in

    training_dict: Dictionary of training and validation values 

    labels: List of labels for test set

    preds: List of model predictions after passed through argmax()

    results: Dictionary of metrics

    tokenizer: Tokenizer to be saved. Defaulted to None.
    """

    output_directory = os.path.join(output_directory, model_type)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    np.save(model_type+"_dogwhistle_train_results.npy", training_dict) #save training dict
    np.save(model_type+"_dogwhistle_test_results.npy", results) #save test metrics
    
    test_predictions = pd.DataFrame([labels, preds]) #save predictions and labels 
    test_predictions = test_predictions.T
    test_predictions = test_predictions.rename(columns={0: 'Labels', 1: 'Predictions'})
    test_predictions.to_csv(model_type+"_dogwhistle_predictions.csv")

    #save models
    if model_type in ["LSTM", "resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"]:
        torch.save(model.state_dict(), file_name+"_model")

    if model_type in ["AlBERT", "BART", "BERT", "DistilBERT", "RoBERTa", "XLNet"]:
        model_to_save = model.module if hasattr(model, 'module') else model 
        model_to_save.save_pretrained(output_directory) 
        tokenizer.save_pretrained(output_directory)

    return print("Saving complete.")


def image_model_saver(image_model, model_type, output_directory, training_dict, labels, preds, results):
    """
    Saves Keras image model and other outputs

    iamge_model: Image model to be saved
    
    model_type (string): Name of model
    
    output_directory: Directory to folder to save file in

    training_dict: Dictionary of training and validation values 

    labels: List of labels for test set

    preds: List of model predictions after passed through argmax()

    results: Dictionary of metrics

    tokenizer: Tokenizer to be saved. Defaulted to None.
    """
    output_directory = os.path.join(output_directory, model_type)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    np.save(model_type+"_dogwhistle_train_results.npy", training_dict) #save training dict
    np.save(model_type+"_dogwhistle_test_results.npy", results) #save test metrics
    
    test_predictions = pd.DataFrame([labels, preds]) #save predictions and labels #TODO: verify this works, attach image_id
    test_predictions = test_predictions.T
    test_predictions.to_csv(model_type+"_dogwhistle_predictions.csv")

    image_model.save("image_model") #save model

    return print("Saving complete.")


def confusion_matrix_plotter(results, save_name, x_tick_labels, y_tick_labels, color):
    """
    Plots confusion matrix

    results: dictionary item containing key of confusion matrix (output of sklearn confusion_matrix())

    x_tick_labels: list of tick labels on x axis

    y_tick_labels: list of tick labels on y axis

    save_name: name of image to be produced with .png

    color: color selection from matplotlib cmap choices: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    """

    df_cm = pd.DataFrame(results.ravel()[0]["confusion_matrix"])

    fig = plt.figure(figsize=(16, 12))
    
    plt.rc('axes', labelsize=14)  
    plt.rc('xtick', labelsize=12)   
    plt.rc('ytick', labelsize=12)       

    #plt.subplot(2, 2, 1)
    g1 = sns.heatmap(df_cm, annot=True, fmt='g', cmap=color)
    g1.set_xlabel('Predicted Label')
    g1.set_ylabel('True Label', rotation=0) 
    g1.xaxis.set_ticklabels(x_tick_labels, rotation=0) 
    g1.yaxis.set_ticklabels(y_tick_labels, rotation=0) 

    plt.show()

    return fig.savefig(save_name,bbox_inches='tight')

